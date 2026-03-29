"""
CNN + CAE EMG 手势识别 - PyTorch版本
支持老数据集(2ch CSV)和Ninapro_DB1(10ch MAT)
滑动窗口 + 按repetition划分 + LOSO交叉验证
"""

import numpy as np
import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
class Config:
    DATASET_MODE = 'ninapro'  # 'ninapro' (EMG已抛弃)
    SPLIT_MODE = 'repetition'  # 'repetition' or 'loso'

    # Ninapro
    NINAPRO_PATH = 'F:/zju/emg_recognition/Ninapro_DB1/Ninapro_DB1'
    NINAPRO_SUBJECTS = [f'S{i}' for i in range(1, 23)]  # S1-S22
    NINAPRO_EXERCISE = 3  # E3=23类手势
    NINAPRO_GESTURES = {1: 12, 2: 17, 3: 23}

    # 训练
    WINDOW_SIZE = 200
    STEP_SIZE = 50
    BATCH_SIZE = 256
    EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # repetition划分: 每个手势重复10次, 用rep 9,10做测试
    TEST_REPETITIONS = [9, 10]


# ============ 模型 ============
class CAE(nn.Module):
    def __init__(self, in_channels=1, feature_dim=64):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, feature_dim, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, in_channels, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CNNClassifier(nn.Module):
    def __init__(self, in_channels=1, feature_dim=32, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class CNNAutoencoderClassifier(nn.Module):
    def __init__(self, in_channels=1, feature_dim=64, num_classes=10):
        super(CNNAutoencoderClassifier, self).__init__()
        self.cae = CAE(in_channels, feature_dim)
        # Classifier: 先用额外卷积层处理CAE特征，再用FC
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.cae.encoder(x)  # (batch, 64, H, W)
        return self.classifier(features)


# ============ 数据加载 ============
def extract_freq_features(emg_windows):
    """频域特征提取: (n_windows, n_channels, window_size) -> (n_windows, n_channels*8)"""
    n_windows, n_channels, window_size = emg_windows.shape
    features = []
    for ch in range(n_channels):
        sig = emg_windows[:, ch, :]  # (n_windows, window_size)

        # FFT
        fft_vals = np.fft.rfft(sig, axis=1)  # 只取正频率
        fft_mag = np.abs(fft_vals)  # 幅度谱
        fft_phase = np.angle(fft_vals)

        # 功率谱
        power = fft_mag ** 2

        # 频段划分 (假设归一化频率 0-0.5)
        n_freq = power.shape[1]
        low = n_freq // 4
        mid = n_freq // 2

        # 各频段能量
        low_power = np.sum(power[:, :low], axis=1)
        mid_power = np.sum(power[:, low:mid], axis=1)
        high_power = np.sum(power[:, mid:], axis=1)
        total_power = np.sum(power[:, 1:], axis=1) + 1e-10  # 去掉DC

        # 谱质心 (spectral centroid)
        freqs = np.arange(n_freq)
        centroid = np.sum(fft_mag[:, 1:] * freqs[1:], axis=1) / (np.sum(fft_mag[:, 1:], axis=1) + 1e-10)

        # 峰值频率索引
        peak_idx = np.argmax(fft_mag[:, 1:], axis=1) + 1

        # 平均频率
        mean_freq = np.sum(fft_mag[:, 1:] * freqs[1:], axis=1) / (np.sum(fft_mag[:, 1:], axis=1) + 1e-10)

        features.extend([
            low_power / total_power,
            mid_power / total_power,
            high_power / total_power,
            centroid,
            peak_idx.astype(np.float32),
            np.mean(fft_mag[:, 1:], axis=1),
            np.std(fft_mag[:, 1:], axis=1),
            np.max(fft_mag[:, 1:], axis=1),
        ])

    return np.stack(features, axis=1).astype(np.float32)  # (n_windows, n_channels*8)


def extract_time_features(emg_windows):
    """时域特征: (n_windows, n_channels, window_size) -> (n_windows, n_channels*12)"""
    n_windows, n_channels, window_size = emg_windows.shape
    features = []
    for ch in range(n_channels):
        sig = emg_windows[:, ch, :]
        features.extend([
            np.mean(sig, axis=1),
            np.std(sig, axis=1),
            np.sqrt(np.mean(sig**2, axis=1)),
            np.max(np.abs(sig), axis=1),
            np.mean(np.abs(sig), axis=1),
            np.var(sig, axis=1),
            np.sum(np.diff(np.sign(sig), axis=1) != 0, axis=1) / window_size,
            np.max(sig, axis=1) - np.min(sig, axis=1),
            np.sum(sig**2, axis=1) / window_size,
            np.percentile(sig, 25, axis=1),
            np.percentile(sig, 50, axis=1),
            np.percentile(sig, 75, axis=1),
        ])
    return np.stack(features, axis=1).astype(np.float32)



def load_ninapro_data(exercise=3, window_size=200, step=50):
    """加载Ninapro_DB1 - 用连续段做滑动窗口，保留repetition/subject信息"""
    print(f"[*] 加载Ninapro E{exercise}...")
    all_windows, all_labels, all_reps, all_subjects = [], [], [], []
    num_classes = Config.NINAPRO_GESTURES[exercise]

    for subject in Config.NINAPRO_SUBJECTS:
        filepath = os.path.join(Config.NINAPRO_PATH, f'{subject}_A1_E{exercise}.mat')
        if not os.path.exists(filepath):
            continue

        mat = scipy.io.loadmat(filepath)
        emg = mat['emg'].astype(np.float32)
        stimulus = mat['stimulus'].ravel()
        repetition = mat['repetition'].ravel()

        for label in range(1, num_classes + 1):
            mask = stimulus == label
            # 找连续段: 每个连续段是同一手势的一次执行
            diff = np.diff(mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            # 处理边界
            if mask[0]:
                starts = np.concatenate([[0], starts])
            if mask[-1]:
                ends = np.concatenate([ends, [len(mask)]])

            for seg_start, seg_end in zip(starts, ends):
                seg_len = seg_end - seg_start
                if seg_len < window_size:
                    continue
                rep = int(repetition[seg_start])
                n_win = (seg_len - window_size) // step + 1
                for w in range(n_win):
                    ws = seg_start + w * step
                    all_windows.append(emg[ws:ws+window_size])
                    all_labels.append(label - 1)
                    all_reps.append(rep)
                    all_subjects.append(subject)

        print(f"    {subject} loaded")

    windows = np.stack(all_windows).transpose(0, 2, 1)  # (n, 10, 200)
    labels = np.array(all_labels, dtype=np.int32)
    reps = np.array(all_reps, dtype=np.int32)
    time_feat = extract_time_features(windows)
    freq_feat = extract_freq_features(windows)
    features = np.hstack([time_feat, freq_feat])
    return features, labels, reps, np.array(all_subjects)



def reshape_for_cnn(X, img_h=10, img_w=20):
    """特征重塑为CNN格式"""
    n_features = img_h * img_w
    if X.shape[1] > n_features:
        X = X[:, :n_features]
    elif X.shape[1] < n_features:
        pad = np.zeros((X.shape[0], n_features - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    return X.reshape(-1, 1, img_h, img_w)




# ============ 训练 ============
def train_cae(model, train_loader, device, epochs=50):
    print("[*] CAE预训练...")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.cae.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model.cae(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


def augment(batch_x):
    """数据增强: 噪声 + 幅度缩放"""
    # 随机噪声
    noise = torch.randn_like(batch_x) * 0.02
    batch_x = batch_x + noise
    # 随机幅度缩放
    scale = torch.FloatTensor(batch_x.size(0), 1, 1, 1).uniform_(0.9, 1.1).to(batch_x.device)
    batch_x = batch_x * scale
    return batch_x


def train_classifier(model, train_loader, test_loader, device, epochs=80):
    print("[*] 训练分类器...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = augment(batch_x)  # 数据增强
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        val_acc = 100. * correct / total
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Best: {best_acc:.2f}%")

    return best_acc


def evaluate(model, test_loader, device):
    print("[*] 评估...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"    准确率: {acc*100:.2f}%")
    return acc


# ============ 主函数 ============
def main():
    print("=" * 60)
    print("CNN + CAE EMG Recognition - PyTorch (Ninapro E3)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 设备: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    X, y, reps, subjects = load_ninapro_data(
        Config.NINAPRO_EXERCISE, Config.WINDOW_SIZE, Config.STEP_SIZE
    )
    num_classes = Config.NINAPRO_GESTURES[Config.NINAPRO_EXERCISE]
    print(f"[*] 数据: {X.shape}, 类别: {num_classes}")
    print(f"[*] 标签分布: {np.bincount(y)}")

    # MinMaxScaler归一化到0-1 (CAE sigmoid)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_img = reshape_for_cnn(X, 10, 20)
    print(f"[*] CNN输入: {X_img.shape}")

    if Config.SPLIT_MODE == 'loso':
        # Leave-One-Subject-Out
        unique_subjects = np.unique(subjects)
        all_accs = []
        for test_subj in unique_subjects:
            print(f"\n--- LOSO: 测试 {test_subj} ---")
            test_mask = subjects == test_subj
            train_mask = ~test_mask
            X_train, X_test = X_img[train_mask], X_img[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(X_test) == 0 or len(np.unique(y_test)) < 2:
                print(f"    跳过 {test_subj} (数据不足)")
                continue

            train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)

            model = CNNAutoencoderClassifier(
                in_channels=1, feature_dim=64, num_classes=num_classes
            ).to(device)
            train_cae(model, train_loader, device, epochs=100)
            best_acc = train_classifier(model, train_loader, test_loader, device, epochs=Config.EPOCHS)
            all_accs.append(best_acc)
            print(f"    {test_subj}: {best_acc:.2f}%")

        print(f"\n[*] LOSO 平均准确率: {np.mean(all_accs):.2f}% (+/- {np.std(all_accs):.2f}%)")
    else:
        # 按repetition划分
        test_mask = np.isin(reps, Config.TEST_REPETITIONS)
        train_mask = ~test_mask
        X_train, X_test = X_img[train_mask], X_img[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print(f"[*] 训练集: {X_train.shape}, 测试集: {X_test.shape}")

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)

        model = CNNAutoencoderClassifier(
            in_channels=1, feature_dim=64, num_classes=num_classes
        ).to(device)

        train_cae(model, train_loader, device, epochs=100)
        best_acc = train_classifier(model, train_loader, test_loader, device, epochs=Config.EPOCHS)

        model.load_state_dict(torch.load('best_model.pth', weights_only=True))
        evaluate(model, test_loader, device)

        print(f"\n{'='*60}")
        print(f"数据集: Ninapro E3 ({num_classes}类), 划分: {Config.SPLIT_MODE}")
        print(f"最佳准确率: {best_acc:.2f}%")
        print(f"{'='*60}")

        suffix = f'_ninapro_e3_{Config.SPLIT_MODE}'
        torch.save(model.state_dict(), f'emg_cnn_cae_pytorch{suffix}.pth')
        print(f"[*] 模型已保存: emg_cnn_cae_pytorch{suffix}.pth")


if __name__ == '__main__':
    main()
