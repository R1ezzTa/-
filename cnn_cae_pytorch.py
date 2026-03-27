"""
CNN + CAE EMG 手势识别 - PyTorch版本
支持老数据集(2ch CSV)和Ninapro_DB1(10ch MAT)
滑动窗口 + 多数投票
"""

import numpy as np
import pandas as pd
import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
class Config:
    DATASET_MODE = 'legacy'  # 'legacy', 'ninapro'

    # 老数据集
    LEGACY_DATA_PATH = '../EMG_data'
    LEGACY_SUBJECTS = [f'EMG-S{i}' for i in range(1, 9)]
    LEGACY_GESTURE_FILES = {
        'HC': 'HC-{}.csv', 'I': 'I-I{}.csv', 'L': 'L-L{}.csv',
        'M': 'M-M{}.csv', 'R': 'R-R{}.csv', 'TI': 'T-I{}.csv',
        'TL': 'T-L{}.csv', 'TM': 'T-M{}.csv', 'TR': 'T-R{}.csv',
        'TT': 'T-T{}.csv',
    }

    # Ninapro
    NINAPRO_PATH = '../Ninapro_DB1/Ninapro_DB1'
    NINAPRO_SUBJECTS = [f'S{i}' for i in range(1, 28)]  # S1-S27
    NINAPRO_EXERCISE = 1
    NINAPRO_GESTURES = {1: 12, 2: 17, 3: 23}

    # 训练
    WINDOW_SIZE = 200
    STEP_SIZE = 100
    BATCH_SIZE = 256
    EPOCHS = 80
    LEARNING_RATE = 0.001


# ============ 模型 ============
class CAE(nn.Module):
    def __init__(self, in_channels=1, feature_dim=32):
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
            nn.Dropout(0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class CNNAutoencoderClassifier(nn.Module):
    def __init__(self, in_channels=1, feature_dim=32, num_classes=10):
        super(CNNAutoencoderClassifier, self).__init__()
        self.cae = CAE(in_channels, feature_dim)
        # Classifier接受CAE的32通道特征输出
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(feature_dim * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.cae.encoder(x)  # (batch, 32, H, W)
        return self.fc(features)


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


def load_legacy_data(window_size=200, step=100):
    """加载老数据集(2ch) - 时频特征"""
    print("[*] 加载老数据集...")
    all_windows, all_labels = [], []

    for gesture_idx, (gesture_name, pattern) in enumerate(Config.LEGACY_GESTURE_FILES.items()):
        for subject in Config.LEGACY_SUBJECTS:
            for i in range(1, 7):
                filepath = os.path.join(Config.LEGACY_DATA_PATH, subject, pattern.format(i))
                if not os.path.exists(filepath):
                    continue
                df = pd.read_csv(filepath, header=None)
                signal = df.values.astype(np.float32)
                n_windows = (len(signal) - window_size) // step + 1
                for w in range(n_windows):
                    all_windows.append(signal[w*step:w*step+window_size])
                    all_labels.append(gesture_idx)

    windows = np.stack(all_windows)  # (n, 200, 2)
    labels = np.array(all_labels, dtype=np.int32)
    time_feat = extract_time_features(windows)  # (n, 24)
    freq_feat = extract_freq_features(windows)  # (n, 16)
    features = np.hstack([time_feat, freq_feat])  # (n, 40)
    return features, labels, 10


def load_ninapro_data(exercise=1, window_size=200, step=100, max_per_label=800):
    """加载Ninapro_DB1 - 时频特征"""
    print(f"[*] 加载Ninapro E{exercise}...")
    all_windows, all_labels = [], []
    num_classes = Config.NINAPRO_GESTURES[exercise]

    for subject in Config.NINAPRO_SUBJECTS:
        filepath = os.path.join(Config.NINAPRO_PATH, f'{subject}_A1_E{exercise}.mat')
        if not os.path.exists(filepath):
            continue

        mat = scipy.io.loadmat(filepath)
        emg = mat['emg'].astype(np.float32)
        stimulus = mat['stimulus'].ravel()

        for label in range(1, num_classes + 1):
            indices = np.where(stimulus == label)[0]
            if len(indices) == 0:
                continue

            # 随机采样
            if len(indices) > max_per_label:
                indices = np.random.choice(indices, max_per_label, replace=False)

            for idx in indices:
                start = idx
                end = idx + window_size
                if end <= len(emg):
                    all_windows.append(emg[start:end])
                    all_labels.append(label - 1)

        print(f"    {subject} loaded")

    windows = np.stack(all_windows)  # (n, 200, 10)
    labels = np.array(all_labels, dtype=np.int32)
    time_feat = extract_time_features(windows)  # (n, 120)
    freq_feat = extract_freq_features(windows)  # (n, 80)
    features = np.hstack([time_feat, freq_feat])  # (n, 200)
    return features, labels, num_classes


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
def train_cae(model, train_loader, device, epochs=30):
    print("[*] CAE预训练...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.cae.parameters(), lr=Config.LEARNING_RATE)

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


def train_classifier(model, train_loader, test_loader, device, epochs=80):
    print("[*] 训练分类器...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
        scheduler.step(val_acc)

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
    print("CNN + CAE EMG Recognition - PyTorch")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 设备: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    if Config.DATASET_MODE == 'legacy':
        X, y, num_classes = load_legacy_data(Config.WINDOW_SIZE, Config.STEP_SIZE)
        img_h, img_w = 5, 8  # 40 = 5*8
    else:
        X, y, num_classes = load_ninapro_data(Config.NINAPRO_EXERCISE,
                                              Config.WINDOW_SIZE, Config.STEP_SIZE)
        img_h, img_w = 10, 20  # 200 = 10*20

    print(f"[*] 数据: {X.shape}, 类别: {num_classes}")
    print(f"[*] 标签分布: {np.bincount(y)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_img = reshape_for_cnn(X, img_h, img_w)
    print(f"[*] CNN输入: {X_img.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_img, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)

    model = CNNAutoencoderClassifier(
        in_channels=1, feature_dim=32, num_classes=num_classes
    ).to(device)

    train_cae(model, train_loader, device, epochs=30)
    best_acc = train_classifier(model, train_loader, test_loader, device, epochs=Config.EPOCHS)

    model.load_state_dict(torch.load('best_model.pth'))
    acc = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print(f"数据集: {Config.DATASET_MODE}")
    print(f"最佳准确率: {best_acc:.2f}%")
    print("=" * 60)

    torch.save(model.state_dict(), 'emg_cnn_cae_pytorch.pth')
    print("[*] 模型已保存")


if __name__ == '__main__':
    main()
