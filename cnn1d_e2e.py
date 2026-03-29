"""
端到端1D CNN EMG识别 - 直接在原始EMG信号上训练
使用连续段滑动窗口 + repetition划分
"""

import numpy as np
import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class Config:
    # 数据
    NINAPRO_PATH = 'F:/zju/emg_recognition/Ninapro_DB1/Ninapro_DB1'
    NINAPRO_SUBJECTS = [f'S{i}' for i in range(1, 23)]  # S1-S22
    NINAPRO_EXERCISE = 3  # E3=23类
    NINAPRO_GESTURES = {1: 12, 2: 17, 3: 23}

    # 训练
    WINDOW_SIZE = 200
    STEP_SIZE = 50
    BATCH_SIZE = 256
    EPOCHS = 300
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4  # 增强正则化
    LABEL_SMOOTHING = 0.1

    # repetition划分
    TEST_REPETITIONS = [9, 10]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1, stride=stride) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.se = SEBlock(out_ch)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return torch.relu(out)


class CNN1D(nn.Module):
    def __init__(self, n_channels=10, window_size=200, num_classes=23):
        super(CNN1D, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            nn.MaxPool1d(2),  # 200 -> 100
            nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128),
            ResBlock(128, 128),
            nn.MaxPool1d(2),  # 100 -> 50
            nn.Dropout(0.15),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256),
            ResBlock(256, 256),
            nn.MaxPool1d(2),  # 50 -> 25
            nn.Dropout(0.2),
        )
        self.layer4 = nn.Sequential(
            ResBlock(256, 512),
            ResBlock(512, 512),
            nn.AdaptiveAvgPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)


def load_ninapro_data(exercise=3, window_size=200, step=50):
    """加载Ninapro - 用连续段做滑动窗口"""
    print(f"[*] 加载Ninapro E{exercise}...")
    all_signals, all_labels, all_reps, all_subjects = [], [], [], []
    num_classes = Config.NINAPRO_GESTURES[exercise]

    for subject in Config.NINAPRO_SUBJECTS:
        filepath = os.path.join(Config.NINAPRO_PATH, f'{subject}_A1_E{exercise}.mat')
        if not os.path.exists(filepath):
            continue

        mat = scipy.io.loadmat(filepath)
        emg = mat['emg'].astype(np.float32)  # (n_samples, 10)
        stimulus = mat['stimulus'].ravel()
        repetition = mat['repetition'].ravel()

        for label in range(1, num_classes + 1):
            mask = stimulus == label
            diff = np.diff(mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
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
                    all_signals.append(emg[ws:ws+window_size].T)  # (10, 200)
                    all_labels.append(label - 1)
                    all_reps.append(rep)
                    all_subjects.append(subject)

        print(f"    {subject} loaded")

    signals = np.stack(all_signals)  # (n, 10, 200)
    labels = np.array(all_labels, dtype=np.int32)
    reps = np.array(all_reps, dtype=np.int32)
    return signals, labels, reps, np.array(all_subjects)


def augment(batch_x):
    """数据增强: 噪声 + 缩放"""
    noise = torch.randn_like(batch_x) * 0.01
    batch_x = batch_x + noise
    scale = torch.FloatTensor(batch_x.size(0), 1, 1).uniform_(0.9, 1.1).to(batch_x.device)
    return batch_x * scale


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, test_loader, device, epochs=200):
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = augment(batch_x)
            # Mixup
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4)
            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            # Mixup准确率按加权计算
            total += batch_y.size(0)
            correct += (lam * predicted.eq(y_a).float() + (1 - lam) * predicted.eq(y_b).float()).sum().item()

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
            torch.save(model.state_dict(), 'best_1d_e2e.pth')

        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Best: {best_acc:.2f}%")

    return best_acc


def main():
    print("=" * 60)
    print("端到端1D CNN - Ninapro E3")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 设备: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    signals, labels, reps, subjects = load_ninapro_data(
        Config.NINAPRO_EXERCISE, Config.WINDOW_SIZE, Config.STEP_SIZE
    )
    num_classes = Config.NINAPRO_GESTURES[Config.NINAPRO_EXERCISE]
    print(f"[*] 数据: {signals.shape}, 类别: {num_classes}")

    # 先划分再标准化, 避免数据泄露
    n_samples, n_channels, window_size = signals.shape
    test_mask = np.isin(reps, Config.TEST_REPETITIONS)
    train_mask = ~test_mask
    X_train_raw, X_test_raw = signals[train_mask], signals[test_mask]
    y_train, y_test = labels[train_mask], labels[test_mask]
    # 按通道标准化
    for ch in range(n_channels):
        mean = X_train_raw[:, ch, :].mean()
        std = X_train_raw[:, ch, :].std() + 1e-8
        X_train_raw[:, ch, :] = (X_train_raw[:, ch, :] - mean) / std
        X_test_raw[:, ch, :] = (X_test_raw[:, ch, :] - mean) / std
    print(f"[*] 训练集: {X_train_raw.shape}, 测试集: {X_test_raw.shape}")
    print(f"[*] 标签分布: {np.bincount(y_train)}")

    train_ds = TensorDataset(torch.FloatTensor(X_train_raw), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test_raw), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)

    model = CNN1D(
        n_channels=n_channels,
        window_size=Config.WINDOW_SIZE,
        num_classes=num_classes
    ).to(device)

    best_acc = train(model, train_loader, test_loader, device, epochs=Config.EPOCHS)

    model.load_state_dict(torch.load('best_1d_e2e.pth', weights_only=True))
    model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_list.extend(batch_y.numpy())

    acc = accuracy_score(all_labels_list, all_preds)
    print(f"\n[*] 最终测试准确率: {acc*100:.2f}%")

    print(f"\n{'='*60}")
    print(f"数据集: Ninapro E3 ({num_classes}类), 划分: repetition")
    print(f"最佳准确率: {best_acc:.2f}%")
    print(f"{'='*60}")

    torch.save(model.state_dict(), 'cnn1d_e2e_ninapro_e3.pth')
    print("[*] 模型已保存: cnn1d_e2e_ninapro_e3.pth")


if __name__ == '__main__':
    main()
