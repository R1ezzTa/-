"""
1D CNN EMG 手势识别 - PyTorch版本
直接在原始EMG信号上跑卷积，不提取特征
支持老数据集(2ch CSV)和Ninapro_DB1(10ch MAT)
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
    DATASET_MODE = 'ninapro'  # 'legacy', 'ninapro'

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
    NINAPRO_SUBJECTS = [f'S{i}' for i in range(1, 28)]
    NINAPRO_EXERCISE = 2
    NINAPRO_GESTURES = {1: 12, 2: 17, 3: 23}

    # 训练
    WINDOW_SIZE = 200
    STEP_SIZE = 100
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 0.001


# ============ 1D CNN 模型 ============
class CNN1D(nn.Module):
    def __init__(self, n_channels=2, window_size=200, num_classes=10):
        super(CNN1D, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 200 -> 100

            # Conv block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 50

            # Conv block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 50 -> 25

            # Conv block 4
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # 固定到4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, n_channels, window_size)
        x = self.features(x)
        return self.classifier(x)


# ============ 数据加载 ============
def load_legacy_data(window_size=200, step=100):
    """加载老数据集(2ch)"""
    print("[*] 加载老数据集...")
    all_signals = []
    all_labels = []

    for gesture_idx, (gesture_name, pattern) in enumerate(Config.LEGACY_GESTURE_FILES.items()):
        for subject in Config.LEGACY_SUBJECTS:
            for i in range(1, 7):
                filepath = os.path.join(Config.LEGACY_DATA_PATH, subject, pattern.format(i))
                if not os.path.exists(filepath):
                    continue
                df = pd.read_csv(filepath, header=None)
                signal = df.values.astype(np.float32)  # (n, 2)

                n_windows = (len(signal) - window_size) // step + 1
                for w in range(n_windows):
                    all_signals.append(signal[w*step:w*step+window_size].T)  # (2, 200)
                    all_labels.append(gesture_idx)

    signals = np.stack(all_signals)  # (n, 2, 200)
    labels = np.array(all_labels, dtype=np.int32)
    return signals, labels, 10


def load_ninapro_data(exercise=1, window_size=200, step=100, max_per_label=500):
    """加载Ninapro_DB1"""
    print(f"[*] 加载Ninapro E{exercise}...")
    all_signals = []
    all_labels = []
    num_classes = Config.NINAPRO_GESTURES[exercise]

    for subject in Config.NINAPRO_SUBJECTS:
        filepath = os.path.join(Config.NINAPRO_PATH, f'{subject}_A1_E{exercise}.mat')
        if not os.path.exists(filepath):
            continue

        mat = scipy.io.loadmat(filepath)
        emg = mat['emg'].astype(np.float32)  # (n_samples, 10)
        stimulus = mat['stimulus'].ravel()

        for label in range(1, num_classes + 1):
            indices = np.where(stimulus == label)[0]
            if len(indices) == 0:
                continue

            if len(indices) > max_per_label:
                indices = np.random.choice(indices, max_per_label, replace=False)

            for idx in indices:
                start = idx
                end = idx + window_size
                if end <= len(emg):
                    all_signals.append(emg[start:end].T)  # (10, 200)
                    all_labels.append(label - 1)

        print(f"    {subject} loaded")

    signals = np.stack(all_signals)  # (n, 10, 200)
    labels = np.array(all_labels, dtype=np.int32)
    return signals, labels, num_classes


# ============ 训练 ============
def train_model(model, train_loader, test_loader, device, epochs=100):
    print("[*] 训练...")
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
            torch.save(model.state_dict(), 'best_cnn1d.pth')

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
    print("1D CNN EMG Recognition - PyTorch")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 设备: {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    if Config.DATASET_MODE == 'legacy':
        X, y, num_classes = load_legacy_data(Config.WINDOW_SIZE, Config.STEP_SIZE)
        n_channels = 2
    else:
        X, y, num_classes = load_ninapro_data(Config.NINAPRO_EXERCISE,
                                              Config.WINDOW_SIZE, Config.STEP_SIZE)
        n_channels = 10

    print(f"[*] 数据: {X.shape}, 类别: {num_classes}")
    print(f"[*] 标签分布: {np.bincount(y)}")

    # 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)

    model = CNN1D(n_channels=n_channels, window_size=Config.WINDOW_SIZE,
                  num_classes=num_classes).to(device)

    best_acc = train_model(model, train_loader, test_loader, device, epochs=Config.EPOCHS)

    model.load_state_dict(torch.load('best_cnn1d.pth'))
    acc = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print(f"数据集: {Config.DATASET_MODE}")
    print(f"最佳准确率: {best_acc:.2f}%")
    print("=" * 60)

    torch.save(model.state_dict(), 'cnn1d_emg.pth')
    print("[*] 模型已保存")


if __name__ == '__main__':
    main()
