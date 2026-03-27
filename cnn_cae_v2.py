"""
EMG 手势识别 - CNN + CAE (改进版)
PyTorch + CUDA 版本
支持新旧数据集
"""

import numpy as np
import pandas as pd
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 50)
print("EMG Gesture Classification - CNN + CAE")
print(f"Device: {device}")
print("=" * 50)

# ============ 配置 ============
DATA_SOURCE = 'both'  # 可选: 'old', 'new', 'both'

OLD_DATA_PATH = './EMG_data'
OLD_SUBJECTS = ['EMG-S1', 'EMG-S2', 'EMG-S3', 'EMG-S4', 'EMG-S5',
                'EMG-S6', 'EMG-S7', 'EMG-S8', 'EMG-S9', 'EMG-S10']
OLD_GESTURE_FILES = {
    'HC': 'HC-{}.csv', 'I': 'I-I{}.csv', 'L': 'L-L{}.csv',
    'M': 'M-M{}.csv', 'R': 'R-R{}.csv', 'TI': 'T-I{}.csv',
    'TL': 'T-L{}.csv', 'TM': 'T-M{}.csv', 'TR': 'T-R{}.csv', 'TT': 'T-T{}.csv',
}

NEW_DATA_PATH = '../Ninapro_DB1/Ninapro_DB1'
NEW_SUBJECTS = [f'S{i}' for i in range(1, 28)]
EXERCISE = 1

WINDOW_SIZE = 500
STEP = 250
BATCH_SIZE = 128
EPOCHS_CAE = 200
EPOCHS_CNN = 200
CAE_LR = 0.005


def extract_freq_features(sig):
    """提取频域特征"""
    fft_vals = np.fft.fft(sig)
    fft_freq = np.fft.fftfreq(len(sig))
    magnitude = np.abs(fft_vals)
    pos_mask = fft_freq > 0
    magnitude_pos = magnitude[pos_mask]
    freq_pos = fft_freq[pos_mask]

    if len(magnitude_pos) == 0:
        return [0] * 6

    spectral_energy = np.sum(magnitude_pos ** 2) / len(magnitude_pos)
    peak_idx = np.argmax(magnitude_pos)
    peak_freq = freq_pos[peak_idx] if peak_idx < len(freq_pos) else 0
    spectral_mean = np.mean(magnitude_pos)
    spectral_std = np.std(magnitude_pos)
    dominant_ratio = np.max(magnitude_pos) / (np.sum(magnitude_pos) + 1e-10)
    mag_norm = magnitude_pos / (np.sum(magnitude_pos) + 1e-10)
    spectral_entropy = -np.sum(mag_norm * np.log(mag_norm + 1e-10))

    return [spectral_mean, spectral_std, spectral_energy, peak_freq, dominant_ratio, spectral_entropy]


def extract_features(window, n_channels):
    """提取特征"""
    features = []
    for ch in range(n_channels):
        sig = window[:, ch]
        time_features = [
            np.mean(sig), np.std(sig), np.sqrt(np.mean(sig**2)),
            np.max(np.abs(sig)), np.mean(np.abs(sig)), np.var(sig),
            np.sum(np.diff(np.sign(sig)) != 0) / len(sig),
            (np.max(sig) - np.min(sig)), np.sum(sig**2) / len(sig),
            np.percentile(sig, 25), np.percentile(sig, 50), np.percentile(sig, 75),
        ]
        freq_features = extract_freq_features(sig)
        features.extend(time_features)
        features.extend(freq_features)
    return np.array(features)


def load_old_data():
    print("\n[OLD] Loading data...")
    all_samples, all_labels = [], []
    gesture_list = list(OLD_GESTURE_FILES.keys())

    for gesture_idx, gesture_name in enumerate(gesture_list):
        for subject in OLD_SUBJECTS:
            for i in range(1, 7):
                filepath = os.path.join(OLD_DATA_PATH, subject, OLD_GESTURE_FILES[gesture_name].format(i))
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, header=None)
                    signal = df.values
                    n_windows = (len(signal) - WINDOW_SIZE) // STEP + 1
                    for w in range(n_windows):
                        window = signal[w * STEP:w * STEP + WINDOW_SIZE]
                        all_samples.append(extract_features(window, 2))
                        all_labels.append(gesture_idx)

    X = np.array(all_samples, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"Old data: {X.shape}, classes: {len(np.unique(y))}")
    return X, y


def load_ninapro_data(exercise=1):
    print(f"\n[NEW] Loading Ninapro Exercise {exercise}...")
    gesture_range = range(1, 13) if exercise == 1 else (range(1, 18) if exercise == 2 else range(1, 24))

    all_samples, all_labels = [], []
    for subject in NEW_SUBJECTS:
        filepath = os.path.join(NEW_DATA_PATH, f'{subject}_A1_E{exercise}.mat')
        if not os.path.exists(filepath):
            continue

        mat = sio.loadmat(filepath)
        emg, stimulus = mat['emg'], mat['stimulus'].flatten()

        for gesture_id in gesture_range:
            gesture_mask = stimulus == gesture_id
            if not np.any(gesture_mask):
                continue
            gesture_data = emg[gesture_mask]
            n_windows = (len(gesture_data) - WINDOW_SIZE) // STEP + 1
            for w in range(n_windows):
                window = gesture_data[w * STEP:w * STEP + WINDOW_SIZE]
                all_samples.append(extract_features(window, 10))
                all_labels.append(gesture_id - 1)

    X = np.array(all_samples, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"Ninapro data: {X.shape}, classes: {len(np.unique(y))}")
    return X, y


def load_combined_data():
    X_old, y_old = load_old_data()
    X_new, y_new = load_ninapro_data(EXERCISE)

    # 扩展旧数据到180维
    X_old_expanded = np.zeros((X_old.shape[0], 180), dtype=np.float32)
    for ch in range(2):
        for feat in range(18):
            for new_ch in range(ch * 5, (ch + 1) * 5):
                X_old_expanded[:, new_ch * 18 + feat] = X_old[:, ch * 18 + feat]

    n_old_classes = len(np.unique(y_old))
    y_new_offset = y_new + n_old_classes

    print(f"\nCombined: old {y_old.shape[0]}, new {y_new.shape[0]}")
    X = np.vstack([X_old_expanded, X_new])
    y = np.hstack([y_old, y_new_offset])
    print(f"Combined: {X.shape}, classes: {len(np.unique(y))}")
    return X, y


# ============ 模型 ============

class ImprovedCAE(nn.Module):
    def __init__(self, img_h, img_w):
        super(ImprovedCAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_encoded(self, x):
        return self.encoder(x)


class CNNClassifier(nn.Module):
    def __init__(self, in_channels=128, feat_h=15, feat_w=12, num_classes=10):
        super(CNNClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AugmentedDataset(torch.utils.data.Dataset):
    """带数据增强的数据集"""
    def __init__(self, features, labels, img_h, img_w, augment=True):
        self.features = features
        self.labels = labels
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def augment_feature(self, x):
        """数据增强"""
        # x shape: (C, H, W)
        if not self.augment:
            return x

        # 1. 随机噪声注入
        if np.random.random() < 0.3:
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        # 2. 随机幅度缩放
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        # 3. 随机通道丢弃 (类似Dropout)
        if np.random.random() < 0.2:
            channel_mask = torch.rand(x.shape[0], device=x.device) > 0.2
            x = x * channel_mask.view(-1, 1, 1)

        # 4. 随机水平翻转
        if np.random.random() < 0.3:
            x = torch.flip(x, dims=[2])  # 翻转W维度

        return x

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.augment:
            x = self.augment_feature(x.clone())

        return x, y


# ============ 主程序 ============
print("\n[1] Loading data...")

if DATA_SOURCE == 'old':
    X, y = load_old_data()
    img_h, img_w = 6, 6
elif DATA_SOURCE == 'new':
    X, y = load_ninapro_data(EXERCISE)
    img_h, img_w = 15, 12
else:
    X, y = load_combined_data()
    img_h, img_w = 15, 12

n_classes = len(np.unique(y))
print(f"Data: {X.shape}, Classes: {n_classes}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_img = X_train.reshape(-1, 1, img_h, img_w)
X_test_img = X_test.reshape(-1, 1, img_h, img_w)

X_train_tensor = torch.FloatTensor(X_train_img).to(device)
X_test_tensor = torch.FloatTensor(X_test_img).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

# ============ 训练CAE ============
print("\n[2] Training CAE...")

cae = ImprovedCAE(img_h, img_w).to(device)
optimizer_ae = optim.Adam(cae.parameters(), lr=CAE_LR)
scheduler_ae = optim.lr_scheduler.StepLR(optimizer_ae, step_size=50, gamma=0.5)

for epoch in range(EPOCHS_CAE):
    cae.train()
    total_loss = 0
    for batch_x, _ in train_loader:
        optimizer_ae.zero_grad()
        output = cae(batch_x)
        loss = nn.MSELoss()(output, batch_x)
        loss.backward()
        optimizer_ae.step()
        total_loss += loss.item()
    scheduler_ae.step()

    if (epoch + 1) % 20 == 0:
        print(f"CAE Epoch [{epoch+1}/{EPOCHS_CAE}], Loss: {total_loss/len(train_loader):.6f}")

# 获取编码特征
cae.eval()
with torch.no_grad():
    X_train_encoded = cae.get_encoded(X_train_tensor)
    X_test_encoded = cae.get_encoded(X_test_tensor)

print(f"Encoded shape: {X_train_encoded.shape}")

# ============ 训练CNN ============
print("\n[3] Training CNN...")

in_channels = X_train_encoded.shape[1]
cnn = CNNClassifier(in_channels=in_channels, feat_h=img_h, feat_w=img_w, num_classes=n_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)
scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='max', factor=0.5, patience=10, min_lr=1e-6)

# 使用数据增强
augmented_train_dataset = AugmentedDataset(X_train_encoded, y_train_tensor, img_h, img_w, augment=True)
encoded_train_loader = DataLoader(augmented_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
encoded_test_loader = DataLoader(TensorDataset(X_test_encoded, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)

best_acc = 0
best_model_state = None

for epoch in range(EPOCHS_CNN):
    cnn.train()
    for batch_x, batch_y in encoded_train_loader:
        optimizer_cnn.zero_grad()
        output = cnn(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer_cnn.step()

    cnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in encoded_test_loader:
            outputs = cnn(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_acc = 100 * correct / total
    scheduler_cnn.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = cnn.state_dict().copy()

    if (epoch + 1) % 20 == 0:
        print(f"CNN Epoch [{epoch+1}/{EPOCHS_CNN}], Val Acc: {val_acc:.2f}%")

if best_model_state:
    cnn.load_state_dict(best_model_state)

# ============ 结果 ============
print("\n" + "=" * 50)
print("[4] Results")
print("=" * 50)

cnn.eval()
with torch.no_grad():
    train_pred = torch.argmax(cnn(X_train_encoded), dim=1).cpu().numpy()
    test_pred = torch.argmax(cnn(X_test_encoded), dim=1).cpu().numpy()

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Train Acc: {train_acc*100:.2f}%")
print(f"Test Acc: {test_acc*100:.2f}%")

# 保存
print("\n[5] Saving...")
model_prefix = f'{"combined" if DATA_SOURCE == "both" else DATA_SOURCE}_e{EXERCISE}'
torch.save(cae.state_dict(), f'autoencoder_{model_prefix}.pth')
torch.save(cnn.state_dict(), f'cnn_classifier_{model_prefix}.pth')
print("Done!")
