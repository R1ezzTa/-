"""
改进版 CNN + Convolutional Autoencoder (CAE) 手势识别
PyTorch + CUDA 版本
支持新旧数据集:
- 旧数据集: EMG_data目录, CSV格式, 2通道, 10类手势
- 新数据集: Ninapro_DB1目录, MAT格式, 10通道, 12/17/23类手势
"""

import numpy as np
import pandas as pd
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置设备 (优先使用CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 50)
print("Improved CNN + CAE EMG Gesture Classification")
print(f"PyTorch Version - Device: {device}")
print("=" * 50)

# ============ 配置 ============
# 数据源: 'old' = EMG_data目录, 'new' = Ninapro_DB1目录
DATA_SOURCE = 'both'  # 可选: 'old', 'new', 'both'

# 旧数据集配置
OLD_DATA_PATH = './EMG_data'  # EMG_CNN_CAE_Package/EMG_data
OLD_SUBJECTS = ['EMG-S1', 'EMG-S2', 'EMG-S3', 'EMG-S4', 'EMG-S5',
                'EMG-S6', 'EMG-S7', 'EMG-S8', 'EMG-S9', 'EMG-S10']
OLD_GESTURE_FILES = {
    'HC': 'HC-{}.csv',
    'I': 'I-I{}.csv',
    'L': 'L-L{}.csv',
    'M': 'M-M{}.csv',
    'R': 'R-R{}.csv',
    'TI': 'T-I{}.csv',
    'TL': 'T-L{}.csv',
    'TM': 'T-M{}.csv',
    'TR': 'T-R{}.csv',
    'TT': 'T-T{}.csv',
}

# 新数据集配置
NEW_DATA_PATH = '../Ninapro_DB1/Ninapro_DB1'  # 父目录下的Ninapro_DB1
NEW_SUBJECTS = [f'S{i}' for i in range(1, 28)]  # S1-S27
EXERCISE = 1  # 使用哪个Exercise: 1=12类, 2=17类, 3=23类

# 通用配置
WINDOW_SIZE = 500
STEP = 250
BATCH_SIZE = 128
EPOCHS_CAE = 50
EPOCHS_CNN = 100


def extract_features_old(window):
    """从窗口提取多尺度特征 (旧数据集, 2通道)"""
    features = []
    for ch in range(2):
        sig = window[:, ch]
        features.extend([
            np.mean(sig),
            np.std(sig),
            np.sqrt(np.mean(sig**2)),  # RMS
            np.max(np.abs(sig)),
            np.mean(np.abs(sig)),
            np.var(sig),
            np.sum(np.diff(np.sign(sig)) != 0) / len(sig),  # ZCC
            (np.max(sig) - np.min(sig)),
            np.sum(sig**2) / len(sig),  # 能量
            np.percentile(sig, 25),
            np.percentile(sig, 50),
            np.percentile(sig, 75),
        ])
    return np.array(features)  # 24 features


def extract_features_new(window):
    """从窗口提取多尺度特征 (新数据集, 10通道)"""
    features = []
    for ch in range(10):
        sig = window[:, ch]
        features.extend([
            np.mean(sig),
            np.std(sig),
            np.sqrt(np.mean(sig**2)),  # RMS
            np.max(np.abs(sig)),
            np.mean(np.abs(sig)),
            np.var(sig),
            np.sum(np.diff(np.sign(sig)) != 0) / len(sig),  # ZCC
            (np.max(sig) - np.min(sig)),
            np.sum(sig**2) / len(sig),  # 能量
            np.percentile(sig, 25),
            np.percentile(sig, 50),
            np.percentile(sig, 75),
        ])
    return np.array(features)  # 120 features


def load_old_data():
    """加载旧数据集 (CSV格式)"""
    print("\n[OLD] 加载旧数据集...")
    all_samples = []
    all_labels = []
    gesture_list = list(OLD_GESTURE_FILES.keys())

    for gesture_idx, gesture_name in enumerate(gesture_list):
        for subject in OLD_SUBJECTS:
            file_pattern = OLD_GESTURE_FILES[gesture_name]
            for i in range(1, 7):
                filepath = os.path.join(OLD_DATA_PATH, subject, file_pattern.format(i))
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, header=None)
                    signal = df.values  # (20000, 2)

                    n_windows = (len(signal) - WINDOW_SIZE) // STEP + 1
                    for w in range(n_windows):
                        start = w * STEP
                        window = signal[start:start + WINDOW_SIZE]
                        feat = extract_features_old(window)
                        all_samples.append(feat)
                        all_labels.append(gesture_idx)

    X = np.array(all_samples, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"旧数据形状: {X.shape}, 手势类别: {len(np.unique(y))}")
    return X, y


def load_ninapro_data(exercise=1):
    """加载Ninapro数据集 (MAT格式)"""
    print(f"\n[NEW] 加载Ninapro_DB1 Exercise {exercise}...")

    if exercise == 1:
        max_gesture = 12
        gesture_range = range(1, 13)
    elif exercise == 2:
        max_gesture = 17
        gesture_range = range(1, 18)
    else:
        max_gesture = 23
        gesture_range = range(1, 24)

    all_samples = []
    all_labels = []
    label_offset = 0

    for subject in NEW_SUBJECTS:
        filepath = os.path.join(NEW_DATA_PATH, f'{subject}_A1_E{exercise}.mat')
        if not os.path.exists(filepath):
            continue

        mat = sio.loadmat(filepath)
        emg = mat['emg']
        stimulus = mat['stimulus'].flatten()

        for gesture_id in gesture_range:
            gesture_mask = stimulus == gesture_id
            if not np.any(gesture_mask):
                continue

            gesture_data = emg[gesture_mask]
            n_windows = (len(gesture_data) - WINDOW_SIZE) // STEP + 1
            for w in range(n_windows):
                start = w * STEP
                window = gesture_data[start:start + WINDOW_SIZE]
                feat = extract_features_new(window)
                all_samples.append(feat)
                all_labels.append(label_offset + gesture_id - 1)

        label_offset += max_gesture

    X = np.array(all_samples, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"Ninapro数据形状: {X.shape}, 手势类别: {len(np.unique(y))}")
    return X, y


def load_combined_data():
    """加载并合并新旧数据集"""
    X_old, y_old = load_old_data()
    X_new, y_new = load_ninapro_data(EXERCISE)

    # 旧数据(47400, 24) -> 扩展到120维以匹配新数据
    # 24 = 2ch * 12feat, 120 = 10ch * 12feat
    # 把2通道扩展为10通道（每通道特征复制5份）
    X_old_expanded = np.zeros((X_old.shape[0], 120), dtype=np.float32)
    for ch in range(2):
        for feat in range(12):
            for new_ch in range(ch * 5, (ch + 1) * 5):
                X_old_expanded[:, new_ch * 12 + feat] = X_old[:, ch * 12 + feat]

    print(f"\n合并数据: 旧数据{y_old.shape[0]}样本(扩展后{X_old_expanded.shape}), 新数据{y_new.shape[0]}样本")
    X = np.vstack([X_old_expanded, X_new])
    y = np.hstack([y_old, y_new])

    print(f"合并后形状: {X.shape}, 总手势类别: {len(np.unique(y))}")
    return X, y


# ============ PyTorch 模型定义 ============

class ConvAutoencoder(nn.Module):
    """卷积自编码器"""
    def __init__(self, img_h, img_w):
        super(ConvAutoencoder, self).__init__()
        self.img_h = img_h
        self.img_w = img_w

        # 编码器: kernel=3, padding=1 保持尺寸不变
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_encoded(self, x):
        return self.encoder(x)


class CNNClassifier(nn.Module):
    """CNN分类器"""
    def __init__(self, in_channels=32, feat_h=10, feat_w=12, num_classes=10):
        super(CNNClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 计算flatten后的尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, feat_h, feat_w)
            flatten_size = self.features(dummy).flatten().shape[0]
            print(f"CNN flatten size: {flatten_size}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============ 1. 加载数据 ============
print("\n[1] 加载数据...")

if DATA_SOURCE == 'old':
    X, y = load_old_data()
    img_h, img_w = 4, 6
elif DATA_SOURCE == 'new':
    X, y = load_ninapro_data(EXERCISE)
    img_h, img_w = 10, 12
else:  # 'both'
    X, y = load_combined_data()
    # 120维特征 -> 10x12 或 12x10
    img_h, img_w = 10, 12

print(f"原始数据形状: {X.shape}, 标签形状: {y.shape}")
print(f"类别数: {len(np.unique(y))}")

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============ 2. 重塑为CNN输入格式 ============
n_classes = len(np.unique(y))

X_train_img = X_train.reshape(-1, img_h, img_w, 1)
X_test_img = X_test.reshape(-1, img_h, img_w, 1)

print(f"图像格式数据: {X_train_img.shape}")

# 转换为PyTorch格式 (batch, channel, height, width)
X_train_tensor = torch.FloatTensor(X_train_img).permute(0, 3, 1, 2).to(device)
X_test_tensor = torch.FloatTensor(X_test_img).permute(0, 3, 1, 2).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# One-hot编码
y_train_onehot = torch.FloatTensor(
    np.eye(n_classes)[y_train]
).to(device)
y_test_onehot = torch.FloatTensor(
    np.eye(n_classes)[y_test]
).to(device)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============ 3. 训练CAE进行无监督预训练 ============
print("\n[2] 训练卷积自编码器...")

cae = ConvAutoencoder(img_h, img_w).to(device)
criterion = nn.MSELoss()
optimizer_ae = optim.Adam(cae.parameters(), lr=0.001)

print(f"CAE模型:\n{cae}")

for epoch in range(EPOCHS_CAE):
    cae.train()
    total_loss = 0
    for batch_x, _ in train_loader:
        optimizer_ae.zero_grad()
        output = cae(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer_ae.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS_CAE}], Loss: {total_loss/len(train_loader):.6f}")

# 获取编码特征
print("\n提取编码特征...")
cae.eval()
with torch.no_grad():
    X_train_encoded = cae.get_encoded(X_train_tensor)
    X_test_encoded = cae.get_encoded(X_test_tensor)

print(f"编码特征形状: {X_train_encoded.shape}")

# 调整编码特征图尺寸用于CNN
enc_h, enc_w = X_train_encoded.shape[2], X_train_encoded.shape[3]
in_channels = X_train_encoded.shape[1]

# 创建基于编码特征的DataLoader
encoded_train_dataset = TensorDataset(X_train_encoded, y_train_tensor)
encoded_test_dataset = TensorDataset(X_test_encoded, y_test_tensor)
encoded_train_loader = DataLoader(encoded_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
encoded_test_loader = DataLoader(encoded_test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============ 4. CNN分类器 ============
print("\n[3] 训练CNN分类器...")

cnn = CNNClassifier(in_channels=in_channels, feat_h=enc_h, feat_w=enc_w, num_classes=n_classes).to(device)

cnn = cnn.to(device)
criterion_clf = nn.CrossEntropyLoss()
optimizer_clf = optim.Adam(cnn.parameters(), lr=0.001)

print(f"CNN分类器:\n{cnn}")

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_clf, mode='max', factor=0.5, patience=3, min_lr=1e-6
)
best_acc = 0
best_model_state = None

for epoch in range(EPOCHS_CNN):
    cnn.train()
    for batch_x, batch_y in encoded_train_loader:
        optimizer_clf.zero_grad()
        output = cnn(batch_x)
        loss = criterion_clf(output, batch_y)
        loss.backward()
        optimizer_clf.step()

    # 验证
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
    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = cnn.state_dict().copy()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS_CNN}], Val Acc: {val_acc:.2f}%")

# 加载最佳模型
if best_model_state is not None:
    cnn.load_state_dict(best_model_state)


# ============ 5. 评估 ============
print("\n" + "=" * 50)
print("[4] 评估结果")
print("=" * 50)

cnn.eval()
with torch.no_grad():
    train_outputs = cnn(X_train_encoded)
    test_outputs = cnn(X_test_encoded)

    train_pred = torch.argmax(train_outputs, dim=1).cpu().numpy()
    test_pred = torch.argmax(test_outputs, dim=1).cpu().numpy()

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"训练准确率: {train_acc*100:.2f}%")
print(f"测试准确率: {test_acc*100:.2f}%")


# ============ 6. 保存 ============
print("\n[5] 保存模型...")
model_prefix = f'{"combined" if DATA_SOURCE == "both" else DATA_SOURCE}_e{EXERCISE}'
torch.save(cae.state_dict(), f'autoencoder_{model_prefix}.pth')
torch.save(cnn.state_dict(), f'cnn_classifier_{model_prefix}.pth')
print("模型已保存!")
print("完成!")
