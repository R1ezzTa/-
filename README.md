# EMG 手势识别 - CNN + CAE

基于卷积自编码器(CNN + CAE)的肌电(EMG)手势识别模型，支持新旧两种数据集。

## 项目结构

```
EMG_CNN_CAE_Package/
├── cnn_cae_v2.py      # 主程序 (PyTorch + CUDA)
├── README.md          # 本文件
├── requirements.txt   # 依赖列表
└── EMG_data/          # 旧数据集 (CSV格式, 2通道, 10类)
```

## 数据集

### 旧数据集 (EMG_data)
- **路径**: `./EMG_data`
- **格式**: CSV文件，每行2列（2通道EMG信号）
- **采样率**: ~2000Hz
- **样本**: EMG-S1 ~ EMG-S10 (每个受试者)
- **手势**: 10类 (HC, I, L, M, R, TI, TL, TM, TR, TT)
- **每类动作**: 6次重复

### 新数据集 (Ninapro_DB1)
- **路径**: `../Ninapro_DB1/Ninapro_DB1/` (父目录)
- **格式**: .mat文件 (scipy.io.loadmat可读取)
- **通道数**: 10通道
- **受试者**: S1 ~ S27
- **Exercise**:
  - E1: 12类手势 (标签1-12)
  - E2: 17类手势 (标签1-17)
  - E3: 23类手势 (标签1-23)

## 环境配置

```bash
# 创建conda环境 (推荐)
conda create -n emg_gesture python=3.9 -y
conda activate emg_gesture

# 安装依赖
pip install torch torchvision scikit-learn numpy scipy pandas
```

或直接使用pip安装：
```bash
pip install -r requirements.txt
```

## 使用方法

### 配置

在 `cnn_cae_v2.py` 顶部修改配置：

```python
# 数据源选择
DATA_SOURCE = 'both'  # 可选: 'old', 'new', 'both'

# 新数据集使用哪个Exercise
EXERCISE = 1  # 1=12类, 2=17类, 3=23类
```

### 运行

```bash
# 激活环境
conda activate emg_gesture

# 运行
python cnn_cae_v2.py
```

### 输出

训练完成后会生成：
- `autoencoder_{source}_e{exercise}.pth` - 自编码器模型
- `cnn_classifier_{source}_e{exercise}.pth` - CNN分类器模型

## 模型架构

### 特征提取
使用滑动窗口从原始EMG信号中提取时域特征：
- 均值、标准差、RMS
- 最大值、绝对值均值、方差
- 零交叉率(ZCC)、峰值差距
- 能量、百分位数(25%, 50%, 75%)

### CNN + CAE
1. **CAE预训练**: 无监督方式学习特征表示
2. **CNN分类**: 有监督训练手势分类器

## 数据源说明

| 配置 | 数据源 | 特征维度 | 手势类别 |
|------|--------|----------|----------|
| `old` | EMG_data | 24 (2ch×12) | 10 |
| `new` | Ninapro_DB1 | 120 (10ch×12) | 12/17/23 |
| `both` | 合并 | 120 | 264 (10+254) |

## 硬件要求

- **GPU**: NVIDIA CUDA (推荐)
- **内存**: 8GB+
- **存储**: 2GB+

## 参考

- Ninapro Dataset: http://ninapro.hevs.ch/
- 原始CNN+CAE方法参考项目根目录下的 `Classification-of-electromyographic-hand-gesture-signals-using-machine-learning-techniques-main/`
