# Preprocessing 预处理模块

本目录包含拉曼光谱数据预处理的完整流程，支持并行处理、自动编码检测和完整的数据验证。

## 📁 文件列表

| 文件名 | 功能 | 代码行数 |
|--------|------|---------|
| `preprocess.py` | 完整的预处理流程（ALS基线校正 + L2归一化） | ~466行 |
| `README.md` | 本文档 | - |

**状态**：✅ 最新版本，无冗余文件

---

## 🎯 核心功能

### 1. ALS基线校正
去除拉曼光谱的荧光背景和基线漂移

### 2. L2归一化
统一光谱强度量纲，消除样本间差异

### 3. 高光谱数据加载
自动解析40×40网格数据，构建3D高光谱立方体

### 4. 并行处理
支持多进程并行预处理，显著提升大数据集处理速度

### 5. 自动编码检测
支持7种常见文件编码，自动处理中文文件名

### 6. 数据验证
完整的NaN/Inf检查和异常值清理

---

## 📊 预处理流程图

```
原始CSV文件 (40×40个)
    ↓
【步骤1】加载高光谱数据
    → load_hyperspectral_data()
    → 解析文件名坐标 (X-Y格式)
    → 构建3D立方体 (H, W, 1024)
    ↓
【步骤2】数据清理
    → clean_spectrum()
    → 去除NaN/Inf异常值
    ↓
【步骤3】ALS基线校正
    → als_baseline()
    → 去除荧光背景
    → corrected = spectrum - baseline
    ↓
【步骤4】L2归一化
    → l2_normalize()
    → 统一强度量纲
    → normalized = spectrum / ||spectrum||₂
    ↓
【步骤5】保存预处理数据
    → save_preprocessed_data()
    → 保存为.npy格式
    → preprocessed_data/*.npy
```

---

## 📚 函数API文档

### 核心函数

#### 1. als_baseline()
```python
def als_baseline(y, lam=1e5, p=0.01, niter=10) -> np.ndarray
```

**功能**：ALS (Asymmetric Least Squares) 基线校正

**算法原理**：
- 通过迭代拟合光滑曲线作为基线
- 使用非对称权重区分信号峰和基线
- 采用稀疏矩阵优化计算效率

**参数说明**：

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `y` | np.ndarray | - | 1D数组 | 输入光谱 |
| `lam` | float | 1e5 | 1e3-1e7 | 平滑度参数，越大基线越平滑 |
| `p` | float | 0.01 | 0.001-0.1 | 非对称权重，越小越贴近谷底 |
| `niter` | int | 10 | 5-20 | 迭代次数 |

**返回值**：估计的基线（np.ndarray）

**异常**：
- `ValueError`: 输入参数无效（维度错误、参数超出范围）

**使用示例**：
```python
from preprocessing.preprocess import als_baseline
import numpy as np

# 加载原始光谱
spectrum = np.load('raw_spectrum.npy')

# 计算基线
baseline = als_baseline(spectrum, lam=1e5, p=0.01, niter=10)

# 基线校正
corrected = spectrum - baseline
```

---

#### 2. l2_normalize()
```python
def l2_normalize(spectrum, eps=1e-10) -> np.ndarray
```

**功能**：L2向量归一化

**公式**：
```
normalized = spectrum / ||spectrum||₂
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `spectrum` | np.ndarray | - | 输入光谱 |
| `eps` | float | 1e-10 | 防止除零的小数 |

**返回值**：归一化后的光谱（np.ndarray）

**作用**：
- 消除光谱强度差异
- 使不同样本具有可比性
- 保持光谱形状特征

**使用示例**：
```python
from preprocessing.preprocess import l2_normalize

# 归一化
normalized = l2_normalize(corrected_spectrum)
```

---

#### 3. preprocess_spectrum()
```python
def preprocess_spectrum(spectrum, lam=1e5, p=0.01, niter=10, clean_data=True) -> np.ndarray
```

**功能**：预处理单条光谱（完整流程）

**处理步骤**：
1. 清理异常值（NaN/Inf）
2. ALS基线校正
3. L2归一化

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `spectrum` | np.ndarray | - | 输入光谱 |
| `lam` | float | 1e5 | ALS平滑度参数 |
| `p` | float | 0.01 | ALS非对称权重 |
| `niter` | int | 10 | ALS迭代次数 |
| `clean_data` | bool | True | 是否清理异常值 |

**返回值**：预处理后的光谱（np.ndarray）

**使用示例**：
```python
from preprocessing.preprocess import preprocess_spectrum

# 一步完成预处理
processed = preprocess_spectrum(raw_spectrum)
```

---

#### 4. load_hyperspectral_data()
```python
def load_hyperspectral_data(folder_path, validate=True) -> Tuple[np.ndarray, np.ndarray]
```

**功能**：加载40×40高光谱数据

**处理逻辑**：
1. 扫描文件夹中的所有CSV文件
2. 解析文件名中的坐标信息（X-Y格式）
3. 构建3D高光谱立方体
4. 验证数据完整性

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `folder_path` | str/Path | - | 数据文件夹路径 |
| `validate` | bool | True | 是否验证数据（NaN/Inf检查） |

**返回值**：
- `hypercube`: 高光谱立方体 (H, W, bands)
- `wavenumbers`: 波数数组 (bands,)

**文件名格式要求**：
```
样本名-编号-X坐标-Y坐标.csv
例如：PP+淀粉1-1-X1-Y1.csv
```

**异常**：
- `FileNotFoundError`: 文件夹不存在
- `ValueError`: 数据格式错误或没有有效文件

**使用示例**：
```python
from preprocessing.preprocess import load_hyperspectral_data

# 加载数据
hypercube, wavenumbers = load_hyperspectral_data('dataset/PP+淀粉/PP+淀粉1')
print(f"数据形状: {hypercube.shape}")  # (40, 40, 1024)
```

---

#### 5. preprocess_hyperspectral()
```python
def preprocess_hyperspectral(hypercube, n_jobs=-1, lam=None, p=None, niter=None, show_progress=True) -> np.ndarray
```

**功能**：预处理整个高光谱立方体（支持并行）

**性能优化**：
- 小数据集（<100像素）：串行处理
- 大数据集：多进程并行处理
- 自动选择最优进程数

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hypercube` | np.ndarray | - | 高光谱数据 (H, W, bands) |
| `n_jobs` | int | -1 | 并行进程数，-1表示使用所有CPU核心 |
| `lam` | float | None | ALS平滑度（None从配置读取） |
| `p` | float | None | ALS非对称权重（None从配置读取） |
| `niter` | int | None | ALS迭代次数（None从配置读取） |
| `show_progress` | bool | True | 是否显示进度信息 |

**返回值**：预处理后的高光谱数据 (H, W, bands)

**性能参考**：
- 单进程：40×40×1024 ≈ 5分钟
- 8进程并行：40×40×1024 ≈ 1分钟

**使用示例**：
```python
from preprocessing.preprocess import preprocess_hyperspectral

# 并行预处理（使用所有CPU核心）
processed = preprocess_hyperspectral(hypercube, n_jobs=-1)

# 指定进程数
processed = preprocess_hyperspectral(hypercube, n_jobs=4)

# 串行处理
processed = preprocess_hyperspectral(hypercube, n_jobs=1)
```

---

#### 6. detect_encoding()
```python
def detect_encoding(filepath) -> str
```

**功能**：自动检测CSV文件编码

**支持的编码**：
- gbk, utf-8, utf-8-sig, latin-1, cp1252, gb2312, gb18030

**使用场景**：
- 处理中文文件名
- 避免编码错误导致的乱码

**使用示例**：
```python
from preprocessing.preprocess import detect_encoding

encoding = detect_encoding('dataset/PP+淀粉/样本1.csv')
print(f"检测到的编码: {encoding}")
```

---

#### 7. load_csv_spectrum()
```python
def load_csv_spectrum(filepath, encoding=None, auto_detect=True) -> Tuple[np.ndarray, np.ndarray]
```

**功能**：加载单个CSV光谱文件

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `filepath` | str/Path | - | CSV文件路径 |
| `encoding` | str | None | 文件编码（None表示从配置读取） |
| `auto_detect` | bool | True | 编码错误时是否自动检测 |

**返回值**：
- `wavenumbers`: 波数数组
- `intensities`: 强度数组

**CSV格式要求**：
```csv
波数,强度
400.0,1234.5
401.0,1235.6
...
```

**使用示例**：
```python
from preprocessing.preprocess import load_csv_spectrum

wavenumbers, intensities = load_csv_spectrum('dataset/PP纯谱/PP.csv')
```

---

#### 8. save_preprocessed_data()
```python
def save_preprocessed_data(data, wavenumbers, output_path, name) -> None
```

**功能**：保存预处理后的数据

**输出文件**：
- `{name}_data.npy`: 预处理后的光谱数据
- `{name}_wavenumbers.npy`: 波数数组

**使用示例**：
```python
from preprocessing.preprocess import save_preprocessed_data

save_preprocessed_data(
    processed_data,
    wavenumbers,
    'preprocessed_data',
    'PP+淀粉1'
)
```

---

## 🚀 使用指南

### 场景1：运行完整预处理流程

```bash
# 直接运行脚本
python preprocessing/preprocess.py
```

**处理内容**：
1. 加载淀粉、PP、PE纯谱
2. 预处理纯谱
3. 加载所有混合样本（PP+淀粉、PE+淀粉）
4. 并行预处理所有样本
5. 保存结果到 `preprocessed_data/`

**输出文件**：
```
preprocessed_data/
├── starch_spectrum.npy          # 淀粉纯谱
├── pp_spectrum.npy              # PP纯谱
├── pe_spectrum.npy              # PE纯谱
├── wavenumbers.npy              # 波数轴
├── PP+淀粉1_data.npy            # PP+淀粉样本1
├── PP+淀粉1_wavenumbers.npy
├── PP+淀粉2_data.npy            # PP+淀粉样本2
├── ...
├── PE+淀粉1_data.npy            # PE+淀粉样本1
└── ...
```

---

### 场景2：预处理单条光谱

```python
from preprocessing.preprocess import preprocess_spectrum
import numpy as np

# 加载原始光谱
spectrum = np.load('raw_spectrum.npy')

# 预处理（使用默认参数）
processed = preprocess_spectrum(spectrum)

# 自定义参数
processed = preprocess_spectrum(
    spectrum,
    lam=1e6,      # 更强的平滑
    p=0.001,      # 更贴近谷底
    niter=15      # 更多迭代
)
```

---

### 场景3：预处理高光谱数据

```python
from preprocessing.preprocess import (
    load_hyperspectral_data,
    preprocess_hyperspectral,
    save_preprocessed_data
)

# 加载数据
hypercube, wavenumbers = load_hyperspectral_data('dataset/PP+淀粉/PP+淀粉1')
print(f"数据形状: {hypercube.shape}")  # (40, 40, 1024)

# 并行预处理
processed = preprocess_hyperspectral(
    hypercube,
    n_jobs=-1     # 使用所有CPU核心
)

# 保存结果
save_preprocessed_data(
    processed,
    wavenumbers,
    'preprocessed_data',
    'PP+淀粉1'
)
```

---

### 场景4：自定义预处理流程

```python
from preprocessing.preprocess import (
    als_baseline,
    l2_normalize,
    clean_spectrum
)
import numpy as np

# 加载原始光谱
spectrum = np.load('raw_spectrum.npy')

# 步骤1：清理异常值
spectrum_clean = clean_spectrum(spectrum)

# 步骤2：基线校正
baseline = als_baseline(spectrum_clean, lam=1e5, p=0.01, niter=10)
corrected = spectrum_clean - baseline

# 步骤3：归一化
normalized = l2_normalize(corrected)

# 可选：添加自定义步骤（如平滑）
from scipy.signal import savgol_filter
smoothed = savgol_filter(normalized, window_length=11, polyorder=3)
```

---

## ⚙️ 参数调优指南

### ALS基线校正参数

#### 问题1：基线校正过强（信号峰被削弱）

**症状**：
- 校正后的光谱峰值明显降低
- 特征峰被"削平"

**解决方案**：
```python
# 减小 λ 值
processed = preprocess_spectrum(spectrum, lam=1e4)  # 默认1e5

# 或增大 p 值
processed = preprocess_spectrum(spectrum, p=0.05)   # 默认0.01
```

#### 问题2：基线校正不足（仍有背景）

**症状**：
- 校正后的光谱仍有明显的基线漂移
- 背景荧光未完全去除

**解决方案**：
```python
# 增大 λ 值
processed = preprocess_spectrum(spectrum, lam=1e6)  # 默认1e5

# 或减小 p 值
processed = preprocess_spectrum(spectrum, p=0.001)  # 默认0.01

# 或增加迭代次数
processed = preprocess_spectrum(spectrum, niter=20)  # 默认10
```

#### 问题3：处理速度慢

**症状**：
- 预处理大数据集耗时过长

**解决方案**：
```python
# 使用并行处理
processed = preprocess_hyperspectral(hypercube, n_jobs=-1)

# 或减少迭代次数（牺牲精度换速度）
processed = preprocess_hyperspectral(hypercube, niter=5)
```

---

### 参数推荐值

| 光谱类型 | λ (lam) | p | niter | 说明 |
|---------|---------|---|-------|------|
| **拉曼光谱（默认）** | 1e5 | 0.01 | 10 | 平衡精度和速度 |
| **强荧光背景** | 1e6 | 0.001 | 15 | 更强的基线校正 |
| **弱荧光背景** | 1e4 | 0.05 | 10 | 避免过度校正 |
| **快速预览** | 1e5 | 0.01 | 5 | 快速处理 |
| **高精度分析** | 1e5 | 0.01 | 20 | 更精确的基线 |

---

## 🔍 验证预处理效果

### 方法1：可视化对比

```python
import matplotlib.pyplot as plt
from preprocessing.preprocess import (
    load_csv_spectrum,
    als_baseline,
    l2_normalize
)

# 加载原始光谱
wavenumbers, spectrum = load_csv_spectrum('dataset/PP纯谱/PP.csv')

# 预处理
baseline = als_baseline(spectrum)
corrected = spectrum - baseline
normalized = l2_normalize(corrected)

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 原始光谱
axes[0, 0].plot(wavenumbers, spectrum)
axes[0, 0].set_title('原始光谱')
axes[0, 0].set_xlabel('波数 (cm⁻¹)')
axes[0, 0].set_ylabel('强度')

# 基线
axes[0, 1].plot(wavenumbers, spectrum, label='原始')
axes[0, 1].plot(wavenumbers, baseline, label='基线', linewidth=2)
axes[0, 1].set_title('ALS基线估计')
axes[0, 1].legend()

# 基线校正后
axes[1, 0].plot(wavenumbers, corrected)
axes[1, 0].set_title('基线校正后')
axes[1, 0].set_xlabel('波数 (cm⁻¹)')

# 归一化后
axes[1, 1].plot(wavenumbers, normalized)
axes[1, 1].set_title('L2归一化后')
axes[1, 1].set_xlabel('波数 (cm⁻¹)')

plt.tight_layout()
plt.savefig('preprocessing_comparison.png', dpi=300)
plt.show()
```

### 方法2：统计指标

```python
import numpy as np

# 计算统计指标
print("原始光谱:")
print(f"  均值: {np.mean(spectrum):.2f}")
print(f"  标准差: {np.std(spectrum):.2f}")
print(f"  范围: [{np.min(spectrum):.2f}, {np.max(spectrum):.2f}]")

print("\n归一化后:")
print(f"  L2范数: {np.linalg.norm(normalized):.6f}")  # 应该≈1.0
print(f"  均值: {np.mean(normalized):.6f}")
print(f"  标准差: {np.std(normalized):.6f}")
```

### 方法3：使用可视化脚本

```bash
# 使用项目提供的可视化脚本
python visualization/preprocessing/preprocess_comparison.py
```

---

## 📊 输入输出格式

### 输入格式

#### 1. CSV光谱文件
```csv
波数,强度
400.0,1234.5
401.0,1235.6
402.0,1236.7
...
```

**要求**：
- 至少2列（波数、强度）
- 第1列：波数（cm⁻¹）
- 第2列：拉曼强度
- 支持多种编码（gbk, utf-8等）

#### 2. 高光谱数据文件夹
```
dataset/PP+淀粉/PP+淀粉1/
├── PP+淀粉1-1-X1-Y1.csv
├── PP+淀粉1-2-X1-Y2.csv
├── PP+淀粉1-3-X1-Y3.csv
├── ...
└── PP+淀粉1-1600-X40-Y40.csv
```

**文件名格式**：`样本名-编号-X坐标-Y坐标.csv`

---

### 输出格式

#### 1. 预处理后的光谱数据
```python
# .npy文件（NumPy数组）
data = np.load('preprocessed_data/PP+淀粉1_data.npy')
print(data.shape)  # (40, 40, 1024)
print(data.dtype)  # float64
```

**数据结构**：
- 3D数组：(高度, 宽度, 波段数)
- 数据类型：float64
- 值范围：归一化后，每个像素的L2范数≈1.0

#### 2. 波数数组
```python
wavenumbers = np.load('preprocessed_data/wavenumbers.npy')
print(wavenumbers.shape)  # (1024,)
print(wavenumbers[0], wavenumbers[-1])  # 起始和结束波数
```

---

## 🔗 相关文件

- **配置文件**：`config.yaml` - 预处理参数配置
- **可视化**：`visualization/preprocessing/` - 预处理效果可视化
- **下游模块**：
  - `unmixing/unmix.py` - 光谱解混
  - `training/train_dual.py` - 模型训练
- **工具模块**：`utils/` - 配置管理、日志、数据验证

---

## 🐛 常见问题

### Q1: 编码错误 `UnicodeDecodeError`

**问题**：无法读取CSV文件，提示编码错误

**解决方案**：
```python
# 方法1：自动检测编码
hypercube, wn = load_hyperspectral_data(folder_path, auto_detect=True)

# 方法2：手动指定编码
wavenumbers, intensities = load_csv_spectrum(filepath, encoding='gbk')

# 方法3：在config.yaml中设置默认编码
dataset:
  encoding: 'gbk'  # 或 'utf-8', 'gb2312' 等
```

---

### Q2: 内存不足 `MemoryError`

**问题**：处理大数据集时内存溢出

**解决方案**：
```python
# 方法1：减少并行进程数
processed = preprocess_hyperspectral(hypercube, n_jobs=2)

# 方法2：串行处理
processed = preprocess_hyperspectral(hypercube, n_jobs=1)

# 方法3：分批处理
for i in range(0, len(samples), batch_size):
    batch = samples[i:i+batch_size]
    process_batch(batch)
```

---

### Q3: 预处理结果全为零

**问题**：预处理后的光谱全是0

**可能原因**：
1. 输入光谱范数接近零
2. 基线校正参数不当
3. 输入数据包含NaN/Inf

**解决方案**：
```python
# 检查输入数据
print(f"输入范数: {np.linalg.norm(spectrum)}")
print(f"NaN数量: {np.sum(np.isnan(spectrum))}")
print(f"Inf数量: {np.sum(np.isinf(spectrum))}")

# 启用数据清理
processed = preprocess_spectrum(spectrum, clean_data=True)

# 调整参数
processed = preprocess_spectrum(spectrum, lam=1e4, p=0.05)
```

---

### Q4: 处理速度慢

**问题**：预处理40×40数据需要很长时间

**优化方案**：
```python
# 1. 启用并行处理（最重要）
processed = preprocess_hyperspectral(hypercube, n_jobs=-1)

# 2. 减少迭代次数
processed = preprocess_hyperspectral(hypercube, niter=5)

# 3. 检查CPU核心数
import os
print(f"CPU核心数: {os.cpu_count()}")
```

**性能参考**：
- 单进程：40×40×1024 ≈ 5分钟
- 4进程：40×40×1024 ≈ 2分钟
- 8进程：40×40×1024 ≈ 1分钟

---

## 📚 参考文献

1. **ALS基线校正**：
   - Eilers, P. H. C., & Boelens, H. F. M. (2005). "Baseline Correction with Asymmetric Least Squares Smoothing". *Leiden University Medical Centre Report*.

2. **拉曼光谱预处理**：
   - Zhao, J., et al. (2015). "Automated autofluorescence background subtraction algorithm for biomedical Raman spectroscopy". *Applied Spectroscopy*, 61(11), 1225-1232.

3. **L2归一化**：
   - Rinnan, Å., et al. (2009). "Review of the most common pre-processing techniques for near-infrared spectra". *TrAC Trends in Analytical Chemistry*, 28(10), 1201-1222.

---

## 📝 更新日志

- **2026-01-17**: 更新README，添加完整的API文档和使用指南
- **2025-XX-XX**: 初始版本，基本预处理功能

---

**最后更新**：2026-01-17
