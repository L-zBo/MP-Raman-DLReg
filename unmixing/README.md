# Unmixing 解混与分类模块

本目录包含光谱解混和标签生成脚本，连接预处理与深度学习训练。

## 当前主线脚本

| 文件 | 功能 |
|------|------|
| `unmix.py` | 三端元 NNLS 解混（淀粉 / PP / PE） |
| `abundance_threshold_labeling.py` | 基于固化丰度阈值生成 PP / PE 三级标签 |
| `detect_type.py` | 基于丰度的微塑料类型检测（辅助工具） |

## 标签生成口径

- 固化阈值文件：
  - `output/models/abundance_thresholds/pp_thresholds.json`
  - `output/models/abundance_thresholds/pe_thresholds.json`
- 阈值结构：`{ "n_components": 3, "thresholds": [low, high] }`
- 切分规则：
  - `abundance ≤ low` → 0（无污染）
  - `low < abundance ≤ high` → 1（轻污染）
  - `abundance > high` → 2（重污染）

PP 阈值：`[0.1142, 0.2945]`
PE 阈值：`[0.0994, 0.1913]`

## 完整流程

```
预处理光谱 → unmix.py（NNLS 解混）→ abundance_threshold_labeling.py（阈值切分）→ 三级标签
```

输出：

- 丰度图：`output/abundance/abundance_npy/{dataset}/{sample}_abundance.npy`，形状 `(40, 40, 3)`
- 标签：`preprocessed_data/{sample}_{pp,pe}_labels.npy`

## 使用示例

```python
import numpy as np
from pathlib import Path
from unmixing.unmix import unmix_hyperspectral
from unmixing.abundance_threshold_labeling import AbundanceThresholdLabeler

# NNLS 解混
data = np.load('preprocessed_data/pp_mixed1_data.npy')
endmembers = np.stack([starch, pp, pe])  # 端元矩阵 (3, 1024)
abundance_maps = unmix_hyperspectral(data, endmembers, n_jobs=-1)  # (40, 40, 3)

# 阈值切分
labeler = AbundanceThresholdLabeler().load(
    Path('output/models/abundance_thresholds/pp_thresholds.json')
)
pp_labels = labeler.predict(abundance_maps[:, :, 1])  # PP 三级标签
```

## 端元配置

端元来自 `config.yaml`：

```yaml
dataset:
  endmembers:
    starch: {folder: "淀粉纯谱/玉米淀粉", file: "DATA-105635-X0-Y30-8884.csv"}
    pp: {folder: "PP纯谱", file: "采集图谱.csv"}
    pe: {folder: "PE纯谱", file: "采集图谱.csv"}
```

## 相关文件

- 预处理：`preprocessing/preprocess.py`
- 训练：`training/train_dual.py`
- 配置：`config.yaml`
