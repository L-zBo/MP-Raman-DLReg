# Generalization 说明

本目录用于验证 `RADAR-Net` 在不同品牌、不同淀粉类型样本上的泛化能力。

## 当前主线脚本

- `run_generalization.py`
  - 当前泛化实验主入口

## 当前方法口径

- 主模型：
  - `RADAR-Net`
- 泛化策略：
  - `Combo`
  - 核心思想：统一端元 + 训练集固化丰度阈值

## 依赖文件

- 主模型权重：
  - `output/models/best_model.pth`
- 丰度阈值：
  - `output/models/abundance_thresholds/pp_thresholds.json`
  - `output/models/abundance_thresholds/pe_thresholds.json`

## 输出位置

- `output/generalization/`

## 说明

- 本目录保留的内容仍属于当前有效实验结果，不在归档范围内。
