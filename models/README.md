# Models 说明

本目录保存当前项目的模型定义。

## 当前主线模型

- `dual_head_model.py`
  - `DualHeadRamanCNNLSTM`
  - 当前主模型 `RADAR-Net`
  - 负责 `PP / PE` 双任务预测

## 当前最终对比模型

- `base_models.py`
  - `SMARTNIRClassifier`
  - `ConvTranClassifier`
  - `MambaHSIClassifier`
  - `ResNet50_1D`

## 当前保留的传统/历史基线

`base_models.py` 中仍保留部分旧单头基线与传统方法实现，目的仅为：

- 历史结果复现
- 旧实验兼容
- 归档代码保留

它们不再属于当前主线对比入口。

## 当前推荐使用方式

- 主模型训练：
  - `training/train_dual.py`
- 最终模型对比：
  - `training/model_comparison_Acc.py`
- 消融实验：
  - `training/ablation_study.py`

## 统一口径

- 主模型名、最终模型池、最终参数来源：
  - `utils/model_identity.py`
- 说明文档：
  - `utils/model_identity.md`
