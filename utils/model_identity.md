# model_identity.py 说明

`utils/model_identity.py` 是当前项目的统一模型口径与统一参数源。

它的职责只有一个：

- 集中记录**当前最终敲定方案**中各模型的名称、替换关系、最终采用版本、模型参数、训练参数以及最终排序口径。

注意：

- **箱型图专用参数不在本文件中维护。**
- 箱型图的最终参数口径单独保存在 `visualization/boxplot/boxplot_analysis.py` 中。
- 原因是箱型图实验允许使用与主对比实验不同的专用训练参数，不应强行共用。

## 当前用途

该文件用于统一以下信息：

- 主模型显示名：
  - `PRIMARY_MODEL_NAME`
  - `PRIMARY_MODEL_FULL_NAME`
- 最终对比模型替换关系：
  - `FINAL_COMPARISON_REPLACEMENTS`
- 已明确弃用的候选：
  - `FINAL_DROPPED_CANDIDATES`
- 最终残差网络口径：
  - `FINAL_RESNET_CHOICE`
  - `FINAL_RESNET_CONFIG`
- 最终对比模型顺序：
  - `FINAL_COMPARISON_MODEL_ORDER`
- 新候选模型最终采用参数：
  - `BEST_CANDIDATE_CONFIGS`

## 约束

- 后续新增或调整最终模型参数时，应优先修改本文件。
- 其他脚本不应再各自维护一套“最终参数”副本。
- 唯一例外是箱型图专用参数，它们独立保存在 `visualization/boxplot/boxplot_analysis.py`。
- 允许脚本保留临时试验参数，但**最终口径**必须以本文件为准。

## 当前最终模型池

- `RADAR-Net`
- `XGBoost`
- `SVM`
- `Random Forest`
- `LightGBM`
- `ResNet-50`
- `SMART-NIR`
- `ConvTran`
- `MambaHSI`

## 当前已弃用模型

- `RS-MLP`
- `LITE`
- 老对比口径中的 `1D-CNN`
- 老对比口径中的 `1D-Transformer`
- 老对比口径中的 `PLS-DA`

## 备注

如果项目后续再次收口、删减脚本或重构主入口，`model_identity.py` 应继续作为第一优先级的参数登记文件保留。
