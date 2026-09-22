# 拉曼光谱微塑料污染检测项目

本项目面向淀粉基质中的 `PP / PE` 微塑料污染分级任务，主线方法为：

- 光谱预处理：`ALS + L2 normalization`
- 光谱解混：`NNLS`
- 自动标签生成：基于 `NNLS` 丰度估计的像素级污染分级
- 主模型：`RADAR-Net`
- 最终对比模型池：
  - `XGBoost`
  - `SVM`
  - `Random Forest`
  - `LightGBM`
  - `ResNet-50`
  - `SMART-NIR`
  - `ConvTran`
  - `MambaHSI`

## 当前项目结构

```text
MP-Raman-DLReg/
├── config.yaml
├── README.md
├── requirements.txt
├── preprocessing/
├── unmixing/
├── models/
├── training/
├── testing/
├── generalization/
├── visualization/
└── utils/
```

## 当前主线入口

### 训练

- `training/train_dual.py`
  - 训练 `RADAR-Net`
- `training/ablation_study.py`
  - 运行消融实验
- `training/model_comparison_Acc.py`
  - 运行最终模型池对比实验

### 测试

- `testing/test_pixel_level.py`
  - 生成像素级预测标签
- `testing/test_object_level.py`
  - 生成对象级结果
- `testing/evaluate_all_datasets.py`
  - 全数据集评估

### 可视化

- `visualization/boxplot/boxplot_analysis.py`
  - 最终箱型图主入口
- `visualization/roc_curve/roc_curve.py`
  - 最终 ROC 主入口
- `visualization/loss_curve/loss_curve.py`
  - 训练曲线主入口

## 当前统一口径

- 主模型名与最终对比模型顺序：
  - `utils/model_identity.py`
- 箱型图专用参数：
  - `visualization/boxplot/boxplot_analysis.py`

## 当前推荐输出目录

- 最终对比结果：
  - `output/comparison_final/`
- 最终箱型图：
  - `output/boxplot_final_ranked/`
- 最终 ROC：
  - `output/roc_curve/`
- 最终测试结果：
  - `output/test_results_dual/`

## 说明

- 本仓库只保存代码、配置与说明文档,不含数据与实验结果。
- `logs/`由程序运行时自动创建,清理后无需手工补建。

### 本仓库不入库的内容

以下内容保留在本地,不进入版本库(见 `.gitignore`):

- `dataset/`:仪器采集的原始高光谱数据
- `preprocessed_data/`:预处理中间产物,可由 `preprocessing/preprocess.py` 重建
- `testing/test_true_label/`:像素级参考真值标签,复现评估时在本地放回
- `output/`:全部实验输出,由脚本运行时生成
- `paper/`:论文稿件与投稿材料
- 归档目录与第三方参考 PDF

克隆本仓库后需自行准备数据目录,方可运行预处理、训练与评估流程。
