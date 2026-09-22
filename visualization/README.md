# Visualization 说明

本目录保存当前项目的可视化脚本。

## 当前主线入口

- `boxplot/boxplot_analysis.py`
  - 最终箱型图主入口
- `roc_curve/roc_curve.py`
  - 最终 ROC 主入口
- `loss_curve/loss_curve.py`
  - 训练曲线主入口

## 当前仍保留且可用的模块

- `ablation/`
- `abundance/`
- `attention_heatmap/`
- `classification/`
- `comparison_Acc/`
- `feature_peaks/`
- `gradcam/`
- `hyperparameter_sensitivity/`
- `performance_radar/`
- `predicted_residuals/`
- `prediction_kde/`
- `prediction_scatter/`
- `preprocessing/`
- `representative_spectra/`
- `shap/`
- `spectral_residuals/`
- `spectrum/`
- `spectrum_3d/`
- `stability/`
- `tables/`
- `tsne/`
- `umap/`
- `violin/`

## 当前统一输出口径

- 最终箱型图：
  - `output/boxplot_final_ranked/`
- 最终 ROC：
  - `output/roc_curve/`
- 训练曲线：
  - `output/loss_curve/`
- 分类图对比：
  - `output/comparison_final/classification_comparison_main_table/`

## 说明

- 已废弃或已归档的旧可视化入口，不再在本 README 中列出。
- 新方法的可视化先在独立项目副本中验证，脚本与输出分开保存，审核后再决定是否纳入正式流程。
