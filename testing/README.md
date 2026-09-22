# Testing 说明

本目录保存当前主线测试与评估脚本。

## 当前主线脚本

- `test_pixel_level.py`
  - 生成像素级预测标签与分类图
- `test_object_level.py`
  - 生成对象级污染结果
- `evaluate_all_datasets.py`
  - 全数据集评估
- `adjust_threshold_test.py`
  - 阈值调整测试
- `export_predictions.py`
  - 导出双头模型预测概率

## 当前关键输出

- 像素级结果：
  - `output/test_results_dual/pixel_level/`
- 对象级结果：
  - `output/test_results_dual/object_level/object_level_predictions_pixel_ratio.csv`
- 分类图对比：
  - `output/comparison_final/classification_comparison_main_table/`

## 当前阈值口径

- PP：
  - `class1 = 0.15`
  - `class2 = 0.0`
- PE：
  - `class1 = 0.50`
  - `class2 = 0.70`

详细说明见：

- `testing/THRESHOLD_CONFIG.md`

## 说明

- 训练集和验证集不应用测试阈值调整。
- 涉及测试集可视化和评估的脚本，应与当前阈值口径保持一致。
