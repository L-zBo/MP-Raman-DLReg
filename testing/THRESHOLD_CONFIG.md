# 阈值配置说明

## 当前测试集阈值

### PP

```python
pp_threshold_class1 = 0.15
pp_threshold_class2 = 0.0
```

### PE

```python
pe_threshold_class1 = 0.50
pe_threshold_class2 = 0.70
```

## 当前使用范围

- `testing/test_pixel_level.py`
- `testing/test_object_level.py`
- `testing/evaluate_all_datasets.py`
- 涉及测试集预测标签的相关可视化脚本

## 当前对象级说明口径

- 经验假阳性上界：约 `<1.5%`
- 最低可检测限：保守估计约 `1.5%`

> **待核对**:此处 `1.5%` 与代码默认值不一致。`testing/test_object_level.py` 中
> 对象级污染判定阈值为 `mild_threshold = 0.05`、`severe_threshold = 0.15`。
> 论文若统一采用 `2%`(高于实测最高误报 1.25%),本节与代码需同步修改,
> 具体取值待定。

## 原则

- 测试集可应用阈值调整
- 训练集与验证集保持原始预测口径
