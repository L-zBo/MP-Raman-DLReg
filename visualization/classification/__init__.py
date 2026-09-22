"""
分类图可视化模块

功能：
- 生成微塑料污染分类的分类图
- 支持双头模型（dual）和分离训练（PP/PE）两种模式
- 使用统一的plot_classification_map函数（来自visualization.spectrum.visualize）
"""

from .classification import save_train_classification_maps, save_test_classification_maps

# 注意：plot_classification_map 从 visualization.spectrum.visualize 导入
# 如需使用，请直接从那里导入：
# from visualization.spectrum.visualize import plot_classification_map

__all__ = [
    'save_train_classification_maps',
    'save_test_classification_maps',
]
