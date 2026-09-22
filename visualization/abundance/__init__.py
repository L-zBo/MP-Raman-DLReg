"""
丰度图可视化模块

功能：
- 生成微塑料丰度分布的热力图
- 使用统一的plot_abundance_map函数（来自visualization.spectrum.visualize）
"""

from .abundance import save_train_abundance_maps

__all__ = [
    'save_train_abundance_maps',
]
