"""
分类图生成模块

功能：
- 为训练集和测试集生成PP和PE的分类图
- 保存到 output/classification/ 目录

使用：
from visualization.classification import save_train_classification_maps, save_test_classification_maps
"""
import numpy as np
from pathlib import Path
import sys

# 添加visualization/spectrum到路径以导入完整版本的plot_classification_map
sys.path.insert(0, str(Path(__file__).parent.parent / 'spectrum'))
from visualize import plot_classification_map


def save_train_classification_maps(pp_labels, pe_labels, sample_name, base_output_dir):
    """保存训练集的分类图到 output/classification/ 目录"""
    classification_output_dir = Path(base_output_dir) / 'classification' / sample_name
    classification_output_dir.mkdir(parents=True, exist_ok=True)

    plot_classification_map(
        pp_labels, f'{sample_name} PP Classification',
        classification_output_dir / 'classification_PP.png',
        colors=['black', 'blue', 'red'],
        class_names=['Non-pollution', 'Slight pollution', 'Severe pollution'],
        dpi=300
    )
    plot_classification_map(
        pe_labels, f'{sample_name} PE Classification',
        classification_output_dir / 'classification_PE.png',
        colors=['black', 'blue', 'red'],
        class_names=['Non-pollution', 'Slight pollution', 'Severe pollution'],
        dpi=300
    )
    return classification_output_dir


def save_test_classification_maps(pp_labels, pe_labels, sample_name, base_output_dir):
    """保存测试集的分类图到 output/classification/ 目录"""
    return save_train_classification_maps(pp_labels, pe_labels, sample_name, base_output_dir)
