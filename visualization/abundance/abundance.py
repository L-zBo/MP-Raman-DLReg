"""
丰度图生成模块

功能：
- 为训练集生成PP和PE的丰度图
- 保存到 output/abundance/ 目录

使用：
from visualization.abundance import save_train_abundance_maps
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'spectrum'))
from visualize import plot_abundance_map


def save_train_abundance_maps(abundance, sample_name, base_output_dir):
    """保存训练集的丰度图到 output/abundance/ 目录"""
    abundance_output_dir = Path(base_output_dir) / 'abundance' / sample_name
    abundance_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(abundance_output_dir / 'abundance.npy', abundance)

    plot_abundance_map(
        abundance[:, :, 1], f'{sample_name} PP Abundance',
        abundance_output_dir / 'abundance_PP.png',
        cmap='viridis', dpi=300, colorbar_label='PP Abundance'
    )
    plot_abundance_map(
        abundance[:, :, 2], f'{sample_name} PE Abundance',
        abundance_output_dir / 'abundance_PE.png',
        cmap='viridis', dpi=300, colorbar_label='PE Abundance'
    )
    return abundance_output_dir
