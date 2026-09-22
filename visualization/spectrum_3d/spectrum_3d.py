"""
3D光谱序列可视化

生成图表:
1. 3D光谱瀑布图 (spectrum_3d_waterfall.png)
2. 按类别分组的3D光谱 (spectrum_3d_by_class.png)

输出位置: output/spectrum_3d/PP/ 和 output/spectrum_3d/PE/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 统一颜色方案
CLASS_COLORS = {
    0: '#2ecc71',  # 无污染 - 绿色
    1: '#f39c12',  # 低污染 - 橙色
    2: '#e74c3c'   # 高污染 - 红色
}

CLASS_NAMES = ['Non-pollution', 'Slight pollution', 'Severe pollution']


def load_data(config, task: str):
    """加载预处理数据"""
    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    prefix = 'pp' if task == 'PP' else 'pe'
    label_suffix = 'pp_labels' if task == 'PP' else 'pe_labels'

    spectra = []
    labels = []

    for i in range(1, 6):
        data_file = preprocessed_dir / f'{prefix}_mixed{i}_data.npy'
        label_file = preprocessed_dir / f'{prefix}_mixed{i}_{label_suffix}.npy'

        if data_file.exists():
            data = np.load(data_file)
            # 展平为 (n_samples, n_bands)
            if data.ndim == 3:
                data = data.reshape(-1, data.shape[-1])
            spectra.append(data)

            if label_file.exists():
                label_data = np.load(label_file)
                # 展平标签
                labels.append(label_data.flatten())

    if spectra:
        X = np.concatenate(spectra, axis=0)
        y = np.concatenate(labels, axis=0) if labels else None
        return X, y

    return None, None


def normalize_spectra(spectra: np.ndarray) -> np.ndarray:
    """归一化光谱到0-1范围"""
    min_val = spectra.min(axis=1, keepdims=True)
    max_val = spectra.max(axis=1, keepdims=True)
    return (spectra - min_val) / (max_val - min_val + 1e-8)


def plot_spectrum_3d_waterfall(X: np.ndarray, y: np.ndarray, output_dir: Path, task: str, n_samples: int = 50):
    """绘制3D光谱瀑布图"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 随机选择样本
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    indices = np.sort(indices)

    X_norm = normalize_spectra(X[indices])
    y_selected = y[indices]

    n_bands = X_norm.shape[1]
    x = np.arange(n_bands)

    for i, (spectrum, label) in enumerate(zip(X_norm, y_selected)):
        color = CLASS_COLORS.get(int(label), '#95a5a6')
        ax.plot(x, [i] * n_bands, spectrum, color=color, alpha=0.7, linewidth=0.8)

    ax.set_xlabel('Spectral Band', fontsize=11, labelpad=10)
    ax.set_ylabel('Sample Index', fontsize=11, labelpad=10)
    ax.set_zlabel('Normalized Intensity', fontsize=11, labelpad=10)
    ax.set_title(f'3D Spectrum Waterfall Plot ({task} Task)', fontsize=14, fontweight='bold')

    # 添加图例
    legend_elements = [plt.Line2D([0], [0], color=CLASS_COLORS[i], linewidth=2, label=CLASS_NAMES[i])
                       for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(output_dir / f'spectrum_3d_waterfall_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectrum_3d_by_class(X: np.ndarray, y: np.ndarray, output_dir: Path, task: str, n_per_class: int = 20):
    """绘制按类别分组的3D光谱"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

    n_bands = X.shape[1]
    x = np.arange(n_bands)

    for class_id, ax in enumerate(axes):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) == 0:
            continue

        selected = np.random.choice(class_indices, min(n_per_class, len(class_indices)), replace=False)
        X_class = normalize_spectra(X[selected])

        color = CLASS_COLORS[class_id]

        for i, spectrum in enumerate(X_class):
            ax.plot(x, [i] * n_bands, spectrum, color=color, alpha=0.6, linewidth=0.8)

        ax.set_xlabel('Spectral Band', fontsize=10)
        ax.set_ylabel('Sample', fontsize=10)
        ax.set_zlabel('Intensity', fontsize=10)
        ax.set_title(f'{CLASS_NAMES[class_id]}', fontsize=12, fontweight='bold', color=color)
        ax.view_init(elev=20, azim=-60)

    plt.suptitle(f'3D Spectra by Class ({task} Task)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'spectrum_3d_by_class_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectrum_surface(X: np.ndarray, y: np.ndarray, output_dir: Path, task: str, n_samples: int = 100):
    """绘制光谱曲面图"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 按类别排序
    sorted_indices = np.argsort(y)
    indices = []
    for class_id in range(3):
        class_idx = sorted_indices[y[sorted_indices] == class_id]
        if len(class_idx) > 0:
            selected = np.random.choice(class_idx, min(n_samples // 3, len(class_idx)), replace=False)
            indices.extend(selected)

    indices = np.array(indices)
    X_selected = normalize_spectra(X[indices])
    y_selected = y[indices]

    n_bands = X_selected.shape[1]
    n_samples_actual = len(X_selected)

    # 创建网格
    X_grid, Y_grid = np.meshgrid(np.arange(n_bands), np.arange(n_samples_actual))
    Z_grid = X_selected

    # 创建颜色数组
    colors = np.zeros((n_samples_actual, n_bands, 4))
    for i, label in enumerate(y_selected):
        color = plt.cm.colors.to_rgba(CLASS_COLORS[int(label)])
        colors[i, :] = color

    ax.plot_surface(X_grid, Y_grid, Z_grid, facecolors=colors, alpha=0.8, linewidth=0, antialiased=True)

    ax.set_xlabel('Spectral Band', fontsize=11)
    ax.set_ylabel('Sample Index', fontsize=11)
    ax.set_zlabel('Normalized Intensity', fontsize=11)
    ax.set_title(f'Normalized Sample Spectra Surface ({task} Task)', fontsize=14, fontweight='bold')

    # 图例
    legend_elements = [plt.Line2D([0], [0], color=CLASS_COLORS[i], linewidth=3, label=CLASS_NAMES[i])
                       for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.savefig(output_dir / f'spectrum_surface_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_spectrum_3d_visualizations(config, output_dir: Path):
    """生成3D光谱可视化"""
    logger = get_logger('spectrum_3d')

    output_dir.mkdir(parents=True, exist_ok=True)

    for task in ['PP', 'PE']:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"生成 {task} 任务 3D光谱图...")

        X, y = load_data(config, task)

        if X is not None and y is not None:
            plot_spectrum_3d_waterfall(X, y, task_dir, task)
            plot_spectrum_3d_by_class(X, y, task_dir, task)
            plot_spectrum_surface(X, y, task_dir, task)
            logger.info(f"{task} 3D光谱图已保存到: {task_dir}")
        else:
            logger.warning(f"未找到 {task} 数据")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('spectrum_3d')

    logger.info("=" * 60)
    logger.info("生成3D光谱序列可视化")
    logger.info("=" * 60)

    output_dir = Path(config.paths['output_dir']) / 'spectrum_3d'
    generate_spectrum_3d_visualizations(config, output_dir)

    logger.info("\n3D光谱可视化完成！")
