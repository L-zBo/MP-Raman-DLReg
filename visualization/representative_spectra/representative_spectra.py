"""
代表性光谱对比图 - 联合训练版本

数据来源:
- 原始光谱: dataset/{dataset_type}/{folder}/*.csv
- 预处理光谱: preprocessed_data/{sample}_data.npy
- 分类标签: output/classification/combined/{dataset}/{sample}_labels.npy

输出位置: output/analysis/combined/representative_spectra.png
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    """ALS基线校正"""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def l2_normalize(spectrum):
    """L2归一化"""
    norm = np.linalg.norm(spectrum)
    if norm > 0:
        return spectrum / norm
    return spectrum


def load_representative_spectra(config):
    """Load representative spectra for three pollution levels"""
    logger = get_logger('representative_spectra')

    base_dir = Path(config.base_dir)
    dataset_dir = base_dir / config.paths['dataset_dir']
    preprocessed_dir = Path(config.paths['preprocessed_dir'])

    datasets = config.get('dataset.datasets')

    # Use the first sample from the first dataset
    dataset = datasets[0]
    dataset_type = dataset['type']
    sample = dataset['samples'][0]
    sample_name = sample['name']

    folder = dataset_dir / sample['folder']

    # Labels are in preprocessed_data/{sample_name}_pp_labels.npy
    labels_path = preprocessed_dir / f"{sample_name}_pp_labels.npy"

    preprocessed_path = preprocessed_dir / f"{sample_name}_data.npy"

    if not labels_path.exists():
        logger.error(f"Label file not found: {labels_path}")
        return None

    labels = np.load(labels_path)
    preprocessed_data = np.load(preprocessed_path) if preprocessed_path.exists() else None

    # Read raw spectra
    csv_files = sorted(folder.glob('*.csv'))
    if not csv_files:
        logger.error(f"No CSV files found: {folder}")
        return None

    # Read wavenumber
    try:
        df = pd.read_csv(csv_files[0], encoding='gbk')
    except:
        df = pd.read_csv(csv_files[0], encoding='utf-8')
    wavenumber = df.iloc[:, 0].values.astype(np.float64)

    # Read all raw spectra
    raw_spectra = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='gbk')
        except:
            df = pd.read_csv(csv_file, encoding='utf-8')
        raw_spectra.append(df.iloc[:, 1].values.astype(np.float64))

    raw_spectra = np.array(raw_spectra).reshape(40, 40, -1)

    result = {
        'wavenumber': wavenumber,
        'labels': labels,
        'raw_spectra': raw_spectra,
        'preprocessed_spectra': preprocessed_data
    }

    logger.info(f"Loaded sample: {sample_name}")
    return result


def plot_representative_spectra(data, output_path, title_suffix='Combined'):
    """绘制三种污染等级的代表性光谱（原始vs预处理）"""
    logger = get_logger('representative_spectra')

    wavenumber = data['wavenumber']
    labels = data['labels']
    raw_spectra = data['raw_spectra']
    preprocessed_spectra = data['preprocessed_spectra']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors_raw = ['#636e72', '#74b9ff', '#fab1a0']
    colors_pre = ['#2d3436', '#0984e3', '#d63031']

    for col, label in enumerate([0, 1, 2]):
        # 找到该类别的样本索引
        indices = np.where(labels.flatten() == label)[0]

        if len(indices) == 0:
            logger.warning(f"类别 {label} 无样本")
            continue

        # 取中间位置的样本作为代表
        sample_idx = indices[len(indices) // 2]
        row_idx, col_idx = sample_idx // 40, sample_idx % 40

        raw_spectrum = raw_spectra[row_idx, col_idx]

        # 预处理：基线校正 + 归一化
        if preprocessed_spectra is not None:
            pre_spectrum = preprocessed_spectra[row_idx, col_idx]
        else:
            baseline = als_baseline(raw_spectrum)
            corrected = raw_spectrum - baseline
            pre_spectrum = l2_normalize(corrected)

        # 上排: 原始光谱
        axes[0, col].plot(wavenumber, raw_spectrum, color=colors_raw[col], linewidth=1.5)
        axes[0, col].set_title(f'{class_names[col]} - Original', fontsize=11, fontweight='bold')
        axes[0, col].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=10)
        axes[0, col].set_ylabel('Intensity', fontsize=10)
        axes[0, col].grid(True, alpha=0.3)

        # 下排: 预处理后光谱
        axes[1, col].plot(wavenumber, pre_spectrum, color=colors_pre[col], linewidth=1.5)
        axes[1, col].set_title(f'{class_names[col]} - Preprocessed', fontsize=11, fontweight='bold')
        axes[1, col].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=10)
        axes[1, col].set_ylabel('Normalized Intensity', fontsize=10)
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle('Representative Spectra by Pollution Level\n'
                 'Top: Original | Bottom: After ALS + L2 Normalization',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"保存: {output_path}")


def main():
    config = get_config()
    logger = get_logger('representative_spectra')

    logger.info("=" * 50)
    logger.info("代表性光谱对比图 - 联合训练")
    logger.info("=" * 50)

    output_dir = Path(config.paths['output_dir']) / 'representative_spectra'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n>>> 加载数据...")
    data = load_representative_spectra(config)

    if data is None:
        logger.error("未找到数据")
        return

    logger.info("\n>>> 绘制光谱对比图...")
    plot_representative_spectra(data, output_dir / 'representative_spectra.png', 'Combined')

    logger.info("\n>>> 完成!")


if __name__ == '__main__':
    main()
