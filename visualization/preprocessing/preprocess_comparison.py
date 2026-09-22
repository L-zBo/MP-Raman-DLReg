"""
预处理可视化：
展示ALS基线校正和L2归一化的效果

输出位置：output/preprocessing/
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

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger

# 设置中文字体
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


def load_sample_spectrum(config):
    """加载示例光谱"""
    logger = get_logger('preprocess_vis')

    base_dir = Path(config.base_dir)
    dataset_dir = base_dir / config.paths['dataset_dir']

    datasets = config.get('dataset.datasets')
    pp_dataset = next((d for d in datasets if d['type'] == 'PP+淀粉'), None)

    if pp_dataset is None:
        return None, None

    sample = pp_dataset['samples'][0]
    folder = dataset_dir / sample['folder']
    csv_files = sorted(folder.glob('*.csv'))

    if not csv_files:
        return None, None

    # 读取第一个光谱
    csv_file = csv_files[0]
    try:
        df = pd.read_csv(csv_file, encoding='gbk')
    except:
        df = pd.read_csv(csv_file, encoding='utf-8')

    wavenumber = df.iloc[:, 0].values.astype(np.float64)
    spectrum = df.iloc[:, 1].values.astype(np.float64)

    logger.info(f"加载光谱: {csv_file.name}")
    return wavenumber, spectrum


def plot_als_comparison(wavenumber, spectrum, output_dir):
    """绘制ALS基线校正对比图"""
    logger = get_logger('preprocess_vis')

    # 计算基线
    baseline = als_baseline(spectrum)
    corrected = spectrum - baseline

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 原始光谱 + 基线
    axes[0].plot(wavenumber, spectrum, 'b-', label='Original Spectrum', linewidth=1.5)
    axes[0].plot(wavenumber, baseline, 'r--', label='ALS Baseline', linewidth=1.5)
    axes[0].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    axes[0].set_ylabel('Intensity', fontsize=12)
    axes[0].set_title('Original Spectrum with ALS Baseline', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    # 校正后光谱
    axes[1].plot(wavenumber, corrected, 'g-', label='Baseline Corrected', linewidth=1.5)
    axes[1].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    axes[1].set_ylabel('Intensity', fontsize=12)
    axes[1].set_title('After ALS Baseline Correction', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    plt.tight_layout()
    plt.savefig(output_dir / 'als_baseline_correction.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"ALS基线校正图已保存: {output_dir / 'als_baseline_correction.png'}")


def plot_normalization_comparison(wavenumber, spectrum, output_dir):
    """绘制归一化对比图"""
    logger = get_logger('preprocess_vis')

    # 基线校正
    baseline = als_baseline(spectrum)
    corrected = spectrum - baseline

    # L2归一化
    normalized = l2_normalize(corrected)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 基线校正后
    axes[0].plot(wavenumber, corrected, 'g-', label='Baseline Corrected', linewidth=1.5)
    axes[0].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    axes[0].set_ylabel('Intensity', fontsize=12)
    axes[0].set_title('After Baseline Correction', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    # L2归一化后
    axes[1].plot(wavenumber, normalized, 'm-', label='L2 Normalized', linewidth=1.5)
    axes[1].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    axes[1].set_ylabel('Normalized Intensity', fontsize=12)
    axes[1].set_title('After L2 Normalization', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    plt.tight_layout()
    plt.savefig(output_dir / 'normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"归一化对比图已保存: {output_dir / 'normalization_comparison.png'}")


def plot_preprocessing_pipeline(wavenumber, spectrum, output_dir):
    """绘制完整预处理流程图"""
    logger = get_logger('preprocess_vis')

    # 各阶段处理
    baseline = als_baseline(spectrum)
    corrected = spectrum - baseline
    normalized = l2_normalize(corrected)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 原始光谱
    axes[0, 0].plot(wavenumber, spectrum, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=11)
    axes[0, 0].set_ylabel('Intensity', fontsize=11)
    axes[0, 0].set_title('Step 1: Original Spectrum', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    # 2. 基线拟合
    axes[0, 1].plot(wavenumber, spectrum, 'b-', label='Original', linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(wavenumber, baseline, 'r--', label='Baseline', linewidth=2)
    axes[0, 1].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=11)
    axes[0, 1].set_ylabel('Intensity', fontsize=11)
    axes[0, 1].set_title('Step 2: ALS Baseline Fitting', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    # 3. 基线校正
    axes[1, 0].plot(wavenumber, corrected, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=11)
    axes[1, 0].set_ylabel('Intensity', fontsize=11)
    axes[1, 0].set_title('Step 3: Baseline Corrected', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    # 4. L2归一化
    axes[1, 1].plot(wavenumber, normalized, 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=11)
    axes[1, 1].set_ylabel('Normalized Intensity', fontsize=11)
    axes[1, 1].set_title('Step 4: L2 Normalized (Final)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    plt.suptitle('Preprocessing Pipeline: Raw → ALS Baseline Correction → L2 Normalization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"预处理流程图已保存: {output_dir / 'preprocessing_pipeline.png'}")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('preprocess_vis')

    logger.info("预处理可视化")

    output_dir = Path(config.paths['output_dir']) / 'preprocessing'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载示例光谱
    logger.info("\n>>> 加载示例光谱...")
    wavenumber, spectrum = load_sample_spectrum(config)

    if wavenumber is None:
        logger.error("无法加载光谱数据")
        sys.exit(1)

    # 绘制各种对比图
    logger.info("\n>>> 绘制ALS基线校正对比图...")
    plot_als_comparison(wavenumber, spectrum, output_dir)

    logger.info("\n>>> 绘制归一化对比图...")
    plot_normalization_comparison(wavenumber, spectrum, output_dir)

    logger.info("\n>>> 绘制预处理流程图...")
    plot_preprocessing_pipeline(wavenumber, spectrum, output_dir)

    logger.info("\n>>> 完成！")
