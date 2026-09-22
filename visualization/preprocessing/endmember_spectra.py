"""
端元纯谱可视化：
展示淀粉、PP、PE的纯谱特征

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


def load_endmember_spectra(config):
    """加载端元纯谱"""
    logger = get_logger('endmember_vis')

    base_dir = Path(config.base_dir)
    dataset_dir = base_dir / config.paths['dataset_dir']
    endmembers = config.get('dataset.endmembers')

    spectra = {}

    # 加载淀粉纯谱
    starch_path = dataset_dir / endmembers['starch']['folder'] / endmembers['starch']['file']
    if starch_path.exists():
        try:
            df = pd.read_csv(starch_path, encoding='gbk')
        except:
            df = pd.read_csv(starch_path, encoding='utf-8')
        spectra['starch'] = {
            'wavenumber': df.iloc[:, 0].values.astype(np.float64),
            'intensity': df.iloc[:, 1].values.astype(np.float64),
            'name': 'Starch'
        }
        logger.info(f"加载淀粉纯谱: {starch_path}")

    # 加载PP纯谱
    pp_path = dataset_dir / endmembers['pp']['folder'] / endmembers['pp']['file']
    if pp_path.exists():
        try:
            df = pd.read_csv(pp_path, encoding='gbk')
        except:
            df = pd.read_csv(pp_path, encoding='utf-8')
        spectra['pp'] = {
            'wavenumber': df.iloc[:, 0].values.astype(np.float64),
            'intensity': df.iloc[:, 1].values.astype(np.float64),
            'name': 'PP (Polypropylene)'
        }
        logger.info(f"加载PP纯谱: {pp_path}")

    # 加载PE纯谱
    pe_path = dataset_dir / endmembers['pe']['folder'] / endmembers['pe']['file']
    if pe_path.exists():
        try:
            df = pd.read_csv(pe_path, encoding='gbk')
        except:
            df = pd.read_csv(pe_path, encoding='utf-8')
        spectra['pe'] = {
            'wavenumber': df.iloc[:, 0].values.astype(np.float64),
            'intensity': df.iloc[:, 1].values.astype(np.float64),
            'name': 'PE (Polyethylene)'
        }
        logger.info(f"加载PE纯谱: {pe_path}")

    return spectra


def plot_endmember_spectra(spectra, output_dir):
    """绘制端元纯谱对比图"""
    logger = get_logger('endmember_vis')

    colors = {'starch': '#2ecc71', 'pp': '#3498db', 'pe': '#e74c3c'}

    # 1. 原始纯谱对比
    fig, ax = plt.subplots(figsize=(14, 6))

    for key, data in spectra.items():
        ax.plot(data['wavenumber'], data['intensity'],
                color=colors[key], label=data['name'], linewidth=1.5)

    ax.set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('Endmember Pure Spectra Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    # x-axis from low to high wavenumber (left to right)
    ax.set_xlim([min(spectra['starch']['wavenumber'].min(), spectra.get('pp', spectra['starch'])['wavenumber'].min()),
                 max(spectra['starch']['wavenumber'].max(), spectra.get('pp', spectra['starch'])['wavenumber'].max())])

    plt.tight_layout()
    plt.savefig(output_dir / 'endmember_spectra.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"端元纯谱对比图已保存: {output_dir / 'endmember_spectra.png'}")

    # 2. 预处理后的纯谱对比
    fig, ax = plt.subplots(figsize=(14, 6))

    for key, data in spectra.items():
        # 基线校正 + L2归一化
        baseline = als_baseline(data['intensity'])
        corrected = data['intensity'] - baseline
        normalized = l2_normalize(corrected)

        ax.plot(data['wavenumber'], normalized,
                color=colors[key], label=data['name'], linewidth=1.5)

    ax.set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Preprocessed Endmember Spectra (ALS + L2 Normalization)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    # x-axis from low to high wavenumber (left to right)

    plt.tight_layout()
    plt.savefig(output_dir / 'endmember_spectra_preprocessed.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"预处理后纯谱对比图已保存: {output_dir / 'endmember_spectra_preprocessed.png'}")


def plot_endmember_peaks(spectra, output_dir):
    """绘制带特征峰标注的纯谱图"""
    logger = get_logger('endmember_vis')

    # 特征峰位置
    pp_peaks = [809, 841, 972, 1152, 1330, 1458]
    pe_peaks = [1063, 1130, 1296, 1440]
    starch_peaks = [480, 865, 940, 1085, 1340, 1460]

    colors = {'starch': '#2ecc71', 'pp': '#3498db', 'pe': '#e74c3c'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 淀粉
    if 'starch' in spectra:
        data = spectra['starch']
        baseline = als_baseline(data['intensity'])
        corrected = data['intensity'] - baseline
        normalized = l2_normalize(corrected)

        axes[0].plot(data['wavenumber'], normalized, color=colors['starch'], linewidth=1.5)
        axes[0].set_title('Starch Characteristic Peaks', fontsize=14, fontweight='bold')

        for peak in starch_peaks:
            idx = np.argmin(np.abs(data['wavenumber'] - peak))
            if idx < len(normalized):
                axes[0].axvline(x=peak, color='gray', linestyle='--', alpha=0.5)
                axes[0].annotate(f'{peak}', xy=(peak, normalized[idx]),
                                 xytext=(0, 10), textcoords='offset points',
                                 ha='center', fontsize=9, color='gray')

        axes[0].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=11)
        axes[0].set_ylabel('Normalized Intensity', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        # x-axis from low to high (left to right)

    # PP
    if 'pp' in spectra:
        data = spectra['pp']
        baseline = als_baseline(data['intensity'])
        corrected = data['intensity'] - baseline
        normalized = l2_normalize(corrected)

        axes[1].plot(data['wavenumber'], normalized, color=colors['pp'], linewidth=1.5)
        axes[1].set_title('PP (Polypropylene) Characteristic Peaks', fontsize=14, fontweight='bold')

        for peak in pp_peaks:
            idx = np.argmin(np.abs(data['wavenumber'] - peak))
            if idx < len(normalized):
                axes[1].axvline(x=peak, color='gray', linestyle='--', alpha=0.5)
                axes[1].annotate(f'{peak}', xy=(peak, normalized[idx]),
                                 xytext=(0, 10), textcoords='offset points',
                                 ha='center', fontsize=9, color='gray')

        axes[1].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=11)
        axes[1].set_ylabel('Normalized Intensity', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        # x-axis from low to high (left to right)

    # PE
    if 'pe' in spectra:
        data = spectra['pe']
        baseline = als_baseline(data['intensity'])
        corrected = data['intensity'] - baseline
        normalized = l2_normalize(corrected)

        axes[2].plot(data['wavenumber'], normalized, color=colors['pe'], linewidth=1.5)
        axes[2].set_title('PE (Polyethylene) Characteristic Peaks', fontsize=14, fontweight='bold')

        for peak in pe_peaks:
            idx = np.argmin(np.abs(data['wavenumber'] - peak))
            if idx < len(normalized):
                axes[2].axvline(x=peak, color='gray', linestyle='--', alpha=0.5)
                axes[2].annotate(f'{peak}', xy=(peak, normalized[idx]),
                                 xytext=(0, 10), textcoords='offset points',
                                 ha='center', fontsize=9, color='gray')

        axes[2].set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=11)
        axes[2].set_ylabel('Normalized Intensity', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        # x-axis from low to high (left to right)

    plt.tight_layout()
    plt.savefig(output_dir / 'endmember_peaks.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"特征峰标注图已保存: {output_dir / 'endmember_peaks.png'}")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('endmember_vis')

    logger.info("端元纯谱可视化")

    output_dir = Path(config.paths['output_dir']) / 'preprocessing'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载端元纯谱
    logger.info("\n>>> 加载端元纯谱...")
    spectra = load_endmember_spectra(config)

    if not spectra:
        logger.error("无法加载端元纯谱")
        sys.exit(1)

    # 绘制纯谱对比图
    logger.info("\n>>> 绘制端元纯谱对比图...")
    plot_endmember_spectra(spectra, output_dir)

    # 绘制特征峰标注图
    logger.info("\n>>> 绘制特征峰标注图...")
    plot_endmember_peaks(spectra, output_dir)

    logger.info("\n>>> 完成！")
