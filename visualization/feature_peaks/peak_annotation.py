"""
特征峰标注可视化：
在混合光谱上标注PP/PE特征峰位置

输出位置：output/feature_peaks/combined/
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


# 特征峰定义
PP_PEAKS = {
    809: 'C-C stretching',
    841: 'CH₂ rocking',
    972: 'CH₃ rocking',
    1152: 'C-C stretching',
    1330: 'CH₂ twisting',
    1458: 'CH₂ bending'
}

PE_PEAKS = {
    1063: 'C-C stretching',
    1130: 'C-C stretching',
    1296: 'CH₂ twisting',
    1440: 'CH₂ bending'
}

STARCH_PEAKS = {
    480: 'Skeletal mode',
    865: 'C-O-C stretching',
    940: 'α-1,4 glycosidic',
    1085: 'C-O stretching',
    1340: 'CH₂ twisting',
    1460: 'CH₂ bending'
}


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


def load_mixed_spectrum(config, dataset_type='PP+淀粉'):
    """加载混合光谱"""
    logger = get_logger('peak_annotation')

    base_dir = Path(config.base_dir)
    dataset_dir = base_dir / config.paths['dataset_dir']

    datasets = config.get('dataset.datasets')
    target_dataset = next((d for d in datasets if d['type'] == dataset_type), None)

    if target_dataset is None:
        return None, None

    sample = target_dataset['samples'][0]
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


def plot_pp_peaks(wavenumber, spectrum, output_dir):
    """绘制PP特征峰标注图"""
    logger = get_logger('peak_annotation')

    # 预处理
    baseline = als_baseline(spectrum)
    corrected = spectrum - baseline
    normalized = l2_normalize(corrected)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(wavenumber, normalized, 'b-', linewidth=1.5, label='Mixed Spectrum')

    # 标注PP特征峰
    for peak, description in PP_PEAKS.items():
        idx = np.argmin(np.abs(wavenumber - peak))
        if idx < len(normalized):
            ax.axvline(x=peak, color='#3498db', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.annotate(f'{peak}\n{description}',
                        xy=(peak, normalized[idx]),
                        xytext=(0, 20), textcoords='offset points',
                        ha='center', fontsize=9, color='#3498db',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('PP (Polypropylene) Characteristic Peaks in Mixed Spectrum', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    plt.tight_layout()
    plt.savefig(output_dir / 'pp_characteristic_peaks.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"PP特征峰标注图已保存: {output_dir / 'pp_characteristic_peaks.png'}")


def plot_pe_peaks(wavenumber, spectrum, output_dir):
    """绘制PE特征峰标注图"""
    logger = get_logger('peak_annotation')

    # 预处理
    baseline = als_baseline(spectrum)
    corrected = spectrum - baseline
    normalized = l2_normalize(corrected)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(wavenumber, normalized, 'b-', linewidth=1.5, label='Mixed Spectrum')

    # 标注PE特征峰
    for peak, description in PE_PEAKS.items():
        idx = np.argmin(np.abs(wavenumber - peak))
        if idx < len(normalized):
            ax.axvline(x=peak, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.annotate(f'{peak}\n{description}',
                        xy=(peak, normalized[idx]),
                        xytext=(0, 20), textcoords='offset points',
                        ha='center', fontsize=9, color='#e74c3c',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('PE (Polyethylene) Characteristic Peaks in Mixed Spectrum', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    plt.tight_layout()
    plt.savefig(output_dir / 'pe_characteristic_peaks.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"PE特征峰标注图已保存: {output_dir / 'pe_characteristic_peaks.png'}")


def plot_all_peaks_comparison(wavenumber, spectrum, output_dir):
    """绘制所有特征峰对比图"""
    logger = get_logger('peak_annotation')

    # 预处理
    baseline = als_baseline(spectrum)
    corrected = spectrum - baseline
    normalized = l2_normalize(corrected)

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(wavenumber, normalized, 'k-', linewidth=1.5, label='Mixed Spectrum')

    # 标注淀粉特征峰
    for peak in STARCH_PEAKS.keys():
        idx = np.argmin(np.abs(wavenumber - peak))
        if idx < len(normalized):
            ax.axvline(x=peak, color='#2ecc71', linestyle=':', alpha=0.6, linewidth=1.5)

    # 标注PP特征峰
    for peak in PP_PEAKS.keys():
        idx = np.argmin(np.abs(wavenumber - peak))
        if idx < len(normalized):
            ax.axvline(x=peak, color='#3498db', linestyle='--', alpha=0.6, linewidth=1.5)

    # 标注PE特征峰
    for peak in PE_PEAKS.keys():
        idx = np.argmin(np.abs(wavenumber - peak))
        if idx < len(normalized):
            ax.axvline(x=peak, color='#e74c3c', linestyle='-.', alpha=0.6, linewidth=1.5)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=1.5, label='Mixed Spectrum'),
        Line2D([0], [0], color='#2ecc71', linestyle=':', linewidth=1.5, label='Starch Peaks'),
        Line2D([0], [0], color='#3498db', linestyle='--', linewidth=1.5, label='PP Peaks'),
        Line2D([0], [0], color='#e74c3c', linestyle='-.', linewidth=1.5, label='PE Peaks'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.set_xlabel(r'Wavenumber ($cm^{-1}$)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title('Characteristic Peaks Comparison: Starch vs PP vs PE', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # x轴从低到高（左到右）

    plt.tight_layout()
    plt.savefig(output_dir / 'all_peaks_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"特征峰对比图已保存: {output_dir / 'all_peaks_comparison.png'}")


def plot_peak_table(output_dir):
    """生成特征峰表格图"""
    logger = get_logger('peak_annotation')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # 准备表格数据
    table_data = []
    table_data.append(['Material', r'Peak ($cm^{-1}$)', 'Assignment'])

    for peak, desc in STARCH_PEAKS.items():
        table_data.append(['Starch', str(peak), desc])

    for peak, desc in PP_PEAKS.items():
        table_data.append(['PP', str(peak), desc])

    for peak, desc in PE_PEAKS.items():
        table_data.append(['PE', str(peak), desc])

    # 创建表格
    table = ax.table(cellText=table_data[1:],
                     colLabels=table_data[0],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.2, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 设置行颜色
    colors = {'Starch': '#d5f5e3', 'PP': '#d6eaf8', 'PE': '#fadbd8'}
    for i, row in enumerate(table_data[1:], 1):
        material = row[0]
        for j in range(3):
            table[(i, j)].set_facecolor(colors.get(material, 'white'))

    ax.set_title('Characteristic Raman Peaks of Starch, PP, and PE', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'peak_assignment_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"特征峰表格已保存: {output_dir / 'peak_assignment_table.png'}")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('peak_annotation')

    logger.info("特征峰标注可视化（联合训练）")

    output_dir = Path(config.paths['output_dir']) / 'feature_peaks'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载PP混合光谱
    logger.info("\n>>> 加载PP混合光谱...")
    wavenumber_pp, spectrum_pp = load_mixed_spectrum(config, 'PP+淀粉')

    if wavenumber_pp is not None:
        logger.info("\n>>> 绘制PP特征峰标注图...")
        plot_pp_peaks(wavenumber_pp, spectrum_pp, output_dir)

    # 加载PE混合光谱
    logger.info("\n>>> 加载PE混合光谱...")
    wavenumber_pe, spectrum_pe = load_mixed_spectrum(config, 'PE+淀粉')

    if wavenumber_pe is not None:
        logger.info("\n>>> 绘制PE特征峰标注图...")
        plot_pe_peaks(wavenumber_pe, spectrum_pe, output_dir)

    # 绘制对比图
    if wavenumber_pp is not None:
        logger.info("\n>>> 绘制特征峰对比图...")
        plot_all_peaks_comparison(wavenumber_pp, spectrum_pp, output_dir)

    # 生成特征峰表格
    logger.info("\n>>> 生成特征峰表格...")
    plot_peak_table(output_dir)

    logger.info("\n>>> 完成！")
