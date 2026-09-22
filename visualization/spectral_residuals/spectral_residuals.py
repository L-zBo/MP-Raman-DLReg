"""
光谱残差分析 - 双头模型版本
评估NNLS光谱解混的质量
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_config, get_logger
from visualization.spectrum.visualize import setup_matplotlib

logger = get_logger(__name__)


def load_endmembers(config):
    """加载端元光谱"""
    preprocessed_dir = Path(config['paths']['preprocessed_dir'])

    starch = np.load(preprocessed_dir / 'starch_spectrum.npy')
    pp = np.load(preprocessed_dir / 'pp_spectrum.npy')
    pe = np.load(preprocessed_dir / 'pe_spectrum.npy')
    wavenumbers = np.load(preprocessed_dir / 'wavenumbers.npy')

    # 端元矩阵: shape [3, 1024] - [淀粉, PP, PE]
    endmembers = np.stack([starch, pp, pe], axis=0)

    return endmembers, wavenumbers


def select_representative_pixels(abundance_map, num_per_category=3):
    """
    选择代表性像素进行可视化

    参数:
        abundance_map: 丰度图 [H, W, 3]
        num_per_category: 每个类别选择的像素数

    返回:
        selected_pixels: [(y, x, category_name), ...]
    """
    H, W, _ = abundance_map.shape

    # 计算总微塑料丰度 (PP + PE)
    mp_abundance = abundance_map[:, :, 1] + abundance_map[:, :, 2]

    # 定义污染等级阈值
    low_threshold = 0.2
    high_threshold = 0.5

    # 分类像素
    no_pollution_mask = mp_abundance < low_threshold
    light_pollution_mask = (mp_abundance >= low_threshold) & (mp_abundance < high_threshold)
    heavy_pollution_mask = mp_abundance >= high_threshold

    selected_pixels = []

    # 从每个类别中随机选择像素
    categories = [
        (no_pollution_mask, 'no'),
        (light_pollution_mask, 'light'),
        (heavy_pollution_mask, 'heavy')
    ]

    for mask, category_name in categories:
        if np.any(mask):
            # 获取该类别的所有像素坐标
            coords = np.argwhere(mask)

            # 随机选择
            if len(coords) > num_per_category:
                indices = np.random.choice(len(coords), num_per_category, replace=False)
                selected_coords = coords[indices]
            else:
                selected_coords = coords

            for y, x in selected_coords:
                selected_pixels.append((y, x, category_name))

    return selected_pixels


def plot_spectral_residual(original_spectrum, fitted_spectrum, residual, wavenumbers,
                           sample_name, pixel_y, pixel_x, category_name, output_path):
    """
    绘制光谱残差图

    参数:
        original_spectrum: 原始观测光谱 [1024]
        fitted_spectrum: NNLS拟合光谱 [1024]
        residual: 残差 [1024]
        wavenumbers: 拉曼位移 [1024]
        sample_name: 样本名称
        pixel_y, pixel_x: 像素坐标
        category_name: 污染类别名称
        output_path: 输出路径
    """
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 6))

    # 计算残差偏移量（将残差曲线向下平移）
    residual_offset = -0.2

    # 绘制原始光谱（黑色实线）
    ax.plot(wavenumbers, original_spectrum, 'k-', linewidth=1.5, label='Original Spectrum')

    # 绘制拟合光谱（红色虚线）
    ax.plot(wavenumbers, fitted_spectrum, 'r--', linewidth=1.5, label='NNLS Fitted Spectrum')

    # 绘制残差（蓝色实线，底部）
    ax.plot(wavenumbers, residual + residual_offset, 'b-', linewidth=1.0, label='Residual (offset)')

    # 添加零线参考
    ax.axhline(y=residual_offset, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # 计算RMSE
    rmse = np.sqrt(np.mean(residual**2))

    # 设置标题和标签
    ax.set_title(f'Spectral Unmixing Residual Analysis - {sample_name}\nPixel: ({pixel_x}, {pixel_y}) | Category: {category_name} | RMSE: {rmse:.4f}',
                fontsize=12, pad=15)
    ax.set_xlabel(r'Raman Shift ($cm^{-1}$)', fontsize=11)
    ax.set_ylabel('Normalized Intensity', fontsize=11)

    # 设置图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"已保存光谱残差图: {output_path}")


def process_sample(sample_name, original_spectra, abundance_map, endmembers, wavenumbers, output_dir):
    """
    处理单个样本的光谱残差分析

    参数:
        sample_name: 样本名称
        original_spectra: 原始光谱 [H, W, 1024]
        abundance_map: 丰度图 [H, W, 3]
        endmembers: 端元矩阵 [3, 1024]
        wavenumbers: 拉曼位移 [1024]
        output_dir: 输出目录
    """
    # 选择代表性像素
    selected_pixels = select_representative_pixels(abundance_map, num_per_category=3)

    logger.info(f"样本 {sample_name}: 选择了 {len(selected_pixels)} 个代表性像素")

    for pixel_y, pixel_x, category_name in selected_pixels:
        # 获取原始光谱
        original_spectrum = original_spectra[pixel_y, pixel_x, :]

        # 获取丰度
        abundance = abundance_map[pixel_y, pixel_x, :]  # [3] - [淀粉, PP, PE]

        # 重建拟合光谱
        fitted_spectrum = abundance @ endmembers  # [3] @ [3, 1024] = [1024]

        # 计算残差
        residual = original_spectrum - fitted_spectrum

        # 生成输出文件名
        output_filename = f'pixel_{pixel_y:02d}_{pixel_x:02d}_{category_name}.png'
        output_path = output_dir / sample_name / output_filename

        # 绘制残差图
        plot_spectral_residual(
            original_spectrum, fitted_spectrum, residual, wavenumbers,
            sample_name, pixel_y, pixel_x, category_name, output_path
        )


def main():
    """主函数"""
    config = get_config()

    # 设置路径
    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    abundance_base_dir = Path(config.paths['output_dir']) / 'abundance'
    output_dir = Path(config.paths['output_dir']) / 'spectral_residuals'

    logger.info("开始生成光谱残差分析图（双头模型）...")

    # 加载端元光谱
    endmembers, wavenumbers = load_endmembers(config)
    logger.info(f"已加载端元光谱，形状: {endmembers.shape}")

    # 从config动态获取所有样本
    datasets = config.dataset.get('datasets', [])

    for dataset_info in datasets:
        dataset_type = dataset_info.get('type', '')
        samples = dataset_info.get('samples', [])

        for sample_info in samples:
            sample_name = sample_info.get('name', '')

            # 加载原始光谱数据
            spectra_path = preprocessed_dir / f'{sample_name}_data.npy'
            if not spectra_path.exists():
                logger.warning(f"未找到光谱数据: {spectra_path}")
                continue

            original_spectra = np.load(spectra_path)

            # 加载丰度数据 - 使用样本名作为文件夹名
            abundance_path = abundance_base_dir / sample_name / 'abundance.npy'
            if not abundance_path.exists():
                logger.warning(f"未找到丰度数据: {abundance_path}")
                continue

            abundance_map = np.load(abundance_path)

            # 处理样本
            logger.info(f"处理样本: {sample_name} (类型: {dataset_type})")
            process_sample(sample_name, original_spectra, abundance_map, endmembers, wavenumbers, output_dir)

    logger.info(f"光谱残差分析完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
