"""
光谱解混模块：基于NNLS的线性混合模型
支持并行处理以提高性能
"""
import numpy as np
from scipy.optimize import nnls
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_config, get_logger

# 全局变量用于并行处理时共享端元矩阵
_shared_endmembers = None
_shared_normalize = True


def _init_worker(endmembers, normalize):
    """初始化工作进程的共享数据"""
    global _shared_endmembers, _shared_normalize
    _shared_endmembers = endmembers
    _shared_normalize = normalize


def unmix_pixel(mixed_spectrum, endmembers, normalize=True):
    """
    对单个像素进行光谱解混

    Args:
        mixed_spectrum: 混合光谱 (n_bands,)
        endmembers: 端元矩阵 (n_endmembers, n_bands)
        normalize: 是否归一化丰度

    Returns:
        abundances: 丰度数组 (n_endmembers,)
    """
    abundances, _ = nnls(endmembers.T, mixed_spectrum)
    # 归一化丰度
    if normalize:
        total = abundances.sum()
        if total > 0:
            abundances = abundances / total
    return abundances


def _unmix_single_pixel(mixed_spectrum):
    """解混单个像素的辅助函数（用于并行处理，使用共享数据）"""
    global _shared_endmembers, _shared_normalize
    return unmix_pixel(mixed_spectrum, _shared_endmembers, _shared_normalize)


def unmix_hyperspectral(hypercube, endmembers, n_jobs=None, normalize=None):
    """
    对整个高光谱数据进行解混（支持并行处理）

    Args:
        hypercube: 高光谱数据 (H, W, bands)
        endmembers: 端元矩阵 (n_endmembers, bands)
        n_jobs: 并行进程数，None表示从配置读取，-1表示使用所有CPU
        normalize: 是否归一化丰度

    Returns:
        abundance_maps: 丰度图 (H, W, n_endmembers)
    """
    config = get_config()
    logger = get_logger('unmix')

    # 从配置获取参数
    if n_jobs is None:
        n_jobs = config.get('unmixing.n_jobs', -1)
    if normalize is None:
        normalize = config.get('unmixing.normalize', True)

    h, w, bands = hypercube.shape
    n_endmembers = endmembers.shape[0]
    total_pixels = h * w

    # 将3D数据展平为2D
    spectra_flat = hypercube.reshape(-1, bands)

    # 决定是否使用并行处理
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1 and total_pixels > 100:
        logger.info(f"使用 {n_jobs} 个进程并行解混 {total_pixels} 个像素...")

        # 并行处理（使用initializer在子进程中初始化共享数据）
        abundances_flat = np.zeros((total_pixels, n_endmembers))
        with ProcessPoolExecutor(max_workers=n_jobs,
                                  initializer=_init_worker,
                                  initargs=(endmembers, normalize)) as executor:
            futures = {executor.submit(_unmix_single_pixel, spectra_flat[i]): i
                      for i in range(total_pixels)}

            for future in as_completed(futures):
                i = futures[future]
                try:
                    abundances_flat[i] = future.result()
                except Exception as e:
                    logger.warning(f"像素 {i} 解混失败: {e}")
                    abundances_flat[i] = np.zeros(n_endmembers)
    else:
        logger.info(f"串行解混 {total_pixels} 个像素...")
        abundances_flat = np.zeros((total_pixels, n_endmembers))
        for i in range(total_pixels):
            abundances_flat[i] = unmix_pixel(spectra_flat[i], endmembers, normalize)

    # 恢复为3D形状
    return abundances_flat.reshape(h, w, n_endmembers)


def unmix_hyperspectral_vectorized(hypercube, endmembers, normalize=True):
    """
    向量化解混（使用矩阵运算，但仍需逐像素求解NNLS）

    Args:
        hypercube: 高光谱数据 (H, W, bands)
        endmembers: 端元矩阵 (n_endmembers, bands)
        normalize: 是否归一化丰度

    Returns:
        abundance_maps: 丰度图 (H, W, n_endmembers)
    """
    h, w, bands = hypercube.shape
    n_endmembers = endmembers.shape[0]
    abundance_maps = np.zeros((h, w, n_endmembers))

    for i in range(h):
        for j in range(w):
            abundance_maps[i, j, :] = unmix_pixel(hypercube[i, j, :], endmembers, normalize)

    return abundance_maps


def postprocess_abundance(abundance_map):
    """
    丰度后处理：归一化与异常值检查

    Args:
        abundance_map: 丰度图

    Returns:
        处理后的丰度图
    """
    # 裁剪到[0,1]范围
    abundance_map = np.clip(abundance_map, 0, 1)
    return abundance_map


def process_sample(sample_name, data_dir, output_dir, endmembers, n_jobs=None):
    """
    处理单个样本的解混

    Args:
        sample_name: 样本名称
        data_dir: 数据目录
        output_dir: 输出目录
        endmembers: 端元矩阵
        n_jobs: 并行进程数

    Returns:
        abundance: 丰度数组
    """
    config = get_config()
    logger = get_logger('unmix')
    logger.info(f"解混 {sample_name}...")

    mixed = np.load(data_dir / f'{sample_name}_data.npy')
    abundance = unmix_hyperspectral(mixed, endmembers, n_jobs=n_jobs)
    abundance[:, :, 1] = postprocess_abundance(abundance[:, :, 1])

    # 保存到 abundance 子目录（使用实验方案路径）
    abundance_dir = output_dir / 'abundance' / config.dataset_type / config.experiment_name
    abundance_dir.mkdir(parents=True, exist_ok=True)
    np.save(abundance_dir / f'{sample_name.replace("mixed", "abundance")}.npy', abundance)

    return abundance


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('unmix')

    base_dir = config.base_dir
    data_dir = Path(config.paths['preprocessed_dir'])
    output_dir = Path(config.paths['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # 加载预处理后的三端元纯谱
    starch_spec = np.load(data_dir / 'starch_spectrum.npy')
    pp_spec = np.load(data_dir / 'pp_spectrum.npy')
    pe_spec = np.load(data_dir / 'pe_spectrum.npy')
    # 端元矩阵：[淀粉, PP, PE]
    endmembers = np.vstack([starch_spec, pp_spec, pe_spec])
    logger.info(f"端元矩阵形状: {endmembers.shape} (淀粉, PP, PE)")

    # 解混所有数据集的样本
    datasets = config.get('dataset.datasets')
    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']
        logger.info(f"\n解混数据集: {dataset_type}")

        for sample in samples:
            name = sample['name']
            logger.info(f"解混 {name}...")

            mixed = np.load(data_dir / f'{name}_data.npy')
            abundance = unmix_hyperspectral(mixed, endmembers)

            # 后处理所有丰度通道
            for i in range(abundance.shape[2]):
                abundance[:, :, i] = postprocess_abundance(abundance[:, :, i])

            # 保存到 abundance/abundance_npy 子目录（按数据集类型分类）
            abundance_dir = output_dir / 'abundance' / 'abundance_npy' / dataset_type
            abundance_dir.mkdir(parents=True, exist_ok=True)
            np.save(abundance_dir / f'{name}_abundance.npy', abundance)

            # 打印丰度统计
            logger.info(f"  淀粉丰度: [{abundance[:,:,0].min():.4f}, {abundance[:,:,0].max():.4f}]")
            logger.info(f"  PP丰度: [{abundance[:,:,1].min():.4f}, {abundance[:,:,1].max():.4f}]")
            logger.info(f"  PE丰度: [{abundance[:,:,2].min():.4f}, {abundance[:,:,2].max():.4f}]")

    logger.info("\n解混完成！")
