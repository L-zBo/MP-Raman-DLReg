"""
微塑料类型自动检测模块
基于端元光谱特征自动判断样品中是PP还是PE

方法：
1. 特征峰检测：检测PP/PE各自的特征峰强度
2. 光谱相关性：计算与PP/PE纯谱的相关系数
3. 解混残差：比较二端元解混的残差

输出：每个像素点的微塑料类型判断结果
"""
import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_config, get_logger

# PP和PE的特征峰位置 (cm⁻¹)
PP_PEAKS = [809, 841, 972, 1152, 1330, 1458]
PE_PEAKS = [1063, 1130, 1296, 1440]


def find_nearest_idx(wavenumbers, target):
    """找到最接近目标波数的索引"""
    return np.argmin(np.abs(wavenumbers - target))


def detect_peak_intensity(spectrum, wavenumbers, peak_positions, window=5):
    """检测特征峰强度"""
    intensities = []
    for peak in peak_positions:
        idx = find_nearest_idx(wavenumbers, peak)
        # 取窗口内的最大值
        start = max(0, idx - window)
        end = min(len(spectrum), idx + window + 1)
        intensities.append(np.max(spectrum[start:end]))
    return np.mean(intensities)


def spectral_correlation(spectrum, reference):
    """计算光谱相关系数"""
    return np.corrcoef(spectrum, reference)[0, 1]


def unmix_residual(spectrum, endmembers):
    """计算解混残差"""
    abundances, residual = nnls(endmembers.T, spectrum)
    reconstructed = endmembers.T @ abundances
    return np.sqrt(np.mean((spectrum - reconstructed) ** 2))


def detect_microplastic_type(spectrum, wavenumbers, starch_spec, pp_spec, pe_spec, method='combined'):
    """
    检测单个像素的微塑料类型
    使用 PP峰/PE峰 比值来判断

    Returns:
        'PP' or 'PE', pp_pe_ratio, pe_intensity
    """
    # 关键特征峰
    PP_KEY_PEAKS = [809, 841]  # PP特有峰
    PE_KEY_PEAKS = [1296, 1440]  # PE特有峰

    # 计算特征峰强度
    pp_intensity = sum(spectrum[find_nearest_idx(wavenumbers, p)] for p in PP_KEY_PEAKS)
    pe_intensity = sum(spectrum[find_nearest_idx(wavenumbers, p)] for p in PE_KEY_PEAKS)

    # 计算比值
    ratio = pp_intensity / (pe_intensity + 1e-6)

    # 判断：PP样本中位数约0.90-0.97，PE样本中位数约0.46-0.65
    # 阈值设为0.75（PP最小中位数0.90和PE最大中位数0.65的中间值）
    if ratio > 0.75:
        return 'PP', ratio, pe_intensity
    else:
        return 'PE', ratio, pe_intensity


def detect_hyperspectral(hypercube, wavenumbers, starch_spec, pp_spec, pe_spec):
    """对整个高光谱数据进行微塑料类型检测"""
    h, w, bands = hypercube.shape
    type_map = np.empty((h, w), dtype='<U4')
    pp_pe_ratio_map = np.zeros((h, w))
    pe_intensity_map = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            mp_type, pp_pe_ratio, pe_intensity = detect_microplastic_type(
                hypercube[i, j, :], wavenumbers, starch_spec, pp_spec, pe_spec
            )
            type_map[i, j] = mp_type
            pp_pe_ratio_map[i, j] = pp_pe_ratio
            pe_intensity_map[i, j] = pe_intensity

    return type_map, pp_pe_ratio_map, pe_intensity_map


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('detect_type')

    base_dir = config.base_dir
    data_dir = Path(config.paths['preprocessed_dir'])
    output_dir = Path(config.paths['output_dir']) / 'detection'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载端元纯谱
    starch_spec = np.load(data_dir / 'starch_spectrum.npy')
    pp_spec = np.load(data_dir / 'pp_spectrum.npy')
    pe_spec = np.load(data_dir / 'pe_spectrum.npy')

    logger.info("端元光谱已加载")
    logger.info(f"  淀粉: {starch_spec.shape}")
    logger.info(f"  PP: {pp_spec.shape}")
    logger.info(f"  PE: {pe_spec.shape}")

    # 遍历所有数据集
    datasets = config.get('dataset.datasets')

    results = []

    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']

        logger.info(f"\n{'='*50}")
        logger.info(f"检测数据集: {dataset_type}")
        logger.info(f"{'='*50}")

        for sample in samples:
            name = sample['name']
            data = np.load(data_dir / f'{name}_data.npy')
            wavenumbers = np.load(data_dir / f'{name}_wavenumbers.npy')

            logger.info(f"\n检测 {name}...")

            # 检测微塑料类型
            type_map, pp_pe_ratio_map, pe_intensity_map = detect_hyperspectral(
                data, wavenumbers, starch_spec, pp_spec, pe_spec
            )

            # 统计结果
            pp_count = np.sum(type_map == 'PP')
            pe_count = np.sum(type_map == 'PE')
            total = type_map.size

            pp_ratio = pp_count / total * 100
            pe_ratio = pe_count / total * 100

            # 判断主要类型
            if pp_ratio > 60:
                detected_type = 'PP'
            elif pe_ratio > 60:
                detected_type = 'PE'
            else:
                detected_type = 'MIXED'

            logger.info(f"  PP像素: {pp_count} ({pp_ratio:.1f}%)")
            logger.info(f"  PE像素: {pe_count} ({pe_ratio:.1f}%)")
            logger.info(f"  检测结果: {detected_type}")
            logger.info(f"  真实类型: {'PP' if 'pp_' in name else 'PE'}")
            logger.info(f"  判断{'正确' if (detected_type == 'PP' and 'pp_' in name) or (detected_type == 'PE' and 'pe_' in name) else '错误'}")

            # 保存结果
            np.save(output_dir / f'{name}_type_map.npy', type_map)
            np.save(output_dir / f'{name}_pp_pe_ratio.npy', pp_pe_ratio_map)
            np.save(output_dir / f'{name}_pe_intensity.npy', pe_intensity_map)

            results.append({
                'name': name,
                'true_type': 'PP' if 'pp_' in name else 'PE',
                'detected_type': detected_type,
                'pp_ratio': pp_ratio,
                'pe_ratio': pe_ratio,
                'correct': (detected_type == 'PP' and 'pp_' in name) or (detected_type == 'PE' and 'pe_' in name)
            })

    # 打印总结
    logger.info(f"\n{'='*50}")
    logger.info("检测结果总结")
    logger.info(f"{'='*50}")

    correct_count = sum(1 for r in results if r['correct'])
    logger.info(f"准确率: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")

    logger.info("\n详细结果:")
    logger.info(f"{'样本名':<15} {'真实类型':<8} {'检测类型':<8} {'PP%':<8} {'PE%':<8} {'结果':<6}")
    logger.info("-" * 60)
    for r in results:
        status = 'true' if r['correct'] else 'false'
        logger.info(f"{r['name']:<15} {r['true_type']:<8} {r['detected_type']:<8} {r['pp_ratio']:<8.1f} {r['pe_ratio']:<8.1f} {status}")
