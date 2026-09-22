"""
光谱预处理模块：基线校正(ALS) + L2归一化
支持并行处理以提高性能，包含完整的类型注解和数据验证
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List, Union

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_config, get_logger, validate_spectrum, clean_spectrum

# 常量定义
EPSILON = 1e-10  # 防止除零的小数
DEFAULT_ALS_LAMBDA = 1e5
DEFAULT_ALS_P = 0.01
DEFAULT_ALS_NITER = 10
SUPPORTED_ENCODINGS = ['gbk', 'utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'gb2312', 'gb18030']


def als_baseline(
    y: np.ndarray,
    lam: float = DEFAULT_ALS_LAMBDA,
    p: float = DEFAULT_ALS_P,
    niter: int = DEFAULT_ALS_NITER
) -> np.ndarray:
    """
    ALS (Asymmetric Least Squares) 基线校正

    Args:
        y: 输入光谱
        lam: 平滑度参数 (lambda)，越大基线越平滑
        p: 非对称权重参数，通常在0.001-0.1之间
        niter: 迭代次数

    Returns:
        估计的基线

    Raises:
        ValueError: 如果输入参数无效
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("输入必须是1维numpy数组")

    if len(y) < 3:
        raise ValueError(f"光谱长度必须至少为3，当前长度: {len(y)}")

    if lam <= 0:
        raise ValueError(f"lambda必须大于0，当前值: {lam}")

    if not 0 < p < 1:
        raise ValueError(f"p必须在(0,1)之间，当前值: {p}")

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


def l2_normalize(spectrum: np.ndarray, eps: float = EPSILON) -> np.ndarray:
    """
    L2向量归一化，带有安全的零值处理

    Args:
        spectrum: 输入光谱
        eps: 防止除零的小数

    Returns:
        归一化后的光谱
    """
    norm = np.linalg.norm(spectrum)
    if norm > eps:
        return spectrum / norm
    else:
        # 如果范数接近零，返回零向量并发出警告
        warnings.warn("光谱范数接近零，返回零向量")
        return np.zeros_like(spectrum)


def preprocess_spectrum(
    spectrum: np.ndarray,
    lam: float = DEFAULT_ALS_LAMBDA,
    p: float = DEFAULT_ALS_P,
    niter: int = DEFAULT_ALS_NITER,
    clean_data: bool = True
) -> np.ndarray:
    """
    预处理单条光谱：基线校正 + L2归一化

    Args:
        spectrum: 输入光谱
        lam: ALS平滑度参数
        p: ALS非对称权重
        niter: ALS迭代次数
        clean_data: 是否清理异常值（NaN/Inf）

    Returns:
        预处理后的光谱
    """
    if clean_data:
        spectrum = clean_spectrum(spectrum)

    baseline = als_baseline(spectrum, lam, p, niter)
    corrected = spectrum - baseline
    normalized = l2_normalize(corrected)

    return normalized


def _process_single_spectrum(args: Tuple) -> np.ndarray:
    """处理单条光谱的辅助函数（用于并行处理）"""
    spectrum, lam, p, niter = args
    return preprocess_spectrum(spectrum, lam, p, niter)


def detect_encoding(filepath: Union[str, Path]) -> str:
    """
    自动检测文件编码

    Args:
        filepath: 文件路径

    Returns:
        检测到的编码
    """
    filepath = Path(filepath)

    # 尝试读取文件头部来检测编码
    for encoding in SUPPORTED_ENCODINGS:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                f.read(1024)  # 尝试读取前1KB
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    # 默认返回 utf-8
    return 'utf-8'


def load_csv_spectrum(
    filepath: Union[str, Path],
    encoding: Optional[str] = None,
    auto_detect: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载CSV光谱文件

    Args:
        filepath: CSV文件路径
        encoding: 文件编码，None表示自动检测或从配置读取
        auto_detect: 是否自动检测编码

    Returns:
        (wavenumbers, intensities): 波数和强度数组

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 无法解析文件
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if encoding is None:
        config = get_config()
        encoding = config.get('dataset.encoding', 'gbk')

    # 尝试指定编码
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        if auto_detect:
            # 自动检测编码
            detected_encoding = detect_encoding(filepath)
            try:
                df = pd.read_csv(filepath, encoding=detected_encoding)
            except Exception as e:
                raise ValueError(f"无法解码文件 {filepath}: {e}")
        else:
            raise ValueError(f"使用编码 {encoding} 无法解码文件 {filepath}")

    if df.shape[1] < 2:
        raise ValueError(f"CSV文件格式错误，至少需要2列: {filepath}")

    wavenumbers = df.iloc[:, 0].values.astype(np.float64)
    intensities = df.iloc[:, 1].values.astype(np.float64)

    # 验证数据
    if np.any(np.isnan(wavenumbers)) or np.any(np.isnan(intensities)):
        warnings.warn(f"文件 {filepath.name} 包含NaN值")

    return wavenumbers, intensities


def load_hyperspectral_data(
    folder_path: Union[str, Path],
    validate: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载40x40高光谱数据

    Args:
        folder_path: 数据文件夹路径
        validate: 是否验证数据

    Returns:
        (hypercube, wavenumbers): 高光谱立方体和波数数组

    Raises:
        FileNotFoundError: 文件夹不存在
        ValueError: 数据格式错误
    """
    logger = get_logger('preprocess')
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    files = list(folder_path.glob('*.csv'))
    if not files:
        raise ValueError(f"文件夹中没有CSV文件: {folder_path}")

    # 解析文件名获取坐标
    data_dict = {}
    for f in files:
        parts = f.name.split('-')
        if len(parts) >= 4:
            try:
                x = int(parts[2].replace('X', ''))
                y = int(parts[3].replace('Y', ''))
                data_dict[(x, y)] = f
            except ValueError:
                logger.warning(f"无法解析文件名: {f.name}")
                continue

    if not data_dict:
        raise ValueError(f"没有找到有效的光谱文件: {folder_path}")

    # 确定数据维度
    sample_file = list(data_dict.values())[0]
    wavenumbers, _ = load_csv_spectrum(sample_file)
    n_bands = len(wavenumbers)

    # 获取实际的x和y范围
    xs = [k[0] for k in data_dict.keys()]
    ys = [k[1] for k in data_dict.keys()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    logger.info(f"加载高光谱数据: {height}x{width}x{n_bands}")

    # 构建高光谱立方体
    hypercube = np.zeros((height, width, n_bands))
    loaded_count = 0

    for (x, y), filepath in data_dict.items():
        try:
            _, intensities = load_csv_spectrum(filepath)
            hypercube[y - y_min, x - x_min, :] = intensities
            loaded_count += 1
        except Exception as e:
            logger.warning(f"加载失败 {filepath}: {e}")

    logger.info(f"成功加载 {loaded_count}/{len(data_dict)} 个光谱")

    # 验证数据
    if validate:
        nan_count = np.sum(np.isnan(hypercube))
        inf_count = np.sum(np.isinf(hypercube))
        if nan_count > 0:
            logger.warning(f"数据包含 {nan_count} 个NaN值")
        if inf_count > 0:
            logger.warning(f"数据包含 {inf_count} 个Inf值")

    return hypercube, wavenumbers


def preprocess_hyperspectral(
    hypercube: np.ndarray,
    n_jobs: Optional[int] = None,
    lam: Optional[float] = None,
    p: Optional[float] = None,
    niter: Optional[int] = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    预处理整个高光谱数据立方体（支持并行处理）

    Args:
        hypercube: 高光谱数据 (H, W, bands)
        n_jobs: 并行进程数，None表示从配置读取，-1表示使用所有CPU
        lam: ALS平滑度参数
        p: ALS非对称权重
        niter: ALS迭代次数
        show_progress: 是否显示进度

    Returns:
        预处理后的高光谱数据
    """
    config = get_config()
    logger = get_logger('preprocess')

    # 从配置获取参数
    if n_jobs is None:
        n_jobs = config.get('preprocessing.n_jobs', -1)
    if lam is None:
        lam = config.get('preprocessing.als.lambda', DEFAULT_ALS_LAMBDA)
    if p is None:
        p = config.get('preprocessing.als.p', DEFAULT_ALS_P)
    if niter is None:
        niter = config.get('preprocessing.als.n_iterations', DEFAULT_ALS_NITER)

    h, w, bands = hypercube.shape
    total_pixels = h * w

    # 将3D数据展平为2D
    spectra_flat = hypercube.reshape(-1, bands)

    # 决定是否使用并行处理
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    if n_jobs > 1 and total_pixels > 100:
        logger.info(f"使用 {n_jobs} 个进程并行预处理 {total_pixels} 条光谱...")

        # 准备参数
        args_list = [(spectra_flat[i], lam, p, niter) for i in range(total_pixels)]

        # 并行处理
        processed = np.zeros_like(spectra_flat)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_process_single_spectrum, args): i
                      for i, args in enumerate(args_list)}

            for future in as_completed(futures):
                i = futures[future]
                try:
                    processed[i] = future.result()
                except Exception as e:
                    logger.warning(f"处理像素 {i} 失败: {e}")
                    processed[i] = np.zeros(bands)
    else:
        logger.info(f"串行预处理 {total_pixels} 条光谱...")
        processed = np.zeros_like(spectra_flat)
        for i in range(total_pixels):
            try:
                processed[i] = preprocess_spectrum(spectra_flat[i], lam, p, niter)
            except Exception as e:
                logger.warning(f"处理像素 {i} 失败: {e}")
                processed[i] = np.zeros(bands)

    # 恢复为3D形状
    return processed.reshape(h, w, bands)


def save_preprocessed_data(
    data: np.ndarray,
    wavenumbers: np.ndarray,
    output_path: Union[str, Path],
    name: str
) -> None:
    """
    保存预处理后的数据

    Args:
        data: 预处理后的数据
        wavenumbers: 波数数组
        output_path: 输出目录
        name: 数据名称
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / f'{name}_data.npy', data)
    np.save(output_path / f'{name}_wavenumbers.npy', wavenumbers)


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('preprocess')

    base_dir = config.base_dir
    dataset_dir = Path(config.paths['dataset_dir'])
    output_dir = Path(config.paths['preprocessed_dir'])
    output_dir.mkdir(exist_ok=True)

    # 获取端元配置
    endmembers = config.get('dataset.endmembers')

    # 加载淀粉纯谱
    logger.info("加载淀粉纯谱...")
    starch_path = dataset_dir / endmembers['starch']['folder'] / endmembers['starch']['file']
    starch_wn, starch_spec = load_csv_spectrum(starch_path)
    starch_processed = preprocess_spectrum(
        starch_spec,
        lam=config.get('preprocessing.als.lambda', DEFAULT_ALS_LAMBDA),
        p=config.get('preprocessing.als.p', DEFAULT_ALS_P),
        niter=config.get('preprocessing.als.n_iterations', DEFAULT_ALS_NITER)
    )

    # 加载PP纯谱
    logger.info("加载PP纯谱...")
    pp_path = dataset_dir / endmembers['pp']['folder'] / endmembers['pp']['file']
    pp_wn, pp_spec = load_csv_spectrum(pp_path)
    pp_processed = preprocess_spectrum(
        pp_spec,
        lam=config.get('preprocessing.als.lambda', DEFAULT_ALS_LAMBDA),
        p=config.get('preprocessing.als.p', DEFAULT_ALS_P),
        niter=config.get('preprocessing.als.n_iterations', DEFAULT_ALS_NITER)
    )

    # 加载PE纯谱
    logger.info("加载PE纯谱...")
    pe_path = dataset_dir / endmembers['pe']['folder'] / endmembers['pe']['file']
    pe_wn, pe_spec = load_csv_spectrum(pe_path)
    pe_processed = preprocess_spectrum(
        pe_spec,
        lam=config.get('preprocessing.als.lambda', DEFAULT_ALS_LAMBDA),
        p=config.get('preprocessing.als.p', DEFAULT_ALS_P),
        niter=config.get('preprocessing.als.n_iterations', DEFAULT_ALS_NITER)
    )

    # 保存纯谱
    np.save(output_dir / 'starch_spectrum.npy', starch_processed)
    np.save(output_dir / 'pp_spectrum.npy', pp_processed)
    np.save(output_dir / 'pe_spectrum.npy', pe_processed)
    np.save(output_dir / 'wavenumbers.npy', starch_wn)

    # 加载并预处理所有数据集的混合光谱
    datasets = config.get('dataset.datasets')
    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']
        logger.info(f"\n处理数据集: {dataset_type}")

        for sample in samples:
            name = sample['name']
            folder = sample['folder']
            logger.info(f"加载 {name}...")
            try:
                cube, wn = load_hyperspectral_data(dataset_dir / folder)
                cube_processed = preprocess_hyperspectral(cube)
                save_preprocessed_data(cube_processed, wn, output_dir, name)
            except Exception as e:
                logger.error(f"处理 {name} 失败: {e}")

    # 预处理测试集数据
    logger.info("\n处理测试集数据...")
    test_dirs = {
        'pp_test': dataset_dir / 'test' / 'PP+淀粉' / '1 785mw 2s 1 1 40 40',
        'pe_test': dataset_dir / 'test' / 'PE+淀粉' / '1 785mw 2s 1 1 40 40',
        'pp_pe_mixed_test': dataset_dir / 'test' / 'PP+PE+淀粉' / '1 785mw 2s 1 1 40 40'
    }

    for test_name, test_path in test_dirs.items():
        if test_path.exists():
            logger.info(f"加载 {test_name}...")
            try:
                cube, wn = load_hyperspectral_data(test_path)
                cube_processed = preprocess_hyperspectral(cube)
                save_preprocessed_data(cube_processed, wn, output_dir, test_name)
                logger.info(f"  {test_name} 预处理完成: {cube_processed.shape}")
            except Exception as e:
                logger.error(f"处理 {test_name} 失败: {e}")
        else:
            logger.warning(f"测试集路径不存在: {test_path}")

    logger.info("\n预处理完成！")
