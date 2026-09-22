"""
数据验证模块
提供数据完整性检查和验证功能
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
import warnings

from .config import get_config
from .logger import get_logger


class DataValidationError(Exception):
    """数据验证错误"""
    pass


def validate_spectrum(
    spectrum: np.ndarray,
    expected_length: Optional[int] = None,
    check_nan: bool = True,
    check_inf: bool = True,
    check_negative: bool = False,
    name: str = "spectrum"
) -> Tuple[bool, List[str]]:
    """
    验证单条光谱数据

    Args:
        spectrum: 光谱数组
        expected_length: 期望的长度
        check_nan: 是否检查NaN
        check_inf: 是否检查Inf
        check_negative: 是否检查负值
        name: 数据名称（用于错误消息）

    Returns:
        (is_valid, error_messages): 验证结果和错误消息列表
    """
    errors = []

    if not isinstance(spectrum, np.ndarray):
        errors.append(f"{name}: 必须是numpy数组，当前类型为 {type(spectrum)}")
        return False, errors

    if spectrum.ndim != 1:
        errors.append(f"{name}: 必须是1维数组，当前维度为 {spectrum.ndim}")

    if expected_length is not None and len(spectrum) != expected_length:
        errors.append(f"{name}: 长度应为 {expected_length}，当前为 {len(spectrum)}")

    if check_nan and np.any(np.isnan(spectrum)):
        nan_count = np.sum(np.isnan(spectrum))
        errors.append(f"{name}: 包含 {nan_count} 个NaN值")

    if check_inf and np.any(np.isinf(spectrum)):
        inf_count = np.sum(np.isinf(spectrum))
        errors.append(f"{name}: 包含 {inf_count} 个Inf值")

    if check_negative and np.any(spectrum < 0):
        neg_count = np.sum(spectrum < 0)
        errors.append(f"{name}: 包含 {neg_count} 个负值")

    return len(errors) == 0, errors


def validate_hypercube(
    hypercube: np.ndarray,
    expected_shape: Optional[Tuple[int, int, int]] = None,
    check_nan: bool = True,
    check_inf: bool = True,
    name: str = "hypercube"
) -> Tuple[bool, List[str]]:
    """
    验证高光谱数据立方体

    Args:
        hypercube: 高光谱数据 (H, W, bands)
        expected_shape: 期望的形状
        check_nan: 是否检查NaN
        check_inf: 是否检查Inf
        name: 数据名称

    Returns:
        (is_valid, error_messages): 验证结果和错误消息列表
    """
    errors = []

    if not isinstance(hypercube, np.ndarray):
        errors.append(f"{name}: 必须是numpy数组")
        return False, errors

    if hypercube.ndim != 3:
        errors.append(f"{name}: 必须是3维数组，当前维度为 {hypercube.ndim}")

    if expected_shape is not None and hypercube.shape != expected_shape:
        errors.append(f"{name}: 形状应为 {expected_shape}，当前为 {hypercube.shape}")

    if check_nan and np.any(np.isnan(hypercube)):
        nan_count = np.sum(np.isnan(hypercube))
        errors.append(f"{name}: 包含 {nan_count} 个NaN值")

    if check_inf and np.any(np.isinf(hypercube)):
        inf_count = np.sum(np.isinf(hypercube))
        errors.append(f"{name}: 包含 {inf_count} 个Inf值")

    return len(errors) == 0, errors


def validate_labels(
    labels: np.ndarray,
    num_classes: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    name: str = "labels"
) -> Tuple[bool, List[str]]:
    """
    验证标签数据

    Args:
        labels: 标签数组
        num_classes: 类别数
        expected_shape: 期望的形状
        name: 数据名称

    Returns:
        (is_valid, error_messages): 验证结果和错误消息列表
    """
    errors = []

    if not isinstance(labels, np.ndarray):
        errors.append(f"{name}: 必须是numpy数组")
        return False, errors

    if expected_shape is not None and labels.shape != expected_shape:
        errors.append(f"{name}: 形状应为 {expected_shape}，当前为 {labels.shape}")

    unique_labels = np.unique(labels)

    if num_classes is not None:
        expected_labels = set(range(num_classes))
        actual_labels = set(unique_labels)
        if not actual_labels.issubset(expected_labels):
            unexpected = actual_labels - expected_labels
            errors.append(f"{name}: 包含意外的标签值 {unexpected}")

    if np.any(labels < 0):
        errors.append(f"{name}: 包含负值标签")

    return len(errors) == 0, errors


def validate_abundance(
    abundance: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    name: str = "abundance"
) -> Tuple[bool, List[str]]:
    """
    验证丰度数据

    Args:
        abundance: 丰度数组
        expected_shape: 期望的形状
        name: 数据名称

    Returns:
        (is_valid, error_messages): 验证结果和错误消息列表
    """
    errors = []

    if not isinstance(abundance, np.ndarray):
        errors.append(f"{name}: 必须是numpy数组")
        return False, errors

    if expected_shape is not None and abundance.shape != expected_shape:
        errors.append(f"{name}: 形状应为 {expected_shape}，当前为 {abundance.shape}")

    if np.any(np.isnan(abundance)):
        nan_count = np.sum(np.isnan(abundance))
        errors.append(f"{name}: 包含 {nan_count} 个NaN值")

    if np.any(abundance < 0):
        neg_count = np.sum(abundance < 0)
        warnings.warn(f"{name}: 包含 {neg_count} 个负值，将被裁剪到0")

    if np.any(abundance > 1):
        over_count = np.sum(abundance > 1)
        warnings.warn(f"{name}: 包含 {over_count} 个大于1的值，将被裁剪到1")

    return len(errors) == 0, errors


def validate_file_exists(
    filepath: Union[str, Path],
    raise_error: bool = True
) -> bool:
    """
    验证文件是否存在

    Args:
        filepath: 文件路径
        raise_error: 不存在时是否抛出异常

    Returns:
        文件是否存在
    """
    filepath = Path(filepath)
    if not filepath.exists():
        if raise_error:
            raise FileNotFoundError(f"文件不存在: {filepath}")
        return False
    return True


def validate_directory(
    dirpath: Union[str, Path],
    create_if_missing: bool = True
) -> Path:
    """
    验证目录是否存在，可选择自动创建

    Args:
        dirpath: 目录路径
        create_if_missing: 不存在时是否创建

    Returns:
        目录Path对象
    """
    dirpath = Path(dirpath)
    if not dirpath.exists():
        if create_if_missing:
            dirpath.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"目录不存在: {dirpath}")
    return dirpath


def safe_load_npy(
    filepath: Union[str, Path],
    validate: bool = True,
    expected_shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """
    安全加载npy文件

    Args:
        filepath: 文件路径
        validate: 是否验证数据
        expected_shape: 期望的形状

    Returns:
        加载的数组
    """
    logger = get_logger('validation')
    filepath = Path(filepath)

    validate_file_exists(filepath)

    try:
        data = np.load(filepath)
    except Exception as e:
        raise DataValidationError(f"无法加载文件 {filepath}: {e}")

    if validate:
        if np.any(np.isnan(data)):
            logger.warning(f"{filepath.name}: 包含NaN值")
        if np.any(np.isinf(data)):
            logger.warning(f"{filepath.name}: 包含Inf值")

    if expected_shape is not None and data.shape != expected_shape:
        raise DataValidationError(
            f"数据形状不匹配: 期望 {expected_shape}，实际 {data.shape}"
        )

    return data


def clean_spectrum(
    spectrum: np.ndarray,
    fill_nan: float = 0.0,
    fill_inf: float = 0.0,
    clip_negative: bool = False
) -> np.ndarray:
    """
    清理光谱数据中的异常值

    Args:
        spectrum: 输入光谱
        fill_nan: NaN替换值
        fill_inf: Inf替换值
        clip_negative: 是否将负值裁剪为0

    Returns:
        清理后的光谱
    """
    result = spectrum.copy()

    # 替换NaN
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        result[nan_mask] = fill_nan

    # 替换Inf
    inf_mask = np.isinf(result)
    if np.any(inf_mask):
        result[inf_mask] = fill_inf

    # 裁剪负值
    if clip_negative:
        result = np.clip(result, 0, None)

    return result
