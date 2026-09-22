"""
工具模块
提供配置管理、日志记录、数据验证等公共功能
"""
from .config import Config, get_config, get
from .logger import setup_logger, get_logger, log_info, log_warning, log_error, log_debug
from .validation import (
    DataValidationError,
    validate_spectrum,
    validate_hypercube,
    validate_labels,
    validate_abundance,
    validate_file_exists,
    validate_directory,
    safe_load_npy,
    clean_spectrum,
)
from .model_identity import (
    PRIMARY_MODEL_NAME,
    PRIMARY_MODEL_FULL_NAME,
    RESNET_VARIANT_NAME,
    BEST_CANDIDATE_CONFIGS,
    FINAL_COMPARISON_REPLACEMENTS,
    FINAL_DROPPED_CANDIDATES,
    FINAL_RESNET_CHOICE,
    FINAL_RESNET_CONFIG,
    FINAL_COMPARISON_MODEL_ORDER,
)

__all__ = [
    # 配置
    'Config',
    'get_config',
    'get',
    # 日志
    'setup_logger',
    'get_logger',
    'log_info',
    'log_warning',
    'log_error',
    'log_debug',
    # 验证
    'DataValidationError',
    'validate_spectrum',
    'validate_hypercube',
    'validate_labels',
    'validate_abundance',
    'validate_file_exists',
    'validate_directory',
    'safe_load_npy',
    'clean_spectrum',
    'PRIMARY_MODEL_NAME',
    'PRIMARY_MODEL_FULL_NAME',
    'RESNET_VARIANT_NAME',
    'BEST_CANDIDATE_CONFIGS',
    'FINAL_COMPARISON_REPLACEMENTS',
    'FINAL_DROPPED_CANDIDATES',
    'FINAL_RESNET_CHOICE',
    'FINAL_RESNET_CONFIG',
    'FINAL_COMPARISON_MODEL_ORDER',
]
