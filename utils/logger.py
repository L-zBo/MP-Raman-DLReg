"""
日志管理模块
提供统一的日志记录功能
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from .config import get_config


def setup_logger(
    name: str = "raman",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    设置并返回日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        console: 是否输出到控制台

    Returns:
        配置好的日志记录器
    """
    config = get_config()

    # 从配置获取默认值
    if level is None:
        level = config.get('logging.level', 'INFO')
    if log_file is None:
        log_file = config.get('logging.file')
    if console is None:
        console = config.get('logging.console', True)

    log_format = config.get(
        'logging.format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 清除已有的处理器
    logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = config.base_dir / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "raman") -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# 创建默认日志记录器
_default_logger: Optional[logging.Logger] = None


def log_info(message: str) -> None:
    """记录INFO级别日志"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.info(message)


def log_warning(message: str) -> None:
    """记录WARNING级别日志"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.warning(message)


def log_error(message: str) -> None:
    """记录ERROR级别日志"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.error(message)


def log_debug(message: str) -> None:
    """记录DEBUG级别日志"""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    _default_logger.debug(message)
