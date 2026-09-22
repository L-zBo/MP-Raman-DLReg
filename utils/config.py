"""
配置管理模块
统一加载和管理项目配置
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """配置管理类"""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[Path] = None):
        """单例模式，确保全局只有一个配置实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None):
        if self._initialized and config_path is None:
            return

        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config.yaml'

        self._load_config(config_path)
        self._initialized = True

    def _load_config(self, config_path: Path) -> None:
        """加载配置文件"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        self._base_dir = config_path.parent
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        """将相对路径转换为绝对路径"""
        paths = self._config.get('paths', {})
        for key, value in paths.items():
            if isinstance(value, str):
                paths[key] = str(self._base_dir / value)

    @property
    def base_dir(self) -> Path:
        """项目根目录"""
        return self._base_dir

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项，支持点分隔的嵌套访问

        Args:
            key: 配置键，如 "preprocessing.als.lambda"
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"配置项不存在: {key}")

        return value

    @property
    def paths(self) -> Dict[str, str]:
        """获取所有路径配置"""
        return self._config.get('paths', {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        """获取预处理配置"""
        return self._config.get('preprocessing', {})

    @property
    def training(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self._config.get('training', {})

    @property
    def model(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self._config.get('model', {})

    @property
    def dataset(self) -> Dict[str, Any]:
        """获取数据集配置"""
        return self._config.get('dataset', {})

    @property
    def visualization(self) -> Dict[str, Any]:
        """获取可视化配置"""
        return self._config.get('visualization', {})

    @property
    def experiment_name(self) -> str:
        """获取当前实验名称"""
        return self._config.get('experiment', {}).get('name', 'default')

    @property
    def dataset_type(self) -> str:
        """获取当前数据集类型"""
        return self._config.get('dataset', {}).get('type', 'default')

    def get_output_dir(self, subdir: str = '') -> Path:
        """
        获取输出目录路径

        Args:
            subdir: 子目录名

        Returns:
            完整输出路径: output/{subdir}/{dataset_type}/{experiment_name}
        """
        output_dir = Path(self.paths['output_dir'])
        if subdir:
            output_dir = output_dir / subdir / self.dataset_type / self.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def reload(self, config_path: Optional[Path] = None) -> None:
        """重新加载配置"""
        if config_path is None:
            config_path = self._base_dir / 'config.yaml'
        self._load_config(config_path)


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    获取配置实例

    Args:
        config_path: 配置文件路径（可选）

    Returns:
        Config实例
    """
    return Config(config_path)


# 便捷函数
def get(key: str, default: Any = None) -> Any:
    """获取配置项的便捷函数"""
    return get_config().get(key, default)
