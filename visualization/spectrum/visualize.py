"""
可视化模块 - 支持配置文件
提供丰度图、分类图、光谱对比图等可视化功能
包含跨平台字体兼容性处理
"""
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.figure import Figure

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_config, get_logger

# 常量定义
DEFAULT_DPI = 300
DEFAULT_FIGURE_SIZE = (8, 6)
DEFAULT_COLORMAP_ABUNDANCE = 'viridis'
DEFAULT_COLORMAP_CLASSIFICATION = ['black', 'blue', 'red']

# 模块级标志，避免重复设置字体
_matplotlib_initialized = False


def get_available_chinese_fonts() -> List[str]:
    """
    获取系统可用的中文字体

    Returns:
        可用字体列表
    """
    import matplotlib.font_manager as fm

    chinese_fonts = []
    preferred_fonts = [
        'SimHei', 'Microsoft YaHei', 'STHeiti', 'STSong',
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
        'PingFang SC', 'Hiragino Sans GB', 'Source Han Sans CN'
    ]

    # 获取系统所有字体
    system_fonts = {f.name for f in fm.fontManager.ttflist}

    # 找出可用的中文字体
    for font in preferred_fonts:
        if font in system_fonts:
            chinese_fonts.append(font)

    # 如果没有找到中文字体，使用默认字体
    if not chinese_fonts:
        chinese_fonts = ['DejaVu Sans']

    return chinese_fonts


def setup_matplotlib(use_chinese: bool = True) -> None:
    """
    设置matplotlib参数，包含跨平台字体兼容性

    Args:
        use_chinese: 是否尝试使用中文字体
    """
    global _matplotlib_initialized
    if _matplotlib_initialized:
        return

    config = get_config()

    if use_chinese:
        # 尝试从配置获取字体
        config_fonts = config.get('visualization.font.family', [])

        # 获取系统可用字体
        available_fonts = get_available_chinese_fonts()

        # 合并配置字体和可用字体
        fonts = []
        for font in config_fonts:
            if font in available_fonts or font == 'DejaVu Sans':
                fonts.append(font)

        if not fonts:
            fonts = available_fonts

        try:
            plt.rcParams['font.sans-serif'] = fonts
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            warnings.warn(f"字体设置失败: {e}，使用默认字体")

    _matplotlib_initialized = True


def plot_abundance_map(
    abundance: np.ndarray,
    title: str,
    save_path: Optional[Union[str, Path]] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: Optional[int] = None,
    show_colorbar: bool = True,
    colorbar_label: str = 'Abundance'
) -> Optional[Figure]:
    """绘制丰度分布图"""
    setup_matplotlib()
    config = get_config()

    if dpi is None:
        dpi = config.get('visualization.dpi', DEFAULT_DPI)
    if figsize is None:
        figsize = tuple(config.get('visualization.figure_size', DEFAULT_FIGURE_SIZE))
    if cmap is None:
        cmap = config.get('visualization.colormap.abundance', DEFAULT_COLORMAP_ABUNDANCE)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = abundance.max() if abundance.max() > 0.1 else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(abundance, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_classification_map(
    labels: np.ndarray,
    title: str,
    save_path: Optional[Union[str, Path]] = None,
    colors: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: Optional[int] = None
) -> Optional[Figure]:
    """
    绘制污染等级分布图

    Args:
        labels: 标签数组 (H, W)
        title: 图片标题
        save_path: 保存路径
        colors: 类别颜色列表
        class_names: 类别名称列表
        figsize: 图片大小
        dpi: 分辨率

    Returns:
        如果不保存则返回Figure对象
    """
    setup_matplotlib()
    config = get_config()

    if dpi is None:
        dpi = config.get('visualization.dpi', DEFAULT_DPI)
    if figsize is None:
        figsize = tuple(config.get('visualization.figure_size', DEFAULT_FIGURE_SIZE))
    if colors is None:
        colors = config.get('visualization.colormap.classification', DEFAULT_COLORMAP_CLASSIFICATION)
    if class_names is None:
        class_names = config.get('classification.class_names', ['Non-pollution', 'Slight pollution', 'Severe pollution'])

    cmap = ListedColormap(colors)
    n_classes = len(colors)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(labels, cmap=cmap, vmin=0, vmax=n_classes - 1, aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 创建离散颜色条
    cbar = plt.colorbar(im, ax=ax, ticks=list(range(n_classes)))
    cbar.ax.set_yticklabels(class_names)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_spectra_comparison(
    wavenumbers: np.ndarray,
    spectra: List[np.ndarray],
    labels: List[str],
    title: str = '光谱对比',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: Optional[int] = None,
    alpha: float = 0.8,
    show_legend: bool = True
) -> Optional[Figure]:
    """
    绘制光谱对比图

    Args:
        wavenumbers: 波数数组
        spectra: 光谱列表
        labels: 标签列表
        title: 图片标题
        save_path: 保存路径
        figsize: 图片大小
        dpi: 分辨率
        alpha: 透明度
        show_legend: 是否显示图例

    Returns:
        如果不保存则返回Figure对象
    """
    setup_matplotlib()
    config = get_config()

    if dpi is None:
        dpi = config.get('visualization.dpi', DEFAULT_DPI)

    fig, ax = plt.subplots(figsize=figsize)

    for spectrum, label in zip(spectra, labels):
        ax.plot(wavenumbers, spectrum, label=label, alpha=alpha)

    ax.set_xlabel(r'波数 ($cm^{-1}$)')
    ax.set_ylabel('归一化强度')
    ax.set_title(title)

    if show_legend:
        ax.legend()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_attention_weights(
    attn_weights: np.ndarray,
    wavenumbers: np.ndarray,
    title: str = '模型注意力分布',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 4),
    dpi: Optional[int] = None,
    highlight_peaks: bool = True,
    n_peaks: int = 5
) -> Optional[Figure]:
    """
    绘制注意力权重

    Args:
        attn_weights: 注意力权重 (batch, seq, 1) 或 (seq,)
        wavenumbers: 波数数组
        title: 图片标题
        save_path: 保存路径
        figsize: 图片大小
        dpi: 分辨率
        highlight_peaks: 是否高亮峰值
        n_peaks: 高亮的峰值数量

    Returns:
        如果不保存则返回Figure对象
    """
    setup_matplotlib()
    config = get_config()

    if dpi is None:
        dpi = config.get('visualization.dpi', DEFAULT_DPI)

    # 处理注意力权重形状
    if attn_weights.ndim == 3:
        avg_attn = attn_weights.mean(axis=0).squeeze()
    elif attn_weights.ndim == 2:
        avg_attn = attn_weights.mean(axis=0)
    else:
        avg_attn = attn_weights

    # 确保波数和注意力长度匹配
    min_len = min(len(wavenumbers), len(avg_attn))
    wavenumbers = wavenumbers[:min_len]
    avg_attn = avg_attn[:min_len]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(wavenumbers, avg_attn, 'b-', linewidth=1.5)
    ax.fill_between(wavenumbers, avg_attn, alpha=0.3)

    # 高亮峰值
    if highlight_peaks:
        peak_indices = np.argsort(avg_attn)[-n_peaks:]
        for idx in peak_indices:
            ax.axvline(x=wavenumbers[idx], color='r', linestyle='--', alpha=0.5)
            ax.annotate(
                f'{wavenumbers[idx]:.0f}',
                xy=(wavenumbers[idx], avg_attn[idx]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

    ax.set_xlabel(r'波数 ($cm^{-1}$)')
    ax.set_ylabel('注意力权重')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = '混淆矩阵',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: Optional[int] = None,
    normalize: bool = True,
    cmap: str = 'Blues'
) -> Optional[Figure]:
    """
    绘制混淆矩阵

    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        title: 图片标题
        save_path: 保存路径
        figsize: 图片大小
        dpi: 分辨率
        normalize: 是否归一化
        cmap: 颜色映射

    Returns:
        如果不保存则返回Figure对象
    """
    setup_matplotlib()
    config = get_config()

    if dpi is None:
        dpi = config.get('visualization.dpi', DEFAULT_DPI)
    if class_names is None:
        class_names = config.get('classification.class_names', ['Non-pollution', 'Slight pollution', 'Severe pollution'])

    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        cm = confusion_matrix
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')

    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            if normalize:
                text = f'{cm[i, j]:.1%}'
            else:
                text = f'{cm[i, j]}'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_training_history(
    history: dict,
    title: str = '训练历史',
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 4),
    dpi: Optional[int] = None
) -> Optional[Figure]:
    """
    绘制训练历史曲线

    Args:
        history: 包含'loss', 'accuracy'等键的字典
        title: 图片标题
        save_path: 保存路径
        figsize: 图片大小
        dpi: 分辨率

    Returns:
        如果不保存则返回Figure对象
    """
    setup_matplotlib()
    config = get_config()

    if dpi is None:
        dpi = config.get('visualization.dpi', DEFAULT_DPI)

    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for ax, (key, values) in zip(axes, history.items()):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, 'b-', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def create_comparison_figure(
    data_list: List[np.ndarray],
    titles: List[str],
    main_title: str = '对比图',
    save_path: Optional[Union[str, Path]] = None,
    ncols: int = 3,
    figsize_per_subplot: Tuple[float, float] = (4, 3),
    dpi: Optional[int] = None,
    cmap: str = 'hot'
) -> Optional[Figure]:
    """
    创建多图对比

    Args:
        data_list: 数据列表
        titles: 标题列表
        main_title: 主标题
        save_path: 保存路径
        ncols: 列数
        figsize_per_subplot: 每个子图的大小
        dpi: 分辨率
        cmap: 颜色映射

    Returns:
        如果不保存则返回Figure对象
    """
    setup_matplotlib()
    config = get_config()

    if dpi is None:
        dpi = config.get('visualization.dpi', DEFAULT_DPI)

    n = len(data_list)
    nrows = (n + ncols - 1) // ncols
    figsize = (figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, (data, title) in enumerate(zip(data_list, titles)):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        im = ax.imshow(data, cmap=cmap, aspect='equal')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏多余的子图
    for idx in range(n, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')

    plt.suptitle(main_title, fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('visualize')

    output_dir = Path(config.paths['output_dir'])

    # 遍历所有数据集生成可视化
    datasets = config.get('dataset.datasets')

    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']

        logger.info(f"\n{'='*50}")
        logger.info(f"处理数据集: {dataset_type}")
        logger.info(f"{'='*50}")

        abundance_dir = output_dir / 'abundance' / 'abundance_npy' / dataset_type
        classification_dir = output_dir / 'classification' / 'combined' / dataset_type

        for sample in samples:
            name = sample['name']
            abundance_path = abundance_dir / f'{name}_abundance.npy'
            labels_path = classification_dir / f'{name}_labels.npy'

            if abundance_path.exists():
                abundance = np.load(abundance_path)

                # 只生成1张图：微塑料总丰度图（PP+PE）
                total_mp = abundance[:, :, 1] + abundance[:, :, 2]
                plot_abundance_map(
                    total_mp,
                    f'{name} 微塑料总丰度分布',
                    abundance_dir / f'{name}_abundance.png'
                )
                logger.info(f"生成 {name}_abundance.png")

            if labels_path.exists():
                labels = np.load(labels_path)
                plot_classification_map(
                    labels,
                    f'{name} 污染等级分布',
                    classification_dir / f'{name}_classification.png'
                )
                logger.info(f"生成 {name}_classification.png")

    logger.info("\n可视化完成！")
