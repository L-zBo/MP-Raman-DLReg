"""
注意力权重可视化：
展示 RADAR-Net 模型的注意力权重分布

输出位置：
- 训练集: output/attention_heatmap/PP/train/ 和 output/attention_heatmap/PE/train/
- 测试集: output/attention_heatmap/PP/test/ 和 output/attention_heatmap/PE/test/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# PyTorch
import torch
import torch.nn as nn

from utils import get_config, get_logger

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# 导入双头模型
from models.dual_head_model import DualHeadRamanCNNLSTM


def load_train_data(config, task='PP') -> Tuple[np.ndarray, np.ndarray]:
    """
    加载训练数据集（指定任务：PP或PE）

    Args:
        config: 配置对象
        task: 'PP' 或 'PE'

    Returns:
        X: 预处理后的数据
        y: 标签（PP或PE）
    """
    logger = get_logger('attention')
    data_dir = Path(config.paths['preprocessed_dir'])

    datasets = config.get('dataset.datasets')

    all_data, all_labels = [], []

    for dataset in datasets:
        samples = dataset['samples']

        for sample in samples:
            name = sample['name']
            data_path = data_dir / f'{name}_data.npy'

            # 从预处理数据目录加载标签
            if task == 'PP':
                label_path = data_dir / f'{name}_pp_labels.npy'
            else:
                label_path = data_dir / f'{name}_pe_labels.npy'

            if data_path.exists() and label_path.exists():
                data = np.load(data_path)
                labels = np.load(label_path)
                all_data.append(data.reshape(-1, data.shape[-1]))
                all_labels.append(labels.flatten())

    if len(all_data) == 0:
        raise ValueError(f"未找到 {task} 任务的训练数据和标签文件")

    X = np.vstack(all_data)
    y = np.concatenate(all_labels)

    logger.info(f"[{task}] 训练集: {X.shape[0]} 样本, 特征维度: {X.shape[1]}")
    return X, y


def load_test_data(config, task='PP') -> Tuple[np.ndarray, np.ndarray]:
    """
    加载测试数据集（指定任务：PP或PE）
    使用固定测试集（3200样本 = 1600原始测试 + 1600混合测试）

    Args:
        config: 配置对象
        task: 'PP' 或 'PE'

    Returns:
        X: 预处理后的数据
        y: 标签（PP或PE）
    """
    logger = get_logger('attention')
    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    true_label_dir = Path(__file__).parent.parent.parent / 'testing' / 'test_true_label'

    # 混合测试集数据（PP和PE共用同一份数据，但标签不同）
    mixed_data_path = preprocessed_dir / 'pp_pe_mixed_test_data.npy'
    X_mixed = np.load(mixed_data_path).reshape(-1, 1024)

    if task == 'PP':
        # PP任务：加载PP测试数据 + 混合测试数据(PP标签)
        pp_test_path = preprocessed_dir / 'pp_test_data.npy'
        pp_test_labels = true_label_dir / 'pp_test' / 'pp_labels.npy'
        pp_mixed_labels = true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy'

        X_test = np.load(pp_test_path).reshape(-1, 1024)
        y_test = np.load(pp_test_labels).flatten()
        y_mixed = np.load(pp_mixed_labels).flatten()

        X_combined = np.vstack([X_test, X_mixed])
        y_combined = np.concatenate([y_test, y_mixed])
    else:
        # PE任务：加载PE测试数据 + 混合测试数据(PE标签)
        pe_test_path = preprocessed_dir / 'pe_test_data.npy'
        pe_test_labels = true_label_dir / 'pe_test' / 'pe_labels.npy'
        pe_mixed_labels = true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy'

        X_test = np.load(pe_test_path).reshape(-1, 1024)
        y_test = np.load(pe_test_labels).flatten()
        y_mixed = np.load(pe_mixed_labels).flatten()

        X_combined = np.vstack([X_test, X_mixed])
        y_combined = np.concatenate([y_test, y_mixed])

    logger.info(f"[{task}] 测试集: {X_combined.shape[0]} 样本 (原始1600 + 混合1600)")
    return X_combined, y_combined


def get_attention_from_dual_model(model, X, device, task='PP'):
    """
    从双头模型获取注意力权重

    Args:
        model: 双头模型实例
        X: 输入数据
        device: 设备
        task: 'PP' 或 'PE'

    Returns:
        attention_weights: 注意力权重
    """
    model.eval()
    model.return_attention = True

    X_t = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        result = model(X_t)
        # 双头模型返回 (pp_out, pe_out, pp_attn_w, pe_attn_w) 如果 return_attention=True
        if len(result) == 4:
            pp_out, pe_out, pp_attn_w, pe_attn_w = result
            # 根据任务选择对应的注意力权重
            if task == 'PP':
                attention_weights = pp_attn_w.cpu().numpy()
            else:
                attention_weights = pe_attn_w.cpu().numpy()
        else:
            # 如果模型不支持返回注意力权重，返回 None
            attention_weights = None

    model.return_attention = False
    return attention_weights


def plot_attention_heatmap(attention_weights, y_test, output_dir, raman_shift, data_type='train'):
    """绘制注意力权重热力图"""
    logger = get_logger('attention')

    # 按类别分组
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for class_idx, class_name in enumerate(class_names):
        mask = y_test == class_idx
        class_attention = attention_weights[mask]

        if len(class_attention) > 0:
            # 取平均注意力
            mean_attention = np.mean(class_attention, axis=0)

            ax = axes[class_idx]

            # 使用stride-based映射构建真实波数坐标
            seq_len = class_attention.shape[1]
            raman_indices = np.array([min(i*16+16, len(raman_shift)-1) for i in range(seq_len)])
            x_centers = raman_shift[raman_indices]

            # 构建pcolormesh的边界(比中心多一个点)
            half_step = (x_centers[1] - x_centers[0]) / 2
            x_edges = np.concatenate([[x_centers[0] - half_step],
                                       (x_centers[:-1] + x_centers[1:]) / 2,
                                       [x_centers[-1] + half_step]])
            n_show = min(50, len(class_attention))
            y_edges = np.arange(n_show + 1)

            im = ax.pcolormesh(x_edges, y_edges,
                               class_attention[:n_show],
                               cmap='hot', shading='flat')
            ax.set_xlabel('Raman Shift ($cm^{-1}$)', fontsize=11)
            ax.set_ylabel('Sample Index', fontsize=11)
            ax.set_title(f'{class_name}\n(n={len(class_attention)})', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            plt.colorbar(im, ax=ax, label='Attention Weight')

    title_suffix = 'Training Set' if data_type == 'train' else 'Test Set'
    plt.suptitle(f'Attention Weights by Pollution Class ({title_suffix})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_heatmap_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"注意力热力图已保存: {output_dir / 'attention_heatmap_by_class.png'}")


def plot_attention_distribution(attention_weights, y_test, output_dir, raman_shift, data_type='train'):
    """绘制注意力权重分布图"""
    logger = get_logger('attention')

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    fig, ax = plt.subplots(figsize=(14, 6))

    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = y_test == class_idx
        class_attention = attention_weights[mask]

        if len(class_attention) > 0:
            mean_attention = np.mean(class_attention, axis=0)
            std_attention = np.std(class_attention, axis=0)

            # 将注意力权重映射到拉曼位移（stride-based映射）
            seq_len = len(mean_attention)
            raman_indices = np.array([min(i*16+16, len(raman_shift)-1) for i in range(seq_len)])
            x = raman_shift[raman_indices]

            ax.plot(x, mean_attention, color=color, label=class_name, linewidth=2)
            ax.fill_between(x, mean_attention - std_attention, mean_attention + std_attention,
                            color=color, alpha=0.2)

    ax.set_xlabel('Raman Shift ($cm^{-1}$)', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    title_suffix = 'Training Set' if data_type == 'train' else 'Test Set'
    ax.set_title(f'Average Attention Weight Distribution by Class ({title_suffix})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'attention_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"注意力分布图已保存: {output_dir / 'attention_distribution.png'}")


def plot_attention_vs_spectrum(attention_weights, X_test, y_test, output_dir, raman_shift, data_type='train'):
    """绘制注意力权重与原始光谱的对比（按类别均值，而非单个样本）"""
    logger = get_logger('attention')

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = y_test == class_idx
        indices = np.where(mask)[0]

        if len(indices) > 0:
            # 左图：该类别平均光谱
            mean_spectrum = X_test[mask].mean(axis=0)
            axes[class_idx, 0].plot(raman_shift, mean_spectrum, color=color, linewidth=1)
            axes[class_idx, 0].set_xlabel('Raman Shift ($cm^{-1}$)', fontsize=11)
            axes[class_idx, 0].set_ylabel('Intensity', fontsize=11)
            axes[class_idx, 0].set_title(f'{class_name}: Mean Spectrum (n={len(indices)})', fontsize=12, fontweight='bold')
            axes[class_idx, 0].grid(True, alpha=0.3)

            # 右图：该类别平均注意力权重（使用stride-based映射）
            mean_attention = attention_weights[mask].mean(axis=0)
            seq_len = len(mean_attention)
            raman_indices = np.array([min(i*16+16, len(raman_shift)-1) for i in range(seq_len)])
            x = raman_shift[raman_indices]

            axes[class_idx, 1].bar(x, mean_attention, width=(x[1]-x[0])*0.8, color=color, alpha=0.7)
            axes[class_idx, 1].set_xlabel(r'Raman Shift ($cm^{-1}$)', fontsize=11)
            axes[class_idx, 1].set_ylabel('Attention Weight', fontsize=11)
            axes[class_idx, 1].set_title(f'{class_name}: Mean Attention (n={len(indices)})', fontsize=12, fontweight='bold')
            axes[class_idx, 1].grid(True, alpha=0.3)

    title_suffix = 'Training Set' if data_type == 'train' else 'Test Set'
    plt.suptitle(f'Spectrum vs Attention Weights Comparison ({title_suffix})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_vs_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"光谱-注意力对比图已保存: {output_dir / 'attention_vs_spectrum.png'}")


def generate_attention_visualizations(model, X, y, output_dir, raman_shift, device, task, data_type='train'):
    """
    生成注意力可视化图

    Args:
        model: 双头模型
        X: 数据
        y: 标签
        output_dir: 输出目录
        raman_shift: 拉曼位移
        device: 设备
        task: 'PP' 或 'PE'
        data_type: 'train' 或 'test'
    """
    logger = get_logger('attention')

    # 为了可视化，随机采样一部分数据（避免内存问题）
    max_samples = 3200
    if len(X) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
        logger.info(f"采样 {max_samples} 个样本用于可视化")
    else:
        X_sample = X
        y_sample = y

    # 获取注意力权重
    logger.info(f">>> 获取 {task} {data_type} 注意力权重...")
    attention_weights = get_attention_from_dual_model(model, X_sample, device, task)

    if attention_weights is None:
        logger.warning(f"{task} 任务：模型不支持返回注意力权重，跳过")
        return

    # 绘制注意力热力图
    logger.info(f">>> 绘制 {task} {data_type} 注意力热力图...")
    plot_attention_heatmap(attention_weights, y_sample, output_dir, raman_shift, data_type)

    # 绘制注意力分布图
    logger.info(f">>> 绘制 {task} {data_type} 注意力分布图...")
    plot_attention_distribution(attention_weights, y_sample, output_dir, raman_shift, data_type)

    # 绘制光谱-注意力对比图
    logger.info(f">>> 绘制 {task} {data_type} 光谱-注意力对比图...")
    plot_attention_vs_spectrum(attention_weights, X_sample, y_sample, output_dir, raman_shift, data_type)


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('attention')

    logger.info("注意力权重可视化（双头模型 - PP和PE任务）")
    logger.info("同时生成训练集和测试集的注意力热图")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载波数文件（从预处理数据目录）
    wavenumbers_path = Path(config.paths['preprocessed_dir']) / 'wavenumbers.npy'
    if wavenumbers_path.exists():
        raman_shift = np.load(wavenumbers_path)
        logger.info(f"Loaded wavenumbers: {raman_shift.min():.1f} ~ {raman_shift.max():.1f} cm-1")
    else:
        # 如果文件不存在，使用默认值
        raman_shift = np.linspace(103.785, 3696.98, 1024)
        logger.warning(f"Wavenumber file not found, using default: {raman_shift.min():.1f} ~ {raman_shift.max():.1f} cm-1")

    # 加载双头模型
    model_path = Path(config.paths['output_dir']) / 'models'  / 'best_model.pth'
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        logger.error("请先运行 train_dual.py 训练双头模型")
        import sys
        sys.exit(1)

    num_classes = config.get('model.architecture.num_classes', 3)
    input_len = config.get('model.architecture.input_len', 1024)
    model = DualHeadRamanCNNLSTM(input_len=input_len, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info("双头模型加载成功")

    # 对 PP 和 PE 分别生成注意力可视化
    for task in ['PP', 'PE']:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理 {task} 任务")
        logger.info(f"{'='*60}")

        # ========== 训练集 ==========
        logger.info(f"\n--- {task} 训练集 ---")
        train_output_dir = Path(config.paths['output_dir']) / 'attention_heatmap' / task / 'train'
        train_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f">>> 加载 {task} 训练数据...")
        X_train, y_train = load_train_data(config, task)

        generate_attention_visualizations(
            model, X_train, y_train, train_output_dir, raman_shift, device, task, 'train'
        )

        # ========== 测试集 ==========
        logger.info(f"\n--- {task} 测试集 ---")
        test_output_dir = Path(config.paths['output_dir']) / 'attention_heatmap' / task / 'test'
        test_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f">>> 加载 {task} 测试数据...")
        X_test, y_test = load_test_data(config, task)

        generate_attention_visualizations(
            model, X_test, y_test, test_output_dir, raman_shift, device, task, 'test'
        )

        logger.info(f"\n>>> {task} 任务完成！")

    logger.info("\n" + "="*60)
    logger.info("所有任务完成！")
    logger.info("输出目录结构:")
    logger.info("  output/attention_heatmap/PP/train/  - PP训练集注意力热图")
    logger.info("  output/attention_heatmap/PP/test/   - PP测试集注意力热图")
    logger.info("  output/attention_heatmap/PE/train/  - PE训练集注意力热图")
    logger.info("  output/attention_heatmap/PE/test/   - PE测试集注意力热图")
    logger.info("="*60)
