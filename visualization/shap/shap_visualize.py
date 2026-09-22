"""
Model Interpretability Visualization Module - Using Permutation Feature Importance
生成类似SHAPValue的可视化图，包括：
1. Feature Importance Bar Chart + Importance Distribution
2. 单样本决策路径（瀑布图）
3. Feature Dependence Analysis图

Using Permutation Feature Importance方法，无需SHAP库依赖
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置中文字体（与 config.yaml 中 visualization.font.family 保持一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 添加训练模块路径
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir / 'training'))
sys.path.insert(0, str(base_dir))

from models.dual_head_model import DualHeadRamanCNNLSTM
from utils import get_logger


class FeatureImportanceAnalyzer:
    """使用置换方法计算Feature Importance（支持单头和Dual-head model）"""

    def __init__(self, model, device='cpu', head='pp', dual_head=True):
        """
        Args:
            model: Model instance
            device: Computing device
            head: 分析哪个头（仅Dual-head model有效），'pp' 或 'pe'
            dual_head: 是否为Dual-head model，False 表示Single-head model
        """
        self.model = model
        self.device = device
        self.head = head.lower()
        self.dual_head = dual_head
        self.model.eval()
        self.logger = get_logger('shap')

    def predict_proba(self, X):
        """Get prediction probabilities"""
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            X = X.to(self.device)

            if self.dual_head:
                # Dual-head model
                pp_out, pe_out = self.model(X)
                outputs = pp_out if self.head == 'pp' else pe_out
            else:
                # Single-head model
                result = self.model(X)
                # 处理可能返回注意力权重的情况
                outputs = result[0] if isinstance(result, tuple) else result

            probs = F.softmax(outputs, dim=1)
            return probs.cpu().numpy()

    def compute_permutation_importance(self, X, n_repeats=10, n_features_sample=50):
        """
        计算置换Feature Importance

        Args:
            X: 输入数据
            n_repeats: 每个特征的重复次数
            n_features_sample: 采样的特征数（加速计算）

        Returns:
            importances: Feature Importance数组
            importance_std: 重要性标准差
        """
        baseline_probs = self.predict_proba(X)
        baseline_score = np.mean(np.max(baseline_probs, axis=1))

        n_samples, n_features = X.shape

        # 如果特征数太多，随机采样一部分
        if n_features > n_features_sample:
            # 首先计算一个粗略的重要性估计
            feature_variance = np.var(X, axis=0)
            top_features = np.argsort(feature_variance)[-n_features_sample:]
        else:
            top_features = np.arange(n_features)

        importances = np.zeros(n_features)
        importance_std = np.zeros(n_features)

        self.logger.info(f"计算 {len(top_features)} 个特征的重要性...")

        for feat_idx in tqdm(top_features, desc="计算Feature Importance"):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feat_idx])
                permuted_probs = self.predict_proba(X_permuted)
                permuted_score = np.mean(np.max(permuted_probs, axis=1))
                scores.append(baseline_score - permuted_score)

            importances[feat_idx] = np.mean(scores)
            importance_std[feat_idx] = np.std(scores)

        return importances, importance_std

    def compute_local_importance(self, X, sample_idx, n_perturbations=100):
        """
        计算单样本的局部Feature Importance（类似SHAPValue）

        Args:
            X: 输入数据
            sample_idx: 样本索引
            n_perturbations: 扰动次数

        Returns:
            local_importance: 局部重要性数组
            baseline_prob: 基线预测概率
            predicted_class: 模型预测的类别索引
        """
        sample = X[sample_idx:sample_idx+1]
        baseline_prob = self.predict_proba(sample)

        # 使用模型预测类别，而非硬编码 class=1
        predicted_class = np.argmax(baseline_prob[0])

        n_features = X.shape[1]
        local_importance = np.zeros(n_features)

        # Use mean as reference value
        mean_values = np.mean(X, axis=0)

        for feat_idx in range(n_features):
            # 创建扰动样本
            perturbed = sample.copy()
            perturbed[0, feat_idx] = mean_values[feat_idx]
            perturbed_prob = self.predict_proba(perturbed)

            # 重要性 = 原始预测 - 扰动后预测（针对预测类别）
            local_importance[feat_idx] = baseline_prob[0, predicted_class] - perturbed_prob[0, predicted_class]

        return local_importance, baseline_prob[0], predicted_class


def _find_borderline_sample(analyzer, X_sample, target_confidence=(0.70, 0.80)):
    """
    从样本中找一个置信度在 target_confidence 范围内的样本。
    70-80%置信度下正负贡献均衡且结果可靠。
    如果找不到合适的，退而求其次选置信度最接近0.75的样本。
    """
    probs = analyzer.predict_proba(X_sample)
    max_probs = np.max(probs, axis=1)

    # 优先找 target_confidence 范围内的样本
    low, high = target_confidence
    mask = (max_probs >= low) & (max_probs <= high)
    if np.any(mask):
        candidates = np.where(mask)[0]
        mid = (low + high) / 2
        best = candidates[np.argmin(np.abs(max_probs[candidates] - mid))]
        return int(best)

    # 退而求其次：选离 0.75 最近的
    best = int(np.argmin(np.abs(max_probs - 0.75)))
    return best


def plot_feature_importance(importances, importance_std=None, feature_names=None,
                           top_k=20, save_path=None):
    """
    绘制Feature Importance Bar Chart + Importance Distribution
    """
    n_features = len(importances)
    if feature_names is None:
        feature_names = [f'Wavenumber_{i}' for i in range(n_features)]

    # 获取top_k个最重要的特征
    top_indices = np.argsort(np.abs(importances))[-top_k:][::-1]
    top_values = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    if importance_std is not None:
        top_std = importance_std[top_indices]
    else:
        top_std = None

    # 创建双图布局 — 增大高度，给纵坐标标签留足空间
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, top_k * 0.55)))

    # 左图：条形图
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_values)))
    y_pos = np.arange(len(top_values))

    bars = axes[0].barh(y_pos, top_values[::-1], color=colors[::-1],
                        xerr=top_std[::-1] if top_std is not None else None,
                        capsize=3, alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_names[::-1], fontsize=9)
    axes[0].set_xlabel('Average Feature Importance', fontsize=10)
    axes[0].set_title('(a) Feature Importance Ranking', fontsize=11)
    axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    # Add value labels — 放在条形末端右侧，白色背景防遮挡
    for i, val in enumerate(top_values[::-1]):
        label_x = val + (top_std[::-1][i] if top_std is not None else 0)
        axes[0].text(label_x, i, f'  {val:.4f}', va='center', ha='left',
                     fontsize=7.5, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor='none', alpha=0.7))

    # 右图：重要性分布的条形对比图
    # Show positive/negative contributions
    positive_mask = top_values > 0
    negative_mask = top_values < 0

    axes[1].barh(y_pos[positive_mask[::-1]], top_values[::-1][positive_mask[::-1]],
                 color='#ff4444', alpha=0.7, label='Positive Impact')
    axes[1].barh(y_pos[negative_mask[::-1]], top_values[::-1][negative_mask[::-1]],
                 color='#4444ff', alpha=0.7, label='Negative Impact')

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_names[::-1], fontsize=9)
    axes[1].set_xlabel('Feature Importance', fontsize=10)
    axes[1].set_title('(b) Positive/Negative Impact Distribution', fontsize=11)
    axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].legend(loc='lower right', fontsize=9)

    plt.suptitle('Model Feature Importance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        get_logger('shap').info(f"Feature Importance图已保存: {save_path}")
    plt.close()


def plot_waterfall(local_importance, baseline_prob, feature_names=None,
                   top_k=10, save_path=None, predicted_class=None):
    """
    绘制瀑布图（单样本决策路径）

    Args:
        local_importance: 局部重要性数组
        baseline_prob: 基线预测概率向量
        feature_names: 特征名称列表
        top_k: 显示top_k个最重要特征
        save_path: 图片保存路径
        predicted_class: 模型预测的类别索引，None则使用argmax
    """
    n_features = len(local_importance)
    if feature_names is None:
        feature_names = [f'Wavenumber_{i}' for i in range(n_features)]

    if predicted_class is None:
        predicted_class = np.argmax(baseline_prob)

    # 获取影响最大的特征
    abs_importance = np.abs(local_importance)
    top_indices = np.argsort(abs_importance)[-top_k:]

    # Sort by absolute importance
    sorted_indices = sorted(top_indices, key=lambda i: local_importance[i], reverse=True)

    # Baseline（使用预测类别的概率）
    base_value = baseline_prob[predicted_class] - np.sum(local_importance[sorted_indices])

    # Calculate cumulative values
    cumulative = [base_value]
    for idx in sorted_indices:
        cumulative.append(cumulative[-1] + local_importance[idx])

    final_value = cumulative[-1]

    # 绘图 — 根据特征数动态调高度，避免纵坐标挤压
    n_bars = len(sorted_indices)
    fig_h = max(6, (n_bars + 2) * 0.65)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    y_positions = list(range(len(sorted_indices) + 2))

    # 计算 X 轴范围 - 聚焦在有意义的区域
    all_values = cumulative + [base_value, final_value]
    x_min = min(all_values)
    x_max = max(all_values)
    x_range = x_max - x_min
    x_padding = max(x_range * 0.3, 0.02)  # 至少 2% 或 30% 的范围作为边距

    xlim_min = max(0, x_min - x_padding)
    xlim_max = min(1, x_max + x_padding)

    # 确保范围足够大以显示细节
    if xlim_max - xlim_min < 0.1:
        center = (x_min + x_max) / 2
        xlim_min = max(0, center - 0.05)
        xlim_max = min(1, center + 0.05)

    # 绘制最终 Prediction (顶部)
    ax.barh(len(sorted_indices)+1, final_value - xlim_min, left=xlim_min,
            color='#2ecc71', alpha=0.8, height=0.6, edgecolor='black', linewidth=1.5)
    ax.text(final_value, len(sorted_indices)+1, f'Prediction\n{final_value:.3f}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    # Plot feature contributions (从上到下)
    for i, idx in enumerate(sorted_indices):
        val = local_importance[idx]
        start = cumulative[i]
        color = '#ff0051' if val > 0 else '#008bfb'

        ax.barh(len(sorted_indices) - i, abs(val), left=min(start, start+val),
                color=color, alpha=0.8, height=0.6, edgecolor='black', linewidth=0.5)

        # 添加数值标签 — 窄条时放到外侧，避免遮挡
        sign = '+' if val > 0 else ''
        bar_width_ratio = abs(val) / max(x_range, 1e-6)
        if bar_width_ratio < 0.15:
            # 条形太窄，标签放到条形右侧外面
            label_x = max(start, start + val) + x_range * 0.01
            ha = 'left'
        else:
            label_x = start + val / 2
            ha = 'center'
        ax.text(label_x, len(sorted_indices) - i, f'{sign}{val:.4f}',
                ha=ha, va='center', fontsize=7.5, color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor='none', alpha=0.75))

    # 绘制 Baseline (底部)
    ax.barh(0, base_value - xlim_min, left=xlim_min,
            color='#95a5a6', alpha=0.8, height=0.6, edgecolor='black', linewidth=1.5)
    ax.text(base_value, 0, f'Baseline\n{base_value:.3f}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='black')

    # 添加连接线（虚线）
    for i in range(len(cumulative)):
        if i < len(sorted_indices):
            y_start = len(sorted_indices) - i + 0.3
            y_end = len(sorted_indices) - i - 0.3 if i < len(sorted_indices) - 1 else 0.3
            ax.plot([cumulative[i], cumulative[i]], [y_start, y_end],
                    'k--', alpha=0.4, linewidth=1)

    # 设置 Y 轴标签 — 缩小字体，防止重叠
    labels = ['E[f(X)]'] + [feature_names[idx] for idx in reversed(sorted_indices)] + ['f(x)']
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)

    # 设置 X 轴范围
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xlabel('Model Output Probability', fontsize=11)
    ax.set_title('Feature Contribution Waterfall - Single Sample Decision Path',
                 fontsize=12, fontweight='bold')

    # 添加说明
    ax.text(0.02, 0.98, 'Red: Positive Impact (+)\nBlue: Negative Impact (-)',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 添加网格线
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        get_logger('shap').info(f"Waterfall plot saved: {save_path}")
    plt.close()


def plot_dependence(X, importances, feature_indices, feature_names=None, save_path=None):
    """
    绘制Feature Dependence Analysis图
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'Wavenumber_{i}' for i in range(n_features)]

    n_plots = len(feature_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    for ax, feat_idx in zip(axes, feature_indices):
        feature_values = X[:, feat_idx]

        # 计算每个样本在该特征上的重要性得分
        # Use difference between feature value and overall mean as proxy
        mean_val = np.mean(feature_values)
        importance_proxy = (feature_values - mean_val) * importances[feat_idx]

        # 使用颜色表示Feature Value大小
        scatter = ax.scatter(feature_values, importance_proxy,
                            c=feature_values, cmap='RdBu_r',
                            s=15, alpha=0.6)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel(f'{feature_names[feat_idx]} Value')
        ax.set_ylabel('Importance Contribution')
        ax.set_title(feature_names[feat_idx])

        # 添加趋势线
        z = np.polyfit(feature_values, importance_proxy, 2)
        p = np.poly1d(z)
        x_line = np.linspace(feature_values.min(), feature_values.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.5)

        plt.colorbar(scatter, ax=ax, label='Feature Value')

    plt.suptitle('Feature Dependence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        get_logger('shap').info(f"Dependence plot saved: {save_path}")
    plt.close()


def generate_interpretability_visualizations(model_path, data_path, output_dir,
                                             n_samples=200, device='cpu', config=None):
    """
    生成所有可解释性可视化图

    Args:
        model_path: 模型文件路径
        data_path: 预处理数据目录路径
        output_dir: 输出目录路径
        n_samples: 采样数量
        device: 计算设备
        config: 配置对象（必须提供）
    """
    if config is None:
        raise ValueError("config参数不能为None，请通过 get_config() 获取配置后传入")

    logger = get_logger('shap')
    logger.info("=" * 50)
    logger.info("Model Interpretability Analysis")
    logger.info("=" * 50)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    data_path = Path(data_path)
    logger.info("加载数据...")

    # 动态加载所有样本
    samples = config.get('dataset.samples', [])
    mixed_data = []
    for i in range(1, len(samples) + 1):
        data_file = data_path / f'mixed{i}_data.npy'
        if data_file.exists():
            mixed_data.append(np.load(data_file))

    # 加载波数信息
    wavenumbers = np.load(data_path / 'wavenumbers.npy')
    logger.info(f"波数范围: {wavenumbers.min():.1f} ~ {wavenumbers.max():.1f} cm-1")

    # 合并数据
    X = np.vstack([m.reshape(-1, m.shape[-1]) for m in mixed_data])

    logger.info(f"数据形状: {X.shape}")

    # Load model
    logger.info("Load model...")
    num_classes = config.get('model.architecture.num_classes', 3)
    model = DualHeadRamanCNNLSTM(input_len=X.shape[1], num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")

    # 随机采样
    logger.info(f"采样 {n_samples} 个样本进行分析...")
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sample = X[indices]

    # 创建分析器
    analyzer = FeatureImportanceAnalyzer(model, device)

    # 计算全局Feature Importance
    logger.info("计算全局Feature Importance...")
    importances, importance_std = analyzer.compute_permutation_importance(
        X_sample, n_repeats=5, n_features_sample=100
    )

    # 生成特征名称（使用拉曼位移/波数）
    n_features = X.shape[1]
    feature_names = [f'Peak {int(round(wavenumbers[i]))}' for i in range(n_features)]

    # 找出最重要的特征索引
    top_feature_indices = np.argsort(np.abs(importances))[-3:][::-1]

    # 1. Feature Importance图
    logger.info("生成Feature Importance图...")
    plot_feature_importance(
        importances,
        importance_std=importance_std,
        feature_names=feature_names,
        top_k=15,
        save_path=output_dir / 'shap_value_plot.png'
    )

    # 2. 瀑布图
    logger.info("生成瀑布图...")
    # 选择一个中等置信度的"边界样本"，使正负贡献都出现
    sample_idx = _find_borderline_sample(analyzer, X_sample)
    local_importance, baseline_prob, pred_class = analyzer.compute_local_importance(X_sample, sample_idx)
    plot_waterfall(
        local_importance,
        baseline_prob,
        feature_names=feature_names,
        top_k=10,
        save_path=output_dir / 'shap_waterfall.png',
        predicted_class=pred_class
    )

    # 3. Feature Dependence Analysis图
    logger.info("生成Feature Dependence Analysis图...")
    plot_dependence(
        X_sample,
        importances,
        feature_indices=top_feature_indices.tolist(),
        feature_names=feature_names,
        save_path=output_dir / 'shap_dependence.png'
    )

    logger.info("=" * 50)
    logger.info(f"所有可视化已保存到: {output_dir}")
    logger.info("=" * 50)

    # 保存重要性数据
    np.save(output_dir / 'feature_importances.npy', importances)
    np.save(output_dir / 'feature_importance_std.npy', importance_std)
    logger.info("Feature importance data saved")

    return importances, importance_std


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(base_dir))
    from utils import get_config

    config = get_config()
    logger = get_logger('shap')

    # 使用Dual-head model
    model_path = base_dir / 'output' / 'models'  / 'best_model.pth'

    # 检查模型文件是否存在
    if not model_path.exists():
        logger.error(f"Model file does not exist: {model_path}")
        logger.error("Please run train_dual.py 生成Dual-head model")
        sys.exit(1)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    # 遍历所有数据集类型，分别生成 shap 分析
    datasets_config = config.get('dataset.datasets', [])
    for ds in datasets_config:
        dataset_type = ds['type']
        # 将中文路径转换为英文: "PP+淀粉" -> "PP_starch", "PE+淀粉" -> "PE_starch"
        dataset_type_en = dataset_type.replace('+淀粉', '_starch').replace('+', '_')
        samples = [s['name'] for s in ds['samples']]  # 提取样本名称
        logger.info(f"\n{'='*50}")
        logger.info(f"分析数据集: {dataset_type} -> {dataset_type_en}")
        logger.info(f"{'='*50}")

        # 加载该数据集的数据
        data_path = base_dir / 'preprocessed_data'

        # 加载数据
        mixed_data = []
        for sample_name in samples:
            data_file = data_path / f'{sample_name}_data.npy'
            if data_file.exists():
                mixed_data.append(np.load(data_file))

        if not mixed_data:
            logger.warning(f"Not found {dataset_type} data files, skipping")
            continue

        # 加载波数 (全局共享的波数文件)
        wavenumbers = np.load(data_path / 'wavenumbers.npy')

        # 合并数据
        X = np.vstack([m.reshape(-1, m.shape[-1]) for m in mixed_data])
        logger.info(f"数据形状: {X.shape}")

        # 加载Dual-head model
        num_classes = config.get('model.architecture.num_classes', 3)
        model = DualHeadRamanCNNLSTM(input_len=X.shape[1], num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        # 随机采样
        n_samples = 200
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_sample = X[indices]

        # 生成特征名称
        n_features = X.shape[1]
        feature_names = [f'Peak {int(round(wavenumbers[i]))}' for i in range(n_features)]

        # 对 PP 和 PE 分别进行分析
        for head in ['pp', 'pe']:
            logger.info(f"\n--- 分析 {head.upper()} 头 ---")
            output_dir = base_dir / 'output' / 'shap'  / dataset_type_en / head
            output_dir.mkdir(parents=True, exist_ok=True)

            # 创建分析器（指定分析哪个头）
            analyzer = FeatureImportanceAnalyzer(model, device, head=head)

            # 计算全局Feature Importance
            logger.info("计算全局Feature Importance...")
            importances, importance_std = analyzer.compute_permutation_importance(
                X_sample, n_repeats=5, n_features_sample=100
            )

            # 找出最重要的特征索引
            top_feature_indices = np.argsort(np.abs(importances))[-3:][::-1]

            # 1. Feature Importance图
            logger.info("生成Feature Importance图...")
            plot_feature_importance(
                importances, importance_std=importance_std,
                feature_names=feature_names, top_k=15,
                save_path=output_dir / 'shap_value_plot.png'
            )

            # 2. 瀑布图
            logger.info("生成瀑布图...")
            # 选择一个中等置信度的"边界样本"，使正负贡献都出现
            sample_idx = _find_borderline_sample(analyzer, X_sample)
            local_importance, baseline_prob, pred_class = analyzer.compute_local_importance(X_sample, sample_idx)
            plot_waterfall(
                local_importance, baseline_prob,
                feature_names=feature_names, top_k=10,
                save_path=output_dir / 'shap_waterfall.png',
                predicted_class=pred_class
            )

            # 3. Feature Dependence Analysis图
            logger.info("生成Feature Dependence Analysis图...")
            plot_dependence(
                X_sample, importances,
                feature_indices=top_feature_indices.tolist(),
                feature_names=feature_names,
                save_path=output_dir / 'shap_dependence.png'
            )

            # 保存重要性数据
            np.save(output_dir / 'feature_importances.npy', importances)
            np.save(output_dir / 'feature_importance_std.npy', importance_std)
            logger.info(f"{head.upper()} 头分析结果已保存到: {output_dir}")
