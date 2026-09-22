"""
双头模型预测残差可视化分析
输出位置：output/predicted_residuals/

测试集: 每任务 3200 样本 (1600 原始测试 + 1600 混合测试)

生成日期: 2026-01-23
版本: v3.1.0
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from utils import get_config, get_logger

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 导入双头模型
from models.dual_head_model import DualHeadRamanCNNLSTM

# 随机种子
RANDOM_SEED = 42


def load_test_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载测试集数据

    每个任务的测试集由两部分组成:
    - 原始测试集 (pp_test/pe_test): 1600 样本
    - 混合测试集 (pp_pe_mixed_test): 1600 样本
    - 合计: 3200 样本

    Returns:
        X_pp_test: PP任务测试数据 (3200 samples)
        y_pp_test: PP任务测试标签
        X_pe_test: PE任务测试数据 (3200 samples)
        y_pe_test: PE任务测试标签
    """
    logger = get_logger('residual_dual')
    data_dir = Path(config.paths['preprocessed_dir'])
    true_label_dir = Path(__file__).parent.parent.parent / 'testing' / 'test_true_label'

    # 标签路径
    label_paths = {
        'PP': true_label_dir / 'pp_test' / 'pp_labels.npy',
        'PE': true_label_dir / 'pe_test' / 'pe_labels.npy',
        'PP_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy',
        'PE_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy'
    }

    # ========== PP 测试集 ==========
    # 原始PP测试集
    X_pp_only = np.load(data_dir / 'pp_test_data.npy')
    X_pp_only = X_pp_only.reshape(-1, X_pp_only.shape[-1])
    y_pp_only = np.load(label_paths['PP']).flatten()
    logger.info(f"  [PP] 原始测试集: {X_pp_only.shape[0]} 样本")

    # ========== PE 测试集 ==========
    # 原始PE测试集
    X_pe_only = np.load(data_dir / 'pe_test_data.npy')
    X_pe_only = X_pe_only.reshape(-1, X_pe_only.shape[-1])
    y_pe_only = np.load(label_paths['PE']).flatten()
    logger.info(f"  [PE] 原始测试集: {X_pe_only.shape[0]} 样本")

    # ========== 混合测试集 ==========
    X_mixed = np.load(data_dir / 'pp_pe_mixed_test_data.npy')
    X_mixed = X_mixed.reshape(-1, X_mixed.shape[-1])
    y_pp_mixed = np.load(label_paths['PP_mixed']).flatten()
    y_pe_mixed = np.load(label_paths['PE_mixed']).flatten()
    logger.info(f"  [Mixed] 混合测试集: {X_mixed.shape[0]} 样本")

    # ========== 合并测试集 ==========
    X_pp_test = np.vstack([X_pp_only, X_mixed])
    y_pp_test = np.concatenate([y_pp_only, y_pp_mixed])
    X_pe_test = np.vstack([X_pe_only, X_mixed])
    y_pe_test = np.concatenate([y_pe_only, y_pe_mixed])

    logger.info(f"PP Test Set: {X_pp_test.shape[0]} samples ({X_pp_only.shape[0]} + {X_mixed.shape[0]})")
    logger.info(f"PE Test Set: {X_pe_test.shape[0]} samples ({X_pe_only.shape[0]} + {X_mixed.shape[0]})")

    return X_pp_test, y_pp_test, X_pe_test, y_pe_test


def get_dual_predictions(model, X, device):
    """获取双头模型预测结果和概率"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        pp_outputs, pe_outputs = model(X_tensor)

        pp_probs = F.softmax(pp_outputs, dim=1).cpu().numpy()
        pe_probs = F.softmax(pe_outputs, dim=1).cpu().numpy()

        pp_preds = pp_outputs.argmax(dim=1).cpu().numpy()
        pe_preds = pe_outputs.argmax(dim=1).cpu().numpy()

    return pp_preds, pe_preds, pp_probs, pe_probs


def plot_residual_distribution(y_true, y_pred, probs, output_dir, task_name):
    """绘制残差分布直方图"""
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, name in enumerate(class_names):
        ax = axes[i]
        mask = y_true == i
        if mask.sum() == 0:
            continue

        true_probs = probs[mask, i]
        residuals = 1.0 - true_probs  # 残差 = 1 - 预测为真实类别的概率

        ax.hist(residuals, bins=30, color=['#2ecc71', '#3498db', '#e74c3c'][i],
                alpha=0.7, edgecolor='black')
        ax.axvline(x=residuals.mean(), color='red', linestyle='--',
                   label=f'Mean: {residuals.mean():.3f}')
        ax.set_title(f'{name}\n(n={mask.sum()})', fontsize=12)
        ax.set_xlabel('Residual (1 - P(true class))', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f'{task_name} - Dual Head Model - Prediction Residual Distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'residual_distribution_{task_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_comparison(y_true, y_pred, probs, output_dir, task_name):
    """绘制正确/错误预测的置信度对比"""
    correct_mask = y_true == y_pred

    # 获取预测类别的置信度
    confidence = probs.max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(confidence[correct_mask], bins=30, alpha=0.7, label='Correct',
            color='#2ecc71', edgecolor='black')
    ax.hist(confidence[~correct_mask], bins=30, alpha=0.7, label='Incorrect',
            color='#e74c3c', edgecolor='black')

    ax.axvline(x=confidence[correct_mask].mean(), color='green', linestyle='--',
               label=f'Correct Mean: {confidence[correct_mask].mean():.3f}')
    if (~correct_mask).sum() > 0:
        ax.axvline(x=confidence[~correct_mask].mean(), color='red', linestyle='--',
                   label=f'Incorrect Mean: {confidence[~correct_mask].mean():.3f}')

    ax.set_xlabel('Prediction Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{task_name} - Dual Head Model - Confidence Distribution (Correct vs Incorrect)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'confidence_comparison_{task_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_misclassified_analysis(X, y_true, y_pred, probs, output_dir, task_name):
    """分析误分类样本"""
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    raman_start, raman_end = 103.8, 3697.0
    raman_shifts = np.linspace(raman_start, raman_end, X.shape[1])

    misclassified_mask = y_true != y_pred

    if misclassified_mask.sum() == 0:
        return

    # 按真实类别分组误分类样本
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for true_class in range(3):
        ax = axes[true_class]
        mask = (y_true == true_class) & misclassified_mask
        correct_mask = (y_true == true_class) & ~misclassified_mask

        if mask.sum() > 0:
            # 绘制误分类样本的平均光谱
            mean_spectrum = X[mask].mean(axis=0)
            ax.plot(raman_shifts, mean_spectrum, color='red', linewidth=2,
                    label=f'Misclassified (n={mask.sum()})')

            # 绘制正确分类样本的平均光谱
            if correct_mask.sum() > 0:
                correct_spectrum = X[correct_mask].mean(axis=0)
                ax.plot(raman_shifts, correct_spectrum, color='green', linewidth=2,
                        alpha=0.7, label=f'Correct (n={correct_mask.sum()})')
        else:
            # 没有误分类样本时，只显示正确分类的光谱
            if correct_mask.sum() > 0:
                correct_spectrum = X[correct_mask].mean(axis=0)
                ax.plot(raman_shifts, correct_spectrum, color='green', linewidth=2,
                        alpha=0.7, label=f'Correct (n={correct_mask.sum()})')
                ax.text(0.5, 0.5, 'No misclassified samples\n(100% accuracy)',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=14, color='green', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'True Class: {class_names[true_class]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)
        ax.set_xlim(raman_start, raman_end)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f'{task_name} - Dual Head Model - Misclassified vs Correct Spectra',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'misclassified_spectra_{task_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_heatmap(y_true, probs, output_dir, task_name):
    """绘制预测概率热力图"""
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for true_class in range(3):
        ax = axes[true_class]
        mask = y_true == true_class
        if mask.sum() == 0:
            continue

        class_probs = probs[mask]

        # 按预测概率排序
        sorted_idx = np.argsort(class_probs[:, true_class])[::-1]
        n_show = min(100, len(sorted_idx))

        im = ax.imshow(class_probs[sorted_idx[:n_show]], aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1)
        ax.set_title(f'True: {class_names[true_class]}\n(n={mask.sum()})', fontsize=12)
        ax.set_xlabel('Predicted Class', fontsize=10)
        ax.set_ylabel('Sample (sorted by confidence)', fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Non', 'Slight', 'Severe'], fontsize=9)
        plt.colorbar(im, ax=ax, label='Probability')

    plt.suptitle(f'{task_name} - Dual Head Model - Prediction Probability Heatmap',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'probability_heatmap_{task_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('residual_dual')

    logger.info("="*60)
    logger.info("Prediction Residual Analysis (Dual Head Model)")
    logger.info("="*60)

    output_dir = Path(config.paths['output_dir']) / 'predicted_residuals'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    logger.info("\n>>> Loading test data...")
    X_pp, y_pp, X_pe, y_pe = load_test_data(config)

    logger.info("\n>>> Loading dual head model...")
    model_path = Path(config.paths['output_dir']) / 'models' / 'best_model.pth'
    model = DualHeadRamanCNNLSTM(input_len=X_pp.shape[1], num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    logger.info("\n>>> Getting predictions...")
    # 阈值调整参数
    pp_threshold_class1 = 0.15
    pp_threshold_class2 = 0.0
    pe_threshold_class1 = 0.50
    pe_threshold_class2 = 0.70

    # PP任务预测（不进行标准化，与训练时保持一致）
    pp_pred_raw, _, pp_probs_raw, _ = get_dual_predictions(model, X_pp, device)
    # 应用阈值调整
    pp_probs = pp_probs_raw.copy()
    pp_probs[:, 1] += pp_threshold_class1
    pp_probs[:, 2] += pp_threshold_class2
    # 重新归一化
    pp_probs = pp_probs / pp_probs.sum(axis=1, keepdims=True)
    pp_pred = pp_probs.argmax(axis=1)
    pp_accuracy = (pp_pred == y_pp).mean()
    logger.info(f"PP Test Accuracy (with threshold): {pp_accuracy:.4f}")

    # PE任务预测（不进行标准化，与训练时保持一致）
    _, pe_pred_raw, _, pe_probs_raw = get_dual_predictions(model, X_pe, device)
    # 应用阈值调整
    pe_probs = pe_probs_raw.copy()
    pe_probs[:, 1] += pe_threshold_class1
    pe_probs[:, 2] += pe_threshold_class2
    # 重新归一化
    pe_probs = pe_probs / pe_probs.sum(axis=1, keepdims=True)
    pe_pred = pe_probs.argmax(axis=1)
    pe_accuracy = (pe_pred == y_pe).mean()
    logger.info(f"PE Test Accuracy (with threshold): {pe_accuracy:.4f}")

    # ========== PP Task Analysis ==========
    logger.info("\n>>> Analyzing PP task...")

    logger.info("  Plotting PP residual distribution...")
    plot_residual_distribution(y_pp, pp_pred, pp_probs, output_dir, 'PP')

    logger.info("  Plotting PP confidence comparison...")
    plot_confidence_comparison(y_pp, pp_pred, pp_probs, output_dir, 'PP')

    logger.info("  Plotting PP misclassified samples analysis...")
    plot_misclassified_analysis(X_pp, y_pp, pp_pred, pp_probs, output_dir, 'PP')

    logger.info("  Plotting PP probability heatmap...")
    # 热力图使用原始概率（未经阈值调整），展示模型真实置信度
    plot_probability_heatmap(y_pp, pp_probs_raw, output_dir, 'PP')

    # ========== PE Task Analysis ==========
    logger.info("\n>>> Analyzing PE task...")

    logger.info("  Plotting PE residual distribution...")
    plot_residual_distribution(y_pe, pe_pred, pe_probs, output_dir, 'PE')

    logger.info("  Plotting PE confidence comparison...")
    plot_confidence_comparison(y_pe, pe_pred, pe_probs, output_dir, 'PE')

    logger.info("  Plotting PE misclassified samples analysis...")
    plot_misclassified_analysis(X_pe, y_pe, pe_pred, pe_probs, output_dir, 'PE')

    logger.info("  Plotting PE probability heatmap...")
    # 热力图使用原始概率（未经阈值调整），展示模型真实置信度
    plot_probability_heatmap(y_pe, pe_probs_raw, output_dir, 'PE')

    logger.info("\n>>> Completed!")
    logger.info(f"All results saved to: {output_dir}")

