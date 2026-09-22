"""
预测值vs真实值散点图 - 双头模型版本

展示双头模型在测试集上的表现：
- 使用固定测试集（3200样本 = 1600原始测试 + 1600混合测试）
- 加载预训练双头模型（best_model.pth）
- 应用阈值调整（PP: +0.15/+0.0, PE: +0.50/+0.70）

输出位置: output/prediction_scatter/PP/ 和 output/prediction_scatter/PE/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

from utils import get_config, get_logger
from models.dual_head_model import DualHeadRamanCNNLSTM

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 阈值调整参数
PP_THRESHOLD_CLASS1 = 0.15
PP_THRESHOLD_CLASS2 = 0.0
PE_THRESHOLD_CLASS1 = 0.50
PE_THRESHOLD_CLASS2 = 0.70


def load_test_data(config, task='PP') -> Tuple[np.ndarray, np.ndarray]:
    """
    加载测试数据（3200样本 = 1600原始测试 + 1600混合测试）
    """
    logger = get_logger('prediction_scatter')

    preprocessed_dir = Path(config['paths']['preprocessed_dir'])
    true_label_dir = Path(__file__).parent.parent.parent / 'testing' / 'test_true_label'

    # 标签路径
    label_paths = {
        'pp': true_label_dir / 'pp_test' / 'pp_labels.npy',
        'pe': true_label_dir / 'pe_test' / 'pe_labels.npy',
        'pp_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy',
        'pe_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy'
    }

    if task == 'PP':
        # 加载PP测试集
        pp_data = np.load(preprocessed_dir / 'pp_test_data.npy')
        pp_labels = np.load(label_paths['pp'])
        X_pp_only = pp_data.reshape(-1, pp_data.shape[-1])
        y_pp_only = pp_labels.flatten()

        # 加载混合测试集
        mixed_data = np.load(preprocessed_dir / 'pp_pe_mixed_test_data.npy')
        pp_mixed_labels = np.load(label_paths['pp_mixed'])
        X_mixed = mixed_data.reshape(-1, mixed_data.shape[-1])
        y_pp_mixed = pp_mixed_labels.flatten()

        # 合并
        X_test = np.vstack([X_pp_only, X_mixed])
        y_test = np.concatenate([y_pp_only, y_pp_mixed])
    else:  # PE
        # 加载PE测试集
        pe_data = np.load(preprocessed_dir / 'pe_test_data.npy')
        pe_labels = np.load(label_paths['pe'])
        X_pe_only = pe_data.reshape(-1, pe_data.shape[-1])
        y_pe_only = pe_labels.flatten()

        # 加载混合测试集
        mixed_data = np.load(preprocessed_dir / 'pp_pe_mixed_test_data.npy')
        pe_mixed_labels = np.load(label_paths['pe_mixed'])
        X_mixed = mixed_data.reshape(-1, mixed_data.shape[-1])
        y_pe_mixed = pe_mixed_labels.flatten()

        # 合并
        X_test = np.vstack([X_pe_only, X_mixed])
        y_test = np.concatenate([y_pe_only, y_pe_mixed])

    logger.info(f"[{task}] 测试集: {X_test.shape[0]} 样本 (1600原始 + 1600混合)")
    return X_test, y_test


def get_predictions(X_test: np.ndarray, config, task: str = 'PP') -> Tuple[np.ndarray, np.ndarray]:
    """
    获取双头模型的预测结果
    加载预训练模型，应用阈值调整
    返回：(预测标签, 预测概率)
    """
    logger = get_logger('prediction_scatter')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = Path(config.base_dir) / 'output' / 'models' / 'best_model.pth'

    if not model_path.exists():
        logger.error(f"未找到预训练模型: {model_path}")
        return None, None

    logger.info(f"加载预训练模型: {model_path}")

    # 加载模型
    model = DualHeadRamanCNNLSTM(input_len=X_test.shape[1], num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        pp_outputs, pe_outputs = model(X_test_t)

        if task == 'PP':
            probs = torch.softmax(pp_outputs, dim=1).cpu().numpy()
            # 阈值调整
            probs[:, 1] += PP_THRESHOLD_CLASS1
            probs[:, 2] += PP_THRESHOLD_CLASS2
        else:  # PE
            probs = torch.softmax(pe_outputs, dim=1).cpu().numpy()
            # 阈值调整
            probs[:, 1] += PE_THRESHOLD_CLASS1
            probs[:, 2] += PE_THRESHOLD_CLASS2

        # 重新归一化
        probs = probs / probs.sum(axis=1, keepdims=True)
        # 预测标签
        y_pred = probs.argmax(axis=1)

    return y_pred, probs


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, task: str = 'PP'):
    """绘制预测vs真实值散点图"""
    logger = get_logger('prediction_scatter')

    fig, ax = plt.subplots(figsize=(10, 10))

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # 绿、蓝、红

    # 为每个类别添加抖动的散点
    for label in [0, 1, 2]:
        mask = y_true == label
        if mask.sum() > 0:
            # 减小抖动程度，让点更集中
            jitter_scale = 0.06  # 从0.15减小到0.06
            jitter_x = np.random.normal(0, jitter_scale, mask.sum())
            jitter_y = np.random.normal(0, jitter_scale, mask.sum())

            # 计算该类别的准确率
            class_acc = (y_true[mask] == y_pred[mask]).mean()

            ax.scatter(
                y_true[mask] + jitter_x,
                y_pred[mask] + jitter_y,
                c=colors[label],
                alpha=0.5,
                s=25,
                label=f'{class_names[label]}\n(n={mask.sum()}, acc={class_acc:.1%})'
            )

    # 完美预测对角线
    ax.plot([-0.5, 2.5], [-0.5, 2.5], 'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')

    ax.set_xlabel('True Label', fontsize=14)
    ax.set_ylabel('Predicted Label', fontsize=14)
    ax.set_title(f'Prediction vs True Label ({task} Task, n={len(y_true)})',
                 fontsize=16, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(class_names, rotation=15, ha='right', fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 添加总体准确率标注
    accuracy = (y_true == y_pred).mean()
    ax.text(0.95, 0.05, f'Overall Accuracy: {accuracy:.2%}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"保存: {output_path}")


def plot_confusion_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, task: str = 'PP'):
    """绘制混淆散点图（按预测结果分组）"""
    logger = get_logger('prediction_scatter')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for pred_label in [0, 1, 2]:
        ax = axes[pred_label]

        # 获取预测为该类别的样本
        pred_mask = y_pred == pred_label

        if pred_mask.sum() == 0:
            ax.text(0.5, 0.5, 'No predictions', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_title(f'Predicted: {class_names[pred_label]}', fontsize=12, fontweight='bold')
            continue

        # 统计真实类别分布
        true_labels_for_pred = y_true[pred_mask]
        counts = np.bincount(true_labels_for_pred.astype(int), minlength=3)

        # 绘制柱状图
        bars = ax.bar([0, 1, 2], counts, color=colors, edgecolor='black', linewidth=1)

        # 添加数值标注
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 高亮正确预测
        bars[pred_label].set_edgecolor('gold')
        bars[pred_label].set_linewidth(3)

        ax.set_xlabel('True Label', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Predicted: {class_names[pred_label]}\n(n={pred_mask.sum()})',
                     fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Non', 'Slight', 'Severe'], rotation=0)

        # 添加准确率
        precision = counts[pred_label] / counts.sum() if counts.sum() > 0 else 0
        ax.text(0.95, 0.95, f'Precision: {precision:.1%}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'Confusion Analysis by Predicted Class ({task} Task)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"保存: {output_path}")


def save_prediction_results(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                             output_dir: Path, task: str = 'PP'):
    """保存预测结果到CSV"""
    logger = get_logger('prediction_scatter')

    class_names = ['NonPollution', 'SlightPollution', 'SeverePollution']

    df = pd.DataFrame({
        'TrueLabel': y_true,
        'TrueLabelName': [class_names[int(l)] for l in y_true],
        'PredictedLabel': y_pred,
        'PredictedLabelName': [class_names[int(l)] for l in y_pred],
        'Correct': (y_true == y_pred).astype(int),
        'Prob_NonPollution': probs[:, 0],
        'Prob_SlightPollution': probs[:, 1],
        'Prob_SeverePollution': probs[:, 2],
        'MaxProb': probs.max(axis=1)
    })

    csv_path = output_dir / 'prediction_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"保存预测结果: {csv_path}")

    # 保存统计摘要
    summary = {
        'Total': len(y_true),
        'Correct': int((y_true == y_pred).sum()),
        'Accuracy': float((y_true == y_pred).mean()),
    }

    for label in [0, 1, 2]:
        mask = y_true == label
        if mask.sum() > 0:
            summary[f'Class{label}_Total'] = int(mask.sum())
            summary[f'Class{label}_Correct'] = int((y_true[mask] == y_pred[mask]).sum())
            summary[f'Class{label}_Accuracy'] = float((y_true[mask] == y_pred[mask]).mean())

    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / 'prediction_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"保存统计摘要: {summary_path}")


def process_task(config, task='PP'):
    """处理单个任务"""
    logger = get_logger('prediction_scatter')

    logger.info("=" * 60)
    logger.info(f"预测散点图分析 - {task} Task")
    logger.info("=" * 60)

    # 输出目录
    output_dir = Path(config.paths['output_dir']) / 'prediction_scatter' / task
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试数据
    logger.info(f"\n>>> 加载 {task} 测试数据...")
    X_test, y_test = load_test_data(config, task)

    # 打印类别分布
    unique, counts = np.unique(y_test, return_counts=True)
    logger.info(f"  类别分布: {dict(zip(unique, counts))}")

    # 获取预测结果
    logger.info(f"\n>>> 获取 {task} 预测结果...")
    y_pred, probs = get_predictions(X_test, config, task)

    if y_pred is None:
        logger.error("无法获取预测结果")
        return

    # 计算准确率
    accuracy = (y_test == y_pred).mean()
    logger.info(f"  总体准确率: {accuracy:.2%}")

    # 绘制散点图
    logger.info(f"\n>>> 绘制预测散点图...")
    plot_scatter(y_test, y_pred, output_dir / 'prediction_scatter.png', task)

    # 绘制混淆散点图
    logger.info(f"\n>>> 绘制混淆分析图...")
    plot_confusion_scatter(y_test, y_pred, output_dir / 'confusion_analysis.png', task)

    # 保存预测结果
    logger.info(f"\n>>> 保存预测结果...")
    save_prediction_results(y_test, y_pred, probs, output_dir, task)

    logger.info(f"\n>>> {task} Task 完成！")


def main():
    config = get_config()
    logger = get_logger('prediction_scatter')

    logger.info("=" * 60)
    logger.info("预测散点图分析（双头模型）")
    logger.info("=" * 60)

    # 处理 PP 任务
    process_task(config, task='PP')

    # 处理 PE 任务
    process_task(config, task='PE')

    logger.info("\n" + "=" * 60)
    logger.info("所有任务完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
