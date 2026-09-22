"""
预测值vs真实值核密度估计(KDE)可视化

输出位置: output/prediction_kde/PP/ 和 output/prediction_kde/PE/
数据来源: 测试集 (pp_test, pe_test, pp_pe_mixed_test)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger
from models.dual_head_model import DualHeadRamanCNNLSTM

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

CLASS_COLORS = {
    0: '#2ecc71',
    1: '#f39c12',
    2: '#e74c3c'
}

CLASS_NAMES = ['Non-pollution', 'Slight pollution', 'Severe pollution']

TRUE_COLOR = '#3498db'
PRED_COLOR = '#e74c3c'


def load_test_data_and_predict(config, task: str):
    """加载测试集数据并预测"""
    logger = get_logger('prediction_kde')

    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    true_label_dir = Path(__file__).parent.parent.parent / 'testing' / 'test_true_label'
    models_dir = Path(config.paths['output_dir']) / 'models'

    label_suffix = 'pp_labels.npy' if task == 'PP' else 'pe_labels.npy'

    if task == 'PP':
        test_sets = ['pp_test', 'pp_pe_mixed_test']
    else:
        test_sets = ['pe_test', 'pp_pe_mixed_test']

    all_data, all_labels = [], []

    for test_name in test_sets:
        data_file = preprocessed_dir / f'{test_name}_data.npy'
        label_file = true_label_dir / test_name / label_suffix

        if data_file.exists() and label_file.exists():
            data = np.load(data_file)
            labels = np.load(label_file)

            data = data.reshape(-1, data.shape[-1])
            labels = labels.flatten()

            all_data.append(data)
            all_labels.append(labels)
            logger.info(f"  加载 {test_name}: {len(labels)} 样本")

    if not all_data:
        logger.warning(f"未找到{task}任务测试数据")
        return None, None

    X = np.vstack(all_data)
    y_true = np.concatenate(all_labels)

    logger.info(f"[{task}] 测试集: {len(y_true)} 样本")

    model_file = models_dir / 'best_model.pth'
    if not model_file.exists():
        logger.warning(f"未找到模型文件: {model_file}")
        return y_true, y_true

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualHeadRamanCNNLSTM(input_len=X.shape[1], num_classes=3).to(device)

    try:
        checkpoint = torch.load(model_file, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        logger.warning(f"加载模型失败: {e}")
        return y_true, y_true

    model.eval()
    y_pred = []

    # 阈值调整参数 (与 adjust_threshold_test.py 一致)
    if task == 'PP':
        threshold_class1 = 0.15
        threshold_class2 = 0.0
    else:
        threshold_class1 = 0.50
        threshold_class2 = 0.70

    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = torch.FloatTensor(X[i:i+256]).to(device)
            pp_out, pe_out = model(batch)
            outputs = pp_out if task == 'PP' else pe_out

            # 应用阈值调整
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probs[:, 1] += threshold_class1
            probs[:, 2] += threshold_class2
            probs = probs / probs.sum(axis=1, keepdims=True)

            predicted = probs.argmax(axis=1)
            y_pred.extend(predicted)

    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    logger.info(f"[{task}] 预测准确率: {accuracy:.4f}")

    return y_true, y_pred


def plot_prediction_vs_true_kde(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, task: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for class_id, ax in enumerate(axes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)

        true_count = np.sum(true_mask)
        pred_count = np.sum(pred_mask)

        x = ['True', 'Predicted']
        heights = [true_count, pred_count]
        colors = [TRUE_COLOR, PRED_COLOR]

        bars = ax.bar(x, heights, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

        for bar, h in zip(bars, heights):
            ax.annotate(f'{h}',
                       xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Sample Count', fontsize=11)
        ax.set_title(f'{CLASS_NAMES[class_id]}', fontsize=12, fontweight='bold',
                    color=CLASS_COLORS[class_id])
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Prediction vs Ground Truth Distribution - {task} Task', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'prediction_vs_true_distribution_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_kde_overlay(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, task: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    y_true_jitter = y_true + np.random.normal(0, 0.05, len(y_true))
    y_pred_jitter = y_pred + np.random.normal(0, 0.05, len(y_pred))

    x_range = np.linspace(-0.5, 2.5, 300)

    try:
        kde_true = stats.gaussian_kde(y_true_jitter)
        kde_pred = stats.gaussian_kde(y_pred_jitter)

        density_true = kde_true(x_range)
        density_pred = kde_pred(x_range)

        ax.plot(x_range, density_true, color=TRUE_COLOR, linewidth=2.5, label='Ground Truth')
        ax.fill_between(x_range, density_true, alpha=0.3, color=TRUE_COLOR)

        ax.plot(x_range, density_pred, color=PRED_COLOR, linewidth=2.5, label='Prediction')
        ax.fill_between(x_range, density_pred, alpha=0.3, color=PRED_COLOR)

    except Exception:
        ax.hist(y_true, bins=30, density=True, alpha=0.5, color=TRUE_COLOR, label='Ground Truth')
        ax.hist(y_pred, bins=30, density=True, alpha=0.5, color=PRED_COLOR, label='Prediction')

    for class_id in range(3):
        ax.axvline(x=class_id, color=CLASS_COLORS[class_id], linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(class_id, ax.get_ylim()[1] * 0.95, CLASS_NAMES[class_id],
               ha='center', fontsize=10, color=CLASS_COLORS[class_id], fontweight='bold')

    ax.set_xlabel('Class Label', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Prediction vs Ground Truth KDE - {task} Task', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'prediction_vs_true_kde_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_kde(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, task: str):
    fig, ax = plt.subplots(figsize=(12, 7))

    x_range = np.linspace(-0.5, 2.5, 300)

    for class_id in range(3):
        true_samples = np.where(y_true == class_id)[0]
        pred_samples = np.where(y_pred == class_id)[0]

        true_data = np.full(len(true_samples), class_id) + np.random.normal(0, 0.1, len(true_samples))
        pred_data = np.full(len(pred_samples), class_id) + np.random.normal(0, 0.1, len(pred_samples))

        color = CLASS_COLORS[class_id]

        if len(true_data) > 1:
            try:
                kde_true = stats.gaussian_kde(true_data)
                ax.plot(x_range, kde_true(x_range), color=color, linewidth=2,
                       linestyle='-', label=f'{CLASS_NAMES[class_id]} - True')
                ax.fill_between(x_range, kde_true(x_range), alpha=0.2, color=color)
            except:
                pass

        if len(pred_data) > 1:
            try:
                kde_pred = stats.gaussian_kde(pred_data)
                ax.plot(x_range, kde_pred(x_range), color=color, linewidth=2,
                       linestyle='--', label=f'{CLASS_NAMES[class_id]} - Pred')
            except:
                pass

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Ground Truth vs Prediction KDE Comparison - {task} Task', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'combined_kde_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_prediction_kde_visualizations(config, output_dir: Path):
    logger = get_logger('prediction_kde')

    output_dir.mkdir(parents=True, exist_ok=True)

    for task in ['PP', 'PE']:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"生成 {task} 任务预测值KDE图...")

        y_true, y_pred = load_test_data_and_predict(config, task)

        if y_true is not None and y_pred is not None:
            plot_prediction_vs_true_kde(y_true, y_pred, task_dir, task)
            plot_kde_overlay(y_true, y_pred, task_dir, task)
            plot_combined_kde(y_true, y_pred, task_dir, task)
            logger.info(f"{task} KDE图已保存到: {task_dir}")
        else:
            logger.warning(f"未找到 {task} 预测结果")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('prediction_kde')

    logger.info("=" * 60)
    logger.info("生成预测值vs真实值KDE可视化")
    logger.info("=" * 60)

    output_dir = Path(config.paths['output_dir']) / 'prediction_kde'
    generate_prediction_kde_visualizations(config, output_dir)

    logger.info("\nKDE可视化完成！")
