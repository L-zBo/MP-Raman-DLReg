"""
训练曲线脚本。

功能：
- 读取训练历史
- 输出训练曲线与任务动态曲线

输出：
- output/loss_curve/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger, PRIMARY_MODEL_NAME

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_training_history(history_path: Path):
    """从JSON文件加载训练历史"""
    logger = get_logger('loss_curve')

    if not history_path.exists():
        logger.error(f"训练历史文件不存在: {history_path}")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    logger.info(f"加载训练历史: {len(history.get('train_loss', []))} epochs")
    return history


def plot_loss_curve(history: dict, output_path: Path, title: str = 'Training Curves'):
    """绘制单个模型的训练损失曲线"""
    logger = get_logger('loss_curve')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curve', fontsize=13, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 准确率曲线
    train_acc = [acc * 100 for acc in history['train_acc']]
    val_acc = [acc * 100 for acc in history['val_acc']]
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, 'r--', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Accuracy Curve', fontsize=13, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 学习率变化
    if 'lr' in history and history['lr']:
        axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nData Not Available',
                       ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')

    # 4. 过拟合分析（训练-验证准确率差距）
    train_val_gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    gap_percent = [gap * 100 for gap in train_val_gap]
    axes[1, 1].plot(epochs, gap_percent, 'm-', linewidth=2, label='Train-Val Gap')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(epochs, 0, gap_percent,
                            where=[gap > 0 for gap in gap_percent],
                            alpha=0.3, color='red', label='Overfitting Region')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Train-Val Accuracy Gap (%)', fontsize=12)
    axes[1, 1].set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"损失曲线已保存: {output_path}")


def plot_task_metric_curves(history: dict, metric_key: str, ylabel: str, title: str, output_path: Path):
    """绘制 PP / PE 双任务动态曲线"""
    logger = get_logger('loss_curve')

    pp_values = [v * 100 for v in history['train_metrics']['PP'][metric_key]]
    pe_values = [v * 100 for v in history['train_metrics']['PE'][metric_key]]
    epochs = range(1, len(pp_values) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(epochs, pp_values, color='#1f77b4', linewidth=2.4, label='PP')
    ax.plot(epochs, pe_values, color='#d62728', linewidth=2.4, label='PE')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_ylim(min(min(pp_values), min(pe_values)) - 5, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"任务动态曲线已保存: {output_path}")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('loss_curve')
    output_dir = Path(config.paths['output_dir']) / 'loss_curve'

    logger.info(f"杈撳嚭鐩綍: {output_dir}")
    logger.info("=" * 60)
    logger.info(f"{PRIMARY_MODEL_NAME} 训练损失曲线可视化")
    logger.info("=" * 60)

    # 设置输出目录
    output_dir = Path(config.paths['output_dir']) / 'loss_curve'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练历史文件路径
    history_path = Path(config.paths['output_dir']) / 'models' / 'training_history.json'

    logger.info(f"\n>>> 加载训练历史...")
    logger.info(f"  文件路径: {history_path}")

    history = load_training_history(history_path)

    if history is None:
        logger.error("\n未找到训练历史文件！")
        logger.error("请先运行 training/train_dual.py 训练模型")
        sys.exit(1)

    # 绘制损失曲线
    logger.info(f"\n>>> 绘制训练损失曲线...")
    plot_loss_curve(history, output_dir / 'training_curves.png',
                   f'{PRIMARY_MODEL_NAME} Training Curves')

    logger.info(f"\n>>> 绘制 PP / PE Accuracy 动态曲线...")
    plot_task_metric_curves(
        history,
        metric_key='accuracy',
        ylabel='Accuracy (%)',
        title=f'{PRIMARY_MODEL_NAME} Training Accuracy',
        output_path=output_dir / 'admic_accuracy_dynamics.png',
    )

    logger.info(f"\n>>> 绘制 PP / PE F1 动态曲线...")
    plot_task_metric_curves(
        history,
        metric_key='f1',
        ylabel='F1-Score (%)',
        title=f'{PRIMARY_MODEL_NAME} Training F1-Score',
        output_path=output_dir / 'admic_f1_dynamics.png',
    )

    logger.info("\n" + "=" * 60)
    logger.info("完成！")
    logger.info(f"输出位置: {output_dir}")
    logger.info("=" * 60)
