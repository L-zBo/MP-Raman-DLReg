"""
Ablation Study Visualization

Generated figures:
1. Accuracy comparison bar chart (accuracy_comparison.png)
2. F1 score comparison chart (f1_comparison.png)
3. Component contribution analysis (contribution_analysis.png)
4. Comprehensive metrics radar chart (radar_comparison.png)
5. All metrics comparison chart (all_metrics_comparison.png)

Output: output/ablation/PP/ and output/ablation/PE/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger, PRIMARY_MODEL_NAME

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

MODEL_COLORS = {
    PRIMARY_MODEL_NAME: '#2ecc71',
    'CNN-LSTM': '#3498db',
    'CNN-Attention': '#e74c3c',
    'CNN-only': '#9b59b6'
}

MODEL_LABELS = {
    PRIMARY_MODEL_NAME: f'{PRIMARY_MODEL_NAME}\n(Full Model)',
    'CNN-LSTM': 'CNN-LSTM\n(w/o Attention)',
    'CNN-Attention': 'CNN-Attention\n(w/o BiLSTM)',
    'CNN-only': 'CNN-only\n(w/o Attention & BiLSTM)'
}


def plot_accuracy_comparison(results_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    models = results_df['Model'].tolist()
    means = results_df['Accuracy_mean'].tolist()
    stds = results_df['Accuracy_std'].tolist()

    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.4f}\n+/-{std:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Ablation Study - Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    full_acc = results_df[results_df['Model'] == PRIMARY_MODEL_NAME]['Accuracy_mean'].values[0]
    ax.axhline(y=full_acc, color='green', linestyle='--', alpha=0.7, label=f'Full Model Baseline: {full_acc:.4f}')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_comparison(results_df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = results_df['Model'].tolist()
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]

    ax1 = axes[0]
    means = results_df['F1_macro_mean'].tolist()
    stds = results_df['F1_macro_std'].tolist()
    bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    for bar, mean in zip(bars, means):
        ax1.annotate(f'{mean:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('F1 Score - Macro', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=9)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    means = results_df['F1_weighted_mean'].tolist()
    stds = results_df['F1_weighted_std'].tolist()
    bars = ax2.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    for bar, mean in zip(bars, means):
        ax2.annotate(f'{mean:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('F1 Score - Weighted', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_contribution_analysis(results_df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    full_model = results_df[results_df['Model'] == PRIMARY_MODEL_NAME].iloc[0]
    full_acc = full_model['Accuracy_mean']
    full_f1 = full_model['F1_macro_mean']

    ablation_models = ['CNN-LSTM', 'CNN-Attention', 'CNN-only']
    component_names = ['Attention', 'BiLSTM', 'Attention & BiLSTM']

    acc_contributions = []
    f1_contributions = []

    for model in ablation_models:
        row = results_df[results_df['Model'] == model].iloc[0]
        acc_contributions.append(full_acc - row['Accuracy_mean'])
        f1_contributions.append(full_f1 - row['F1_macro_mean'])

    x = np.arange(len(component_names))
    colors = ['#3498db', '#e74c3c', '#9b59b6']

    ax1 = axes[0]
    bars = ax1.bar(x, acc_contributions, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, acc_contributions):
        ax1.annotate(f'{val:+.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Removed Component', fontsize=11)
    ax1.set_ylabel('Accuracy Drop', fontsize=11)
    ax1.set_title('Component Contribution to Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(component_names, fontsize=10)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    bars = ax2.bar(x, f1_contributions, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, f1_contributions):
        ax2.annotate(f'{val:+.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Removed Component', fontsize=11)
    ax2.set_ylabel('F1 Score Drop', fontsize=11)
    ax2.set_title('Component Contribution to F1 Score', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(component_names, fontsize=10)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'contribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_comparison(results_df: pd.DataFrame, output_dir: Path):
    from math import pi

    metrics = ['Accuracy', 'F1_macro', 'Precision', 'Recall']
    metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']

    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.set_theta_offset(pi / 4)
    ax.set_theta_direction(-1)

    for _, row in results_df.iterrows():
        model = row['Model']
        values = [row[f'{m}_mean'] for m in metrics]
        values += values[:1]

        color = MODEL_COLORS.get(model, '#95a5a6')
        ax.plot(angles, values, 'o-', linewidth=2, label=MODEL_LABELS.get(model, model), color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_title('Ablation Study - Metrics Comparison - Macro', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05))

    plt.tight_layout()
    plt.savefig(output_dir / 'radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_metrics_comparison(results_df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 8))

    models = results_df['Model'].tolist()
    metrics = ['Accuracy', 'F1_macro', 'Precision', 'Recall']
    metric_labels = ['Accuracy', 'F1', 'Precision', 'Recall']

    x = np.arange(len(models))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        means = results_df[f'{metric}_mean'].tolist()
        stds = results_df[f'{metric}_std'].tolist()
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=label, color=color, edgecolor='black')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study - All Metrics Comparison - Macro', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_ablation_visualizations(output_dir: Path):
    logger = get_logger('ablation_visualization')

    results_path = output_dir / 'ablation_results.csv'
    if not results_path.exists():
        logger.error(f"Ablation results not found: {results_path}")
        logger.info("Please run training/ablation_study.py first")
        return

    results_df = pd.read_csv(results_path)
    logger.info(f"Loaded ablation results: {len(results_df)} models")

    logger.info("\n>>> Plotting accuracy comparison...")
    plot_accuracy_comparison(results_df, output_dir)

    logger.info(">>> Plotting F1 score comparison...")
    plot_f1_comparison(results_df, output_dir)

    logger.info(">>> Plotting contribution analysis...")
    plot_contribution_analysis(results_df, output_dir)

    logger.info(">>> Plotting radar comparison...")
    plot_radar_comparison(results_df, output_dir)

    logger.info(">>> Plotting all metrics comparison...")
    plot_all_metrics_comparison(results_df, output_dir)

    logger.info(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('ablation_visualization')

    logger.info("=" * 60)
    logger.info("Ablation Study Visualization")
    logger.info("=" * 60)

    base_output_dir = Path(config.paths['output_dir']) / 'ablation'

    logger.info("\n" + "=" * 60)
    logger.info("PP Task Visualization")
    logger.info("=" * 60)
    generate_ablation_visualizations(base_output_dir / 'PP')

    logger.info("\n" + "=" * 60)
    logger.info("PE Task Visualization")
    logger.info("=" * 60)
    generate_ablation_visualizations(base_output_dir / 'PE')

    logger.info("\n" + "=" * 60)
    logger.info("Ablation visualization completed!")
    logger.info("=" * 60)
