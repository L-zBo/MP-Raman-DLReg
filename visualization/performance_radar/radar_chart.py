import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def create_radar_chart(csv_path, task_name, output_dir):
    """
    为单个任务（PP或PE）创建性能雷达图

    Args:
        csv_path: 性能指标CSV文件路径
        task_name: 任务名称 ('PP' 或 'PE')
        output_dir: 输出目录
    """
    # 读取数据
    df = pd.read_csv(csv_path)

    # 提取指标列（排除Model列）- CSV中的列名
    metrics_csv = ['Accuracy (Macro)', 'F1-Score (Macro)', 'Recall (Macro)', 'Precision (Macro)']
    # 显示用的标签（移除 Macro）
    metrics_display = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
    models = df['Model'].values

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(metrics_csv), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    # 颜色方案
    colors = sns.color_palette("husl", len(models))

    # 绘制每个模型的数据
    for idx, model in enumerate(models):
        values = df[df['Model'] == model][metrics_csv].values[0].tolist()
        values += values[:1]  # 闭合

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # 设置标签 - 使用简化的显示名称
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_display, size=11, fontweight='bold')
    ax.tick_params(axis='x', pad=15)  # 将标签往外移动
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 标题和图例
    plt.title(f'Performance Radar Chart - {task_name} Task', size=14, weight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # 保存
    output_path = Path(output_dir) / f'radar_chart_{task_name}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def create_comparison_radar(pp_csv, pe_csv, output_dir):
    """
    创建PP和PE并排对比的雷达图

    Args:
        pp_csv: PP任务的CSV文件路径
        pe_csv: PE任务的CSV文件路径
        output_dir: 输出目录
    """
    df_pp = pd.read_csv(pp_csv)
    df_pe = pd.read_csv(pe_csv)

    # CSV中的列名
    metrics_csv = ['Accuracy (Macro)', 'F1-Score (Macro)', 'Recall (Macro)', 'Precision (Macro)']
    # 显示用的标签（移除 Macro）
    metrics_display = ['Accuracy', 'F1-Score', 'Recall', 'Precision']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(metrics_csv), endpoint=False).tolist()
    angles += angles[:1]

    colors = sns.color_palette("husl", len(df_pp))

    # PP雷达图
    for idx, model in enumerate(df_pp['Model'].values):
        values = df_pp[df_pp['Model'] == model][metrics_csv].values[0].tolist()
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax1.fill(angles, values, alpha=0.15, color=colors[idx])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_display, size=10, fontweight='bold')
    ax1.tick_params(axis='x', pad=15)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('PP Task', size=12, weight='bold', pad=15)

    # PE雷达图
    for idx, model in enumerate(df_pe['Model'].values):
        values = df_pe[df_pe['Model'] == model][metrics_csv].values[0].tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax2.fill(angles, values, alpha=0.15, color=colors[idx])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_display, size=10, fontweight='bold')
    ax2.tick_params(axis='x', pad=15)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('PE Task', size=12, weight='bold', pad=15)

    fig.suptitle('Performance Radar Chart - PP vs PE Comparison', size=14, weight='bold', y=0.98)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9)

    output_path = Path(output_dir) / 'radar_chart_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == '__main__':
    # 数据路径
    pp_csv = 'output/comparison_Acc/PP/comparison_results.csv'
    pe_csv = 'output/comparison_Acc/PE/comparison_results.csv'
    output_dir = 'output/performance_radar'

    # 生成单个雷达图
    create_radar_chart(pp_csv, 'PP', output_dir)
    create_radar_chart(pe_csv, 'PE', output_dir)

    # 生成对比雷达图
    create_comparison_radar(pp_csv, pe_csv, output_dir)

    print("\nAll radar charts generated successfully!")
