"""
小提琴图分析：
直接读取 cross_validation.py 输出的 cv_results.json，
使用 Balanced Accuracy（= Recall_Macro）作为评估指标
（类别不平衡下比 Overall Accuracy 更可靠）

输出位置：output/violin/PP/ 和 output/violin/PE/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_config, get_logger

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_cv_results(cv_results_path: Path) -> Dict[str, List[float]]:
    """从 cv_results.json 加载交叉验证结果的 recall(macro) 数据"""
    logger = get_logger('violin')

    if not cv_results_path.exists():
        raise FileNotFoundError(f"交叉验证结果文件不存在: {cv_results_path}")

    with open(cv_results_path, 'r') as f:
        raw = json.load(f)

    results = {}
    for model_name, metrics in raw.items():
        if 'recall' in metrics and len(metrics['recall']) > 0:
            results[model_name] = metrics['recall']
            logger.info(f"  {model_name}: {len(metrics['recall'])} folds, "
                        f"Recall_Macro = {np.mean(metrics['recall']):.4f} +/- {np.std(metrics['recall']):.4f}")
        else:
            logger.warning(f"  {model_name}: 缺少 recall 数据，跳过")

    return results


def plot_violin(results: Dict[str, List[float]], output_path: Path, task: str = 'PP'):
    """绘制小提琴图 - Y轴为 Balanced Accuracy (%)，数据来源为 Recall_Macro"""
    logger = get_logger('violin')

    # 准备数据（Balanced Accuracy = Recall_Macro）
    data = []
    for model_name, recall_values in results.items():
        for val in recall_values:
            data.append({'Model': model_name, 'Balanced_Accuracy': val * 100})

    df = pd.DataFrame(data)

    # 按 Balanced Accuracy 均值降序排列模型
    model_means = df.groupby('Model')['Balanced_Accuracy'].mean().sort_values(ascending=False)
    model_order = model_means.index.tolist()

    # 颜色方案
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c', '#95a5a6']
    palette = dict(zip(model_order, colors[:len(model_order)]))

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.violinplot(x='Model', y='Balanced_Accuracy', data=df, order=model_order,
                   palette=palette, inner='box', ax=ax)

    # 添加散点（显示每个fold的真实值）
    sns.stripplot(x='Model', y='Balanced_Accuracy', data=df, order=model_order,
                  color='black', alpha=0.3, size=4, ax=ax)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=12)
    ax.set_title(f'Cross-Validation Balanced Accuracy Distribution - {task} Task\n'
                 f'(3 repeats × 5 folds = 15 data points per model)',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

    # 添加均值+标准差标注
    for i, model in enumerate(model_order):
        model_data = df[df['Model'] == model]['Balanced_Accuracy']
        mean_val = model_data.mean()
        std_val = model_data.std()
        ax.annotate(f'{mean_val:.1f}±{std_val:.1f}%',
                    xy=(i, mean_val), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.9))

    # 添加网格线
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"小提琴图已保存: {output_path}")

    # 保存统计数据
    stats_path = output_path.parent / 'violin_stats.csv'
    stats_data = []
    for model in model_order:
        model_data = df[df['Model'] == model]['Balanced_Accuracy']
        stats_data.append({
            'Model': model,
            'Mean(%)': round(model_data.mean(), 2),
            'Std(%)': round(model_data.std(), 2),
            'Min(%)': round(model_data.min(), 2),
            'Max(%)': round(model_data.max(), 2),
            'Median(%)': round(model_data.median(), 2),
            'N': len(model_data)
        })
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"统计数据已保存: {stats_path}")

    # 打印统计表
    logger.info(f"\n{'='*70}")
    logger.info(f"  {task} Task - Balanced Accuracy 统计")
    logger.info(f"{'='*70}")
    for row in stats_data:
        logger.info(f"  {row['Model']:25s}  {row['Mean(%)']:6.2f}±{row['Std(%)']:5.2f}%  "
                     f"[{row['Min(%)']:.2f}, {row['Max(%)']:.2f}]")
    logger.info(f"{'='*70}")


def process_task(config, task='PP'):
    """处理单个任务：读取cv_results.json并绘制小提琴图"""
    logger = get_logger('violin')

    logger.info("=" * 60)
    logger.info(f"小提琴图分析 - {task} Task")
    logger.info("=" * 60)

    # 输出目录
    output_dir = Path(config.paths['output_dir']) / 'violin' / task
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取交叉验证结果
    cv_results_path = Path(config.paths['output_dir']) / 'models' / 'cross_validation' / task / 'cv_results.json'
    logger.info(f"\n>>> 读取交叉验证结果: {cv_results_path}")
    results = load_cv_results(cv_results_path)

    if not results:
        logger.error(f"没有找到有效的交叉验证结果！")
        return

    logger.info(f"  共加载 {len(results)} 个模型的结果")

    # 绘制小提琴图
    logger.info(f"\n>>> 绘制小提琴图 ({task})...")
    plot_violin(results, output_dir / 'balanced_accuracy_violin.png', task=task)

    logger.info(f"\n>>> {task} Task 完成！")


def main():
    """主函数"""
    config = get_config()
    logger = get_logger('violin')

    logger.info("=" * 60)
    logger.info("小提琴图分析 (Balanced Accuracy = Recall Macro)")
    logger.info("数据来源: cross_validation.py -> cv_results.json")
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
