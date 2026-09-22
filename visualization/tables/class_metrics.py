"""
类别级别指标表生成
从测试集读取真实标签和预测标签，计算分类性能指标

输出: output/tables/class_metrics_PP.csv, class_metrics_PE.csv, class_metrics.tex
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils import get_config, get_logger


TEST_SAMPLES = {
    'PP': ['pp_test', 'pp_pe_mixed_test'],
    'PE': ['pe_test', 'pp_pe_mixed_test']
}

CLASS_NAMES = ['Non-pollution', 'Slight pollution', 'Severe pollution']


def load_test_labels(config, task: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载测试集的真实标签和预测标签"""
    logger = get_logger('class_metrics')

    output_dir = Path(config.paths['output_dir'])
    true_label_dir = Path(__file__).parent.parent.parent / 'testing' / 'test_true_label'

    y_true_list, y_pred_list = [], []
    task_lower = task.lower()

    for sample in TEST_SAMPLES[task]:
        true_path = true_label_dir / sample / f'{task_lower}_labels.npy'
        pred_path = output_dir / 'test_results_dual' / 'pixel_level' / sample / f'{task_lower}_labels.npy'

        if not true_path.exists():
            logger.warning(f"真实标签不存在: {true_path}")
            continue
        if not pred_path.exists():
            logger.warning(f"预测标签不存在: {pred_path}")
            continue

        y_true = np.load(true_path).flatten()
        y_pred = np.load(pred_path).flatten()

        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        logger.info(f"  加载 {sample}: {len(y_true)} 样本")

    if not y_true_list:
        return None, None

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def calculate_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """计算每个类别的指标"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    row_sums = cm.sum(axis=1)
    class_acc = np.divide(cm.diagonal(), row_sums, where=row_sums != 0, out=np.zeros_like(row_sums, dtype=float))

    rows = []
    for i, name in enumerate(CLASS_NAMES):
        rows.append({
            'Class': i,
            'Class Name': name,
            'Precision': f'{precision[i]:.4f}',
            'Recall': f'{recall[i]:.4f}',
            'F1-Score': f'{f1[i]:.4f}',
            'Accuracy': f'{class_acc[i]:.4f}',
            'Support': int(support[i])
        })

    rows.append({
        'Class': '-',
        'Class Name': 'Macro Avg',
        'Precision': f'{np.mean(precision):.4f}',
        'Recall': f'{np.mean(recall):.4f}',
        'F1-Score': f'{np.mean(f1):.4f}',
        'Accuracy': f'{np.mean(class_acc):.4f}',
        'Support': int(np.sum(support))
    })

    return pd.DataFrame(rows)


def generate_latex(results: Dict[str, pd.DataFrame]) -> str:
    """生成LaTeX三线表"""
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(r'\caption{Per-class Classification Metrics on Test Set}')
    latex.append(r'\label{tab:class_metrics}')
    latex.append(r'\begin{tabular}{llccccc}')
    latex.append(r'\toprule')
    latex.append(r'Task & Class & Precision & Recall & F1-Score & Accuracy & Support \\')
    latex.append(r'\midrule')

    for task, df in results.items():
        for idx, row in df.iterrows():
            task_col = task if idx == 0 else ''
            latex.append(
                f"{task_col} & {row['Class Name']} & {row['Precision']} & "
                f"{row['Recall']} & {row['F1-Score']} & {row['Accuracy']} & {row['Support']} \\\\"
            )
        if task != list(results.keys())[-1]:
            latex.append(r'\midrule')

    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)


def main():
    config = get_config()
    logger = get_logger('class_metrics')

    logger.info("=" * 60)
    logger.info("生成类别级别指标表（测试集）")
    logger.info("=" * 60)

    output_dir = Path(config.paths['output_dir']) / 'tables' / 'class_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for task in ['PP', 'PE']:
        logger.info(f"\n>>> 处理 {task} 任务...")

        y_true, y_pred = load_test_labels(config, task)

        if y_true is None:
            logger.warning(f"{task} 任务无可用数据")
            continue

        logger.info(f"  总样本数: {len(y_true)}")
        logger.info(f"  真实标签分布: {dict(zip(*np.unique(y_true, return_counts=True)))}")
        logger.info(f"  预测标签分布: {dict(zip(*np.unique(y_pred, return_counts=True)))}")

        df = calculate_class_metrics(y_true, y_pred)
        results[task] = df

        csv_path = output_dir / f'class_metrics_{task}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"  CSV已保存: {csv_path}")

    if results:
        latex_content = generate_latex(results)
        tex_path = output_dir / 'class_metrics.tex'
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        logger.info(f"\nLaTeX已保存: {tex_path}")

    logger.info("\n" + "=" * 60)
    for task, df in results.items():
        print(f"\n{task} Task Classification Metrics:")
        print(df.to_string(index=False))
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
