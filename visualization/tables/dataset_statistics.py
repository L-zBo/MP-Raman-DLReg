"""
数据集统计表生成

按 Train/Val/Test 划分统计各污染级别样本数量

输出格式:
| Task  | Data Set | Non pollution | Slight pollution | Severe pollution |
|-------|----------|---------------|------------------|------------------|
| PP    | Train    | xxx           | xxx              | xxx              |
|       | Val      | xxx           | xxx              | xxx              |
|       | Test     | xxx           | xxx              | xxx              |
| PE    | Train    | xxx           | xxx              | xxx              |
|       | Val      | xxx           | xxx              | xxx              |
|       | Test     | xxx           | xxx              | xxx              |
| PP+PE | Test(PP) | xxx           | xxx              | xxx              |
|       | Test(PE) | xxx           | xxx              | xxx              |
| Total |          | xxx           | xxx              | xxx              |

输出: output/tables/dataset_statistics/
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger


def load_split_statistics(config) -> dict:
    """
    按 Train/Val/Test 划分加载数据集统计信息

    Returns:
        dict: {
            'PP': {'Train': {0: n, 1: n, 2: n}, 'Val': {...}, 'Test': {...}},
            'PE': {'Train': {0: n, 1: n, 2: n}, 'Val': {...}, 'Test': {...}}
        }
    """
    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    testing_dir = Path('testing/test_true_label')

    # 从config获取train/val划分
    split_config = config.get('training.split')
    pp_train_idx = split_config.get('PP+淀粉', {}).get('train', [1, 2, 3, 4])
    pp_val_idx = split_config.get('PP+淀粉', {}).get('val', [5])
    pe_train_idx = split_config.get('PE+淀粉', {}).get('train', [1, 2, 3, 4])
    pe_val_idx = split_config.get('PE+淀粉', {}).get('val', [5])

    stats = {
        'PP': {'Train': Counter(), 'Val': Counter(), 'Test': Counter()},
        'PE': {'Train': Counter(), 'Val': Counter(), 'Test': Counter()},
        'PP+PE': {'Test(PP)': Counter(), 'Test(PE)': Counter()}
    }

    # PP Train 数据
    for i in pp_train_idx:
        label_file = preprocessed_dir / f'pp_mixed{i}_pp_labels.npy'
        if label_file.exists():
            labels = np.load(label_file).flatten()
            stats['PP']['Train'].update(labels.tolist())

    # PP Val 数据
    for i in pp_val_idx:
        label_file = preprocessed_dir / f'pp_mixed{i}_pp_labels.npy'
        if label_file.exists():
            labels = np.load(label_file).flatten()
            stats['PP']['Val'].update(labels.tolist())

    # PP Test 数据
    pp_test_file = testing_dir / 'pp_test' / 'pp_labels.npy'
    if pp_test_file.exists():
        labels = np.load(pp_test_file).flatten()
        stats['PP']['Test'].update(labels.tolist())

    # PE Train 数据
    for i in pe_train_idx:
        label_file = preprocessed_dir / f'pe_mixed{i}_pe_labels.npy'
        if label_file.exists():
            labels = np.load(label_file).flatten()
            stats['PE']['Train'].update(labels.tolist())

    # PE Val 数据
    for i in pe_val_idx:
        label_file = preprocessed_dir / f'pe_mixed{i}_pe_labels.npy'
        if label_file.exists():
            labels = np.load(label_file).flatten()
            stats['PE']['Val'].update(labels.tolist())

    # PE Test 数据
    pe_test_file = testing_dir / 'pe_test' / 'pe_labels.npy'
    if pe_test_file.exists():
        labels = np.load(pe_test_file).flatten()
        stats['PE']['Test'].update(labels.tolist())

    # PP+PE 混合测试集
    pp_pe_pp_file = testing_dir / 'pp_pe_mixed_test' / 'pp_labels.npy'
    pp_pe_pe_file = testing_dir / 'pp_pe_mixed_test' / 'pe_labels.npy'
    if pp_pe_pp_file.exists():
        labels = np.load(pp_pe_pp_file).flatten()
        stats['PP+PE']['Test(PP)'].update(labels.tolist())
    if pp_pe_pe_file.exists():
        labels = np.load(pp_pe_pe_file).flatten()
        stats['PP+PE']['Test(PE)'].update(labels.tolist())

    return stats


def generate_dataset_detail(config, output_dir: Path):
    """生成详细数据集表（每个样本集的具体信息）"""
    logger = get_logger('dataset_statistics')

    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    testing_dir = Path('testing/test_true_label')

    split_config = config.get('training.split')
    pp_train_idx = split_config.get('PP+淀粉', {}).get('train', [1, 2, 3, 4])
    pp_val_idx = split_config.get('PP+淀粉', {}).get('val', [5])
    pe_train_idx = split_config.get('PE+淀粉', {}).get('train', [1, 2, 3, 4])
    pe_val_idx = split_config.get('PE+淀粉', {}).get('val', [5])

    rows = []

    # PP Train
    for i in pp_train_idx:
        data_file = preprocessed_dir / f'pp_mixed{i}_data.npy'
        label_file = preprocessed_dir / f'pp_mixed{i}_pp_labels.npy'
        if data_file.exists():
            data = np.load(data_file)
            n_samples = data.shape[0] * data.shape[1] if data.ndim == 3 else len(data)
            label_dist = {}
            if label_file.exists():
                labels = np.load(label_file).flatten()
                counts = Counter(labels.tolist())
                label_dist = {0: counts.get(0, 0), 1: counts.get(1, 0), 2: counts.get(2, 0)}
            rows.append({
                'Task': 'PP',
                'Split': 'Train',
                'Dataset': f'pp_mixed{i}',
                'Samples': n_samples,
                'Spatial Size': '40×40',
                'Spectral Bands': 1024,
                'Non pollution': label_dist.get(0, 0),
                'Slight pollution': label_dist.get(1, 0),
                'Severe pollution': label_dist.get(2, 0)
            })

    # PP Val
    for i in pp_val_idx:
        data_file = preprocessed_dir / f'pp_mixed{i}_data.npy'
        label_file = preprocessed_dir / f'pp_mixed{i}_pp_labels.npy'
        if data_file.exists():
            data = np.load(data_file)
            n_samples = data.shape[0] * data.shape[1] if data.ndim == 3 else len(data)
            label_dist = {}
            if label_file.exists():
                labels = np.load(label_file).flatten()
                counts = Counter(labels.tolist())
                label_dist = {0: counts.get(0, 0), 1: counts.get(1, 0), 2: counts.get(2, 0)}
            rows.append({
                'Task': 'PP',
                'Split': 'Val',
                'Dataset': f'pp_mixed{i}',
                'Samples': n_samples,
                'Spatial Size': '40×40',
                'Spectral Bands': 1024,
                'Non pollution': label_dist.get(0, 0),
                'Slight pollution': label_dist.get(1, 0),
                'Severe pollution': label_dist.get(2, 0)
            })

    # PP Test
    pp_test_file = testing_dir / 'pp_test' / 'pp_labels.npy'
    if pp_test_file.exists():
        labels = np.load(pp_test_file).flatten()
        counts = Counter(labels.tolist())
        rows.append({
            'Task': 'PP',
            'Split': 'Test',
            'Dataset': 'pp_test',
            'Samples': len(labels),
            'Spatial Size': '40×40',
            'Spectral Bands': 1024,
            'Non pollution': counts.get(0, 0),
            'Slight pollution': counts.get(1, 0),
            'Severe pollution': counts.get(2, 0)
        })

    # PE Train
    for i in pe_train_idx:
        data_file = preprocessed_dir / f'pe_mixed{i}_data.npy'
        label_file = preprocessed_dir / f'pe_mixed{i}_pe_labels.npy'
        if data_file.exists():
            data = np.load(data_file)
            n_samples = data.shape[0] * data.shape[1] if data.ndim == 3 else len(data)
            label_dist = {}
            if label_file.exists():
                labels = np.load(label_file).flatten()
                counts = Counter(labels.tolist())
                label_dist = {0: counts.get(0, 0), 1: counts.get(1, 0), 2: counts.get(2, 0)}
            rows.append({
                'Task': 'PE',
                'Split': 'Train',
                'Dataset': f'pe_mixed{i}',
                'Samples': n_samples,
                'Spatial Size': '40×40',
                'Spectral Bands': 1024,
                'Non pollution': label_dist.get(0, 0),
                'Slight pollution': label_dist.get(1, 0),
                'Severe pollution': label_dist.get(2, 0)
            })

    # PE Val
    for i in pe_val_idx:
        data_file = preprocessed_dir / f'pe_mixed{i}_data.npy'
        label_file = preprocessed_dir / f'pe_mixed{i}_pe_labels.npy'
        if data_file.exists():
            data = np.load(data_file)
            n_samples = data.shape[0] * data.shape[1] if data.ndim == 3 else len(data)
            label_dist = {}
            if label_file.exists():
                labels = np.load(label_file).flatten()
                counts = Counter(labels.tolist())
                label_dist = {0: counts.get(0, 0), 1: counts.get(1, 0), 2: counts.get(2, 0)}
            rows.append({
                'Task': 'PE',
                'Split': 'Val',
                'Dataset': f'pe_mixed{i}',
                'Samples': n_samples,
                'Spatial Size': '40×40',
                'Spectral Bands': 1024,
                'Non pollution': label_dist.get(0, 0),
                'Slight pollution': label_dist.get(1, 0),
                'Severe pollution': label_dist.get(2, 0)
            })

    # PE Test
    pe_test_file = testing_dir / 'pe_test' / 'pe_labels.npy'
    if pe_test_file.exists():
        labels = np.load(pe_test_file).flatten()
        counts = Counter(labels.tolist())
        rows.append({
            'Task': 'PE',
            'Split': 'Test',
            'Dataset': 'pe_test',
            'Samples': len(labels),
            'Spatial Size': '40×40',
            'Spectral Bands': 1024,
            'Non pollution': counts.get(0, 0),
            'Slight pollution': counts.get(1, 0),
            'Severe pollution': counts.get(2, 0)
        })

    # PP+PE 混合测试集 - PP维度
    pp_pe_pp_file = testing_dir / 'pp_pe_mixed_test' / 'pp_labels.npy'
    if pp_pe_pp_file.exists():
        labels = np.load(pp_pe_pp_file).flatten()
        counts = Counter(labels.tolist())
        rows.append({
            'Task': 'PP+PE',
            'Split': 'Test(PP)',
            'Dataset': 'pp_pe_mixed_test',
            'Samples': len(labels),
            'Spatial Size': '40×40',
            'Spectral Bands': 1024,
            'Non pollution': counts.get(0, 0),
            'Slight pollution': counts.get(1, 0),
            'Severe pollution': counts.get(2, 0)
        })

    # PP+PE 混合测试集 - PE维度
    pp_pe_pe_file = testing_dir / 'pp_pe_mixed_test' / 'pe_labels.npy'
    if pp_pe_pe_file.exists():
        labels = np.load(pp_pe_pe_file).flatten()
        counts = Counter(labels.tolist())
        rows.append({
            'Task': 'PP+PE',
            'Split': 'Test(PE)',
            'Dataset': 'pp_pe_mixed_test',
            'Samples': len(labels),
            'Spatial Size': '40×40',
            'Spectral Bands': 1024,
            'Non pollution': counts.get(0, 0),
            'Slight pollution': counts.get(1, 0),
            'Severe pollution': counts.get(2, 0)
        })

    df_detail = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_detail.to_csv(output_dir / 'dataset_detail.csv', index=False)
    logger.info(f"详细数据集表已保存到: {output_dir / 'dataset_detail.csv'}")

    return df_detail


def generate_dataset_table(config, output_dir: Path):
    """生成数据集统计表（按用户指定格式）"""
    logger = get_logger('dataset_statistics')

    stats = load_split_statistics(config)

    # 构建表格数据
    rows = []
    total_counts = Counter()

    # PP 任务
    for i, split in enumerate(['Train', 'Val', 'Test']):
        counts = stats['PP'][split]
        non_p, slight, severe = counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)
        row = {
            'Task': 'PP' if i == 0 else '',
            'Data Set': split,
            'Non pollution': non_p,
            'Slight pollution': slight,
            'Severe pollution': severe,
            'TOTAL': non_p + slight + severe
        }
        rows.append(row)
        for cls in [0, 1, 2]:
            total_counts[cls] += counts.get(cls, 0)

    # PE 任务
    for i, split in enumerate(['Train', 'Val', 'Test']):
        counts = stats['PE'][split]
        non_p, slight, severe = counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)
        row = {
            'Task': 'PE' if i == 0 else '',
            'Data Set': split,
            'Non pollution': non_p,
            'Slight pollution': slight,
            'Severe pollution': severe,
            'TOTAL': non_p + slight + severe
        }
        rows.append(row)
        for cls in [0, 1, 2]:
            total_counts[cls] += counts.get(cls, 0)

    # PP+PE 混合测试集
    for i, split in enumerate(['Test(PP)', 'Test(PE)']):
        counts = stats['PP+PE'][split]
        non_p, slight, severe = counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)
        row = {
            'Task': 'PP+PE' if i == 0 else '',
            'Data Set': split,
            'Non pollution': non_p,
            'Slight pollution': slight,
            'Severe pollution': severe,
            'TOTAL': non_p + slight + severe
        }
        rows.append(row)
        for cls in [0, 1, 2]:
            total_counts[cls] += counts.get(cls, 0)

    # Total 行
    grand_total = total_counts.get(0, 0) + total_counts.get(1, 0) + total_counts.get(2, 0)
    rows.append({
        'Task': 'Total',
        'Data Set': '',
        'Non pollution': total_counts.get(0, 0),
        'Slight pollution': total_counts.get(1, 0),
        'Severe pollution': total_counts.get(2, 0),
        'TOTAL': grand_total
    })

    df = pd.DataFrame(rows)

    # 保存CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'dataset_statistics.csv', index=False)

    # 生成LaTeX表格
    latex_content = generate_latex_table(df)
    with open(output_dir / 'dataset_statistics.tex', 'w', encoding='utf-8') as f:
        f.write(latex_content)

    logger.info(f"数据集统计表已保存到: {output_dir}")

    # 打印表格
    print_formatted_table(stats, total_counts)

    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """生成LaTeX表格代码（符合学术论文格式）"""
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(r'\caption{Assign spectral dataset}')
    latex.append(r'\label{tab:dataset_statistics}')
    latex.append(r'\begin{tabular}{llcccc}')
    latex.append(r'\toprule')
    latex.append(r'Task & Data Set & Non pollution & Slight pollution & Severe pollution & TOTAL \\')
    latex.append(r'\midrule')

    for idx, row in df.iterrows():
        task = row['Task']
        data_set = row['Data Set']
        non_pol = row['Non pollution']
        slight = row['Slight pollution']
        severe = row['Severe pollution']
        total = row['TOTAL']

        # Total行和PP+PE行前加横线
        if task in ('Total', 'PP+PE'):
            latex.append(r'\midrule')

        latex.append(f'{task} & {data_set} & {non_pol} & {slight} & {severe} & {total} \\\\')

    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)


def print_formatted_table(stats: dict, total_counts: Counter):
    """打印格式化表格到控制台"""
    print("\n" + "=" * 85)
    print("Assign spectral dataset")
    print("=" * 85)
    print(f"{'Task':<8} {'Data Set':<10} {'Non pollution':>15} {'Slight pollution':>18} {'Severe pollution':>18} {'TOTAL':>8}")
    print("-" * 85)

    # PP
    for i, split in enumerate(['Train', 'Val', 'Test']):
        task_name = 'PP' if i == 0 else ''
        counts = stats['PP'][split]
        non_p, slight, severe = counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)
        print(f"{task_name:<8} {split:<10} {non_p:>15} {slight:>18} {severe:>18} {non_p+slight+severe:>8}")

    print("-" * 85)

    # PE
    for i, split in enumerate(['Train', 'Val', 'Test']):
        task_name = 'PE' if i == 0 else ''
        counts = stats['PE'][split]
        non_p, slight, severe = counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)
        print(f"{task_name:<8} {split:<10} {non_p:>15} {slight:>18} {severe:>18} {non_p+slight+severe:>8}")

    print("-" * 85)

    # PP+PE
    for i, split in enumerate(['Test(PP)', 'Test(PE)']):
        task_name = 'PP+PE' if i == 0 else ''
        counts = stats['PP+PE'][split]
        non_p, slight, severe = counts.get(0, 0), counts.get(1, 0), counts.get(2, 0)
        print(f"{task_name:<8} {split:<10} {non_p:>15} {slight:>18} {severe:>18} {non_p+slight+severe:>8}")

    print("-" * 85)

    # Total
    grand_total = total_counts.get(0, 0) + total_counts.get(1, 0) + total_counts.get(2, 0)
    print(f"{'Total':<8} {'':<10} {total_counts.get(0, 0):>15} {total_counts.get(1, 0):>18} {total_counts.get(2, 0):>18} {grand_total:>8}")
    print("=" * 85)


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('dataset_statistics')

    logger.info("=" * 60)
    logger.info("生成数据集统计表")
    logger.info("=" * 60)

    output_dir = Path(config.paths['output_dir']) / 'tables' / 'dataset_statistics'

    # 生成汇总统计表
    df = generate_dataset_table(config, output_dir)

    # 生成详细数据集表
    df_detail = generate_dataset_detail(config, output_dir)

    print("\n详细数据集信息:")
    print(df_detail.to_string(index=False))
