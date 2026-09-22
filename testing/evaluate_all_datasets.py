"""
全数据集评估脚本。

功能：
- 评估训练集、验证集、测试集
- 输出主要指标与报告

输出：
- output/evaluation/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from models.dual_head_model import DualHeadRamanCNNLSTM
from utils.config import get_config
from utils.logger import get_logger

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_dataset(config, dataset_type='train'):
    """加载指定数据集。"""
    data_dir = Path(config.paths['preprocessed_dir'])
    output_dir = Path(config.paths['output_dir'])
    true_label_dir = Path(__file__).parent / 'test_true_label'

    if dataset_type == 'test':
        X_pp_test = np.load(data_dir / 'pp_test_data.npy')
        X_pe_test = np.load(data_dir / 'pe_test_data.npy')
        X_mixed_test = np.load(data_dir / 'pp_pe_mixed_test_data.npy')

        y_pp_test = np.load(true_label_dir / 'pp_test' / 'pp_labels.npy')
        y_pe_test = np.load(true_label_dir / 'pe_test' / 'pe_labels.npy')
        y_pp_mixed = np.load(true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy')
        y_pe_mixed = np.load(true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy')

        X_pp = np.vstack([
            X_pp_test.reshape(-1, X_pp_test.shape[-1]),
            X_mixed_test.reshape(-1, X_mixed_test.shape[-1])
        ])
        X_pe = np.vstack([
            X_pe_test.reshape(-1, X_pe_test.shape[-1]),
            X_mixed_test.reshape(-1, X_mixed_test.shape[-1])
        ])

        y_pp = np.concatenate([y_pp_test.flatten(), y_pp_mixed.flatten()])
        y_pe = np.concatenate([y_pe_test.flatten(), y_pe_mixed.flatten()])

        return X_pp, y_pp, X_pe, y_pe

    datasets = config.get('dataset.datasets')
    split_config = config.get('training.split')

    data_list, pp_labels_list, pe_labels_list = [], [], []

    for dataset in datasets:
        dataset_type_name = dataset['type']
        samples = dataset['samples']
        split = split_config.get(dataset_type_name, {'train': [], 'val': []})

        for i, sample in enumerate(samples, 1):
            if (dataset_type == 'train' and i in split['train']) or \
               (dataset_type == 'val' and i in split['val']):
                name = sample['name']
                data = np.load(data_dir / f'{name}_data.npy')
                pp_labels = np.load(data_dir / f'{name}_pp_labels.npy')
                pe_labels = np.load(data_dir / f'{name}_pe_labels.npy')

                data_list.append(data.reshape(-1, data.shape[-1]))
                pp_labels_list.append(pp_labels.flatten())
                pe_labels_list.append(pe_labels.flatten())

    X = np.vstack(data_list)
    y_pp = np.concatenate(pp_labels_list)
    y_pe = np.concatenate(pe_labels_list)

    return X, y_pp, X, y_pe


def evaluate_task(model, X, y_true, device, task='PP', threshold_adjust_class1=0.0, threshold_adjust_class2=0.0):
    """评估单个任务。"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        if task == 'PP':
            pp_out, _ = model(X_tensor)
            probs = torch.softmax(pp_out, dim=1).cpu().numpy()
        else:
            _, pe_out = model(X_tensor)
            probs = torch.softmax(pe_out, dim=1).cpu().numpy()

    # 应用阈值调整
    probs[:, 1] += threshold_adjust_class1
    probs[:, 2] += threshold_adjust_class2

    # 重新归一化
    probs = probs / probs.sum(axis=1, keepdims=True)

    # 预测
    y_pred = probs.argmax(axis=1)

    # 计算各种指标
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_true': y_true
    }


def print_results(results, dataset_name, task, logger):
    """打印评估结果"""
    logger.info(f"\n{'='*80}")
    logger.info(f"{dataset_name} - {task}任务评估结果")
    logger.info(f"{'='*80}")

    logger.info(f"\n总体指标:")
    logger.info(f"  准确率 (Accuracy):           {results['accuracy']:.4f}")
    logger.info(f"  精确率 (Macro):              {results['precision_macro']:.4f}")
    logger.info(f"  召回率 (Macro):              {results['recall_macro']:.4f}")
    logger.info(f"  F1分数 (Macro):              {results['f1_macro']:.4f}")
    logger.info(f"  精确率 (Weighted):           {results['precision_weighted']:.4f}")
    logger.info(f"  召回率 (Weighted):           {results['recall_weighted']:.4f}")
    logger.info(f"  F1分数 (Weighted):           {results['f1_weighted']:.4f}")

    logger.info(f"\n各类别详细指标:")
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    logger.info(f"{'类别':<20} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'样本数':<10}")
    logger.info("-" * 80)

    for i, class_name in enumerate(class_names):
        support = (results['y_true'] == i).sum()
        logger.info(f"{class_name:<20} {results['precision_per_class'][i]:<12.4f} "
                   f"{results['recall_per_class'][i]:<12.4f} "
                   f"{results['f1_per_class'][i]:<12.4f} "
                   f"{support:<10}")

    logger.info(f"\n混淆矩阵:")
    logger.info(results['confusion_matrix'])


def main():
    logger = get_logger('evaluate_all')
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"使用设备: {device}")

    # 加载模型
    model_path = Path(config.paths['output_dir']) / 'models' / 'best_model.pth'
    logger.info(f"加载模型: {model_path}")

    # 创建输出目录
    output_dir = Path(config.paths['output_dir']) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 存储所有结果
    all_results = {}

    # 评估三个数据集
    for dataset_type in ['train', 'val', 'test']:
        logger.info(f"\n{'='*80}")
        logger.info(f"加载{dataset_type}数据集...")
        logger.info(f"{'='*80}")

        if dataset_type == 'test':
            X_pp, y_pp, X_pe, y_pe = load_dataset(config, dataset_type)
            logger.info(f"PP测试集大小: {len(y_pp)} 样本")
            logger.info(f"PE测试集大小: {len(y_pe)} 样本")
        else:
            X, y_pp, _, y_pe = load_dataset(config, dataset_type)
            X_pp = X
            X_pe = X
            logger.info(f"数据集大小: {len(y_pp)} 样本")

        # 加载模型
        model = DualHeadRamanCNNLSTM(input_len=X_pp.shape[1], num_classes=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # 最终阈值配置
        # PP任务：轻微污染+0.15
        # PE任务：轻微污染+0.50, 严重污染+0.70
        pp_threshold_class1 = 0.15
        pp_threshold_class2 = 0.0
        pe_threshold_class1 = 0.50
        pe_threshold_class2 = 0.70

        # 评估PP任务（应用阈值调整）
        pp_results = evaluate_task(model, X_pp, y_pp, device, 'PP',
                                   pp_threshold_class1, pp_threshold_class2)
        print_results(pp_results, dataset_type.upper(), 'PP', logger)

        # 评估PE任务（应用阈值调整）
        pe_results = evaluate_task(model, X_pe, y_pe, device, 'PE',
                                   pe_threshold_class1, pe_threshold_class2)
        print_results(pe_results, dataset_type.upper(), 'PE', logger)

        # 保存结果
        all_results[dataset_type] = {
            'PP': pp_results,
            'PE': pe_results
        }

    # 生成汇总表格
    logger.info(f"\n{'='*80}")
    logger.info("所有数据集性能汇总")
    logger.info(f"{'='*80}")

    # PP任务汇总
    logger.info(f"\nPP任务汇总:")
    logger.info(f"{'数据集':<15} {'准确率':<12} {'精确率(M)':<12} {'召回率(M)':<12} {'F1分数(M)':<12}")
    logger.info("-" * 80)
    for dataset_type in ['train', 'val', 'test']:
        r = all_results[dataset_type]['PP']
        logger.info(f"{dataset_type.upper():<15} {r['accuracy']:<12.4f} "
                   f"{r['precision_macro']:<12.4f} {r['recall_macro']:<12.4f} "
                   f"{r['f1_macro']:<12.4f}")

    # PE任务汇总
    logger.info(f"\nPE任务汇总:")
    logger.info(f"{'数据集':<15} {'准确率':<12} {'精确率(M)':<12} {'召回率(M)':<12} {'F1分数(M)':<12}")
    logger.info("-" * 80)
    for dataset_type in ['train', 'val', 'test']:
        r = all_results[dataset_type]['PE']
        logger.info(f"{dataset_type.upper():<15} {r['accuracy']:<12.4f} "
                   f"{r['precision_macro']:<12.4f} {r['recall_macro']:<12.4f} "
                   f"{r['f1_macro']:<12.4f}")

    # 各类别详细汇总
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    for task in ['PP', 'PE']:
        logger.info(f"\n{task}任务 - 各类别召回率对比:")
        logger.info(f"{'数据集':<15} {class_names[0]:<20} {class_names[1]:<20} {class_names[2]:<20}")
        logger.info("-" * 80)
        for dataset_type in ['train', 'val', 'test']:
            r = all_results[dataset_type][task]
            logger.info(f"{dataset_type.upper():<15} "
                       f"{r['recall_per_class'][0]:<20.4f} "
                       f"{r['recall_per_class'][1]:<20.4f} "
                       f"{r['recall_per_class'][2]:<20.4f}")

    # 生成可视化图表
    logger.info("\n>>> 生成可视化图表...")
    plot_confusion_matrices(all_results, output_dir, class_names)
    plot_metrics_comparison(all_results, output_dir)
    plot_class_distribution(all_results, output_dir, class_names)
    plot_performance_radar(all_results, output_dir, class_names)

    # 保存结果到文件
    report_path = output_dir / 'all_datasets_evaluation.txt'
    logger.info(f"\n评估报告已保存到: {report_path}")
    logger.info(f"可视化图表已保存到: {output_dir}")


def plot_confusion_matrices(all_results, output_dir, class_names):
    """绘制混淆矩阵（测试集）- 显示数量和百分比"""
    logger = get_logger('evaluate_all')

    for task in ['PP', 'PE']:
        test_results = all_results['test'][task]
        cm = test_results['confusion_matrix']

        # 计算每行的百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

        # 创建自定义标注：数量 + 百分比
        labels = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                labels[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'{task} Task - Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = output_dir / f'confusion_matrix_{task}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  保存: {save_path}")


def plot_metrics_comparison(all_results, output_dir):
    """绘制指标对比图（所有数据集）- 分组柱状图样式"""
    logger = get_logger('evaluate_all')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    datasets = ['train', 'val', 'test']
    x = np.arange(len(datasets))
    width = 0.25

    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']

    # 配色方案：蓝色和绿色
    pp_color = '#3498db'  # 蓝色
    pe_color = '#2ecc71'  # 绿色

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        pp_values = [all_results[ds]['PP'][metric] for ds in datasets]
        pe_values = [all_results[ds]['PE'][metric] for ds in datasets]

        bars1 = ax.bar(x - width/2, pp_values, width, label='PP Classifier', color=pp_color, edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, pe_values, width, label='PE Classifier', color=pe_color, edgecolor='white', linewidth=1.5)

        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([ds.capitalize() for ds in datasets], fontsize=10)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Performance Metrics Comparison Across Datasets', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_path = output_dir / 'metrics_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  保存: {save_path}")


def plot_class_distribution(all_results, output_dir, class_names):
    """绘制类别分布图（测试集召回率）- 绿黄红配色"""
    logger = get_logger('evaluate_all')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, task in enumerate(['PP', 'PE']):
        ax = axes[idx]
        test_results = all_results['test'][task]
        recall_per_class = test_results['recall_per_class']

        # 配色：绿、黄、红
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # 绿色、黄色、红色
        bars = ax.bar(class_names, recall_per_class, color=colors, edgecolor='white', linewidth=1.5)

        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title(f'{task} Task - Per-Class Recall (Test Set)', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

        # 添加数值标签
        for bar, val in zip(bars, recall_per_class):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = output_dir / 'class_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  保存: {save_path}")


def plot_performance_radar(all_results, output_dir, class_names):
    """绘制性能雷达图（测试集）- PP和PE任务"""
    logger = get_logger('evaluate_all')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))

    # 指标：每个类别的召回率 + 宏平均指标
    categories = ['Non-pollution\nRecall', 'Slight\nRecall', 'Severe\nRecall',
                  'Precision\n(Macro)', 'F1-Score\n(Macro)']
    N = len(categories)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for idx, task in enumerate(['PP', 'PE']):
        ax = axes[idx]
        test_results = all_results['test'][task]

        # 提取数据
        recall_per_class = test_results['recall_per_class'].tolist()
        precision_macro = test_results['precision_macro']
        f1_macro = test_results['f1_macro']

        values = recall_per_class + [precision_macro, f1_macro]
        values += values[:1]  # 闭合

        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db', label=f'{task} Task')
        ax.fill(angles, values, alpha=0.25, color='#3498db')

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_title(f'{task} Task - Performance Radar Chart', fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = output_dir / 'performance_radar.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  保存: {save_path}")


if __name__ == '__main__':
    main()
