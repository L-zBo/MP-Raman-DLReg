"""
泛化实验主脚本。

功能：
- 评估 RADAR-Net 的跨基质泛化能力
- 输出 generalization 结果

输出：
- output/generalization/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import json
from typing import Dict, Tuple
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from utils import get_config, get_logger, PRIMARY_MODEL_NAME
from preprocessing.preprocess import (
    load_hyperspectral_data,
    preprocess_hyperspectral,
    preprocess_spectrum,
    load_csv_spectrum
)
from unmixing.unmix import unmix_hyperspectral, postprocess_abundance
from models.dual_head_model import DualHeadRamanCNNLSTM


def load_unified_endmembers(config, dataset_dir: Path) -> np.ndarray:
    """
    加载统一端元矩阵（原始玉米淀粉）

    使用与训练集相同的端元进行 NNLS 分解，保证泛化数据的丰度分布与训练集一致。
    """
    logger = get_logger('generalization')
    endmember_config = config.get('dataset.endmembers')
    gen_config = config.get('generalization')

    starch_path = dataset_dir / endmember_config['starch']['folder'] / endmember_config['starch']['file']
    _, starch_spec = load_csv_spectrum(starch_path)
    starch_processed = preprocess_spectrum(starch_spec)
    logger.info(f"  统一端元 - 淀粉: {starch_path.name}")

    shared = gen_config.get('shared_endmembers')
    pp_path = dataset_dir / shared['pp']['folder'] / shared['pp']['file']
    _, pp_spec = load_csv_spectrum(pp_path)
    pp_processed = preprocess_spectrum(pp_spec)
    logger.info(f"  统一端元 - PP: {pp_path.name}")

    pe_path = dataset_dir / shared['pe']['folder'] / shared['pe']['file']
    _, pe_spec = load_csv_spectrum(pe_path)
    pe_processed = preprocess_spectrum(pe_spec)
    logger.info(f"  统一端元 - PE: {pe_path.name}")

    return np.vstack([starch_processed, pp_processed, pe_processed])


def load_abundance_thresholds(task: str) -> list:
    """加载训练集固化的丰度阈值。"""
    path = Path(f'output/models/abundance_thresholds/{task.lower()}_thresholds.json')
    if not path.exists():
        raise FileNotFoundError(f"丰度阈值文件不存在: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return data['thresholds']


def label_with_thresholds(target_abundance: np.ndarray, task: str) -> np.ndarray:
    """按训练集阈值将丰度切成三级标签 (0=无, 1=轻, 2=重)。"""
    thresholds = load_abundance_thresholds(task)
    labels = np.zeros(len(target_abundance), dtype=int)
    labels[target_abundance >= thresholds[0]] = 1
    labels[target_abundance >= thresholds[1]] = 2
    return labels


def process_generalization_sample(
    sample_path: Path,
    endmembers: np.ndarray,
    task: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """处理单个泛化样本：预处理 → NNLS 解混 → 丰度阈值标签。"""
    logger = get_logger('generalization')

    hypercube, wavenumbers = load_hyperspectral_data(sample_path)
    logger.info(f"    加载数据: {hypercube.shape}")

    hypercube_processed = preprocess_hyperspectral(hypercube, n_jobs=4)

    abundance = unmix_hyperspectral(hypercube_processed, endmembers, n_jobs=4)
    for i in range(3):
        abundance[:, :, i] = postprocess_abundance(abundance[:, :, i])

    if task == 'PP':
        target_abundance = abundance[:, :, 1].flatten()
    else:
        target_abundance = abundance[:, :, 2].flatten()

    labels = label_with_thresholds(target_abundance, task)

    h, w, bands = hypercube_processed.shape
    spectra_flat = hypercube_processed.reshape(-1, bands)
    abundance_flat = abundance.reshape(-1, 3)

    return spectra_flat, abundance_flat, labels


def load_model(config, device: torch.device) -> DualHeadRamanCNNLSTM:
    """加载训练好的 RADAR-Net 模型权重。"""
    logger = get_logger('generalization')
    models_dir = Path(config.paths['models_dir'])
    model_path = models_dir / 'best_model.pth'

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = DualHeadRamanCNNLSTM(
        input_len=config.get('model.architecture.input_len', 1024),
        num_classes=config.get('model.architecture.num_classes', 3)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  模型参数: input_len={config.get('model.architecture.input_len', 1024)}, "
                f"num_classes={config.get('model.architecture.num_classes', 3)}")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  模型权重: {model_path.name}")

    return model


def predict(
    model: DualHeadRamanCNNLSTM,
    spectra: np.ndarray,
    task: str,
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:
    """使用 RADAR-Net 推理。"""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(spectra), batch_size):
            batch = spectra[i:i+batch_size]
            X = torch.FloatTensor(batch).to(device)
            pp_out, pe_out = model(X)

            if task == 'PP':
                preds = pp_out.argmax(dim=1).cpu().numpy()
            else:
                preds = pe_out.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)

    return np.concatenate(all_preds)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算评估指标。"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }


def run_generalization_experiment():
    """
    运行泛化实验 - Combo 方案。

    方法：统一端元 + 训练集固化丰度阈值。
    """
    config = get_config()
    logger = get_logger('generalization')

    logger.info("=" * 70)
    logger.info(f"泛化实验 - {PRIMARY_MODEL_NAME} (Combo方案)")
    logger.info("=" * 70)
    logger.info("方法: 统一端元 + 训练集固化丰度阈值")
    logger.info("  - 统一端元: 原始玉米淀粉，保证丰度分布与训练集一致")
    logger.info("  - 固化阈值: 避免在泛化数据上重新估计分布")

    dataset_dir = Path(config.paths['dataset_dir'])
    gen_config = config.get('generalization')
    output_dir = Path(gen_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n使用设备: {device}")

    logger.info("\n>>> 加载统一端元...")
    endmembers = load_unified_endmembers(config, dataset_dir)
    logger.info(f"  端元矩阵形状: {endmembers.shape}")

    logger.info("\n>>> 加载训练集丰度阈值...")
    pp_thresholds = load_abundance_thresholds('PP')
    pe_thresholds = load_abundance_thresholds('PE')
    logger.info(f"  PP阈值: {pp_thresholds}")
    logger.info(f"  PE阈值: {pe_thresholds}")

    logger.info(f"\n>>> 加载 {PRIMARY_MODEL_NAME} 模型...")
    model = load_model(config, device)

    all_results = {}

    for brand_config in gen_config['datasets']:
        brand_name = brand_config['name']
        brand_display = brand_config['brand']
        starch_type = brand_config['starch_type']

        logger.info(f"\n{'='*60}")
        logger.info(f"测试品牌: {brand_display} ({starch_type})")
        logger.info(f"{'='*60}")

        all_results[brand_name] = {
            'brand': brand_display,
            'starch_type': starch_type
        }

        brand_output_dir = output_dir / brand_name
        (brand_output_dir / 'preprocessed').mkdir(parents=True, exist_ok=True)
        (brand_output_dir / 'abundance').mkdir(parents=True, exist_ok=True)
        (brand_output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        (brand_output_dir / 'predictions').mkdir(parents=True, exist_ok=True)

        for task in ['PP', 'PE']:
            sample_folder = brand_config['samples'].get(task)
            if not sample_folder:
                logger.warning(f"  {task} 样本未配置，跳过")
                continue

            sample_path = dataset_dir / brand_config['base_path'] / sample_folder
            if not sample_path.exists():
                logger.warning(f"  {task} 样本路径不存在: {sample_path}")
                continue

            logger.info(f"\n  >>> {task} 任务")

            spectra, abundance, labels = process_generalization_sample(
                sample_path, endmembers, task
            )

            np.save(brand_output_dir / 'preprocessed' / f'{task}_spectra.npy', spectra)
            np.save(brand_output_dir / 'abundance' / f'{task}_abundance.npy', abundance)
            np.save(brand_output_dir / 'labels' / f'{task}_labels.npy', labels)

            unique, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique.tolist(), counts.tolist()))
            logger.info(f"      标签分布: {label_dist}")

            logger.info(f"      运行推理...")
            predictions = predict(model, spectra, task, device)

            np.save(brand_output_dir / 'predictions' / f'{task}_predictions.npy', predictions)

            metrics = calculate_metrics(labels, predictions)
            all_results[brand_name][task] = metrics

            cm = confusion_matrix(labels, predictions)

            logger.info(f"      ----------------------------------------")
            logger.info(f"      Accuracy:          {metrics['accuracy']:.4f}")
            logger.info(f"      Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            logger.info(f"      F1 (Macro):        {metrics['f1_macro']:.4f}")
            logger.info(f"      Precision (Macro): {metrics['precision_macro']:.4f}")
            logger.info(f"      Recall (Macro):    {metrics['recall_macro']:.4f}")
            logger.info(f"      混淆矩阵:")
            for row in cm:
                logger.info(f"        {row}")

    results_dir = output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / 'generalization_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    generate_summary_csv(all_results, results_dir)
    print_summary(all_results, logger)

    logger.info(f"\n{'='*70}")
    logger.info("泛化实验完成！(Combo方案: 统一端元 + 训练集固化丰度阈值)")
    logger.info(f"结果保存至: {results_dir}")
    logger.info(f"{'='*70}")

    return all_results


def generate_summary_csv(results: Dict, output_dir: Path):
    """生成 CSV 汇总表。"""
    rows = []

    for brand_name, brand_data in results.items():
        if 'brand' not in brand_data:
            continue

        for task in ['PP', 'PE']:
            if task not in brand_data:
                continue

            metrics = brand_data[task]
            rows.append({
                'Brand': brand_data['brand'],
                'Starch_Type': brand_data['starch_type'],
                'Task': task,
                'Accuracy': metrics['accuracy'],
                'Balanced_Accuracy': metrics['balanced_accuracy'],
                'F1_Macro': metrics['f1_macro'],
                'Precision_Macro': metrics['precision_macro'],
                'Recall_Macro': metrics['recall_macro']
            })

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(['Starch_Type', 'Brand', 'Task'])
        df.to_csv(output_dir / 'generalization_summary.csv', index=False)

        task_summary = df.groupby('Task').agg({
            'Accuracy': ['mean', 'std'],
            'Balanced_Accuracy': ['mean', 'std'],
            'F1_Macro': ['mean', 'std']
        }).round(4)
        task_summary.to_csv(output_dir / 'generalization_by_task.csv')


def print_summary(results: Dict, logger):
    """打印泛化实验总结。"""
    logger.info(f"\n{'='*70}")
    logger.info("泛化实验结果总结 (Combo方案)")
    logger.info(f"{'='*70}")

    all_balanced_acc = []
    all_f1 = []

    for brand_name, brand_data in results.items():
        if 'brand' not in brand_data:
            continue

        brand = brand_data['brand']
        starch = brand_data['starch_type']

        logger.info(f"\n{brand} ({starch}):")

        for task in ['PP', 'PE']:
            if task not in brand_data:
                continue
            m = brand_data[task]
            logger.info(f"  {task}: Acc={m['accuracy']:.4f}, "
                       f"BalAcc={m['balanced_accuracy']:.4f}, "
                       f"F1={m['f1_macro']:.4f}")
            all_balanced_acc.append(m['balanced_accuracy'])
            all_f1.append(m['f1_macro'])

    if all_balanced_acc:
        logger.info(f"\n整体平均:")
        logger.info(f"  Balanced Accuracy: {np.mean(all_balanced_acc):.4f} ± {np.std(all_balanced_acc):.4f}")
        logger.info(f"  F1 Macro:          {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")


if __name__ == '__main__':
    run_generalization_experiment()
