"""
最终对比实验主入口。

功能：
- 评估最终模型池
- 输出 comparison_final 结果

输出：
- output/comparison_final/PP/comparison_results.csv
- output/comparison_final/PE/comparison_results.csv
- output/comparison_final/final_comparison_results.csv
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import random
from pathlib import Path
from typing import Dict, Tuple, Callable
import warnings

warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    get_config,
    get_logger,
    PRIMARY_MODEL_NAME,
    BEST_CANDIDATE_CONFIGS,
    FINAL_RESNET_CHOICE,
    FINAL_RESNET_CONFIG,
    FINAL_COMPARISON_MODEL_ORDER,
)
from models import SMARTNIRClassifier, ConvTranClassifier, MambaHSIClassifier, ResNet50_1D
from models.dual_head_model import DualHeadRamanCNNLSTM


RANDOM_SEED = 42

PP_THRESHOLD_CLASS1 = 0.15
PP_THRESHOLD_CLASS2 = 0.0
PE_THRESHOLD_CLASS1 = 0.50
PE_THRESHOLD_CLASS2 = 0.70


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_pp_mixedaware_dataset(
    X_pp: np.ndarray,
    y_pp: np.ndarray,
    X_pe: np.ndarray,
    neg_ratio: float,
    seed_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if neg_ratio <= 0:
        return X_pp, y_pp

    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    sample_size = min(len(X_pe), max(1, int(round(len(X_pp) * neg_ratio))))
    indices = rng.choice(len(X_pe), size=sample_size, replace=False)
    X_neg = X_pe[indices]
    y_neg = np.zeros(sample_size, dtype=y_pp.dtype)
    return np.vstack([X_pp, X_neg]), np.concatenate([y_pp, y_neg])


def _load_split_data(config, split_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger = get_logger('comparison')
    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    datasets = config.get('dataset.datasets')

    pp_data, pp_labels = [], []
    pe_data, pe_labels = [], []

    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']
        split = config.get('training.split', {}).get(dataset_type, {'train': [1, 2, 3, 4], 'val': [5]})

        for i, sample in enumerate(samples, 1):
            if i not in split[split_key]:
                continue

            name = sample['name']
            data_path = preprocessed_dir / f'{name}_data.npy'
            if not data_path.exists():
                continue

            data = np.load(data_path).reshape(-1, 1024)

            if 'PP' in dataset_type or 'pp' in name:
                label_path = preprocessed_dir / f'{name}_pp_labels.npy'
                if label_path.exists():
                    pp_data.append(data)
                    pp_labels.append(np.load(label_path).flatten())

            if 'PE' in dataset_type or 'pe' in name:
                label_path = preprocessed_dir / f'{name}_pe_labels.npy'
                if label_path.exists():
                    pe_data.append(data)
                    pe_labels.append(np.load(label_path).flatten())

    X_pp = np.vstack(pp_data)
    y_pp = np.concatenate(pp_labels)
    X_pe = np.vstack(pe_data)
    y_pe = np.concatenate(pe_labels)

    logger.info(f"[{split_key}] PP: {X_pp.shape[0]} 样本 | PE: {X_pe.shape[0]} 样本")
    return X_pp, y_pp, X_pe, y_pe


def load_pp_pe_separate_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_split_data(config, 'train')


def load_pp_pe_separate_val_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_split_data(config, 'val')


def load_pp_pe_separate_test_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger = get_logger('comparison')
    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    true_label_dir = Path(__file__).parent.parent / 'testing' / 'test_true_label'

    label_paths = {
        'pp': true_label_dir / 'pp_test' / 'pp_labels.npy',
        'pe': true_label_dir / 'pe_test' / 'pe_labels.npy',
        'pp_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy',
        'pe_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy',
    }

    pp_data = np.load(preprocessed_dir / 'pp_test_data.npy').reshape(-1, 1024)
    pe_data = np.load(preprocessed_dir / 'pe_test_data.npy').reshape(-1, 1024)
    mixed_data = np.load(preprocessed_dir / 'pp_pe_mixed_test_data.npy').reshape(-1, 1024)

    y_pp = np.concatenate([
        np.load(label_paths['pp']).flatten(),
        np.load(label_paths['pp_mixed']).flatten(),
    ])
    y_pe = np.concatenate([
        np.load(label_paths['pe']).flatten(),
        np.load(label_paths['pe_mixed']).flatten(),
    ])

    X_pp = np.vstack([pp_data, mixed_data])
    X_pe = np.vstack([pe_data, mixed_data])

    logger.info(f"[test] PP: {X_pp.shape[0]} 样本 | PE: {X_pe.shape[0]} 样本")
    return X_pp, y_pp, X_pe, y_pe


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.max()
    return torch.tensor(weights, dtype=torch.float32)


def create_sample_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.max()
    return torch.tensor(weights[labels], dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'accuracy_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'y_pred': y_pred,
    }


def sanitize_model_name(model_name: str) -> str:
    return model_name.lower().replace(' ', '_').replace('-', '_')


def get_traditional_factories() -> Dict[str, Callable[[], object]]:
    return {
        'SVM': lambda: SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_SEED),
        'Random Forest': lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        'XGBoost': lambda: XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='mlogloss',
        ) if HAS_XGBOOST else None,
        'LightGBM': lambda: LGBMClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.04,
            num_leaves=12,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ) if HAS_LIGHTGBM else None,
    }


def evaluate_traditional_models(
    *,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_dir: Path,
) -> Dict[str, Dict[str, object]]:
    logger = get_logger('comparison')
    factories = get_traditional_factories()
    task_suffix = task.lower()

    scaler_path = model_dir / f'scaler_{task_suffix}.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler().fit(X_train)
        joblib.dump(scaler, scaler_path)
        logger.info(f"保存传统模型共享 Scaler: {scaler_path}")

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: Dict[str, Dict[str, object]] = {}
    for model_name in ['XGBoost', 'SVM', 'Random Forest', 'LightGBM']:
        factory = factories[model_name]
        model = None if factory is None else factory()
        model_path = model_dir / f"{sanitize_model_name(model_name)}_{task_suffix}.pkl"

        if model_path.exists():
            try:
                model = joblib.load(model_path)
                logger.info(f"加载 {model_name}: {model_path.name}")
            except Exception as exc:
                logger.warning(f"加载 {model_name} 失败，将重新训练: {exc}")
                model = None

        if model is None:
            if factory is None:
                logger.warning(f"{model_name} 当前环境不可用，跳过")
                continue
            logger.info(f"训练 {model_name}...")
            model = factory()
            model.fit(X_train_scaled, y_train)
            joblib.dump(model, model_path)
            logger.info(f"保存 {model_name}: {model_path.name}")

        y_pred = model.predict(X_test_scaled)
        results[model_name] = compute_metrics(y_test, y_pred)

    return results


def get_candidate_spec(model_name: str, task: str, input_len: int) -> Dict[str, object]:
    model_dir = Path(get_config().base_dir) / 'output' / 'models' / 'traditional'

    if model_name == 'SMART-NIR':
        key = 'pp_best' if task == 'PP' else 'pe_best'
        base = BEST_CANDIDATE_CONFIGS['SMART-NIR'][key]
        checkpoint = 'smart_nir_pp_final.pth' if task == 'PP' else 'smart_nir_pe_final.pth'
        scaler = 'scaler_smart_nir_pp_final.pkl' if task == 'PP' else 'scaler_smart_nir_pe_final.pkl'
        model = SMARTNIRClassifier(input_len=input_len, num_classes=3, **base['model_kwargs'])
    elif model_name == 'ConvTran':
        key = 'pp_best' if task == 'PP' else 'pe_best'
        base = BEST_CANDIDATE_CONFIGS['ConvTran'][key]
        checkpoint = 'convtran_pp_final.pth' if task == 'PP' else 'convtran_pe_final.pth'
        scaler = 'scaler_convtran_pp_final.pkl' if task == 'PP' else 'scaler_convtran_pe_final.pkl'
        model = ConvTranClassifier(input_len=input_len, num_classes=3, **base['model_kwargs'])
    elif model_name == 'MambaHSI':
        key = 'pp_best' if task == 'PP' else 'pe_best'
        base = BEST_CANDIDATE_CONFIGS['MambaHSI'][key]
        checkpoint = 'mambahsi_pp_final.pth' if task == 'PP' else 'mambahsi_pe_final.pth'
        scaler = 'scaler_mambahsi_pp_final.pkl' if task == 'PP' else 'scaler_mambahsi_pe_final.pkl'
        model = MambaHSIClassifier(input_len=input_len, num_classes=3, **base['model_kwargs'])
    elif model_name == FINAL_RESNET_CHOICE:
        key = 'pp_best' if task == 'PP' else 'pe_best'
        base = FINAL_RESNET_CONFIG[key]
        checkpoint = 'resnet50_pp_final.pth' if task == 'PP' else 'resnet50_pe_final.pth'
        scaler = 'scaler_resnet50_pp_final.pkl' if task == 'PP' else 'scaler_resnet50_pe_final.pkl'
        model = ResNet50_1D(input_len=input_len, num_classes=3, **base['model_kwargs'])
    else:
        raise ValueError(f'未知模型: {model_name}')

    return {
        'model': model,
        'checkpoint_path': model_dir / checkpoint,
        'scaler_path': model_dir / scaler,
        'train_kwargs': dict(base.get('train_kwargs', {})),
    }


def train_single_head_classifier(
    *,
    model_name: str,
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_path: Path,
    checkpoint_path: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int = 64,
    patience: int = 8,
    label_smoothing: float = 0.0,
    use_weighted_sampler: bool = True,
) -> Dict[str, object]:
    logger = get_logger('comparison')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))

    if use_weighted_sampler:
        sampler = WeightedRandomSampler(create_sample_weights(y_train), len(y_train), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(y_train).to(device),
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_state = None
    best_val_f1 = -1.0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_pred.extend(preds)
                val_true.extend(y_batch.numpy())

        val_f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"{model_name} 早停于 epoch {epoch + 1}, best val F1={best_val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"保存 {model_name}: {checkpoint_path.name}")

    return evaluate_single_head_model(model, X_test, y_test, scaler, device)


def evaluate_single_head_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
) -> Dict[str, object]:
    X_test_scaled = scaler.transform(X_test)
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test_scaled).to(device))
        y_pred = logits.argmax(dim=1).cpu().numpy()
    return compute_metrics(y_test, y_pred)


def evaluate_or_train_candidate(
    *,
    model_name: str,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_other_train: np.ndarray | None,
    X_other_val: np.ndarray | None,
) -> Dict[str, object]:
    logger = get_logger('comparison')
    spec = get_candidate_spec(model_name, task, X_train.shape[1])
    checkpoint_path = spec['checkpoint_path']
    scaler_path = spec['scaler_path']
    train_kwargs = dict(spec['train_kwargs'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path.exists() and scaler_path.exists():
        model = spec['model'].to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        scaler = joblib.load(scaler_path)
        logger.info(f"加载 {model_name} [{task}]: {checkpoint_path.name}")
        return evaluate_single_head_model(model, X_test, y_test, scaler, device)

    logger.info(f"未找到 {model_name} [{task}] 最终权重，按最终参数训练...")

    if task == 'PP' and 'neg_ratio_train' in train_kwargs:
        X_train_final, y_train_final = build_pp_mixedaware_dataset(
            X_train, y_train, X_other_train, train_kwargs.pop('neg_ratio_train'), seed_offset=0
        )
        X_val_final, y_val_final = build_pp_mixedaware_dataset(
            X_val, y_val, X_other_val, train_kwargs.pop('neg_ratio_val'), seed_offset=100
        )
    else:
        train_kwargs.pop('neg_ratio_train', None)
        train_kwargs.pop('neg_ratio_val', None)
        X_train_final, y_train_final = X_train, y_train
        X_val_final, y_val_final = X_val, y_val

    return train_single_head_classifier(
        model_name=f'{model_name}[{task}]',
        model=spec['model'],
        X_train=X_train_final,
        y_train=y_train_final,
        X_val=X_val_final,
        y_val=y_val_final,
        X_test=X_test,
        y_test=y_test,
        scaler_path=scaler_path,
        checkpoint_path=checkpoint_path,
        **train_kwargs,
    )


def evaluate_radar_net(task: str, X_test: np.ndarray, y_test: np.ndarray, config) -> Dict[str, object]:
    logger = get_logger('comparison')
    model_path = Path(config.base_dir) / 'output' / 'models' / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'未找到 RADAR-Net 权重: {model_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualHeadRamanCNNLSTM(input_len=X_test.shape[1], num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        pp_logits, pe_logits = model(torch.FloatTensor(X_test).to(device))
        if task == 'PP':
            probs = torch.softmax(pp_logits, dim=1).cpu().numpy()
            probs[:, 1] += PP_THRESHOLD_CLASS1
            probs[:, 2] += PP_THRESHOLD_CLASS2
        else:
            probs = torch.softmax(pe_logits, dim=1).cpu().numpy()
            probs[:, 1] += PE_THRESHOLD_CLASS1
            probs[:, 2] += PE_THRESHOLD_CLASS2
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = probs.argmax(axis=1)

    logger.info(f"加载 {PRIMARY_MODEL_NAME} [{task}]: {model_path.name}")
    return compute_metrics(y_test, y_pred)


def plot_metrics_comparison(results: Dict[str, Dict[str, object]], output_dir: Path, task: str) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_models = [m for m in FINAL_COMPARISON_MODEL_ORDER if m in results]

    df = pd.DataFrame([
        {
            'Model': model_name,
            'Accuracy': results[model_name]['accuracy'],
            'Accuracy (Macro)': results[model_name]['accuracy_macro'],
            'F1-Score (Macro)': results[model_name]['f1_score'],
            'Recall (Macro)': results[model_name]['recall'],
            'Precision (Macro)': results[model_name]['precision'],
        }
        for model_name in ordered_models
    ])
    df.to_csv(output_dir / 'comparison_results.csv', index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ordered_models))
    width = 0.2
    metrics = [
        ('Accuracy (Macro)', '#74b9ff', -1.5 * width),
        ('F1-Score (Macro)', '#55efc4', -0.5 * width),
        ('Recall (Macro)', '#fab1a0', 0.5 * width),
        ('Precision (Macro)', '#DDA0DD', 1.5 * width),
    ]

    for metric_name, color, offset in metrics:
        values = df[metric_name].tolist()
        bars = ax.bar(x + offset, values, width, label=metric_name, color=color, edgecolor='none')
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax.set_ylabel('Score')
    ax.set_title(f'Final Comparison Metrics ({task} Task)')
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_models, rotation=20, ha='right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    return df


def main():
    config = get_config()
    logger = get_logger('comparison')
    set_seed()

    logger.info("=" * 70)
    logger.info("最终对比实验主入口")
    logger.info("模型池: RADAR-Net + 4传统模型 + 4个最终单头候选")
    logger.info("=" * 70)

    X_pp_train, y_pp_train, X_pe_train, y_pe_train = load_pp_pe_separate_data(config)
    X_pp_val, y_pp_val, X_pe_val, y_pe_val = load_pp_pe_separate_val_data(config)
    X_pp_test, y_pp_test, X_pe_test, y_pe_test = load_pp_pe_separate_test_data(config)

    model_dir = Path(config.base_dir) / 'output' / 'models' / 'traditional'
    task_bundles = {
        'PP': {
            'train': (X_pp_train, y_pp_train),
            'val': (X_pp_val, y_pp_val),
            'test': (X_pp_test, y_pp_test),
            'other_train': X_pe_train,
            'other_val': X_pe_val,
        },
        'PE': {
            'train': (X_pe_train, y_pe_train),
            'val': (X_pe_val, y_pe_val),
            'test': (X_pe_test, y_pe_test),
            'other_train': None,
            'other_val': None,
        },
    }

    combined_rows = []

    for task, bundle in task_bundles.items():
        logger.info("\n" + "=" * 70)
        logger.info(f">>> {task} 任务")
        logger.info("=" * 70)

        X_train, y_train = bundle['train']
        X_val, y_val = bundle['val']
        X_test, y_test = bundle['test']

        results = {}
        results.update(
            evaluate_traditional_models(
                task=task,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_dir=model_dir,
            )
        )

        results[PRIMARY_MODEL_NAME] = evaluate_radar_net(task, X_test, y_test, config)

        for model_name in [FINAL_RESNET_CHOICE, 'SMART-NIR', 'ConvTran', 'MambaHSI']:
            results[model_name] = evaluate_or_train_candidate(
                model_name=model_name,
                task=task,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                X_other_train=bundle['other_train'],
                X_other_val=bundle['other_val'],
            )

        output_dir = Path(config.base_dir) / 'output' / 'comparison_final' / task
        df = plot_metrics_comparison(results, output_dir, task)
        for _, row in df.iterrows():
            combined_rows.append({
                'Task': task,
                'Model': row['Model'],
                'Accuracy': row['Accuracy'],
                'Accuracy (Macro)': row['Accuracy (Macro)'],
                'F1-Score (Macro)': row['F1-Score (Macro)'],
                'Recall (Macro)': row['Recall (Macro)'],
                'Precision (Macro)': row['Precision (Macro)'],
            })
        logger.info("\n" + df.to_string(index=False))

    combined_df = pd.DataFrame(combined_rows)
    combined_output = Path(config.base_dir) / 'output' / 'comparison_final'
    combined_output.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(combined_output / 'final_comparison_results.csv', index=False)
    logger.info(f"\n总汇总已保存: {combined_output / 'final_comparison_results.csv'}")


if __name__ == '__main__':
    main()
