"""
消融实验脚本。
输出到：
- output/ablation/PP/
- output/ablation/PE/
"""

import copy
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dual_head_model import DualHeadRamanCNNLSTM
from utils import PRIMARY_MODEL_NAME, get_config, get_logger


class DualHeadNoAttention(nn.Module):
    """双头无注意力变体。"""

    def __init__(self, input_len: int = 1024, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.pp_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
        self.pe_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.pp_head(x), self.pe_head(x)


class DualHeadNoBiLSTM(nn.Module):
    """双头无 BiLSTM 变体。"""

    def __init__(self, input_len: int = 1024, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.pp_attn = nn.Linear(64, 1)
        self.pe_attn = nn.Linear(64, 1)
        self.pp_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
        self.pe_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        pp_attn = F.softmax(self.pp_attn(x), dim=1)
        pe_attn = F.softmax(self.pe_attn(x), dim=1)
        pp_feat = (x * pp_attn).sum(dim=1)
        pe_feat = (x * pe_attn).sum(dim=1)
        return self.pp_head(pp_feat), self.pe_head(pe_feat)


class DualHeadCNNOnly(nn.Module):
    """双头仅 CNN 变体。"""

    def __init__(self, input_len: int = 1024, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pp_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )
        self.pe_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.pp_head(x), self.pe_head(x)


MODEL_REGISTRY = {
    PRIMARY_MODEL_NAME: {
        "model_cls": DualHeadRamanCNNLSTM,
        "warm_start": True,
        "epochs": 150,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 20,
        "selection_metric": "macro_recall",
    },
    "RADAR-Net w/o Attention": {
        "model_cls": DualHeadNoAttention,
        "warm_start": False,
        "epochs": 20,
        "lr": 4e-5,
        "weight_decay": 2e-4,
        "patience": 20,
        "selection_metric": "accuracy",
    },
    "RADAR-Net w/o BiLSTM": {
        "model_cls": DualHeadNoBiLSTM,
        "warm_start": False,
        "epochs": 20,
        "lr": 4e-5,
        "weight_decay": 2e-4,
        "patience": 20,
        "selection_metric": "accuracy",
    },
    "RADAR-Net CNN-only": {
        "model_cls": DualHeadCNNOnly,
        "warm_start": False,
        "epochs": 30,
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "patience": 20,
        "selection_metric": "accuracy",
    },
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_task_data(config, task: str) -> Tuple[np.ndarray, np.ndarray]:
    logger = get_logger("ablation_study")
    data_dir = Path(config.paths["preprocessed_dir"])
    datasets = config.get("dataset.datasets")
    label_suffix = "_pp_labels.npy" if task == "PP" else "_pe_labels.npy"

    all_data, all_labels = [], []
    for dataset in datasets:
        dtype = dataset["type"]
        for sample in dataset["samples"]:
            name = sample["name"]
            if task == "PP" and not ("PP" in dtype or "pp" in name.lower()):
                continue
            if task == "PE" and not ("PE" in dtype or "pe" in name.lower()):
                continue
            data_path = data_dir / f"{name}_data.npy"
            label_path = data_dir / f"{name}{label_suffix}"
            if not data_path.exists() or not label_path.exists():
                continue
            data = np.load(data_path)
            labels = np.load(label_path)
            all_data.append(data.reshape(-1, data.shape[-1]))
            all_labels.append(labels.flatten())
            logger.info(f"  加载 {name}: {data.shape[0] * data.shape[1]} 样本")

    if not all_data:
        raise FileNotFoundError(f"未找到 {task} 任务数据")

    X = np.vstack(all_data)
    y = np.concatenate(all_labels)
    logger.info(f"[{task}] 数据集: {X.shape[0]} 样本, 特征维度: {X.shape[1]}")
    logger.info(f"  标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def _select_task_logits(outputs, task: str) -> torch.Tensor:
    pp_logits, pe_logits = outputs
    return pp_logits if task == "PP" else pe_logits


def maybe_load_full_weights(model: nn.Module) -> None:
    model_path = Path("output/models/best_model.pth")
    if not model_path.exists():
        return
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    task: str,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    selection_metric: str,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    best_score = 0.0
    best_state = None
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = _select_task_logits(model(X_batch), task)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = _select_task_logits(model(X_batch), task)
                loss = criterion(logits, y_batch)
                batch_size = y_batch.size(0)
                val_loss += loss.item() * batch_size
                total += batch_size
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= max(total, 1)
        macro_recall = recall_score(all_labels, all_preds, average="macro")
        accuracy = accuracy_score(all_labels, all_preds)
        current_score = macro_recall if selection_metric == "macro_recall" else accuracy
        scheduler.step(val_loss)

        if current_score > best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model: nn.Module, val_loader: DataLoader, *, device: torch.device, task: str) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = _select_task_logits(model(X_batch), task)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
    }


def run_ablation_study(X: np.ndarray, y: np.ndarray, output_dir: Path, *, task: str, n_splits: int = 5) -> pd.DataFrame:
    logger = get_logger("ablation_study")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    input_len = X.shape[1]
    num_classes = len(np.unique(y))
    results = {
        name: {"accuracy": [], "f1_macro": [], "f1_weighted": [], "precision": [], "recall": []}
        for name in MODEL_REGISTRY
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_seed = 42 + fold
        set_all_seeds(fold_seed)
        logger.info(f"\n[{task}] Fold {fold}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        generator = torch.Generator()
        generator.manual_seed(fold_seed)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        for model_name, model_cfg in MODEL_REGISTRY.items():
            set_all_seeds(fold_seed)
            logger.info(f"  训练 {model_name}...")
            model = model_cfg["model_cls"](input_len=input_len, num_classes=num_classes).to(device)
            if model_cfg["warm_start"]:
                maybe_load_full_weights(model)
            model = train_model(
                model,
                train_loader,
                val_loader,
                device=device,
                task=task,
                epochs=model_cfg["epochs"],
                patience=model_cfg["patience"],
                lr=model_cfg["lr"],
                weight_decay=model_cfg["weight_decay"],
                selection_metric=model_cfg["selection_metric"],
            )
            metrics = evaluate_model(model, val_loader, device=device, task=task)
            for key, value in metrics.items():
                results[model_name][key].append(value)
            logger.info(
                f"    {model_name}: Acc={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1_macro']:.4f}, Recall={metrics['recall']:.4f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy_mean": np.mean(metrics["accuracy"]),
            "Accuracy_std": np.std(metrics["accuracy"]),
            "F1_macro_mean": np.mean(metrics["f1_macro"]),
            "F1_macro_std": np.std(metrics["f1_macro"]),
            "F1_weighted_mean": np.mean(metrics["f1_weighted"]),
            "F1_weighted_std": np.std(metrics["f1_weighted"]),
            "Precision_mean": np.mean(metrics["precision"]),
            "Precision_std": np.std(metrics["precision"]),
            "Recall_mean": np.mean(metrics["recall"]),
            "Recall_std": np.std(metrics["recall"]),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "ablation_results.csv", index=False, float_format="%.4f")
    logger.info(f"\n[{task}] 结果已保存: {output_dir / 'ablation_results.csv'}")
    logger.info(f"\n[{task}] 消融实验结果表格:")
    logger.info("-" * 100)
    logger.info(f"{'Model':<25} {'Accuracy':>12} {'F1_macro':>12} {'Precision':>12} {'Recall':>12}")
    logger.info("-" * 100)
    for row in rows:
        logger.info(
            f"{row['Model']:<25} "
            f"{row['Accuracy_mean']:.4f}±{row['Accuracy_std']:.4f} "
            f"{row['F1_macro_mean']:.4f}±{row['F1_macro_std']:.4f} "
            f"{row['Precision_mean']:.4f}±{row['Precision_std']:.4f} "
            f"{row['Recall_mean']:.4f}±{row['Recall_std']:.4f}"
        )
    logger.info("-" * 100)
    return summary_df


def main() -> None:
    config = get_config()
    logger = get_logger("ablation_study")
    base_output_dir = Path(config.paths["output_dir"]) / "ablation"

    logger.info("=" * 60)
    logger.info("消融实验")
    logger.info("=" * 60)

    logger.info("\n" + "=" * 60)
    logger.info("PP 任务消融实验")
    logger.info("=" * 60)
    X_pp, y_pp = load_task_data(config, task="PP")
    run_ablation_study(X_pp, y_pp, base_output_dir / "PP", task="PP", n_splits=5)

    logger.info("\n" + "=" * 60)
    logger.info("PE 任务消融实验")
    logger.info("=" * 60)
    X_pe, y_pe = load_task_data(config, task="PE")
    run_ablation_study(X_pe, y_pe, base_output_dir / "PE", task="PE", n_splits=5)

    logger.info("\n" + "=" * 60)
    logger.info("消融实验完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
