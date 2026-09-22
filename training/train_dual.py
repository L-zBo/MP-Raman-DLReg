"""
双头模型训练脚本。

功能：
- 训练 RADAR-Net
- 输出模型权重与训练历史

输出：
- output/models/best_model.pth
- output/models/final_two_stage_model.pth
- output/models/training_history.json
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import random
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_config, get_logger
from models.dual_head_model import DualHeadRamanCNNLSTM, DualHeadLoss, FocalLoss


def set_random_seed(seed: int) -> None:
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DualLabelDataset(Dataset):
    """双标签数据集"""

    def __init__(
        self,
        spectra: np.ndarray,
        pp_labels: np.ndarray,
        pe_labels: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.01
    ):
        self.spectra = torch.FloatTensor(spectra)
        self.pp_labels = torch.LongTensor(pp_labels)
        self.pe_labels = torch.LongTensor(pe_labels)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.pp_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.spectra[idx].clone()
        if self.augment and np.random.random() > 0.5:
            spec = spec + torch.randn_like(spec) * self.noise_std
        return spec, self.pp_labels[idx], self.pe_labels[idx]


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_weights: Optional[Dict] = None
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, output_dir: Path, keep_last_n: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoints: List[Path] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> Path:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        while len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)

        return checkpoint_path


def compute_class_weights(labels: np.ndarray, num_classes: int = 3, enhanced: bool = True) -> List[float]:
    """计算类别权重。"""
    counter = Counter(labels.flatten())
    total = sum(counter.values())

    if enhanced:
        beta = 0.9999
        weights = []
        for i in range(num_classes):
            n = counter.get(i, 1)
            effective_n = (1 - beta**n) / (1 - beta)
            weights.append(1.0 / effective_n)
    else:
        weights = []
        for i in range(num_classes):
            freq = counter.get(i, 1) / total
            weights.append(1.0 / (freq + 0.1))

    max_weight = max(weights)
    return [w / max_weight for w in weights]


def compute_sample_weights(pp_labels: np.ndarray, pe_labels: np.ndarray) -> np.ndarray:
    """计算采样权重。"""
    pp_counter = Counter(pp_labels.flatten())
    pp_total = sum(pp_counter.values())
    pp_class_weights = {}
    for label, count in pp_counter.items():
        freq = count / pp_total
        pp_class_weights[label] = 1.0 / np.sqrt(freq)
    pp_max = max(pp_class_weights.values())
    pp_class_weights = {k: v / pp_max for k, v in pp_class_weights.items()}

    pe_counter = Counter(pe_labels.flatten())
    pe_total = sum(pe_counter.values())
    pe_class_weights = {}
    for label, count in pe_counter.items():
        freq = count / pe_total
        pe_class_weights[label] = 1.0 / np.sqrt(freq)
    pe_max = max(pe_class_weights.values())
    pe_class_weights = {k: v / pe_max for k, v in pe_class_weights.items()}

    sample_weights = np.array([
        max(pp_class_weights[pp_label], pe_class_weights[pe_label])
        for pp_label, pe_label in zip(pp_labels, pe_labels)
    ])

    return sample_weights


def compute_epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算单个任务在一个 epoch 上的四个验证指标。"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def train_dual_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
    pp_train_labels: Optional[np.ndarray] = None,
    pe_train_labels: Optional[np.ndarray] = None,
    use_focal: bool = True,
    focal_gamma: float = 2.0,
    gamma_pp: Optional[float] = None,
    gamma_pe: Optional[float] = None,
    label_smoothing: float = 0.0,
    pp_task_weight: float = 1.0,
    pe_task_weight: float = 1.0,
    custom_pp_weights: Optional[List[float]] = None,
    custom_pe_weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """训练双头模型。"""
    logger = get_logger('train_dual')

    model = model.to(device)

    if custom_pp_weights is not None:
        pp_weights = custom_pp_weights
    else:
        pp_weights = compute_class_weights(pp_train_labels, enhanced=True) if pp_train_labels is not None else [1.0, 1.0, 1.0]

    if custom_pe_weights is not None:
        pe_weights = custom_pe_weights
    else:
        pe_weights = compute_class_weights(pe_train_labels, enhanced=True) if pe_train_labels is not None else [1.0, 1.0, 1.0]

    actual_gamma_pp = gamma_pp if gamma_pp is not None else focal_gamma
    actual_gamma_pe = gamma_pe if gamma_pe is not None else focal_gamma

    logger.info(f"PP类别权重: {pp_weights}" + (" (自定义)" if custom_pp_weights else " (自动计算)"))
    logger.info(f"PE类别权重: {pe_weights}" + (" (自定义)" if custom_pe_weights else " (自动计算)"))
    logger.info(f"PP任务权重: {pp_task_weight}, PE任务权重: {pe_task_weight}")
    logger.info(f"使用Focal Loss: {use_focal}, gamma_pp={actual_gamma_pp}, gamma_pe={actual_gamma_pe}, label_smoothing={label_smoothing}")

    pp_weights_tensor = torch.FloatTensor(pp_weights).to(device)
    pe_weights_tensor = torch.FloatTensor(pe_weights).to(device)

    criterion = DualHeadLoss(
        pp_weight=pp_task_weight,
        pe_weight=pe_task_weight,
        class_weights_pp=pp_weights_tensor,
        class_weights_pe=pe_weights_tensor,
        use_focal=use_focal,
        gamma=focal_gamma,
        gamma_pp=gamma_pp,
        gamma_pe=gamma_pe,
        label_smoothing=label_smoothing
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    config = get_config()
    patience = config.get('training.early_stopping.patience', 50)
    early_stopper = EarlyStopping(patience=patience)
    checkpoint_manager = CheckpointManager(output_dir) if output_dir else None

    best_score = 0
    best_results = {}

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
        'train_metrics': {
            'PP': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'PE': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        },
        'val_metrics': {
            'PP': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'PE': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }
    }

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_pp_true, train_pp_pred = [], []
        train_pe_true, train_pe_pred = [], []

        for spectra, pp_labels, pe_labels in train_loader:
            spectra = spectra.to(device)
            pp_labels = pp_labels.to(device)
            pe_labels = pe_labels.to(device)

            optimizer.zero_grad()
            pp_out, pe_out = model(spectra)
            total_loss, pp_loss, pe_loss = criterion(pp_out, pe_out, pp_labels, pe_labels)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += total_loss.item()

            # 计算训练准确率
            with torch.no_grad():
                pp_pred = pp_out.argmax(dim=1)
                pe_pred = pe_out.argmax(dim=1)
                train_pp_true.append(pp_labels.cpu().numpy())
                train_pp_pred.append(pp_pred.cpu().numpy())
                train_pe_true.append(pe_labels.cpu().numpy())
                train_pe_pred.append(pe_pred.cpu().numpy())

        scheduler.step()

        # 验证阶段
        model.eval()
        pp_correct, pe_correct, total = 0, 0, 0
        val_loss = 0
        pp_class_correct = [0, 0, 0]
        pp_class_total = [0, 0, 0]
        pe_class_correct = [0, 0, 0]
        pe_class_total = [0, 0, 0]
        pp_val_true, pp_val_pred = [], []
        pe_val_true, pe_val_pred = [], []

        with torch.no_grad():
            for spectra, pp_labels, pe_labels in val_loader:
                spectra = spectra.to(device)
                pp_labels = pp_labels.to(device)
                pe_labels = pe_labels.to(device)

                pp_out, pe_out = model(spectra)
                total_loss, pp_loss, pe_loss = criterion(pp_out, pe_out, pp_labels, pe_labels)
                val_loss += total_loss.item()

                pp_pred = pp_out.argmax(dim=1)
                pe_pred = pe_out.argmax(dim=1)

                pp_val_true.append(pp_labels.cpu().numpy())
                pp_val_pred.append(pp_pred.cpu().numpy())
                pe_val_true.append(pe_labels.cpu().numpy())
                pe_val_pred.append(pe_pred.cpu().numpy())

                total += pp_labels.size(0)
                pp_correct += pp_pred.eq(pp_labels).sum().item()
                pe_correct += pe_pred.eq(pe_labels).sum().item()

                for i in range(3):
                    pp_mask = pp_labels == i
                    pp_class_total[i] += pp_mask.sum().item()
                    pp_class_correct[i] += (pp_pred[pp_mask] == i).sum().item()

                    pe_mask = pe_labels == i
                    pe_class_total[i] += pe_mask.sum().item()
                    pe_class_correct[i] += (pe_pred[pe_mask] == i).sum().item()

        pp_acc = pp_correct / total
        pe_acc = pe_correct / total
        avg_acc = (pp_acc + pe_acc) / 2

        pp_class_accs = [pp_class_correct[i] / pp_class_total[i] if pp_class_total[i] > 0 else 0 for i in range(3)]
        pe_class_accs = [pe_class_correct[i] / pe_class_total[i] if pe_class_total[i] > 0 else 0 for i in range(3)]
        train_pp_metrics = compute_epoch_metrics(np.concatenate(train_pp_true), np.concatenate(train_pp_pred))
        train_pe_metrics = compute_epoch_metrics(np.concatenate(train_pe_true), np.concatenate(train_pe_pred))
        pp_metrics = compute_epoch_metrics(np.concatenate(pp_val_true), np.concatenate(pp_val_pred))
        pe_metrics = compute_epoch_metrics(np.concatenate(pe_val_true), np.concatenate(pe_val_pred))

        # 计算训练准确率
        train_pp_acc = train_pp_metrics['accuracy']
        train_pe_acc = train_pe_metrics['accuracy']
        train_avg_acc = (train_pp_acc + train_pe_acc) / 2

        # 记录训练历史
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_avg_acc)
        history['val_acc'].append(avg_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['train_metrics']['PP']['accuracy'].append(train_pp_metrics['accuracy'])
        history['train_metrics']['PP']['precision'].append(train_pp_metrics['precision'])
        history['train_metrics']['PP']['recall'].append(train_pp_metrics['recall'])
        history['train_metrics']['PP']['f1'].append(train_pp_metrics['f1'])
        history['train_metrics']['PE']['accuracy'].append(train_pe_metrics['accuracy'])
        history['train_metrics']['PE']['precision'].append(train_pe_metrics['precision'])
        history['train_metrics']['PE']['recall'].append(train_pe_metrics['recall'])
        history['train_metrics']['PE']['f1'].append(train_pe_metrics['f1'])
        history['val_metrics']['PP']['accuracy'].append(pp_metrics['accuracy'])
        history['val_metrics']['PP']['precision'].append(pp_metrics['precision'])
        history['val_metrics']['PP']['recall'].append(pp_metrics['recall'])
        history['val_metrics']['PP']['f1'].append(pp_metrics['f1'])
        history['val_metrics']['PE']['accuracy'].append(pe_metrics['accuracy'])
        history['val_metrics']['PE']['precision'].append(pe_metrics['precision'])
        history['val_metrics']['PE']['recall'].append(pe_metrics['recall'])
        history['val_metrics']['PE']['f1'].append(pe_metrics['f1'])

        # 综合评分
        score = avg_acc * 0.5 + (pp_class_accs[0] + pe_class_accs[0]) * 0.15 + (pp_class_accs[1] + pe_class_accs[1]) * 0.1

        is_best = score > best_score
        if is_best:
            best_score = score
            best_results = {
                'epoch': epoch + 1,
                'pp_acc': pp_acc,
                'pe_acc': pe_acc,
                'avg_acc': avg_acc,
                'pp_metrics': pp_metrics,
                'pe_metrics': pe_metrics,
                'pp_class_accs': pp_class_accs,
                'pe_class_accs': pe_class_accs
            }

        if checkpoint_manager:
            metrics = {'pp_acc': pp_acc, 'pe_acc': pe_acc, 'avg_acc': avg_acc, 'score': score}
            checkpoint_manager.save(model, optimizer, epoch + 1, metrics, is_best)

        if early_stopper(score, model):
            logger.info(f"早停触发于 epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}')
            logger.info(f'  Train PP Acc={train_pp_acc:.4f}, Train PE Acc={train_pe_acc:.4f}, Train Avg={train_avg_acc:.4f}')
            logger.info(f'  Val PP Acc={pp_acc:.4f}, Val PE Acc={pe_acc:.4f}, Val Avg={avg_acc:.4f}')
            logger.info(
                f"  Train PP Metrics: Acc={train_pp_metrics['accuracy']:.4f}, "
                f"Precision={train_pp_metrics['precision']:.4f}, "
                f"Recall={train_pp_metrics['recall']:.4f}, F1={train_pp_metrics['f1']:.4f}"
            )
            logger.info(
                f"  Train PE Metrics: Acc={train_pe_metrics['accuracy']:.4f}, "
                f"Precision={train_pe_metrics['precision']:.4f}, "
                f"Recall={train_pe_metrics['recall']:.4f}, F1={train_pe_metrics['f1']:.4f}"
            )
            logger.info(
                f"  PP Metrics: Acc={pp_metrics['accuracy']:.4f}, "
                f"Precision={pp_metrics['precision']:.4f}, "
                f"Recall={pp_metrics['recall']:.4f}, F1={pp_metrics['f1']:.4f}"
            )
            logger.info(
                f"  PE Metrics: Acc={pe_metrics['accuracy']:.4f}, "
                f"Precision={pe_metrics['precision']:.4f}, "
                f"Recall={pe_metrics['recall']:.4f}, F1={pe_metrics['f1']:.4f}"
            )
            logger.info(f'  PP类别: {[f"{a:.3f}" for a in pp_class_accs]}')
            logger.info(f'  PE类别: {[f"{a:.3f}" for a in pe_class_accs]}')

    # 保存训练历史到 JSON 文件
    if output_dir:
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"训练历史已保存: {history_path}")

    return best_results


def finetune_head(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    head: str,  # 'pp' or 'pe'
    epochs: int = 30,
    lr: float = 1e-4,
    device: str = 'cpu',
    train_labels: Optional[np.ndarray] = None,
    use_focal: bool = True,
    focal_gamma: float = 2.0
) -> Dict[str, Any]:
    """微调单个分类头。"""
    logger = get_logger('train_dual')
    logger.info(f"\n=== 微调{head.upper()}分类头 ===")

    model = model.to(device)

    # 冻结共享层
    model.freeze_shared_layers()

    # 冻结另一个分类头
    if head == 'pp':
        model.freeze_pe_head()
        model.unfreeze_pp_head()
    else:
        model.freeze_pp_head()
        model.unfreeze_pe_head()

    logger.info(f"可训练参数: {model.get_trainable_params()}")

    # 计算类别权重
    weights = compute_class_weights(train_labels, enhanced=True) if train_labels is not None else [1.0, 1.0, 1.0]
    weights_tensor = torch.FloatTensor(weights).to(device)

    if use_focal:
        criterion = FocalLoss(alpha=weights_tensor, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # 只优化可训练参数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    best_acc = 0
    best_class_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for spectra, pp_labels, pe_labels in train_loader:
            spectra = spectra.to(device)
            labels = pp_labels.to(device) if head == 'pp' else pe_labels.to(device)

            optimizer.zero_grad()
            pp_out, pe_out = model(spectra)
            out = pp_out if head == 'pp' else pe_out
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct, total = 0, 0
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]

        with torch.no_grad():
            for spectra, pp_labels, pe_labels in val_loader:
                spectra = spectra.to(device)
                labels = pp_labels.to(device) if head == 'pp' else pe_labels.to(device)

                pp_out, pe_out = model(spectra)
                out = pp_out if head == 'pp' else pe_out
                pred = out.argmax(dim=1)

                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

                for i in range(3):
                    mask = labels == i
                    class_total[i] += mask.sum().item()
                    class_correct[i] += (pred[mask] == i).sum().item()

        acc = correct / total
        class_accs = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(3)]

        if acc > best_acc:
            best_acc = acc
            best_class_accs = class_accs

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.4f}')
            logger.info(f'  类别准确率: {[f"{a:.3f}" for a in class_accs]}')

    # 解冻所有层
    model.unfreeze_shared_layers()
    model.unfreeze_pp_head()
    model.unfreeze_pe_head()

    return {'acc': best_acc, 'class_accs': best_class_accs}


def train_two_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cpu',
    output_dir: Optional[Path] = None,
    pp_train_labels: Optional[np.ndarray] = None,
    pe_train_labels: Optional[np.ndarray] = None,
    stage1_epochs: int = 50,
    stage2_epochs: int = 30,
    use_focal: bool = True,
    focal_gamma: float = 2.0
) -> Dict[str, Any]:
    """
    两阶段训练：
    阶段1：训练整个模型（共享层 + 双分类头）
    阶段2：冻结共享层，分别微调PP和PE分类头
    """
    logger = get_logger('train_dual')

    # ========== 阶段1：联合训练 ==========
    logger.info("\n" + "="*50)
    logger.info("阶段1：联合训练整个模型")
    logger.info("="*50)

    stage1_results = train_dual_model(
        model, train_loader, val_loader,
        epochs=stage1_epochs,
        lr=5e-4,
        device=device,
        output_dir=output_dir,
        pp_train_labels=pp_train_labels,
        pe_train_labels=pe_train_labels,
        use_focal=use_focal,
        focal_gamma=focal_gamma
    )

    logger.info(f"\n阶段1完成:")
    logger.info(f"  PP准确率: {stage1_results['pp_acc']:.4f}")
    logger.info(f"  PE准确率: {stage1_results['pe_acc']:.4f}")
    logger.info(f"  PP类别: {[f'{a:.3f}' for a in stage1_results['pp_class_accs']]}")
    logger.info(f"  PE类别: {[f'{a:.3f}' for a in stage1_results['pe_class_accs']]}")

    # ========== 阶段2：分别微调分类头 ==========
    logger.info("\n" + "="*50)
    logger.info("阶段2：分别微调PP和PE分类头")
    logger.info("="*50)

    # 微调PP分类头
    pp_finetune_results = finetune_head(
        model, train_loader, val_loader,
        head='pp',
        epochs=stage2_epochs,
        lr=1e-4,
        device=device,
        train_labels=pp_train_labels,
        use_focal=use_focal,
        focal_gamma=focal_gamma
    )

    # 微调PE分类头
    pe_finetune_results = finetune_head(
        model, train_loader, val_loader,
        head='pe',
        epochs=stage2_epochs,
        lr=1e-4,
        device=device,
        train_labels=pe_train_labels,
        use_focal=use_focal,
        focal_gamma=focal_gamma
    )

    # ========== 最终评估 ==========
    logger.info("\n" + "="*50)
    logger.info("最终评估")
    logger.info("="*50)

    model.eval()
    pp_correct, pe_correct, total = 0, 0, 0
    pp_class_correct = [0, 0, 0]
    pp_class_total = [0, 0, 0]
    pe_class_correct = [0, 0, 0]
    pe_class_total = [0, 0, 0]

    with torch.no_grad():
        for spectra, pp_labels, pe_labels in val_loader:
            spectra = spectra.to(device)
            pp_labels = pp_labels.to(device)
            pe_labels = pe_labels.to(device)

            pp_out, pe_out = model(spectra)
            pp_pred = pp_out.argmax(dim=1)
            pe_pred = pe_out.argmax(dim=1)

            total += pp_labels.size(0)
            pp_correct += pp_pred.eq(pp_labels).sum().item()
            pe_correct += pe_pred.eq(pe_labels).sum().item()

            for i in range(3):
                pp_mask = pp_labels == i
                pp_class_total[i] += pp_mask.sum().item()
                pp_class_correct[i] += (pp_pred[pp_mask] == i).sum().item()

                pe_mask = pe_labels == i
                pe_class_total[i] += pe_mask.sum().item()
                pe_class_correct[i] += (pe_pred[pe_mask] == i).sum().item()

    pp_acc = pp_correct / total
    pe_acc = pe_correct / total
    pp_class_accs = [pp_class_correct[i] / pp_class_total[i] if pp_class_total[i] > 0 else 0 for i in range(3)]
    pe_class_accs = [pe_class_correct[i] / pe_class_total[i] if pe_class_total[i] > 0 else 0 for i in range(3)]

    # 保存最终模型
    if output_dir:
        final_path = output_dir / 'final_two_stage_model.pth'
        torch.save(model.state_dict(), final_path)
        logger.info(f"最终模型已保存: {final_path}")

    return {
        'pp_acc': pp_acc,
        'pe_acc': pe_acc,
        'avg_acc': (pp_acc + pe_acc) / 2,
        'pp_class_accs': pp_class_accs,
        'pe_class_accs': pe_class_accs,
        'stage1_results': stage1_results,
        'pp_finetune': pp_finetune_results,
        'pe_finetune': pe_finetune_results
    }


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('train_dual')

    seed = config.get('training.random_seed', 42)
    set_random_seed(seed)

    base_dir = config.base_dir
    data_dir = Path(config.paths['preprocessed_dir'])
    output_dir = Path(config.paths['output_dir']) / 'models' 
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"设备: {device}")

    # 加载数据
    datasets = config.get('dataset.datasets')
    split_config = config.get('training.split')

    train_data, train_pp_labels, train_pe_labels = [], [], []
    val_data, val_pp_labels, val_pe_labels = [], [], []

    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']
        split = split_config.get(dataset_type, {'train': [], 'val': []})

        # 标签从 preprocessed_data 目录读取
        labels_dir = Path(config.paths['preprocessed_dir'])

        logger.info(f"\n加载数据集: {dataset_type}")

        for i, sample in enumerate(samples, 1):
            name = sample['name']
            data = np.load(data_dir / f'{name}_data.npy')
            pp_labels = np.load(labels_dir / f'{name}_pp_labels.npy')
            pe_labels = np.load(labels_dir / f'{name}_pe_labels.npy')

            data_flat = data.reshape(-1, data.shape[-1])
            pp_labels_flat = pp_labels.flatten()
            pe_labels_flat = pe_labels.flatten()

            if i in split['train']:
                train_data.append(data_flat)
                train_pp_labels.append(pp_labels_flat)
                train_pe_labels.append(pe_labels_flat)
                logger.info(f"  {name} -> 训练集")
            elif i in split['val']:
                val_data.append(data_flat)
                val_pp_labels.append(pp_labels_flat)
                val_pe_labels.append(pe_labels_flat)
                logger.info(f"  {name} -> 验证集")

    X_train = np.vstack(train_data)
    y_pp_train = np.concatenate(train_pp_labels)
    y_pe_train = np.concatenate(train_pe_labels)
    X_val = np.vstack(val_data)
    y_pp_val = np.concatenate(val_pp_labels)
    y_pe_val = np.concatenate(val_pe_labels)

    logger.info(f"\n训练集: {len(y_pp_train)} 样本, 验证集: {len(y_pp_val)} 样本")

    # 打印类别分布
    for name, labels in [('PP训练', y_pp_train), ('PE训练', y_pe_train), ('PP验证', y_pp_val), ('PE验证', y_pe_val)]:
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"{name}类别分布: {dict(zip(unique, counts))}")

    batch_size = config.get('training.batch_size', 32)

    # 计算样本权重用于加权采样
    sample_weights = compute_sample_weights(y_pp_train, y_pe_train)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    logger.info(f"使用加权采样器，样本权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

    train_loader = DataLoader(
        DualLabelDataset(X_train, y_pp_train, y_pe_train, augment=True),
        batch_size=batch_size,
        sampler=sampler  # 使用加权采样器，不用shuffle
    )
    val_loader = DataLoader(
        DualLabelDataset(X_val, y_pp_val, y_pe_val),
        batch_size=batch_size
    )

    model = DualHeadRamanCNNLSTM(input_len=X_train.shape[1], num_classes=3)
    logger.info("\n开始训练双头模型（Focal Loss + 双标签联合加权采样）...")

    # 从配置文件读取PP和PE的类别权重
    custom_pp_weights = config.get('training.class_weights_pp', None)
    custom_pe_weights = config.get('training.class_weights_pe', None)

    # 从配置文件读取训练轮数
    epochs = config.get('training.epochs', 100)
    # 标签从 preprocessed_data 目录读取
    results = train_dual_model(
        model, train_loader, val_loader,
        epochs=epochs,
        device=device,
        output_dir=output_dir,
        pp_train_labels=y_pp_train,
        pe_train_labels=y_pe_train,
        use_focal=True,
        focal_gamma=2.0,
        gamma_pp=2.5,       # 适度关注边界样本
        gamma_pe=2.5,       # 适度关注边界样本
        label_smoothing=0.1,
        pp_task_weight=1.0,   # 平衡的任务权重
        pe_task_weight=1.0,
        custom_pp_weights=custom_pp_weights,  # 从配置文件读取
        custom_pe_weights=custom_pe_weights   # 从配置文件读取
    )

    if results:
        logger.info(f"\n===== 最佳结果 (Epoch {results['epoch']}) =====")
        logger.info(f"  PP准确率: {results['pp_acc']:.4f}")
        logger.info(f"  PE准确率: {results['pe_acc']:.4f}")
        logger.info(f"  平均准确率: {results['avg_acc']:.4f}")
        logger.info(f"  PP类别准确率: {results['pp_class_accs']}")
        logger.info(f"  PE类别准确率: {results['pe_class_accs']}")
