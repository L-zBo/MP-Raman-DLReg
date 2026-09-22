"""
最终箱型图脚本。

功能：
- 生成训练阶段箱型图
- 生成最终模型池排序版箱型图

输出：
- output/boxplot_final_ranked/PP/
- output/boxplot_final_ranked/PE/
"""

import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

# XGBoost
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

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    get_config,
    get_logger,
    PRIMARY_MODEL_NAME,
)

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 导入模型类
from models import SMARTNIRClassifier, ConvTranClassifier, MambaHSIClassifier, ResNet50_1D
from models.dual_head_model import DualHeadRamanCNNLSTM


def load_all_data(config, task='PP') -> Tuple[np.ndarray, np.ndarray]:
    """加载数据集"""
    logger = get_logger('boxplot')
    data_dir = Path(config.paths['preprocessed_dir'])
    output_dir = Path(config.paths['output_dir'])

    datasets = config.get('dataset.datasets')

    all_data, all_labels = [], []
    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']

        for sample in samples:
            name = sample['name']
            data_path = data_dir / f'{name}_data.npy'

            # 根据任务加载对应的标签（从preprocessed_data目录）
            if task == 'PP':
                label_path = data_dir / f'{name}_pp_labels.npy'
            else:  # PE
                label_path = data_dir / f'{name}_pe_labels.npy'

            if data_path.exists() and label_path.exists():
                data = np.load(data_path)
                labels = np.load(label_path)
                all_data.append(data.reshape(-1, data.shape[-1]))
                all_labels.append(labels.flatten())

    X_pre = np.vstack(all_data)
    y = np.concatenate(all_labels)

    logger.info(f"数据集 ({task}): {X_pre.shape[0]} 样本, 特征维度: {X_pre.shape[1]}")
    return X_pre, y


def train_cnn_lstm_with_history(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    device: str = 'cuda',
    task: str = 'PP',
    patience: int = 10
) -> Dict[str, List[float]]:
    """训练双头模型并记录每个epoch的指标"""
    logger = get_logger('boxplot')

    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    if not torch.cuda.is_available():
        device = 'cpu'

    # 准备数据
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 加载双头模型
    model_path = Path('output/models/best_model.pth')
    model = DualHeadRamanCNNLSTM(input_len=X_train.shape[1], num_classes=3).to(device)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 动态学习率调整
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }

    # 早停机制变量
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pp_outputs, pe_outputs = model(X_batch)

            # 根据任务选择输出
            outputs = pp_outputs if task == 'PP' else pe_outputs
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pp_outputs, pe_outputs = model(X_batch)

                # 根据任务选择输出
                outputs = pp_outputs if task == 'PP' else pe_outputs
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        current_val_acc = val_correct / val_total
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(current_val_acc)
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)

        # 更新学习率
        scheduler.step(val_loss / val_total)

        # 早停检查
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # 每10个epoch打印一次进度
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs}: Val Acc = {current_val_acc:.4f}")

        # 早停触发（仅当patience不为None时）
        if patience is not None and patience_counter >= patience:
            logger.info(f"    早停触发于 Epoch {epoch+1}，最佳验证准确率: {best_val_acc:.4f}")
            break

    return history


def load_raw_data(config) -> np.ndarray:
    """加载数据集"""
    logger = get_logger('boxplot')
    import pandas as pd

    base_dir = Path(config.base_dir)
    dataset_dir = base_dir / config.paths['dataset_dir']
    datasets = config.get('dataset.datasets')

    all_data = []
    for dataset in datasets:
        samples = dataset['samples']
        for sample in samples:
            folder = dataset_dir / sample['folder']
            csv_files = sorted(folder.glob('*.csv'))
            spectra = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, encoding='gbk')
                    spectrum = df.iloc[:, 1].values.astype(np.float64)
                except:
                    df = pd.read_csv(csv_file, encoding='utf-8')
                    spectrum = df.iloc[:, 1].values.astype(np.float64)
                spectra.append(spectrum)
            data = np.array(spectra).reshape(40, 40, -1)
            all_data.append(data.reshape(-1, data.shape[-1]))

    X_raw = np.vstack(all_data)
    logger.info(f"数据集: {X_raw.shape[0]} 样本, 特征维度: {X_raw.shape[1]}")
    return X_raw


def train_cnn_lstm_cv(X_train, y_train, X_val, y_val, max_epochs=200, patience=20, device='cpu', task='PP', progress_label=''):
    """
    训练双头模型用于交叉验证
    - 固定随机种子
    - 动态学习率
    - 早停机制（基于宏平均准确率）
    - 返回宏平均准确率
    """
    logger = get_logger('boxplot')
    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 加载双头模型
    model_path = Path('output/models/best_model.pth')
    model = DualHeadRamanCNNLSTM(input_len=X_train.shape[1], num_classes=3).to(device)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 动态学习率调整
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # 早停机制
    best_acc_macro = 0.0
    patience_counter = 0

    epoch_bar = tqdm(
        range(1, max_epochs + 1),
        desc=f'[{progress_label}] Epoch',
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
    )
    for epoch_idx in epoch_bar:
        # 训练阶段
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        train_bar = tqdm(
            train_loader,
            desc=f'[{progress_label}] Train {epoch_idx}/{max_epochs}',
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        )
        for X_batch, y_batch in train_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pp_outputs, pe_outputs = model(X_batch)

            # 根据任务选择输出
            outputs = pp_outputs if task == 'PP' else pe_outputs
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            batch_size_actual = y_batch.size(0)
            running_train_loss += loss.detach().item() * batch_size_actual
            train_samples += batch_size_actual
            train_bar.set_postfix(
                loss=f'{loss.detach().item():.4f}',
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        train_bar.close()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_bar = tqdm(
            val_loader,
            desc=f'[{progress_label}] Val {epoch_idx}/{max_epochs}',
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        )
        with torch.no_grad():
            for X_batch, y_batch in val_bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pp_outputs, pe_outputs = model(X_batch)

                # 根据任务选择输出
                outputs = pp_outputs if task == 'PP' else pe_outputs
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * y_batch.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        val_bar.close()

        val_loss /= len(all_labels)
        # 计算宏平均准确率
        val_acc_macro = recall_score(all_labels, all_preds, average='macro')
        avg_train_loss = running_train_loss / max(train_samples, 1)

        # 更新学习率
        scheduler.step(val_loss)
        epoch_bar.set_postfix(
            train_loss=f'{avg_train_loss:.4f}',
            val_macro=f'{val_acc_macro:.4f}',
            best=f'{best_acc_macro:.4f}',
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        logger.info(
            f"    [{progress_label}] Epoch {epoch_idx}/{max_epochs}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Macro Acc={val_acc_macro:.4f}, "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        # 早停检查（基于宏平均准确率）
        if val_acc_macro > best_acc_macro:
            best_acc_macro = val_acc_macro
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"    [{progress_label}] Early stop at epoch {epoch_idx}, "
                    f"best Val Macro Acc={best_acc_macro:.4f}"
                )
                break

    epoch_bar.close()
    del train_loader, val_loader, train_dataset, val_dataset, model, optimizer, scheduler, criterion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_acc_macro


def get_training_stages_data(X: np.ndarray, y: np.ndarray, task: str = 'PP') -> Dict[str, List[float]]:
    """获取训练过程分阶段的数据（自适应区间划分）"""
    logger = get_logger('boxplot')
    logger.info(f"\n>>> 获取训练迭代次数数据（50 epochs，自适应区间，{task} Task）...")

    # 固定随机种子
    np.random.seed(42)

    # 划分数据
    n_samples = len(X)
    split_idx = int(n_samples * 0.8)
    indices = np.random.permutation(n_samples)

    X_train, y_train = X[indices[:split_idx]], y[indices[:split_idx]]
    X_val, y_val = X[indices[split_idx:]], y[indices[split_idx:]]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 训练50个epoch并记录历史（不使用早停，需要完整曲线）
    total_epochs = 50
    logger.info(f"  开始训练 {total_epochs} epochs...")
    history = train_cnn_lstm_with_history(X_train, y_train, X_val, y_val, epochs=total_epochs, device=device, task=task, patience=None)

    val_acc = np.array(history['val_acc'])

    # 自适应阶段划分：基于准确率变化率和稳定性
    stages = adaptive_stage_split(val_acc, logger)

    return stages


def adaptive_stage_split(val_acc: np.ndarray, logger) -> Dict[str, List[float]]:
    """
    自适应划分训练阶段（适配50个epoch）
    基于准确率曲线自动划分，确保4个阶段的均值呈递增趋势（从低到高）
    """
    n_epochs = len(val_acc)

    # 计算累积均值曲线，用于找到自然的阶段边界
    cumulative_mean = np.array([np.mean(val_acc[:i+1]) for i in range(n_epochs)])

    # 计算准确率的变化率（一阶差分）
    acc_diff = np.diff(val_acc)

    # 计算滑动窗口均值（窗口大小=5）
    window_size = 5
    smoothed_acc = np.convolve(val_acc, np.ones(window_size)/window_size, mode='valid')

    # 策略：基于准确率提升的自然分段
    # 找到准确率显著提升的关键点

    # 计算每个位置到结束的平均准确率
    suffix_means = np.array([np.mean(val_acc[i:]) for i in range(n_epochs)])

    # 初始边界估计
    # Stage 1: 初始学习阶段（准确率最低的区间）
    # Stage 4: 收敛阶段（准确率最高且稳定的区间）

    # 使用分位数方法找到边界
    acc_25 = np.percentile(val_acc, 25)
    acc_50 = np.percentile(val_acc, 50)
    acc_75 = np.percentile(val_acc, 75)

    # 找到各阶段的自然边界
    stage1_end = 5  # 默认值
    for i in range(3, min(15, n_epochs)):
        if val_acc[i] > acc_25 and np.mean(val_acc[i:i+3]) > acc_25:
            stage1_end = i
            break

    # 找到收敛开始点（准确率稳定在高位）
    convergence_start = n_epochs - 10  # 默认值
    for i in range(n_epochs - 5, max(stage1_end + 10, 15), -1):
        window_mean = np.mean(val_acc[i-5:i])
        window_std = np.std(val_acc[i-5:i])
        if window_mean < acc_75 or window_std > 0.005:
            convergence_start = i
            break

    # 确保边界合理
    stage1_end = max(3, min(stage1_end, 12))
    convergence_start = max(stage1_end + 15, min(convergence_start, n_epochs - 8))

    # 中间阶段的划分 - 基于准确率曲线的形状
    middle_range = convergence_start - stage1_end
    mid_point = stage1_end + middle_range // 2

    # 构建初始阶段
    boundaries = [0, stage1_end, mid_point, convergence_start, n_epochs]

    # 验证并调整：确保每个阶段的均值递增
    stage_means = []
    for i in range(4):
        stage_vals = val_acc[boundaries[i]:boundaries[i+1]]
        stage_means.append(np.mean(stage_vals))

    # 如果均值不是递增的，调整边界
    max_iterations = 10
    for _ in range(max_iterations):
        needs_adjustment = False
        for i in range(3):
            if stage_means[i] >= stage_means[i+1]:
                needs_adjustment = True
                # 调整边界：将分界点向后移动
                if i < 2:
                    boundaries[i+1] = min(boundaries[i+1] + 2, boundaries[i+2] - 2)
                else:
                    boundaries[i+1] = max(boundaries[i+1] - 2, boundaries[i] + 2)

        if not needs_adjustment:
            break

        # 重新计算均值
        stage_means = []
        for i in range(4):
            stage_vals = val_acc[boundaries[i]:boundaries[i+1]]
            if len(stage_vals) > 0:
                stage_means.append(np.mean(stage_vals))
            else:
                stage_means.append(0)

    # 最终确保边界有效
    boundaries[1] = max(3, min(boundaries[1], boundaries[2] - 3))
    boundaries[2] = max(boundaries[1] + 3, min(boundaries[2], boundaries[3] - 3))
    boundaries[3] = max(boundaries[2] + 3, min(boundaries[3], n_epochs - 3))

    # 构建阶段字典
    stages = {
        f'Stage 1\n(Epoch 1-{boundaries[1]})': val_acc[boundaries[0]:boundaries[1]].tolist(),
        f'Stage 2\n(Epoch {boundaries[1]+1}-{boundaries[2]})': val_acc[boundaries[1]:boundaries[2]].tolist(),
        f'Stage 3\n(Epoch {boundaries[2]+1}-{boundaries[3]})': val_acc[boundaries[2]:boundaries[3]].tolist(),
        f'Stage 4\n(Epoch {boundaries[3]+1}-{n_epochs})': val_acc[boundaries[3]:n_epochs].tolist(),
    }

    # 打印每个阶段的统计信息
    logger.info("  阶段划分结果（自适应，确保均值递增）：")
    for stage_name, values in stages.items():
        if len(values) > 0:
            logger.info(f"    {stage_name.replace(chr(10), ' ')}: {len(values)} points, "
                       f"Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, "
                       f"Range=[{np.min(values):.4f}, {np.max(values):.4f}]")

    return stages


def plot_training_iterations_boxplot(stages_data: Dict[str, List[float]], output_dir: Path, task: str = 'PP'):
    """绘制训练迭代次数变化箱线图（带散点显示）"""
    logger = get_logger('boxplot')

    fig, ax = plt.subplots(figsize=(12, 7))

    stage_names = list(stages_data.keys())
    stage_values = [stages_data[name] for name in stage_names]

    # 颜色配置 - 不同颜色区分不同阶段
    colors = ['#B19CD9', '#77DD77', '#89CFF0', '#FFB6C1']  # 浅紫、浅绿、浅蓝、浅粉

    # 创建箱线图
    bp = ax.boxplot(stage_values, labels=stage_names, patch_artist=True, widths=0.5)

    # 设置箱体颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # 设置中位线为深红色
    for median in bp['medians']:
        median.set_color('#8B0000')
        median.set_linewidth(2)

    # 设置whisker和cap
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)

    # 隐藏默认的fliers（异常点）
    for flier in bp['fliers']:
        flier.set_visible(False)

    # 添加散点（jittered scatter）- 关键特性！
    for i, (values, color) in enumerate(zip(stage_values, colors)):
        # 生成水平方向的抖动（jitter）
        x_jitter = np.random.normal(i + 1, 0.08, size=len(values))
        # 绘制散点，透明度设置让重叠可见
        ax.scatter(x_jitter, values, c=color, alpha=0.7, s=50, edgecolors='white', linewidths=0.5, zorder=3)

    # 添加均值点（菱形标记）
    means = [np.mean(v) for v in stage_values]
    ax.scatter(range(1, len(means)+1), means, color='#228B22', marker='D', s=80, zorder=4,
               edgecolors='white', linewidths=1.5, label='Mean')

    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Stage', fontsize=12, fontweight='bold')
    ax.set_title(f'{PRIMARY_MODEL_NAME} Training Progress - {task} Task', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)

    # 设置y轴范围，留出空间
    all_values = [v for vals in stage_values for v in vals]
    y_min = min(all_values) - 0.02
    y_max = max(all_values) + 0.02
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    save_path = output_dir / 'training_iterations_boxplot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"保存: {save_path}")


def plot_accuracy_boxplot(cv_results: Dict[str, List[float]], output_dir: Path, task: str = 'PP'):
    """绘制模型准确率对比箱线图（带散点显示）"""
    logger = get_logger('boxplot')

    # 按平均准确率排序
    sorted_models = sorted(cv_results.keys(), key=lambda m: np.mean(cv_results[m]), reverse=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    model_names = sorted_models
    model_values = [cv_results[name] for name in model_names]

    # 颜色配置 - 所有模型使用统一的浅色系
    colors = ['#81ecec', '#ffeaa7', '#fab1a0', '#74b9ff', '#a29bfe', '#DDA0DD', '#F39C12', '#E74C3C', '#98D8C8']

    # 创建箱线图
    bp = ax.boxplot(model_values, labels=model_names, patch_artist=True, widths=0.5)

    # 设置箱体颜色
    for i, (patch, _) in enumerate(zip(bp['boxes'], model_names)):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # 设置中位线为深红色
    for median in bp['medians']:
        median.set_color('#8B0000')
        median.set_linewidth(2)

    # 设置whisker和cap
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)

    # 隐藏默认的fliers
    for flier in bp['fliers']:
        flier.set_visible(False)

    # 添加散点（jittered scatter）
    for i, (values, _) in enumerate(zip(model_values, model_names)):
        x_jitter = np.random.normal(i + 1, 0.06, size=len(values))
        ax.scatter(x_jitter, values, c=colors[i % len(colors)], alpha=0.8, s=80, edgecolors='white', linewidths=1, zorder=3)

    # 添加均值点（菱形标记）
    means = [np.mean(v) for v in model_values]
    ax.scatter(range(1, len(means)+1), means, color='#228B22', marker='D', s=100, zorder=4,
               edgecolors='white', linewidths=1.5, label='Mean')

    ax.set_ylabel('Macro Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Accuracy Comparison - {task} Task', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 设置x轴标签
    ax.set_xticklabels(model_names, rotation=15, ha='right')

    # 在箱线图上方添加均值标注
    for i, (mean, values) in enumerate(zip(means, model_values)):
        max_val = max(values)
        ax.text(i+1, max_val + 0.008, f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(loc='lower right', fontsize=10)

    # 自适应纵轴范围
    all_values = [v for values in model_values for v in values]
    y_min = min(all_values) - 0.03
    y_max = max(all_values) + 0.03
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    save_path = output_dir / 'accuracy_boxplot_ranked.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"保存: {save_path}")

    # 保存统计数据到CSV
    stats_data = []
    for model in sorted_models:
        values = cv_results[model]
        row = {
            'Model': model,
            'Mean': round(np.mean(values), 4),
            'Std': round(np.std(values), 4),
            'Min': round(np.min(values), 4),
            'Max': round(np.max(values), 4),
        }
        for i, v in enumerate(values):
            row[f'Fold{i+1}'] = round(v, 4)
        stats_data.append(row)

    df = pd.DataFrame(stats_data)
    csv_path = output_dir / 'accuracy_boxplot_ranked_stats.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"保存: {csv_path}")

    return df


FINAL_BOXPLOT_MODEL_ORDER = [
    'SMART-NIR',
    'ConvTran',
    'MambaHSI',
    'ResNet-50',
    PRIMARY_MODEL_NAME,
    'XGBoost',
    'SVM',
    'Random Forest',
    'LightGBM',
]

CANDIDATE_MODEL_CLASSES = {
    'SMART-NIR': SMARTNIRClassifier,
    'ConvTran': ConvTranClassifier,
    'MambaHSI': MambaHSIClassifier,
}


def build_pp_mixedaware_dataset(
    X_pp: np.ndarray,
    y_pp: np.ndarray,
    X_pe: np.ndarray,
    neg_ratio: float,
    seed_offset: int = 0,
):
    if neg_ratio <= 0:
        return X_pp, y_pp

    rng = np.random.default_rng(42 + seed_offset)
    sample_size = min(len(X_pe), max(1, int(round(len(X_pp) * neg_ratio))))
    indices = rng.choice(len(X_pe), size=sample_size, replace=False)
    X_neg = X_pe[indices]
    y_neg = np.zeros(sample_size, dtype=y_pp.dtype)
    return np.vstack([X_pp, X_neg]), np.concatenate([y_pp, y_neg])


BOXPLOT_SINGLE_HEAD_CONFIGS = {
    'SMART-NIR': {
        'PP': {
            'model_kwargs': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 4,
                'patch_size': 16,
                'branch_channels': 16,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'neg_ratio_train': 0.25,
                'neg_ratio_val': 0.25,
                'epochs': 20,
                'lr': 5e-6,
                'weight_decay': 4e-4,
                'batch_size': 64,
                'patience': 16,
                'label_smoothing': 0.02,
                'use_weighted_sampler': False,
            },
        },
        'PE': {
            'model_kwargs': {
                'd_model': 48,
                'num_heads': 4,
                'num_layers': 3,
                'patch_size': 16,
                'branch_channels': 12,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'epochs': 30,
                'lr': 1.2e-5,
                'weight_decay': 4e-4,
                'batch_size': 64,
                'patience': 12,
                'label_smoothing': 0.02,
                'use_weighted_sampler': True,
            },
        },
    },
    'ConvTran': {
        'PP': {
            'model_kwargs': {
                'emb_size': 96,
                'num_heads': 4,
                'num_layers': 2,
                'dim_ff': 192,
                'patch_size': 16,
                'conv_expansion': 3,
                'dropout': 0.32,
            },
            'train_kwargs': {
                'neg_ratio_train': 0.50,
                'neg_ratio_val': 0.50,
                'epochs': 95,
                'lr': 5e-5,
                'weight_decay': 0.0012,
                'batch_size': 64,
                'patience': 20,
                'label_smoothing': 0.06,
                'use_weighted_sampler': True,
            },
        },
        'PE': {
            'model_kwargs': {
                'emb_size': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dim_ff': 160,
                'patch_size': 16,
                'conv_expansion': 2,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'epochs': 65,
                'lr': 2e-4,
                'weight_decay': 3e-4,
                'batch_size': 64,
                'patience': 14,
                'label_smoothing': 0.02,
                'use_weighted_sampler': True,
            },
        },
    },
    'MambaHSI': {
        'PP': {
            'model_kwargs': {
                'd_model': 96,
                'patch_size': 16,
                'num_layers': 4,
                'num_groups': 8,
                'dropout': 0.15,
            },
            'train_kwargs': {
                'neg_ratio_train': 0.25,
                'neg_ratio_val': 0.25,
                'epochs': 90,
                'lr': 8e-5,
                'weight_decay': 8e-4,
                'batch_size': 64,
                'patience': 18,
                'label_smoothing': 0.05,
                'use_weighted_sampler': False,
            },
        },
        'PE': {
            'model_kwargs': {
                'd_model': 96,
                'patch_size': 16,
                'num_layers': 4,
                'num_groups': 8,
                'dropout': 0.10,
            },
            'train_kwargs': {
                'epochs': 50,
                'lr': 1.5e-4,
                'weight_decay': 2e-4,
                'batch_size': 64,
                'patience': 12,
                'label_smoothing': 0.02,
                'use_weighted_sampler': True,
            },
        },
    },
    'ResNet-50': {
        'PP': {
            'model_kwargs': {
                'dropout': 0.25,
            },
            'train_kwargs': {
                'neg_ratio_train': 0.50,
                'neg_ratio_val': 0.25,
                'epochs': 100,
                'lr': 5e-5,
                'weight_decay': 0.0012,
                'batch_size': 64,
                'patience': 20,
                'label_smoothing': 0.06,
                'use_weighted_sampler': False,
            },
        },
        'PE': {
            'model_kwargs': {
                'dropout': 0.0,
            },
            'train_kwargs': {
                'epochs': 80,
                'lr': 1e-4,
                'weight_decay': 8e-4,
                'batch_size': 64,
                'patience': 18,
                'label_smoothing': 0.03,
                'use_weighted_sampler': True,
            },
        },
    },
}

TRADITIONAL_MODEL_FACTORIES = {
    'XGBoost': lambda: XGBClassifier(
        n_estimators=40, max_depth=2, learning_rate=0.02,
        random_state=42, use_label_encoder=False, eval_metric='mlogloss'
    ) if HAS_XGBOOST else None,
    'SVM': lambda: SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Random Forest': lambda: RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
    ),
    'LightGBM': lambda: LGBMClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.04,
        num_leaves=12, random_state=42, n_jobs=-1, verbose=-1
    ) if HAS_LIGHTGBM else None,
}


def train_single_head_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int = 64,
    patience: int = 10,
    label_smoothing: float = 0.0,
    use_weighted_sampler: bool = False,
    progress_label: str = '',
) -> float:
    logger = get_logger('boxplot_final')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))

    pin_memory = device.type == 'cuda'

    if use_weighted_sampler:
        counts = np.bincount(y_train, minlength=3).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / counts
        sample_weights = torch.tensor(weights[y_train], dtype=torch.float32)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = model.to(device)
    class_weights = np.bincount(y_train, minlength=3).astype(np.float32)
    class_weights[class_weights == 0] = 1.0
    class_weights = class_weights.sum() / class_weights
    class_weights = class_weights / class_weights.max()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_state = None
    best_val_macro_acc = -1.0
    no_improve = 0

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f'[{progress_label}] Epoch',
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
    )
    for epoch_idx in epoch_bar:
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        train_bar = tqdm(
            train_loader,
            desc=f'[{progress_label}] Train {epoch_idx}/{epochs}',
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        )
        for X_batch, y_batch in train_bar:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            batch_size_actual = y_batch.size(0)
            running_train_loss += loss.detach().item() * batch_size_actual
            train_samples += batch_size_actual
            train_bar.set_postfix(
                loss=f'{loss.detach().item():.4f}',
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        train_bar.close()

        model.eval()
        val_preds = []
        val_true = []
        val_bar = tqdm(
            val_loader,
            desc=f'[{progress_label}] Val {epoch_idx}/{epochs}',
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        )
        with torch.no_grad():
            for X_batch, y_batch in val_bar:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    logits = model(X_batch)
                pred = logits.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
        val_bar.close()

        val_macro_acc = recall_score(val_true, val_preds, average='macro')
        scheduler.step(val_macro_acc)
        avg_train_loss = running_train_loss / max(train_samples, 1)
        epoch_bar.set_postfix(
            train_loss=f'{avg_train_loss:.4f}',
            val_macro=f'{val_macro_acc:.4f}',
            best=f'{best_val_macro_acc if best_val_macro_acc >= 0 else val_macro_acc:.4f}',
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        logger.info(
            f"    [{progress_label}] Epoch {epoch_idx}/{epochs}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Macro Acc={val_macro_acc:.4f}, "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_macro_acc > best_val_macro_acc:
            best_val_macro_acc = val_macro_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(
                f"    [{progress_label}] Early stop at epoch {epoch_idx}, "
                f"best Val Macro Acc={best_val_macro_acc:.4f}"
            )
            break

    epoch_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val_scaled).to(device)
        preds = model(X_val_t).argmax(dim=1).cpu().numpy()
    result = recall_score(y_val, preds, average='macro')

    del X_val_t
    del train_loader, val_loader, train_dataset, val_dataset
    del model, optimizer, scheduler, scaler_amp, criterion
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def _strip_neg_ratio_train_kwargs(train_kwargs: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in train_kwargs.items() if not k.startswith('neg_ratio')}


def _build_cv_context(config) -> Dict[str, Dict[str, object]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_pp, y_pp = load_all_data(config, task='PP')
    X_pe, y_pe = load_all_data(config, task='PE')

    return {
        'PP': {
            'X': X_pp,
            'y': y_pp,
            'splits': list(skf.split(X_pp, y_pp)),
            'X_other': X_pe,
        },
        'PE': {
            'X': X_pe,
            'y': y_pe,
            'splits': list(skf.split(X_pe, y_pe)),
            'X_other': None,
        },
    }


def _init_results(selected_models: List[str], selected_tasks: List[str]) -> Dict[str, Dict[str, List[float]]]:
    return {
        task: {model_name: [] for model_name in selected_models}
        for task in selected_tasks
    }


def _resolve_single_head_bundle(model_name: str, task: str, input_len: int):
    raw_cfg = BOXPLOT_SINGLE_HEAD_CONFIGS[model_name][task]
    train_kwargs = dict(raw_cfg['train_kwargs'])
    if task == 'PP':
        train_kwargs = _strip_neg_ratio_train_kwargs(train_kwargs)

    model_class = ResNet50_1D if model_name == 'ResNet-50' else CANDIDATE_MODEL_CLASSES[model_name]
    return model_class, {'input_len': input_len, 'num_classes': 3, **raw_cfg['model_kwargs']}, train_kwargs, raw_cfg


def _get_fold_arrays(context: Dict[str, Dict[str, object]], task: str, fold: int):
    task_context = context[task]
    X_pre = task_context['X']
    y = task_context['y']
    train_idx, val_idx = task_context['splits'][fold - 1]
    return (
        X_pre[train_idx],
        y[train_idx],
        X_pre[val_idx],
        y[val_idx],
        task_context['X_other'],
        X_pre.shape[1],
    )


def _run_single_head_model_fold(*, model_name: str, task: str, fold: int, context: Dict[str, Dict[str, object]]) -> float:
    logger = get_logger('boxplot_final')
    X_train, y_train, X_val, y_val, X_other, input_len = _get_fold_arrays(context, task, fold)
    model_class, model_init_kwargs, train_kwargs, raw_cfg = _resolve_single_head_bundle(model_name, task, input_len)

    if task == 'PP':
        X_train_final, y_train_final = build_pp_mixedaware_dataset(
            X_train, y_train, X_other, raw_cfg['train_kwargs']['neg_ratio_train'], seed_offset=fold
        )
        X_val_final, y_val_final = build_pp_mixedaware_dataset(
            X_val, y_val, X_other, raw_cfg['train_kwargs']['neg_ratio_val'], seed_offset=100 + fold
        )
    else:
        X_train_final, y_train_final = X_train, y_train
        X_val_final, y_val_final = X_val, y_val

    logger.info(f"  开始训练 {model_name}...")
    acc = train_single_head_fold(
        model_class(**model_init_kwargs),
        X_train_final,
        y_train_final,
        X_val_final,
        y_val_final,
        progress_label=f'{task} Fold{fold} {model_name}',
        **train_kwargs,
    )
    logger.info(f"  {model_name}: {acc:.4f}")
    return acc


def _run_radar_fold(*, task: str, fold: int, context: Dict[str, Dict[str, object]]) -> float:
    logger = get_logger('boxplot_final')
    X_train, y_train, X_val, y_val, _, _ = _get_fold_arrays(context, task, fold)
    logger.info(f"  开始训练 {PRIMARY_MODEL_NAME}...")
    acc = train_cnn_lstm_cv(
        X_train, y_train, X_val, y_val,
        max_epochs=150,
        patience=20,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        task=task,
        progress_label=f'{task} Fold{fold} {PRIMARY_MODEL_NAME}',
    )
    logger.info(f"  {PRIMARY_MODEL_NAME}: {acc:.4f}")
    return acc


def _run_traditional_fold(*, model_name: str, task: str, fold: int, context: Dict[str, Dict[str, object]]) -> float:
    logger = get_logger('boxplot_final')
    X_train, y_train, X_val, y_val, _, _ = _get_fold_arrays(context, task, fold)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    factory = TRADITIONAL_MODEL_FACTORIES[model_name]
    model = factory()
    if model is None:
        raise RuntimeError(f'{model_name} 当前环境不可用')
    logger.info(f"  开始训练 {model_name}...")
    model.fit(X_train_std, y_train)
    preds = model.predict(X_val_std)
    acc = recall_score(y_val, preds, average='macro')
    logger.info(f"  {model_name}: {acc:.4f}")
    return acc


def run_model_task_fold(
    *,
    model_name: str,
    task: str,
    fold: int,
    context: Dict[str, Dict[str, object]],
    results: Dict[str, Dict[str, List[float]]],
):
    logger = get_logger('boxplot_final')
    logger.info(f"\n=== {model_name} | Fold {fold}/5 | {task} ===")

    if model_name in CANDIDATE_MODEL_CLASSES or model_name == 'ResNet-50':
        acc = _run_single_head_model_fold(model_name=model_name, task=task, fold=fold, context=context)
    elif model_name == PRIMARY_MODEL_NAME:
        acc = _run_radar_fold(task=task, fold=fold, context=context)
    else:
        acc = _run_traditional_fold(model_name=model_name, task=task, fold=fold, context=context)

    results[task][model_name].append(acc)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_final_cv_results_all_tasks(
    config,
    *,
    selected_models: List[str],
    selected_tasks: List[str],
    selected_folds: List[int],
) -> Dict[str, Dict[str, List[float]]]:
    logger = get_logger('boxplot_final')
    context = _build_cv_context(config)
    results = _init_results(selected_models, selected_tasks)

    for model_name in selected_models:
        logger.info("\n" + "=" * 60)
        logger.info(f"开始执行模型: {model_name}")
        logger.info("=" * 60)
        for fold in selected_folds:
            for task in ['PP', 'PE']:
                if task not in selected_tasks:
                    continue
                run_model_task_fold(
                    model_name=model_name,
                    task=task,
                    fold=fold,
                    context=context,
                    results=results,
                )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Final scheme boxplot cross-validation runner')
    parser.add_argument('--models', nargs='+', choices=FINAL_BOXPLOT_MODEL_ORDER, default=FINAL_BOXPLOT_MODEL_ORDER)
    parser.add_argument('--tasks', nargs='+', choices=['PP', 'PE'], default=['PP', 'PE'])
    parser.add_argument('--folds', nargs='+', type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    parser.add_argument('--skip-final-plots', action='store_true')
    parser.add_argument('--skip-stage-boxplot', action='store_true')
    return parser.parse_args()


def plot_accuracy_boxplot_final(cv_results: Dict[str, List[float]], output_dir: Path, task: str = 'PP'):
    logger = get_logger('boxplot_final')
    model_names = [m for m in FINAL_BOXPLOT_MODEL_ORDER if m in cv_results and cv_results[m]]
    if not model_names:
        logger.warning(f"{task} 任务没有可绘制的结果，跳过箱型图")
        return None

    model_values = [cv_results[name] for name in model_names]
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#81ecec', '#ffeaa7', '#fab1a0', '#74b9ff', '#98D8C8', '#a29bfe', '#DDA0DD', '#F39C12', '#E74C3C']
    bp = ax.boxplot(model_values, labels=model_names, patch_artist=True, widths=0.5)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('#8B0000')
        median.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)
    for flier in bp['fliers']:
        flier.set_visible(False)

    for i, values in enumerate(model_values):
        x_jitter = np.random.normal(i + 1, 0.06, size=len(values))
        ax.scatter(x_jitter, values, c=colors[i % len(colors)], alpha=0.8, s=80, edgecolors='white', linewidths=1, zorder=3)

    means = [np.mean(v) for v in model_values]
    ax.scatter(range(1, len(means) + 1), means, color='#228B22', marker='D', s=100, zorder=4,
               edgecolors='white', linewidths=1.5, label='Mean')

    ax.set_ylabel('Macro Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Final Scheme Model Accuracy Comparison - {task} Task', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_xticklabels(model_names, rotation=15, ha='right')

    for i, (mean, values) in enumerate(zip(means, model_values)):
        ax.text(i + 1, max(values) + 0.008, f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(loc='lower right', fontsize=10)
    all_values = [v for values in model_values for v in values]
    ax.set_ylim(min(all_values) - 0.03, max(all_values) + 0.03)

    plt.tight_layout()
    save_path = output_dir / 'accuracy_boxplot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"保存: {save_path}")

    stats_data = []
    for model in model_names:
        values = cv_results[model]
        row = {
            'Model': model,
            'Mean': round(np.mean(values), 4),
            'Std': round(np.std(values), 4),
            'Min': round(np.min(values), 4),
            'Max': round(np.max(values), 4),
        }
        for i, v in enumerate(values):
            row[f'Fold{i+1}'] = round(v, 4)
        stats_data.append(row)

    df = pd.DataFrame(stats_data)
    csv_path = output_dir / 'accuracy_boxplot_stats.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"保存: {csv_path}")
    return df


def process_task_final(config, task='PP', cv_results=None):
    logger = get_logger('boxplot_final')
    if not cv_results or not any(cv_results.values()):
        logger.warning(f"{task} Task 没有结果，跳过最终输出")
        return

    output_dir = Path(config.base_dir) / 'output' / 'boxplot_final_ranked' / task
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"最终方案箱型图 - {task} Task")
    logger.info("=" * 60)
    stats_df = plot_accuracy_boxplot_final(cv_results, output_dir, task=task)
    if stats_df is not None:
        logger.info("\n" + stats_df.to_string(index=False))


def main():
    args = parse_args()
    config = get_config()
    logger = get_logger('boxplot_final')

    logger.info("\n>>> 开始最终方案交叉验证...")
    logger.info(f"  模型顺序: {args.models}")
    logger.info(f"  任务顺序: {args.tasks}")
    logger.info(f"  折数: {args.folds}")

    all_results = build_final_cv_results_all_tasks(
        config,
        selected_models=args.models,
        selected_tasks=args.tasks,
        selected_folds=args.folds,
    )

    if not args.skip_final_plots:
        for task in args.tasks:
            process_task_final(config, task=task, cv_results=all_results[task])

    if not args.skip_stage_boxplot:
        logger.info(f"\n>>> 生成 {PRIMARY_MODEL_NAME} 训练轮次箱型图 (PP)...")
        X_pp, y_pp = load_all_data(config, task='PP')
        output_dir_pp = Path(config.base_dir) / 'output' / 'boxplot_final_ranked' / 'PP'
        output_dir_pp.mkdir(parents=True, exist_ok=True)
        stages_data = get_training_stages_data(X_pp, y_pp, task='PP')
        plot_training_iterations_boxplot(stages_data, output_dir_pp, task='PP')


if __name__ == '__main__':
    main()
