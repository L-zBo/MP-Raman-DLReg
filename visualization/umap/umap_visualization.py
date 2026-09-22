"""
UMAP降维可视化：
将高维特征降维到2D/3D空间，展示不同类别的分布

使用固定测试集（3200样本 = 1600原始测试 + 1600混合测试）
使用双头模型（DualHeadRamanCNNLSTM）提取特征，分别对 PP 和 PE 任务进行可视化

输出位置：output/umap/PP/ 和 output/umap/PE/

UMAP vs t-SNE:
- UMAP 更快，更好地保持全局结构
- UMAP 产生更紧凑的聚类，避免"蛇形"伪影
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# sklearn & umap
from sklearn.preprocessing import StandardScaler
import umap

# PyTorch
import torch
import torch.nn.functional as F

from utils import get_config, get_logger, PRIMARY_MODEL_NAME

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 导入双头模型
from models.dual_head_model import DualHeadRamanCNNLSTM


def load_test_data(config, task='PP') -> Tuple[np.ndarray, np.ndarray]:
    """
    加载固定测试集（3200样本 = 1600原始测试 + 1600混合测试）
    - PP任务: pp_test(1600) + pp_pe_mixed_test(1600, PP标签) = 3200样本
    - PE任务: pe_test(1600) + pp_pe_mixed_test(1600, PE标签) = 3200样本
    """
    logger = get_logger('umap')

    preprocessed_dir = Path(config['paths']['preprocessed_dir'])
    true_label_dir = Path(__file__).parent.parent.parent / 'testing' / 'test_true_label'

    # 混合测试集数据（PP和PE共用同一份数据，但标签不同）
    mixed_data_path = preprocessed_dir / 'pp_pe_mixed_test_data.npy'
    X_mixed = np.load(mixed_data_path).reshape(-1, 1024)

    if task == 'PP':
        # PP任务：加载PP测试数据 + 混合测试数据(PP标签)
        pp_test_path = preprocessed_dir / 'pp_test_data.npy'
        pp_test_labels = true_label_dir / 'pp_test' / 'pp_labels.npy'
        pp_mixed_labels = true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy'

        X_test = np.load(pp_test_path).reshape(-1, 1024)
        y_test = np.load(pp_test_labels).flatten()
        y_mixed = np.load(pp_mixed_labels).flatten()

        X_combined = np.vstack([X_test, X_mixed])
        y_combined = np.concatenate([y_test, y_mixed])
    else:
        # PE任务：加载PE测试数据 + 混合测试数据(PE标签)
        pe_test_path = preprocessed_dir / 'pe_test_data.npy'
        pe_test_labels = true_label_dir / 'pe_test' / 'pe_labels.npy'
        pe_mixed_labels = true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy'

        X_test = np.load(pe_test_path).reshape(-1, 1024)
        y_test = np.load(pe_test_labels).flatten()
        y_mixed = np.load(pe_mixed_labels).flatten()

        X_combined = np.vstack([X_test, X_mixed])
        y_combined = np.concatenate([y_test, y_mixed])

    logger.info(f"[{task}] 测试集: {X_combined.shape[0]} 样本 (原始1600 + 混合1600)")
    return X_combined, y_combined


def extract_features(X: np.ndarray, model_path: Path, task: str = 'PP') -> np.ndarray:
    """
    使用双头模型提取特征（注意力加权后的特征）

    Args:
        X: 输入数据
        model_path: 双头模型路径
        task: 'PP' 或 'PE'，决定使用哪个头的特征

    Returns:
        注意力加权后的特征向量
    """
    logger = get_logger('umap')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not model_path.exists():
        logger.warning(f"模型文件不存在: {model_path}，使用原始特征")
        return X

    # 加载双头模型
    model = DualHeadRamanCNNLSTM(input_len=X.shape[1], num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 提取注意力加权后的特征
    features = []
    batch_size = 256

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(device)

            # 通过共享的CNN
            x = batch.unsqueeze(1)  # [B, 1, L]
            x = model.conv(x)  # [B, 64, L/16]

            # 通过共享的LSTM
            x = x.permute(0, 2, 1)  # [B, L/16, 64]
            lstm_out, _ = model.lstm(x)  # [B, L/16, 128]

            # 使用对应任务的注意力层
            if task == 'PP':
                attn_w = F.softmax(model.pp_attn(lstm_out), dim=1)
            else:
                attn_w = F.softmax(model.pe_attn(lstm_out), dim=1)

            x = (lstm_out * attn_w).sum(dim=1)  # [B, 128]
            features.append(x.cpu().numpy())

    features = np.vstack(features)
    logger.info(f"提取特征维度: {features.shape}")
    return features


def plot_umap_2d(X: np.ndarray, y: np.ndarray, output_path: Path, title: str = 'UMAP 2D'):
    """绘制2D UMAP图"""
    logger = get_logger('umap')

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP降维
    logger.info("执行UMAP降维（2D）...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,        # 邻域大小
        min_dist=0.3,          # 最小距离，越大越分散
        metric='euclidean',
        random_state=42,
        n_jobs=-1
    )
    X_umap = reducer.fit_transform(X_scaled)

    # 类别名称和颜色
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=color, marker=marker, label=name,
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"UMAP 2D图已保存: {output_path}")


def plot_umap_3d(X: np.ndarray, y: np.ndarray, output_path: Path, title: str = 'UMAP 3D'):
    """绘制3D UMAP图"""
    logger = get_logger('umap')
    from mpl_toolkits.mplot3d import Axes3D

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP降维到3D
    logger.info("执行UMAP降维（3D）...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=30,
        min_dist=0.3,
        metric='euclidean',
        random_state=42,
        n_jobs=-1
    )
    X_umap = reducer.fit_transform(X_scaled)

    # 类别名称和颜色
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    # 绘图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], X_umap[mask, 2],
                   c=color, marker=marker, label=name,
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.3)

    ax.set_xlabel('UMAP Dim 1', fontsize=10)
    ax.set_ylabel('UMAP Dim 2', fontsize=10)
    ax.set_zlabel('UMAP Dim 3', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"UMAP 3D图已保存: {output_path}")


def plot_umap_comparison(X_raw: np.ndarray, X_features: np.ndarray, y: np.ndarray,
                         output_dir: Path, task: str = 'PP'):
    """对比原始特征和模型特征的UMAP"""
    logger = get_logger('umap')

    # 标准化
    scaler_raw = StandardScaler()
    X_raw_scaled = scaler_raw.fit_transform(X_raw)

    scaler_feat = StandardScaler()
    X_feat_scaled = scaler_feat.fit_transform(X_features)

    # UMAP参数
    umap_params = {
        'n_components': 2,
        'n_neighbors': 30,
        'min_dist': 0.3,
        'metric': 'euclidean',
        'random_state': 42,
        'n_jobs': -1
    }

    # UMAP降维
    logger.info("执行UMAP降维（原始特征）...")
    reducer_raw = umap.UMAP(**umap_params)
    X_umap_raw = reducer_raw.fit_transform(X_raw_scaled)

    logger.info("执行UMAP降维（模型特征）...")
    reducer_feat = umap.UMAP(**umap_params)
    X_umap_feat = reducer_feat.fit_transform(X_feat_scaled)

    # 类别名称和颜色
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 原始特征
    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        axes[0].scatter(X_umap_raw[mask, 0], X_umap_raw[mask, 1],
                        c=color, marker=marker, label=name,
                        alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    axes[0].set_xlabel('UMAP Dimension 1', fontsize=12)
    axes[0].set_ylabel('UMAP Dimension 2', fontsize=12)
    axes[0].set_title('Original Features', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 模型特征
    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        axes[1].scatter(X_umap_feat[mask, 0], X_umap_feat[mask, 1],
                        c=color, marker=marker, label=name,
                        alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    axes[1].set_xlabel('UMAP Dimension 1', fontsize=12)
    axes[1].set_ylabel('UMAP Dimension 2', fontsize=12)
    axes[1].set_title(f'{PRIMARY_MODEL_NAME} Features ({task})', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'UMAP Comparison: Original vs Model Features ({task} Task, n={len(y)})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"UMAP对比图已保存: {output_dir / 'umap_comparison.png'}")


def main():
    """主函数"""
    config = get_config()
    logger = get_logger('umap')

    logger.info("=" * 60)
    logger.info("UMAP降维可视化（使用测试集 + 双头模型）")
    logger.info("=" * 60)

    model_path = Path(config.paths['output_dir']) / 'models' / 'best_model.pth'

    for task in ['PP', 'PE']:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理 {task} 任务")
        logger.info(f"{'='*60}")

        output_dir = Path(config.paths['output_dir']) / 'umap' / task
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载测试数据
        logger.info("\n>>> 加载测试数据...")
        X, y = load_test_data(config, task)

        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"  测试集类别分布: {dict(zip(unique, counts))}")

        # 采样（如果数据量太大）
        if len(X) > 5000:
            logger.info(f"数据量较大 ({len(X)})，随机采样5000个样本...")
            np.random.seed(42)
            indices = np.random.choice(len(X), 5000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y

        # 绘制原始特征UMAP（2D）
        logger.info("\n>>> 绘制原始特征UMAP 2D...")
        plot_umap_2d(X_sample, y_sample, output_dir / 'umap_original_2d.png',
                     title=f'UMAP 2D: Original Spectral Features ({task} Task, n={len(y_sample)})')

        # 绘制原始特征UMAP（3D）
        logger.info("\n>>> 绘制原始特征UMAP 3D...")
        plot_umap_3d(X_sample, y_sample, output_dir / 'umap_original_3d.png',
                     title=f'UMAP 3D: Original Spectral Features ({task} Task, n={len(y_sample)})')

        # 提取模型特征
        logger.info("\n>>> 提取双头模型特征...")
        X_features = extract_features(X_sample, model_path, task)

        # 绘制模型特征UMAP（2D）
        logger.info("\n>>> 绘制模型特征UMAP 2D...")
        plot_umap_2d(X_features, y_sample, output_dir / 'umap_model_features_2d.png',
                     title=f'UMAP 2D: {PRIMARY_MODEL_NAME} Features ({task} Task, n={len(y_sample)})')

        # 绘制模型特征UMAP（3D）
        logger.info("\n>>> 绘制模型特征UMAP 3D...")
        plot_umap_3d(X_features, y_sample, output_dir / 'umap_model_features_3d.png',
                     title=f'UMAP 3D: {PRIMARY_MODEL_NAME} Features ({task} Task, n={len(y_sample)})')

        # 绘制对比图
        logger.info("\n>>> 绘制对比图...")
        plot_umap_comparison(X_sample, X_features, y_sample, output_dir, task)

        logger.info(f"\n>>> {task} 任务完成！")

    logger.info("\n" + "=" * 60)
    logger.info("所有任务完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
