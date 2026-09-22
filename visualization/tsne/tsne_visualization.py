"""
t-SNE降维可视化：
将高维特征降维到2D/3D空间，展示不同类别的分布

使用固定测试集（3200样本 = 1600原始测试 + 1600混合测试）
使用双头模型（DualHeadRamanCNNLSTM）提取特征，分别对 PP 和 PE 任务进行可视化

输出位置：output/tsne/PP/ 和 output/tsne/PE/
输出内容：
  - tsne_original_2d.png      原始特征2D
  - tsne_original_3d.png      原始特征3D
  - tsne_model_features_2d.png 模型特征2D
  - tsne_model_features_3d.png 模型特征3D
  - tsne_comparison.png       对比图
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

# sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_config, get_logger, PRIMARY_MODEL_NAME

# 设置中文字体
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
    logger = get_logger('tsne')

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
    logger = get_logger('tsne')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not model_path.exists():
        logger.warning(f"模型文件不存在: {model_path}，使用原始特征")
        return X

    # 加载双头模型
    model = DualHeadRamanCNNLSTM(input_len=X.shape[1], num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 提取注意力加权后的特征（与分类器输入相同）
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
                attn_w = F.softmax(model.pp_attn(lstm_out), dim=1)  # [B, L/16, 1]
            else:
                attn_w = F.softmax(model.pe_attn(lstm_out), dim=1)  # [B, L/16, 1]

            x = (lstm_out * attn_w).sum(dim=1)  # [B, 128] - 注意力加权特征

            features.append(x.cpu().numpy())

    features = np.vstack(features)
    logger.info(f"提取特征维度: {features.shape}")
    return features


def plot_tsne(X: np.ndarray, y: np.ndarray, output_path: Path, title: str = 't-SNE Visualization',
              use_pca: bool = True, pca_components: int = 50):
    """
    绘制t-SNE图

    Args:
        X: 输入特征
        y: 标签
        output_path: 输出路径
        title: 图表标题
        use_pca: 是否先用PCA预降维（推荐对高维数据使用）
        pca_components: PCA目标维度
    """
    logger = get_logger('tsne')
    from sklearn.decomposition import PCA

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA预降维（减少噪声，加速t-SNE，避免"蛇形"伪影）
    if use_pca and X_scaled.shape[1] > pca_components:
        logger.info(f"PCA预降维: {X_scaled.shape[1]}D → {pca_components}D")
        pca = PCA(n_components=pca_components, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        logger.info(f"PCA解释方差: {pca.explained_variance_ratio_.sum():.2%}")

    # t-SNE降维（优化参数避免蛇形伪影）
    logger.info("执行t-SNE降维...")
    tsne = TSNE(
        n_components=2,
        perplexity=100,           # 增大perplexity，考虑更大邻域
        early_exaggeration=24,    # 增大early_exaggeration，聚类更紧凑
        learning_rate='auto',     # 自动调整学习率
        max_iter=2000,            # 增加迭代次数确保收敛
        init='pca',               # 用PCA初始化，更稳定
        random_state=42,
        n_jobs=-1
    )
    X_tsne = tsne.fit_transform(X_scaled)

    # 类别名称和颜色
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=color, marker=marker, label=name,
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"t-SNE图已保存: {output_path}")


def plot_tsne_3d(X: np.ndarray, y: np.ndarray, output_path: Path, title: str = 't-SNE 3D',
                 use_pca: bool = True, pca_components: int = 50, task: str = 'PP'):
    """
    绘制3D t-SNE图

    Args:
        X: 输入特征
        y: 标签
        output_path: 输出路径
        title: 图表标题
        use_pca: 是否先用PCA预降维
        pca_components: PCA目标维度
        task: 'PP' 或 'PE'，用于设置不同的视角
    """
    logger = get_logger('tsne')
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA预降维
    if use_pca and X_scaled.shape[1] > pca_components:
        logger.info(f"PCA预降维: {X_scaled.shape[1]}D → {pca_components}D")
        pca = PCA(n_components=pca_components, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        logger.info(f"PCA解释方差: {pca.explained_variance_ratio_.sum():.2%}")

    # t-SNE降维到3D
    logger.info("执行t-SNE降维（3D）...")
    tsne = TSNE(
        n_components=3,
        perplexity=100,
        early_exaggeration=24,
        learning_rate='auto',
        max_iter=2000,
        init='random',  # 3D时使用random初始化
        random_state=42,
        n_jobs=-1
    )
    X_tsne = tsne.fit_transform(X_scaled)

    # 类别名称和颜色
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    # 绘图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], X_tsne[mask, 2],
                   c=color, marker=marker, label=name,
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.3)

    ax.set_xlabel('t-SNE Dim 1', fontsize=10)
    ax.set_ylabel('t-SNE Dim 2', fontsize=10)
    ax.set_zlabel('t-SNE Dim 3', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)

    # 设置视角：PE任务使用不同的视角以获得更好的可视化效果
    if task == 'PE':
        ax.view_init(elev=25, azim=45)  # PE: 仰角25°, 方位角45°
    else:
        ax.view_init(elev=30, azim=-60)  # PP: 默认视角

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"t-SNE 3D图已保存: {output_path}")


def plot_tsne_comparison(X_raw: np.ndarray, X_features: np.ndarray, y: np.ndarray,
                         output_dir: Path, task: str = 'PP'):
    """对比原始特征和模型特征的t-SNE（优化版）"""
    logger = get_logger('tsne')
    from sklearn.decomposition import PCA

    # ========== 原始特征处理 ==========
    scaler_raw = StandardScaler()
    X_raw_scaled = scaler_raw.fit_transform(X_raw)

    # PCA预降维（1024D → 50D）
    logger.info("原始特征PCA预降维: 1024D → 50D")
    pca_raw = PCA(n_components=50, random_state=42)
    X_raw_pca = pca_raw.fit_transform(X_raw_scaled)
    logger.info(f"原始特征PCA解释方差: {pca_raw.explained_variance_ratio_.sum():.2%}")

    # ========== 模型特征处理 ==========
    scaler_feat = StandardScaler()
    X_feat_scaled = scaler_feat.fit_transform(X_features)

    # ========== t-SNE降维 ==========
    tsne_params = {
        'n_components': 2,
        'perplexity': 100,           # 增大邻域范围
        'early_exaggeration': 24,    # 增强聚类紧凑性
        'learning_rate': 'auto',
        'max_iter': 2000,            # 增加迭代
        'init': 'pca',               # PCA初始化
        'random_state': 42,
        'n_jobs': -1
    }

    logger.info("执行t-SNE降维（原始特征）...")
    tsne_raw = TSNE(**tsne_params)
    X_tsne_raw = tsne_raw.fit_transform(X_raw_pca)

    logger.info("执行t-SNE降维（模型特征）...")
    tsne_feat = TSNE(**tsne_params)
    X_tsne_feat = tsne_feat.fit_transform(X_feat_scaled)

    # 类别名称和颜色
    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 原始特征
    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        axes[0].scatter(X_tsne_raw[mask, 0], X_tsne_raw[mask, 1],
                        c=color, marker=marker, label=name,
                        alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].set_title('Original Features', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 模型特征
    for i, (name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = y == i
        axes[1].scatter(X_tsne_feat[mask, 0], X_tsne_feat[mask, 1],
                        c=color, marker=marker, label=name,
                        alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_title(f'{PRIMARY_MODEL_NAME} Features ({task})', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f't-SNE Comparison: Original vs Model Features ({task} Task, n={len(y)})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'tsne_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"t-SNE对比图已保存: {output_dir / 'tsne_comparison.png'}")


def main():
    """主函数"""
    config = get_config()
    logger = get_logger('tsne')

    logger.info("=" * 60)
    logger.info("t-SNE降维可视化（使用测试集 + 双头模型）")
    logger.info("=" * 60)

    model_path = Path(config.paths['output_dir']) / 'models' / 'best_model.pth'

    for task in ['PP', 'PE']:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理 {task} 任务")
        logger.info(f"{'='*60}")

        output_dir = Path(config.paths['output_dir']) / 'tsne' / task
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

        # 绘制原始特征t-SNE（2D）
        logger.info("\n>>> 绘制原始特征t-SNE 2D...")
        plot_tsne(X_sample, y_sample, output_dir / 'tsne_original_2d.png',
                  title=f't-SNE 2D: Original Spectral Features ({task} Task, n={len(y_sample)})')

        # 绘制原始特征t-SNE（3D）
        logger.info("\n>>> 绘制原始特征t-SNE 3D...")
        plot_tsne_3d(X_sample, y_sample, output_dir / 'tsne_original_3d.png',
                     title=f't-SNE 3D: Original Spectral Features ({task} Task, n={len(y_sample)})',
                     task=task)

        # 提取模型特征
        logger.info("\n>>> 提取双头模型特征...")
        X_features = extract_features(X_sample, model_path, task)

        # 绘制模型特征t-SNE（2D）
        logger.info("\n>>> 绘制模型特征t-SNE 2D...")
        plot_tsne(X_features, y_sample, output_dir / 'tsne_model_features_2d.png',
                  title=f't-SNE 2D: {PRIMARY_MODEL_NAME} Features ({task} Task, n={len(y_sample)})')

        # 绘制模型特征t-SNE（3D）
        logger.info("\n>>> 绘制模型特征t-SNE 3D...")
        plot_tsne_3d(X_features, y_sample, output_dir / 'tsne_model_features_3d.png',
                     title=f't-SNE 3D: {PRIMARY_MODEL_NAME} Features ({task} Task, n={len(y_sample)})',
                     task=task)

        # 绘制对比图
        logger.info("\n>>> 绘制对比图...")
        plot_tsne_comparison(X_sample, X_features, y_sample, output_dir, task)

        logger.info(f"\n>>> {task} 任务完成！")

    logger.info("\n" + "=" * 60)
    logger.info("所有任务完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
