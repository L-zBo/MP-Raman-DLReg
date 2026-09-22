"""
ROC曲线（Receiver Operating Characteristic）：
展示真阳性率-假阳性率权衡关系，计算AUC值

使用固定测试集（3200样本 = 1600原始测试 + 1600混合测试）
直接加载预训练模型（无需重新训练），用于当前最终对比方案的 ROC/AUC 可视化

输出位置：output/roc_curve/PP/ 和 output/roc_curve/PE/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
import joblib

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# sklearn
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc

# PyTorch
import torch
import torch.nn.functional as F

from utils import (
    get_config,
    get_logger,
    PRIMARY_MODEL_NAME,
    BEST_CANDIDATE_CONFIGS,
    FINAL_COMPARISON_MODEL_ORDER,
    FINAL_RESNET_CHOICE,
    FINAL_RESNET_CONFIG,
)

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 导入模型类
from models.base_models import (
    SMARTNIRClassifier,
    ConvTranClassifier,
    MambaHSIClassifier,
    ResNet50_1D,
)
from models.dual_head_model import DualHeadRamanCNNLSTM

# 阈值调整参数
# 注意：ROC曲线使用原始概率，不应用阈值调整，以保持曲线的公平性和可比性
# 阈值调整仅用于最终分类决策（如混淆矩阵），不影响ROC/AUC计算
PP_THRESHOLD_CLASS1 = 0.15
PP_THRESHOLD_CLASS2 = 0.0
PE_THRESHOLD_CLASS1 = 0.50
PE_THRESHOLD_CLASS2 = 0.70


def load_test_data(config, task='PP') -> Tuple[np.ndarray, np.ndarray]:
    """
    加载测试数据（3200样本 = 1600原始测试 + 1600混合测试）
    """
    logger = get_logger('roc_curve')

    preprocessed_dir = Path(config['paths']['preprocessed_dir'])
    true_label_dir = Path(__file__).parent.parent.parent / 'testing' / 'test_true_label'

    label_paths = {
        'pp': true_label_dir / 'pp_test' / 'pp_labels.npy',
        'pe': true_label_dir / 'pe_test' / 'pe_labels.npy',
        'pp_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy',
        'pe_mixed': true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy'
    }

    if task == 'PP':
        pp_data = np.load(preprocessed_dir / 'pp_test_data.npy')
        pp_labels = np.load(label_paths['pp'])
        X_pp_only = pp_data.reshape(-1, pp_data.shape[-1])
        y_pp_only = pp_labels.flatten()

        mixed_data = np.load(preprocessed_dir / 'pp_pe_mixed_test_data.npy')
        pp_mixed_labels = np.load(label_paths['pp_mixed'])
        X_mixed = mixed_data.reshape(-1, mixed_data.shape[-1])
        y_pp_mixed = pp_mixed_labels.flatten()

        X_test = np.vstack([X_pp_only, X_mixed])
        y_test = np.concatenate([y_pp_only, y_pp_mixed])
    else:
        pe_data = np.load(preprocessed_dir / 'pe_test_data.npy')
        pe_labels = np.load(label_paths['pe'])
        X_pe_only = pe_data.reshape(-1, pe_data.shape[-1])
        y_pe_only = pe_labels.flatten()

        mixed_data = np.load(preprocessed_dir / 'pp_pe_mixed_test_data.npy')
        pe_mixed_labels = np.load(label_paths['pe_mixed'])
        X_mixed = mixed_data.reshape(-1, mixed_data.shape[-1])
        y_pe_mixed = pe_mixed_labels.flatten()

        X_test = np.vstack([X_pe_only, X_mixed])
        y_test = np.concatenate([y_pe_only, y_pe_mixed])

    logger.info(f"[{task}] 测试集: {X_test.shape[0]} 样本 (1600原始 + 1600混合)")
    return X_test, y_test


def load_saved_models(config, task: str = 'PP') -> Dict:
    """
    加载预训练模型
    无需重新训练！
    """
    logger = get_logger('roc_curve')

    model_dir = Path(config.base_dir) / 'output' / 'models' / 'traditional'
    task_suffix = f'_{task.lower()}'

    models = {}

    # 加载 Scaler
    scaler_path = model_dir / f'scaler{task_suffix}.pkl'
    if scaler_path.exists():
        models['scaler'] = joblib.load(scaler_path)
        logger.info(f"  加载 Scaler: {scaler_path.name}")

    # 加载传统模型
    traditional_models = {
        'SVM': f'svm{task_suffix}.pkl',
        'Random Forest': f'random_forest{task_suffix}.pkl',
        'XGBoost': f'xgboost{task_suffix}.pkl',
        'LightGBM': f'lightgbm{task_suffix}.pkl',
    }

    for name, filename in traditional_models.items():
        model_path = model_dir / filename
        if model_path.exists():
            models[name] = joblib.load(model_path)
            logger.info(f"  加载 {name}: {filename}")

    return models


def get_model_probs(models: Dict, X_test: np.ndarray, task: str = 'PP') -> Dict[str, np.ndarray]:
    """
    使用已加载的模型获取预测概率
    """
    logger = get_logger('roc_curve')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = models.get('scaler')
    if scaler is None:
        logger.warning("未找到 Scaler，使用原始数据")
        X_test_scaled = X_test
    else:
        X_test_scaled = scaler.transform(X_test)

    all_probs = {}

    # SVM 特殊处理（使用 decision_function + softmax）
    if 'SVM' in models:
        model = models['SVM']
        logger.info(f"  获取 SVM 概率...")
        try:
            # 尝试使用 predict_proba
            probs = model.predict_proba(X_test_scaled)
        except AttributeError:
            # 如果没有 probability=True，使用 decision_function
            decision = model.decision_function(X_test_scaled)
            # 使用 softmax 转换为概率
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            probs = exp_decision / exp_decision.sum(axis=1, keepdims=True)
        all_probs['SVM'] = probs

    # Random Forest 和 XGBoost（有 predict_proba）
    for name in ['Random Forest', 'XGBoost', 'LightGBM']:
        if name in models:
            model = models[name]
            logger.info(f"  获取 {name} 概率...")
            probs = model.predict_proba(X_test_scaled)
            all_probs[name] = probs

    return all_probs


def _load_candidate_bundle(model_name: str, task: str, input_len: int, config, device: torch.device):
    model_dir = Path(config.base_dir) / 'output' / 'models' / 'traditional'

    if model_name == 'SMART-NIR':
        if task == 'PP':
            kwargs = BEST_CANDIDATE_CONFIGS['SMART-NIR']['pp_best']['model_kwargs']
            model_path = model_dir / 'smart_nir_pp_final.pth'
            scaler_path = model_dir / 'scaler_smart_nir_pp_final.pkl'
        else:
            kwargs = {
                'd_model': 48,
                'num_heads': 4,
                'num_layers': 3,
                'patch_size': 16,
                'branch_channels': 12,
                'dropout': 0.15,
            }
            model_path = model_dir / 'smart_nir_pe_final.pth'
            scaler_path = model_dir / 'scaler_smart_nir_pe_final.pkl'
        model = SMARTNIRClassifier(input_len=input_len, num_classes=3, **kwargs).to(device)
    elif model_name == 'ConvTran':
        if task == 'PP':
            kwargs = BEST_CANDIDATE_CONFIGS['ConvTran']['pp_best']['model_kwargs']
            model_path = model_dir / 'convtran_pp_final.pth'
            scaler_path = model_dir / 'scaler_convtran_pp_final.pkl'
        else:
            kwargs = BEST_CANDIDATE_CONFIGS['ConvTran']['pe_best']['model_kwargs']
            model_path = model_dir / 'convtran_pe_final.pth'
            scaler_path = model_dir / 'scaler_convtran_pe_final.pkl'
        model = ConvTranClassifier(input_len=input_len, num_classes=3, **kwargs).to(device)
    elif model_name == 'MambaHSI':
        if task == 'PP':
            kwargs = BEST_CANDIDATE_CONFIGS['MambaHSI']['pp_best']['model_kwargs']
            model_path = model_dir / 'mambahsi_pp_final.pth'
            scaler_path = model_dir / 'scaler_mambahsi_pp_final.pkl'
        else:
            kwargs = BEST_CANDIDATE_CONFIGS['MambaHSI']['pe_best']['model_kwargs']
            model_path = model_dir / 'mambahsi_pe_final.pth'
            scaler_path = model_dir / 'scaler_mambahsi_pe_final.pkl'
        model = MambaHSIClassifier(input_len=input_len, num_classes=3, **kwargs).to(device)
    elif model_name == FINAL_RESNET_CHOICE:
        if task == 'PP':
            kwargs = FINAL_RESNET_CONFIG['pp_best']['model_kwargs']
            model_path = model_dir / 'resnet50_pp_final.pth'
            scaler_path = model_dir / 'scaler_resnet50_pp_final.pkl'
        else:
            kwargs = FINAL_RESNET_CONFIG['pe_best']['model_kwargs']
            model_path = model_dir / 'resnet50_pe_final.pth'
            scaler_path = model_dir / 'scaler_resnet50_pe_final.pkl'
        model = ResNet50_1D(input_len=input_len, num_classes=3, **kwargs).to(device)
    else:
        return None

    if not model_path.exists() or not scaler_path.exists():
        return None

    scaler = joblib.load(scaler_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, scaler


def get_final_candidate_probs(X_test: np.ndarray, config, task: str = 'PP') -> Dict[str, np.ndarray]:
    logger = get_logger('roc_curve')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for model_name in ['SMART-NIR', 'ConvTran', 'MambaHSI', FINAL_RESNET_CHOICE]:
        bundle = _load_candidate_bundle(model_name, task, X_test.shape[1], config, device)
        if bundle is None:
            logger.warning(f"  未找到 {model_name} 的最终方案权重或 scaler")
            continue

        model, scaler = bundle
        X_scaled = scaler.transform(X_test)
        X_t = torch.FloatTensor(X_scaled).to(device)
        logger.info(f"  获取 {model_name} 概率...")
        with torch.no_grad():
            outputs = model(X_t)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
        results[model_name] = probs

    return results


def get_cnn_lstm_probs(X_test: np.ndarray, config, task: str = 'PP') -> np.ndarray:
    """
    获取预训练 RADAR-Net 的预测概率
    使用原始概率（不应用阈值调整，保持ROC曲线的公平性）
    """
    logger = get_logger('roc_curve')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = Path(config.base_dir) / 'output' / 'models' / 'best_model.pth'

    if not model_path.exists():
        logger.warning(f"未找到预训练模型: {model_path}")
        return None

    model = DualHeadRamanCNNLSTM(input_len=X_test.shape[1], num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        pp_outputs, pe_outputs = model(X_test_t)
        if task == 'PP':
            probs = torch.softmax(pp_outputs, dim=1).cpu().numpy()
        else:
            probs = torch.softmax(pe_outputs, dim=1).cpu().numpy()

    return probs


def plot_roc_curves(all_probs: Dict[str, np.ndarray], y_test: np.ndarray,
                    output_dir: Path, task: str = 'PP'):
    """绘制ROC曲线"""
    logger = get_logger('roc_curve')

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    model_order = FINAL_COMPARISON_MODEL_ORDER
    model_order = [m for m in model_order if m in all_probs]

    model_colors = {
        'SVM': '#FF6B6B',
        'Random Forest': '#4ECDC4',
        'XGBoost': '#96CEB4',
        'LightGBM': '#F39C12',
        'ResNet-50': '#DDA0DD',
        'SMART-NIR': '#5DADE2',
        'ConvTran': '#E67E22',
        'MambaHSI': '#AF7AC5',
        PRIMARY_MODEL_NAME: '#98D8C8'
    }

    auc_scores = {model: [] for model in model_order}

    # 计算AUC
    for model_name in model_order:
        probs = all_probs[model_name]
        for class_idx in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
            roc_auc = auc(fpr, tpr)
            auc_scores[model_name].append(roc_auc)

    # 图1: RADAR-Net 分类别ROC曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_colors = ['#2ecc71', '#3498db', '#e74c3c']

    if PRIMARY_MODEL_NAME in all_probs:
        probs = all_probs[PRIMARY_MODEL_NAME]
        for class_idx in range(n_classes):
            ax = axes[class_idx]
            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], probs[:, class_idx])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=class_colors[class_idx],
                    label=f'{PRIMARY_MODEL_NAME}\n(AUC={roc_auc:.3f})', linewidth=2.5)
            ax.fill_between(fpr, tpr, alpha=0.2, color=class_colors[class_idx])
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'{class_names[class_idx]}', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])

    plt.suptitle(f'{PRIMARY_MODEL_NAME} ROC Curves by Class ({task} Task, n={len(y_test)})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"保存: {output_dir / 'roc_curves_by_class.png'}")

    # 图2: 所有模型 Micro-Average ROC曲线
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name in model_order:
        probs = all_probs[model_name]
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)

        linewidth = 3 if model_name == PRIMARY_MODEL_NAME else 2
        ax.plot(fpr, tpr, color=model_colors[model_name],
                label=f'{model_name} (AUC={roc_auc:.3f})', linewidth=linewidth)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'Micro-Average ROC Curve ({task} Task, n={len(y_test)})', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_average.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"保存: {output_dir / 'roc_curves_average.png'}")

    # 图3: AUC对比柱状图
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(model_order))
    width = 0.25

    bar_colors = ['#2ecc71', '#3498db', '#e74c3c']
    for i, class_name in enumerate(class_names):
        aucs = [auc_scores[model][i] for model in model_order]
        bars = ax.bar(x + i * width, aucs, width, label=class_name, color=bar_colors[i])

        for j, v in enumerate(aucs):
            ax.text(x[j] + i * width, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title(f'AUC Comparison by Model and Class ({task} Task)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_order, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    plt.savefig(output_dir / 'auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"保存: {output_dir / 'auc_comparison.png'}")

    # 保存AUC数据
    auc_data = []
    for model_name in model_order:
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), all_probs[model_name].ravel())
        row = {
            'Model': model_name,
            'AUC_NonPollution': auc_scores[model_name][0],
            'AUC_SlightPollution': auc_scores[model_name][1],
            'AUC_SeverePollution': auc_scores[model_name][2],
            'AUC_Micro': auc(fpr, tpr)
        }
        auc_data.append(row)

    df = pd.DataFrame(auc_data)
    df = df.sort_values('AUC_Micro', ascending=False)
    df.to_csv(output_dir / 'auc_scores.csv', index=False)
    logger.info(f"保存: {output_dir / 'auc_scores.csv'}")

    return auc_scores


def main():
    """主函数"""
    config = get_config()
    logger = get_logger('roc_curve')

    logger.info("=" * 60)
    logger.info("ROC曲线分析（加载已训练模型，无需重新训练）")
    logger.info("=" * 60)

    for task in ['PP', 'PE']:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理 {task} 任务")
        logger.info(f"{'='*60}")

        output_dir = Path(config.paths['output_dir']) / 'roc_curve' / task
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载测试数据
        logger.info(f"\n>>> 加载 {task} 测试数据...")
        X_test, y_test = load_test_data(config, task)

        unique, counts = np.unique(y_test, return_counts=True)
        logger.info(f"  测试集类别分布: {dict(zip(unique, counts))}")

        # 加载已保存的模型（无需重新训练！）
        logger.info(f"\n>>> 加载已训练的模型...")
        models = load_saved_models(config, task)

        # 获取模型概率
        logger.info(f"\n>>> 获取模型预测概率...")
        all_probs = get_model_probs(models, X_test, task)

        # 加载最终候选模型
        logger.info(f"  获取最终候选模型概率...")
        all_probs.update(get_final_candidate_probs(X_test, config, task))

        # 加载 RADAR-Net
        logger.info(f"  获取 {PRIMARY_MODEL_NAME} 概率...")
        cnn_lstm_probs = get_cnn_lstm_probs(X_test, config, task)
        if cnn_lstm_probs is not None:
            all_probs[PRIMARY_MODEL_NAME] = cnn_lstm_probs

        # 绘制ROC曲线
        logger.info(f"\n>>> 绘制 {task} ROC曲线...")
        plot_roc_curves(all_probs, y_test, output_dir, task)

        logger.info(f"\n>>> {task} 任务完成！")

    logger.info("\n" + "=" * 60)
    logger.info("所有任务完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
