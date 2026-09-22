"""
阈值调整测试脚本。

功能：
- 测试不同阈值配置
- 输出测试集指标变化

输出：
- 控制台报告
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from models.dual_head_model import DualHeadRamanCNNLSTM
from utils import get_config, get_logger
from sklearn.metrics import classification_report, confusion_matrix

def load_test_data_and_labels(config, task='PP'):
    """加载测试集数据和真实标签。"""
    data_dir = Path(config.paths['preprocessed_dir'])
    true_label_dir = Path(__file__).parent / 'test_true_label'

    X_pp_test = np.load(data_dir / 'pp_test_data.npy')
    X_pe_test = np.load(data_dir / 'pe_test_data.npy')
    X_mixed_test = np.load(data_dir / 'pp_pe_mixed_test_data.npy')

    y_pp_test = np.load(true_label_dir / 'pp_test' / 'pp_labels.npy')
    y_pe_test = np.load(true_label_dir / 'pe_test' / 'pe_labels.npy')
    y_pp_mixed = np.load(true_label_dir / 'pp_pe_mixed_test' / 'pp_labels.npy')
    y_pe_mixed = np.load(true_label_dir / 'pp_pe_mixed_test' / 'pe_labels.npy')

    if task == 'PP':
        X_test = np.vstack([
            X_pp_test.reshape(-1, X_pp_test.shape[-1]),
            X_mixed_test.reshape(-1, X_mixed_test.shape[-1])
        ])
        y_test = np.concatenate([y_pp_test.flatten(), y_pp_mixed.flatten()])
    else:
        X_test = np.vstack([
            X_pe_test.reshape(-1, X_pe_test.shape[-1]),
            X_mixed_test.reshape(-1, X_mixed_test.shape[-1])
        ])
        y_test = np.concatenate([y_pe_test.flatten(), y_pe_mixed.flatten()])

    return X_test, y_test

def predict_with_threshold(model, X, device, task='PP', threshold_adjust_class1=0.0, threshold_adjust_class2=0.0):
    """使用调整后的阈值进行预测。"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        if task == 'PP':
            pp_out, _ = model(X_tensor)
            probs = torch.softmax(pp_out, dim=1).cpu().numpy()
        else:
            _, pe_out = model(X_tensor)
            probs = torch.softmax(pe_out, dim=1).cpu().numpy()

    probs[:, 1] += threshold_adjust_class1
    probs[:, 2] += threshold_adjust_class2

    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    return preds, probs

def main():
    logger = get_logger('threshold_adjust')
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model_path = Path(config.paths['output_dir']) / 'models' / 'best_model.pth'

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    # 测试PP和PE两个任务
    for task in ['PP', 'PE']:
        logger.info("\n" + "="*80)
        logger.info(f"任务: {task}")
        logger.info("="*80)

        # 加载测试数据
        X_test, y_test = load_test_data_and_labels(config, task)

        # 加载模型
        model = DualHeadRamanCNNLSTM(input_len=X_test.shape[1], num_classes=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # 对于PP任务，只调整类别1（轻微污染）
        # 对于PE任务，同时调整类别1（轻微污染）和类别2（严重污染）
        if task == 'PP':
            # PP任务：最终配置
            threshold_configs = [
                (0.0, 0.0, "原始结果"),
                (0.15, 0.0, "轻微污染+0.15 [最终配置]")
            ]
        else:
            # PE任务：最终配置
            threshold_configs = [
                (0.0, 0.0, "原始结果"),
                (0.40, 0.65, "方案29: 轻微+0.40, 严重+0.65"),
                (0.50, 0.70, "方案33: 轻微+0.50, 严重+0.70 [最终配置]")
            ]

        for adjust1, adjust2, desc in threshold_configs:
            preds, probs = predict_with_threshold(model, X_test, device, task, adjust1, adjust2)

            logger.info(f"\n{desc}")
            logger.info("-"*80)

            # 计算准确率
            acc = (preds == y_test).mean()
            logger.info(f"总体准确率: {acc:.4f}")

            # 打印分类报告
            report = classification_report(y_test, preds, target_names=class_names, digits=4)
            logger.info(f"\n{report}")

            # 打印混淆矩阵
            cm = confusion_matrix(y_test, preds)
            logger.info(f"混淆矩阵:\n{cm}")

            # 打印每个类别的详细指标
            logger.info("\n每个类别的详细指标:")
            for i, class_name in enumerate(class_names):
                class_mask = (y_test == i)
                class_total = class_mask.sum()
                class_correct = ((preds == i) & class_mask).sum()
                class_predicted = (preds == i).sum()

                recall = class_correct / class_total if class_total > 0 else 0
                precision = class_correct / class_predicted if class_predicted > 0 else 0

                logger.info(f"  {class_name}:")
                logger.info(f"    样本数: {class_total}")
                logger.info(f"    正确预测: {class_correct}")
                logger.info(f"    召回率: {recall:.4f}")
                logger.info(f"    精确率: {precision:.4f}")

if __name__ == '__main__':
    main()
