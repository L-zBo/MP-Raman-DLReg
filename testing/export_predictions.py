"""
预测概率导出脚本。

功能：
- 导出代表性像素的预测概率
- 输出 CSV

输出：
- output/PredictedProbabilities/train_predictions.csv
- output/PredictedProbabilities/test_predictions.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.dual_head_model import DualHeadRamanCNNLSTM
from utils import get_config, get_logger

logger = get_logger(__name__)


def load_test_data(config):
    """加载测试数据集。"""
    preprocessed_dir = Path(config['paths']['preprocessed_dir'])

    test_samples = [
        'pp_mixed1', 'pp_mixed2', 'pp_mixed3', 'pp_mixed4', 'pp_mixed5',
        'pe_mixed1', 'pe_mixed2', 'pe_mixed3', 'pe_mixed4', 'pe_mixed5'
    ]

    samples_data = {}

    for sample_name in test_samples:
        data_path = preprocessed_dir / f'{sample_name}_data.npy'
        pp_labels_path = preprocessed_dir / f'{sample_name}_pp_labels.npy'
        pe_labels_path = preprocessed_dir / f'{sample_name}_pe_labels.npy'

        if not data_path.exists():
            logger.warning(f"未找到样本数据: {data_path}")
            continue

        data = np.load(data_path)
        pp_labels = np.load(pp_labels_path)
        pe_labels = np.load(pe_labels_path)

        samples_data[sample_name] = {
            'data': data,
            'pp_labels': pp_labels,
            'pe_labels': pe_labels
        }

    return samples_data


def select_representative_pixels(pp_labels, pe_labels, num_per_category=10):
    """从每个污染等级中选择代表性像素。"""
    H, W = pp_labels.shape

    selected_pixels = []

    for level in [0, 1, 2]:
        mask = (pp_labels == level) | (pe_labels == level)

        if np.any(mask):
            coords = np.argwhere(mask)

            if len(coords) > num_per_category:
                indices = np.random.choice(len(coords), num_per_category, replace=False)
                selected_coords = coords[indices]
            else:
                selected_coords = coords

            level_names = {0: '无污染', 1: '轻度污染', 2: '重度污染'}
            level_name = level_names[level]

            for y, x in selected_coords:
                selected_pixels.append((y, x, level_name))

    return selected_pixels


def predict_probabilities(model, spectra, device):
    """预测 PP / PE 类别概率与标签。"""
    model.eval()

    dataset = TensorDataset(torch.FloatTensor(spectra))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    pp_probs_list = []
    pe_probs_list = []

    with torch.no_grad():
        for (batch_spectra,) in dataloader:
            batch_spectra = batch_spectra.to(device)

            pp_logits, pe_logits = model(batch_spectra)
            pp_probs = torch.softmax(pp_logits, dim=1)
            pe_probs = torch.softmax(pe_logits, dim=1)

            pp_probs_list.append(pp_probs.cpu().numpy())
            pe_probs_list.append(pe_probs.cpu().numpy())

    pp_probs = np.vstack(pp_probs_list)
    pe_probs = np.vstack(pe_probs_list)

    pp_preds = np.argmax(pp_probs, axis=1)
    pe_preds = np.argmax(pe_probs, axis=1)

    return pp_probs, pe_probs, pp_preds, pe_preds


def export_predictions_to_csv(samples_data, model, device, output_path, is_test_set=False):
    """导出预测概率到 CSV。"""
    records = []

    for sample_name, sample_info in samples_data.items():
        data = sample_info['data']  # [H, W, 1024]
        pp_labels = sample_info['pp_labels']  # [H, W]
        pe_labels = sample_info['pe_labels']  # [H, W]

        # 选择代表性像素
        selected_pixels = select_representative_pixels(pp_labels, pe_labels, num_per_category=10)

        logger.info(f"样本 {sample_name}: 选择了 {len(selected_pixels)} 个像素")

        # 提取选中像素的光谱
        spectra_list = []
        pixel_info_list = []

        for y, x, pollution_level in selected_pixels:
            spectrum = data[y, x, :]
            spectra_list.append(spectrum)
            pixel_info_list.append({
                'sample_name': sample_name,
                'true_label_PP': pp_labels[y, x],
                'true_label_PE': pe_labels[y, x]
            })

        if len(spectra_list) == 0:
            continue

        spectra = np.array(spectra_list)  # [N, 1024]

        # 预测概率
        pp_probs, pe_probs, pp_preds, pe_preds = predict_probabilities(model, spectra, device)

        # 构建记录
        for i, pixel_info in enumerate(pixel_info_list):
            # 判断是否同时检测到PP和PE污染（预测标签都>0）
            both_detected = 'Yes' if (pp_preds[i] > 0 and pe_preds[i] > 0) else 'No'

            record = {
                'sample_name': pixel_info['sample_name'],
                'true_label_PP': pixel_info['true_label_PP'],
                'true_label_PE': pixel_info['true_label_PE'],
                'pred_label_PP': pp_preds[i],
                'pred_label_PE': pe_preds[i],
                'prob_PP_class0': round(pp_probs[i, 0], 4),
                'prob_PP_class1': round(pp_probs[i, 1], 4),
                'prob_PP_class2': round(pp_probs[i, 2], 4),
                'prob_PE_class0': round(pe_probs[i, 0], 4),
                'prob_PE_class1': round(pe_probs[i, 1], 4),
                'prob_PE_class2': round(pe_probs[i, 2], 4),
                'confidence_PP': round(np.max(pp_probs[i]), 4),
                'confidence_PE': round(np.max(pe_probs[i]), 4),
                'both_PP_PE_detected': both_detected
            }
            records.append(record)

    # 创建DataFrame并保存
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    logger.info(f"预测概率已保存到: {output_path}")
    logger.info(f"总共导出了 {len(records)} 条记录")

    # 统计同时检测到PP和PE的像素数量
    both_count = df[df['both_PP_PE_detected'] == 'Yes'].shape[0]
    logger.info(f"同时检测到PP和PE污染的像素: {both_count} 个 ({both_count/len(records)*100:.2f}%)")


def load_test_set_data(config):
    """加载测试集数据"""
    preprocessed_dir = Path(config['paths']['preprocessed_dir'])
    true_label_dir = Path(__file__).parent / 'test_true_label'

    test_samples = ['pp_test', 'pe_test', 'pp_pe_mixed_test']
    samples_data = {}

    for sample_name in test_samples:
        data_path = preprocessed_dir / f'{sample_name}_data.npy'
        pp_labels_path = true_label_dir / f'{sample_name}' / 'pp_labels.npy'
        pe_labels_path = true_label_dir / f'{sample_name}' / 'pe_labels.npy'

        if not data_path.exists():
            logger.warning(f"未找到测试集数据: {data_path}")
            continue

        if not pp_labels_path.exists() or not pe_labels_path.exists():
            logger.warning(f"未找到测试集标签: {pp_labels_path} 或 {pe_labels_path}")
            continue

        data = np.load(data_path)  # [H, W, 1024]
        pp_labels = np.load(pp_labels_path)  # [H, W]
        pe_labels = np.load(pe_labels_path)  # [H, W]

        samples_data[sample_name] = {
            'data': data,
            'pp_labels': pp_labels,
            'pe_labels': pe_labels
        }

    return samples_data


def main():
    """主函数"""
    config = get_config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载模型
    model_path = Path(config.paths['output_dir']) / 'models' / 'best_model.pth'
    if not model_path.exists():
        logger.error(f"未找到模型文件: {model_path}")
        return

    model = DualHeadRamanCNNLSTM(input_len=1024, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    logger.info(f"已加载模型: {model_path}")

    output_base_dir = Path(config.paths['output_dir']) / 'PredictedProbabilities'

    # 1. 导出训练集样本的预测概率
    logger.info("\n=== 处理训练集样本 ===")
    logger.info("加载训练集数据...")
    train_samples_data = load_test_data(config)
    logger.info(f"加载了 {len(train_samples_data)} 个训练集样本")

    train_output_path = output_base_dir / 'train_predictions.csv'
    logger.info("开始导出训练集预测概率...")
    export_predictions_to_csv(train_samples_data, model, device, train_output_path, is_test_set=False)

    # 2. 导出测试集的预测概率
    logger.info("\n=== 处理测试集样本 ===")
    logger.info("加载测试集数据...")
    test_samples_data = load_test_set_data(config)
    logger.info(f"加载了 {len(test_samples_data)} 个测试集样本")

    if len(test_samples_data) > 0:
        test_output_path = output_base_dir / 'test_predictions.csv'
        logger.info("开始导出测试集预测概率...")
        export_predictions_to_csv(test_samples_data, model, device, test_output_path, is_test_set=True)

    logger.info("\n完成！")


if __name__ == '__main__':
    main()
