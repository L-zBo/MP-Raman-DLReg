"""
像素级预测脚本。

功能：
- 生成像素级预测标签
- 输出分类图与测试结果

输出：
- output/test_results_dual/pixel_level/
- output/classification/
"""
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_config, get_logger
from preprocessing.preprocess import load_hyperspectral_data, preprocess_hyperspectral
from unmixing.unmix import unmix_hyperspectral, postprocess_abundance
from models.dual_head_model import DualHeadRamanCNNLSTM
from visualization.classification import save_test_classification_maps


def load_dual_model(checkpoint_path: Path, device: torch.device) -> DualHeadRamanCNNLSTM:
    """加载双头模型"""
    logger = get_logger('test_pixel')

    model = DualHeadRamanCNNLSTM(input_len=1024, num_classes=3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"加载双头模型 (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        logger.info("加载双头模型权重")

    model.to(device)
    model.eval()
    return model


def pixel_level_prediction(
    model: DualHeadRamanCNNLSTM,
    spectra: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    pp_threshold_class1: float = 0.15,
    pp_threshold_class2: float = 0.0,
    pe_threshold_class1: float = 0.50,
    pe_threshold_class2: float = 0.70
) -> Tuple[np.ndarray, np.ndarray]:
    """像素级预测"""
    h, w, bands = spectra.shape
    spectra_flat = spectra.reshape(-1, bands)
    n_samples = spectra_flat.shape[0]

    pp_predictions = []
    pe_predictions = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = spectra_flat[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            pp_out, pe_out = model(batch_tensor)

            pp_probs = torch.softmax(pp_out, dim=1).cpu().numpy()
            pp_probs[:, 1] += pp_threshold_class1
            pp_probs[:, 2] += pp_threshold_class2
            pp_probs = pp_probs / pp_probs.sum(axis=1, keepdims=True)
            pp_preds = pp_probs.argmax(axis=1)

            pe_probs = torch.softmax(pe_out, dim=1).cpu().numpy()
            pe_probs[:, 1] += pe_threshold_class1
            pe_probs[:, 2] += pe_threshold_class2
            pe_probs = pe_probs / pe_probs.sum(axis=1, keepdims=True)
            pe_preds = pe_probs.argmax(axis=1)

            pp_predictions.extend(pp_preds)
            pe_predictions.extend(pe_preds)

    pp_labels = np.array(pp_predictions).reshape(h, w)
    pe_labels = np.array(pe_predictions).reshape(h, w)

    return pp_labels, pe_labels


def process_sample_pixel_level(
    sample_name: str,
    sample_folder: Path,
    endmembers: np.ndarray,
    model: DualHeadRamanCNNLSTM,
    device: torch.device,
    output_dir: Path,
    mp_type: str
) -> Dict:
    """处理单个样本的像素级预测。"""
    logger = get_logger('test_pixel')

    logger.info(f"\n{'='*60}")
    logger.info(f"【像素级预测】样本: {sample_name} (实际类型: {mp_type})")
    logger.info(f"{'='*60}")

    logger.info("加载高光谱数据...")
    hypercube, wavenumbers = load_hyperspectral_data(sample_folder)
    logger.info(f"  数据形状: {hypercube.shape}")

    logger.info("预处理光谱数据...")
    processed = preprocess_hyperspectral(hypercube, n_jobs=4)
    logger.info(f"  预处理完成，形状: {processed.shape}")

    logger.info("光谱解混...")
    abundance = unmix_hyperspectral(processed, endmembers, n_jobs=4)
    for i in range(abundance.shape[2]):
        abundance[:, :, i] = postprocess_abundance(abundance[:, :, i])

    logger.info("  丰度统计:")
    logger.info(f"    PP丰度: mean={abundance[:,:,1].mean():.4f}")
    logger.info(f"    PE丰度: mean={abundance[:,:,2].mean():.4f}")

    logger.info("像素级预测...")
    pp_labels, pe_labels = pixel_level_prediction(model, processed, device)

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    pp_unique, pp_counts = np.unique(pp_labels, return_counts=True)
    logger.info("  PP像素级分类统计:")
    for u, c in zip(pp_unique, pp_counts):
        logger.info(f"    {class_names[u]}: {c} ({c/pp_labels.size*100:.1f}%)")

    pe_unique, pe_counts = np.unique(pe_labels, return_counts=True)
    logger.info("  PE像素级分类统计:")
    for u, c in zip(pe_unique, pe_counts):
        logger.info(f"    {class_names[u]}: {c} ({c/pe_labels.size*100:.1f}%)")

    logger.info("保存结果和可视化...")
    test_output_dir = output_dir / 'test_results_dual' / 'pixel_level' / sample_name
    test_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(test_output_dir / 'pp_labels.npy', pp_labels)
    np.save(test_output_dir / 'pe_labels.npy', pe_labels)

    save_test_classification_maps(pp_labels, pe_labels, sample_name, output_dir)
    logger.info(f"  分类图已保存到: output/classification/{sample_name}/")

    result = {
        'sample_name': sample_name,
        'actual_type': mp_type,
        'pp_abundance_mean': float(abundance[:,:,1].mean()),
        'pe_abundance_mean': float(abundance[:,:,2].mean()),
        'pp_pixel_stats': {class_names[u]: int(c) for u, c in zip(pp_unique, pp_counts)},
        'pe_pixel_stats': {class_names[u]: int(c) for u, c in zip(pe_unique, pe_counts)}
    }

    with open(test_output_dir / 'result_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"像素级预测结果\n")
        f.write(f"{'='*40}\n")
        f.write(f"样本名称: {sample_name}\n")
        f.write(f"实际类型: {mp_type}\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n丰度统计:\n")
        f.write(f"  PP平均丰度: {abundance[:,:,1].mean():.4f}\n")
        f.write(f"  PE平均丰度: {abundance[:,:,2].mean():.4f}\n")
        f.write(f"\nPP像素级分类:\n")
        for u, c in zip(pp_unique, pp_counts):
            f.write(f"  {class_names[u]}: {c} ({c/pp_labels.size*100:.1f}%)\n")
        f.write(f"\nPE像素级分类:\n")
        for u, c in zip(pe_unique, pe_counts):
            f.write(f"  {class_names[u]}: {c} ({c/pe_labels.size*100:.1f}%)\n")

    return result


def main():
    """主函数"""
    logger = get_logger('test_pixel')
    config = get_config()

    logger.info("="*60)
    logger.info("像素级预测测试")
    logger.info("输出：PP分类图 + PE分类图")
    logger.info("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / 'dataset'
    preprocessed_dir = base_dir / 'preprocessed_data'
    output_dir = base_dir / 'output'

    test_samples = [
        {
            'name': 'pp_test',
            'folder': dataset_dir / 'test' / 'PP+淀粉' / '1 785mw 2s 1 1 40 40',
            'type': 'PP'
        },
        {
            'name': 'pe_test',
            'folder': dataset_dir / 'test' / 'PE+淀粉' / '1 785mw 2s 1 1 40 40',
            'type': 'PE'
        },
        {
            'name': 'pp_pe_mixed_test',
            'folder': dataset_dir / 'test' / 'PP+PE+淀粉' / '1 785mw 2s 1 1 40 40',
            'type': 'PP+PE'
        }
    ]

    logger.info("\n加载端元光谱...")
    starch_spec = np.load(preprocessed_dir / 'starch_spectrum.npy')
    pp_spec = np.load(preprocessed_dir / 'pp_spectrum.npy')
    pe_spec = np.load(preprocessed_dir / 'pe_spectrum.npy')
    endmembers = np.vstack([starch_spec, pp_spec, pe_spec])

    model_dir = output_dir / 'models' 
    best_model_path = model_dir / 'best_model.pth'

    if best_model_path.exists():
        model_path = best_model_path
        logger.info(f"使用模型: best_model.pth")
    else:
        checkpoints = list(model_dir.glob('checkpoint_epoch_*.pth'))
        if not checkpoints:
            logger.error("找不到双头模型！请先运行 training/train_dual.py")
            return
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        model_path = checkpoints[-1]
        logger.info(f"使用模型: {model_path.name}")

    model = load_dual_model(model_path, device)

    all_results = []
    for sample in test_samples:
        if sample['folder'].exists():
            result = process_sample_pixel_level(
                sample['name'],
                sample['folder'],
                endmembers,
                model,
                device,
                output_dir,
                sample['type']
            )
            all_results.append(result)
        else:
            logger.warning(f"测试数据不存在: {sample['folder']}")

    logger.info("\n" + "="*60)
    logger.info("像素级预测结果汇总")
    logger.info("="*60)

    for r in all_results:
        logger.info(f"\n{r['sample_name']} (实际类型: {r['actual_type']}):")
        logger.info(f"  PP丰度: {r['pp_abundance_mean']:.4f}")
        logger.info(f"  PE丰度: {r['pe_abundance_mean']:.4f}")
        logger.info(f"  PP分类: {r['pp_pixel_stats']}")
        logger.info(f"  PE分类: {r['pe_pixel_stats']}")

    logger.info("\n" + "="*60)
    logger.info("像素级预测完成！")
    logger.info(f"结果保存在: {output_dir / 'test_results_dual' / 'pixel_level'}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
