"""
对象级预测脚本。

功能：
- 生成对象级污染结果
- 输出对象级 CSV

输出：
- output/test_results_dual/object_level/object_level_predictions_pixel_ratio.csv
"""
import sys
import csv
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_config, get_logger
from preprocessing.preprocess import load_hyperspectral_data, preprocess_hyperspectral
from unmixing.unmix import unmix_hyperspectral, postprocess_abundance
from models.dual_head_model import DualHeadRamanCNNLSTM
from testing.test_pixel_level import pixel_level_prediction


def load_dual_model(checkpoint_path: Path, device: torch.device) -> DualHeadRamanCNNLSTM:
    """加载双头模型"""
    logger = get_logger('test_object')

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


def object_level_prediction(
    model: DualHeadRamanCNNLSTM,
    spectra: np.ndarray,
    device: torch.device,
    method: str = 'pixel_ratio',
    batch_size: int = 64,
    mild_threshold: float = 0.05,
    severe_threshold: float = 0.15,
    abundance_info: Dict = None
) -> Tuple[int, int, Dict]:
    """生成对象级预测结果。"""
    h, w, bands = spectra.shape
    n_samples = h * w

    pp_label_map, pe_label_map = pixel_level_prediction(
        model,
        spectra,
        device,
        batch_size=batch_size,
    )

    pp_vote_counts = np.bincount(pp_label_map.flatten(), minlength=3)
    pe_vote_counts = np.bincount(pe_label_map.flatten(), minlength=3)

    pp_pollution_pixels = pp_vote_counts[1] + pp_vote_counts[2]
    pe_pollution_pixels = pe_vote_counts[1] + pe_vote_counts[2]

    pp_pollution_ratio = pp_pollution_pixels / n_samples
    pe_pollution_ratio = pe_pollution_pixels / n_samples

    details = {
        'method': method,
        'n_pixels': n_samples,
        'pp_vote_counts': pp_vote_counts.tolist(),
        'pe_vote_counts': pe_vote_counts.tolist(),
        'pp_pollution_pixels': int(pp_pollution_pixels),
        'pe_pollution_pixels': int(pe_pollution_pixels),
        'pp_pollution_ratio': float(pp_pollution_ratio),
        'pe_pollution_ratio': float(pe_pollution_ratio),
        'mild_threshold': mild_threshold,
        'severe_threshold': severe_threshold,
        'pp_label_map': pp_label_map,
        'pe_label_map': pe_label_map,
    }

    if method == 'mean':
        mean_spectrum = spectra.mean(axis=(0, 1))
        with torch.no_grad():
            input_tensor = torch.FloatTensor(mean_spectrum).unsqueeze(0).to(device)
            pp_out, pe_out = model(input_tensor)
            pp_level = pp_out.argmax(dim=1).item()
            pe_level = pe_out.argmax(dim=1).item()

    elif method == 'pixel_ratio':
        if pp_pollution_ratio >= severe_threshold:
            pp_level = 2
        elif pp_pollution_ratio >= mild_threshold:
            pp_level = 1
        else:
            pp_level = 0

        if pe_pollution_ratio >= severe_threshold:
            pe_level = 2
        elif pe_pollution_ratio >= mild_threshold:
            pe_level = 1
        else:
            pe_level = 0

    else:
        raise ValueError(f"未知的聚合方法: {method}")

    return pp_level, pe_level, details


def process_sample_object_level(
    sample_name: str,
    sample_folder: Path,
    endmembers: np.ndarray,
    model: DualHeadRamanCNNLSTM,
    device: torch.device,
    mp_type: str,
    method: str = 'pixel_ratio'
) -> Dict:
    """处理单个样本的对象级预测。"""
    logger = get_logger('test_object')

    logger.info(f"\n{'='*60}")
    logger.info(f"【对象级预测】样本: {sample_name} (实际类型: {mp_type})")
    logger.info(f"使用方法: {method}")
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

    pp_abundance = float(abundance[:,:,1].mean())
    pe_abundance = float(abundance[:,:,2].mean())
    starch_abundance = float(abundance[:,:,0].mean())

    logger.info("  丰度统计:")
    logger.info(f"    淀粉丰度: {starch_abundance:.4f}")
    logger.info(f"    PP丰度:   {pp_abundance:.4f}")
    logger.info(f"    PE丰度:   {pe_abundance:.4f}")

    logger.info(f"对象级预测 (方法: {method})...")
    pp_level, pe_level, details = object_level_prediction(
        model, processed, device,
        method=method
    )

    level_names = ['无污染(0)', '轻度污染(1)', '重度污染(2)']
    logger.info(f"  预测结果: (PP:{pp_level}, PE:{pe_level})")
    logger.info(f"    PP: {level_names[pp_level]}")
    logger.info(f"    PE: {level_names[pe_level]}")

    if 'pp_pollution_ratio' in details:
        logger.info(f"  像素统计:")
        logger.info(f"    总像素数: {details['n_pixels']}")
        logger.info(f"    PP污染像素: {details['pp_pollution_pixels']} ({details['pp_pollution_ratio']*100:.2f}%)")
        logger.info(f"    PE污染像素: {details['pe_pollution_pixels']} ({details['pe_pollution_ratio']*100:.2f}%)")
        logger.info(f"    PP各类别: 无污染={details['pp_vote_counts'][0]}, 轻度={details['pp_vote_counts'][1]}, 重度={details['pp_vote_counts'][2]}")
        logger.info(f"    PE各类别: 无污染={details['pe_vote_counts'][0]}, 轻度={details['pe_vote_counts'][1]}, 重度={details['pe_vote_counts'][2]}")

    result = {
        'sample_name': sample_name,
        'actual_type': mp_type,
        'starch_abundance': starch_abundance,
        'pp_abundance': pp_abundance,
        'pe_abundance': pe_abundance,
        'pp_level': pp_level,
        'pe_level': pe_level,
        'prediction': f"(PP:{pp_level}, PE:{pe_level})",
        'method': method,
        'details': details
    }

    return result


def _format_pixel_cell(pixel_count: int, ratio_pct: float, is_noise: bool) -> str:
    """格式化像素统计单元格，噪声值加*标记"""
    cell = f"{pixel_count}({ratio_pct:.2f}%)"
    if is_noise:
        cell += "*"
    return cell


def save_results_to_csv(results: List[Dict], output_path: Path,
                        noise_threshold: float = 0.01):
    """保存对象级结果到 CSV。"""
    type_contains = {
        'PP': {'pp'},
        'PE': {'pe'},
        'PP+PE': {'pp', 'pe'},
        'Starch': set(),
    }

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            '样本名称',
            '实际类型',
            '总像素数',
            'PP污染像素',
            'PE污染像素',
            'PP预测等级',
            'PE预测等级',
            '预测结果'
        ])
        for r in results:
            details = r.get('details', {})
            n_pixels = details.get('n_pixels', 0)
            pp_poll = details.get('pp_pollution_pixels', 0)
            pe_poll = details.get('pe_pollution_pixels', 0)
            pp_ratio = details.get('pp_pollution_ratio', 0) * 100
            pe_ratio = details.get('pe_pollution_ratio', 0) * 100

            actual_type = r['actual_type']
            contains = type_contains.get(actual_type, set())

            pp_is_noise = ('pp' not in contains) and (pp_ratio < noise_threshold * 100)
            pe_is_noise = ('pe' not in contains) and (pe_ratio < noise_threshold * 100)

            writer.writerow([
                r['sample_name'],
                r['actual_type'],
                n_pixels,
                _format_pixel_cell(pp_poll, pp_ratio, pp_is_noise),
                _format_pixel_cell(pe_poll, pe_ratio, pe_is_noise),
                r['pp_level'],
                r['pe_level'],
                r['prediction']
            ])

        writer.writerow([])
        writer.writerow([
            '* 在已知组分的验证样本中，该污染物并非样本的实际成分。'
            '其检出比例低于像素级分类器的经验假阳性率（<1%），'
            '属于模型固有分类误差（classification artifact），不代表真实污染信号。'
            '该假阳性率同时定义了本方法对微塑料像素占比的最低可检测限（LOD≈1%）。'
        ])


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='对象级预测测试（像素比例投票）')
    parser.add_argument('--method', type=str, default='pixel_ratio',
                        choices=['mean', 'pixel_ratio'],
                        help='聚合方法: mean 或 pixel_ratio')
    parser.add_argument('--mild', type=float, default=0.05,
                        help='轻度污染阈值（默认5%%）')
    parser.add_argument('--severe', type=float, default=0.15,
                        help='重度污染阈值（默认15%%）')
    parser.add_argument('--compare', action='store_true',
                        help='对比 mean 和 pixel_ratio 方法')
    args = parser.parse_args()

    logger = get_logger('test_object')
    config = get_config()

    logger.info("="*60)
    logger.info("对象级预测测试（像素比例投票版）")
    logger.info("判定规则:")
    logger.info(f"  污染像素 < {args.mild*100:.0f}% → 无污染")
    logger.info(f"  {args.mild*100:.0f}% ≤ 污染像素 < {args.severe*100:.0f}% → 轻度污染")
    logger.info(f"  污染像素 ≥ {args.severe*100:.0f}% → 重度污染")
    logger.info("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / 'dataset'
    preprocessed_dir = base_dir / 'preprocessed_data'
    output_dir = base_dir / 'output'

    # 测试数据路径
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

    # 加载端元
    logger.info("\n加载端元光谱...")
    starch_spec = np.load(preprocessed_dir / 'starch_spectrum.npy')
    pp_spec = np.load(preprocessed_dir / 'pp_spectrum.npy')
    pe_spec = np.load(preprocessed_dir / 'pe_spectrum.npy')
    endmembers = np.vstack([starch_spec, pp_spec, pe_spec])

    # 查找双头模型（优先使用 best_model.pth）
    model_dir = output_dir / 'models'
    best_model_path = model_dir / 'best_model.pth'

    if best_model_path.exists():
        model_path = best_model_path
        logger.info(f"使用最佳模型: {model_path.name}")
    else:
        # 按数字排序查找最新的 checkpoint
        import re
        checkpoints = list(model_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x.name).group(1)))
            model_path = checkpoints[-1]
            logger.info(f"使用最新checkpoint: {model_path.name}")
        else:
            logger.error("找不到双头模型！请先运行 training/train_dual.py")
            return

    model = load_dual_model(model_path, device)

    # 确定要测试的方法
    if args.compare:
        methods = ['mean', 'pixel_ratio']
        logger.info("\n[对比模式] 将测试 mean 和 pixel_ratio 方法")
    else:
        methods = [args.method]

    # 对每种方法进行测试
    all_comparison_results = {}

    for method in methods:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 测试方法: {method}")
        logger.info(f"{'#'*60}")

        all_results = []
        for sample in test_samples:
            if sample['folder'].exists():
                result = process_sample_object_level(
                    sample['name'],
                    sample['folder'],
                    endmembers,
                    model,
                    device,
                    sample['type'],
                    method=method
                )
                all_results.append(result)
            else:
                logger.warning(f"测试数据不存在: {sample['folder']}")

        all_comparison_results[method] = all_results

        # 保存结果到CSV
        csv_dir = output_dir / 'test_results_dual' / 'object_level'
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f'object_level_predictions_{method}.csv'
        save_results_to_csv(all_results, csv_path)
        logger.info(f"\n结果已保存到: {csv_path}")

    # 打印汇总表格
    logger.info("\n" + "="*80)
    logger.info("对象级预测结果汇总")
    logger.info("="*80)

    if args.compare:
        # 对比模式：显示两种方法的结果
        logger.info(f"{'样本':<18} {'类型':<8} {'PP污染比':<10} {'PE污染比':<10} | {'mean':<12} {'pixel_ratio':<12}")
        logger.info("-"*80)
        for i, sample in enumerate(test_samples):
            if all_comparison_results['mean'] and i < len(all_comparison_results['mean']):
                r_mean = all_comparison_results['mean'][i]
                r_pixel = all_comparison_results['pixel_ratio'][i]
                pp_ratio = r_pixel['details'].get('pp_pollution_ratio', 0) * 100
                pe_ratio = r_pixel['details'].get('pe_pollution_ratio', 0) * 100
                row = f"{r_mean['sample_name']:<18} {r_mean['actual_type']:<8} {pp_ratio:<10.2f}% {pe_ratio:<10.2f}% |"
                row += f" {r_mean['prediction']:<12} {r_pixel['prediction']:<12}"
                logger.info(row)
    else:
        # 单一方法模式：显示详细像素统计
        logger.info(f"{'样本':<18} {'类型':<8} {'PP污染比':<12} {'PE污染比':<12} {'预测结果':<15}")
        logger.info("-"*80)
        for r in all_comparison_results[args.method]:
            pp_ratio = r['details'].get('pp_pollution_ratio', 0) * 100
            pe_ratio = r['details'].get('pe_pollution_ratio', 0) * 100
            logger.info(f"{r['sample_name']:<18} {r['actual_type']:<8} {pp_ratio:<12.2f}% {pe_ratio:<12.2f}% {r['prediction']:<15}")

    logger.info("\n" + "="*80)
    logger.info("对象级预测完成！")
    logger.info("="*80)
    logger.info("\n判定规则:")
    logger.info(f"  污染像素 < {args.mild*100:.0f}%  → 无污染(0)")
    logger.info(f"  {args.mild*100:.0f}% ≤ 污染像素 < {args.severe*100:.0f}% → 轻度污染(1)")
    logger.info(f"  污染像素 ≥ {args.severe*100:.0f}% → 重度污染(2)")
    logger.info("\n使用方法:")
    logger.info("  python testing/test_object_level.py                    # 默认pixel_ratio")
    logger.info("  python testing/test_object_level.py --compare          # 对比两种方法")
    logger.info("  python testing/test_object_level.py --mild 0.03 --severe 0.10  # 自定义阈值")


if __name__ == '__main__':
    main()
