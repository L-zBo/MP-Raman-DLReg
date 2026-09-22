"""
基于丰度阈值的像素级标签生成。

输入：
- NNLS 解混得到的丰度 (output/abundance/abundance_npy/)
- 固化阈值 JSON (output/models/abundance_thresholds/)

输出：
- 三级污染标签 (无 / 轻 / 重)
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_config, get_logger


class AbundanceThresholdLabeler:
    """按固化阈值切分丰度，输出三级标签。"""

    def __init__(self, thresholds: Optional[List[float]] = None, n_classes: int = 3):
        self.n_classes = n_classes
        self.thresholds = thresholds

    def predict(self, abundance: NDArray[np.floating]) -> NDArray[np.integer]:
        if self.thresholds is None:
            raise ValueError("阈值未加载，请先调用 load() 或在构造时传入 thresholds")

        original_shape = abundance.shape
        flat = abundance.flatten()
        labels = np.zeros_like(flat, dtype=np.int32)
        for i, threshold in enumerate(self.thresholds):
            labels[flat > threshold] = i + 1
        return labels.reshape(original_shape)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(
                {'n_components': self.n_classes, 'thresholds': self.thresholds},
                f,
                indent=2,
            )

    def load(self, path: Path) -> 'AbundanceThresholdLabeler':
        with open(path, 'r') as f:
            data = json.load(f)
        self.n_classes = data.get('n_components', data.get('n_classes', 3))
        self.thresholds = data['thresholds']
        return self


def load_all_abundance(config, component: str = 'pp') -> Tuple[NDArray, List[Dict]]:
    logger = get_logger('abundance_threshold')
    output_dir = Path(config.paths['output_dir'])

    all_abundance: List[NDArray] = []
    sample_info: List[Dict] = []

    datasets = config.get('dataset.datasets')
    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']

        if component == 'pp' and 'PP' not in dataset_type:
            continue
        if component == 'pe' and 'PE' not in dataset_type:
            continue

        abundance_dir = output_dir / 'abundance' / 'abundance_npy' / dataset_type
        component_idx = 1 if component == 'pp' else 2

        for sample in samples:
            name = sample['name']
            abundance_path = abundance_dir / f'{name}_abundance.npy'
            if not abundance_path.exists():
                continue
            abundance = np.load(abundance_path)
            all_abundance.append(abundance[:, :, component_idx].flatten())
            sample_info.append({
                'name': name,
                'dataset_type': dataset_type,
                'shape': abundance.shape[:2],
            })

    if not all_abundance:
        logger.warning(f"未找到 {component} 丰度数据")
        return np.array([]), []

    return np.concatenate(all_abundance), sample_info


def label_all_samples(
    config,
    pp_labeler: AbundanceThresholdLabeler,
    pe_labeler: AbundanceThresholdLabeler,
) -> None:
    logger = get_logger('abundance_threshold')
    output_dir = Path(config.paths['output_dir'])

    datasets = config.get('dataset.datasets')
    for dataset in datasets:
        dataset_type = dataset['type']
        samples = dataset['samples']

        logger.info(f"\n处理数据集: {dataset_type}")

        abundance_dir = output_dir / 'abundance' / 'abundance_npy' / dataset_type
        labels_dir = Path(config.paths['preprocessed_dir'])
        labels_dir.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            name = sample['name']
            abundance_path = abundance_dir / f'{name}_abundance.npy'
            if not abundance_path.exists():
                logger.warning(f"  跳过 {name}: 丰度文件不存在")
                continue

            abundance = np.load(abundance_path)
            pp_abundance = abundance[:, :, 1]
            pe_abundance = abundance[:, :, 2]

            if 'PP' in dataset_type and 'PE' not in dataset_type:
                pp_labels = pp_labeler.predict(pp_abundance)
                pe_labels = np.zeros_like(pp_labels, dtype=np.int32)
            elif 'PE' in dataset_type and 'PP' not in dataset_type:
                pp_labels = np.zeros(pe_abundance.shape, dtype=np.int32)
                pe_labels = pe_labeler.predict(pe_abundance)
            else:
                pp_labels = pp_labeler.predict(pp_abundance)
                pe_labels = pe_labeler.predict(pe_abundance)

            np.save(labels_dir / f'{name}_pp_labels.npy', pp_labels)
            np.save(labels_dir / f'{name}_pe_labels.npy', pe_labels)

            pp_counts = [int((pp_labels == i).sum()) for i in range(3)]
            pe_counts = [int((pe_labels == i).sum()) for i in range(3)]
            logger.info(f"  {name}: PP={pp_counts}, PE={pe_counts}")


def main() -> None:
    config = get_config()
    logger = get_logger('abundance_threshold')

    output_dir = Path(config.paths['output_dir'])
    thresholds_dir = output_dir / 'models' / 'abundance_thresholds'

    logger.info("=" * 60)
    logger.info("丰度阈值标签生成")
    logger.info("=" * 60)

    pp_labeler = AbundanceThresholdLabeler().load(thresholds_dir / 'pp_thresholds.json')
    pe_labeler = AbundanceThresholdLabeler().load(thresholds_dir / 'pe_thresholds.json')
    logger.info(f"  PP 阈值: {[f'{t:.4f}' for t in pp_labeler.thresholds]}")
    logger.info(f"  PE 阈值: {[f'{t:.4f}' for t in pe_labeler.thresholds]}")

    logger.info("\n>>> 应用阈值生成所有样本的三级标签...")
    label_all_samples(config, pp_labeler, pe_labeler)

    logger.info("\n" + "=" * 60)
    logger.info("完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
