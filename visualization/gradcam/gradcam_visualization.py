"""
Grad-CAM可视化

生成图表:
1. Grad-CAM热图叠加光谱 (gradcam_samples.png)
2. 各类别平均激活图 (gradcam_by_class.png)

输出位置: output/gradcam/PP/ 和 output/gradcam/PE/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import get_config, get_logger

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class GradCAM:
    """Grad-CAM实现 - 支持双头模型"""

    def __init__(self, model, target_layer, head='pp'):
        """
        Args:
            model: 模型
            target_layer: 目标卷积层
            head: 'pp' 或 'pe'，指定使用哪个头（双头模型时）
        """
        self.model = model
        self.target_layer = target_layer
        self.head = head.lower()
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """生成Grad-CAM热图"""
        self.model.eval()
        output = self.model(input_tensor)

        # 处理双头模型输出
        if isinstance(output, tuple):
            output = output[0] if self.head == 'pp' else output[1]

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()

        # 创建one-hot向量
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        weights = self.gradients.mean(dim=2, keepdim=True)

        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


def load_model_and_data(config, task: str):
    """加载模型和数据"""
    from models.dual_head_model import DualHeadRamanCNNLSTM

    # 强制使用CPU以避免cuDNN LSTM backward问题
    device = torch.device('cpu')
    models_dir = Path(config.paths['output_dir']) / 'models'

    # 加载模型
    model = DualHeadRamanCNNLSTM(
        input_len=config.model['architecture']['input_len'],
        num_classes=config.model['architecture']['num_classes']
    ).to(device)

    model_path = models_dir / 'best_model.pth'
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # 加载数据
    preprocessed_dir = Path(config.paths['preprocessed_dir'])
    output_dir = Path(config.paths['output_dir'])
    prefix = 'pp' if task == 'PP' else 'pe'
    label_suffix = 'pp_labels' if task == 'PP' else 'pe_labels'

    spectra = []
    labels = []
    for i in range(1, 6):
        data_file = preprocessed_dir / f'{prefix}_mixed{i}_data.npy'
        label_file = preprocessed_dir / f'{prefix}_mixed{i}_{label_suffix}.npy'
        if data_file.exists():
            data = np.load(data_file)
            if data.ndim == 3:
                data = data.reshape(-1, data.shape[-1])
            spectra.append(data)
            if label_file.exists():
                label_data = np.load(label_file).flatten()
                labels.append(label_data)

    wavenumbers = None
    wavenumbers_file = preprocessed_dir / 'wavenumbers.npy'
    if wavenumbers_file.exists():
        wavenumbers = np.load(wavenumbers_file)

    if spectra:
        X = np.concatenate(spectra, axis=0)
        y = np.concatenate(labels, axis=0) if labels else None
        return model, X, y, device, wavenumbers

    return model, None, None, device, wavenumbers


def plot_gradcam_samples(model, X, y, device, output_dir: Path, task: str, wavenumbers=None, n_samples: int = 9):
    """绘制Grad-CAM样本图"""
    if X is None:
        return

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']

    # 获取CNN层 - DualHeadRamanCNNLSTM 使用 self.conv Sequential
    if hasattr(model, 'conv') and isinstance(model.conv, nn.Sequential):
        # 获取 Sequential 中的第二个 Conv1d 层 (index 4)
        target_layer = model.conv[4]
    elif hasattr(model, 'conv2'):
        target_layer = model.conv2
    elif hasattr(model, 'conv1'):
        target_layer = model.conv1
    else:
        get_logger('gradcam').error("无法找到合适的卷积层")
        return

    # 指定使用哪个头
    head = 'pp' if task == 'PP' else 'pe'
    gradcam = GradCAM(model, target_layer, head=head)

    # 每类选择样本
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    sample_idx = 0
    for class_id in range(3):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) == 0:
            continue

        selected = np.random.choice(class_indices, min(3, len(class_indices)), replace=False)

        for i, idx in enumerate(selected):
            if sample_idx >= 9:
                break

            row = sample_idx // 3
            col = sample_idx % 3
            ax = axes[row, col]

            # 获取光谱
            spectrum = X[idx]
            # 模型期望输入形状 (batch, features)，内部会自动添加通道维度
            x_input = torch.FloatTensor(spectrum).unsqueeze(0).to(device)

            # 生成Grad-CAM
            try:
                cam = gradcam.generate(x_input, class_id)

                # 插值到原始长度
                cam_interp = np.interp(
                    np.linspace(0, len(cam)-1, len(spectrum)),
                    np.arange(len(cam)),
                    cam
                )

                # 绘制
                if wavenumbers is not None and len(wavenumbers) == len(spectrum):
                    x_axis = wavenumbers
                    x_label = 'Raman Shift (cm⁻¹)'
                else:
                    x_axis = np.arange(len(spectrum))
                    x_label = 'Spectral Band'
                ax.plot(x_axis, spectrum, 'b-', linewidth=1, alpha=0.7, label='Spectrum')

                # 叠加热图
                ax2 = ax.twinx()
                ax2.fill_between(x_axis, 0, cam_interp, alpha=0.3, color='red')
                ax2.plot(x_axis, cam_interp, 'r-', linewidth=1.5, alpha=0.8, label='Grad-CAM')
                ax2.set_ylim(0, 1.2)
                ax2.set_ylabel('Activation', color='red', fontsize=10)

                ax.set_title(f'{class_names[class_id]} (Sample {idx})', fontsize=11)
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('Intensity', color='blue', fontsize=10)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', transform=ax.transAxes)

            sample_idx += 1

    plt.suptitle(f'Grad-CAM Visualization ({task} Task)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'gradcam_samples_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_gradcam_by_class(model, X, y, device, output_dir: Path, task: str, wavenumbers=None, n_samples: int = 50):
    """绘制各类别平均Grad-CAM"""
    if X is None:
        return

    class_names = ['Non-pollution', 'Slight pollution', 'Severe pollution']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    # 获取CNN层 - DualHeadRamanCNNLSTM 使用 self.conv Sequential
    if hasattr(model, 'conv') and isinstance(model.conv, nn.Sequential):
        target_layer = model.conv[4]
    elif hasattr(model, 'conv2'):
        target_layer = model.conv2
    else:
        target_layer = model.conv1

    # 指定使用哪个头
    head = 'pp' if task == 'PP' else 'pe'
    gradcam = GradCAM(model, target_layer, head=head)

    fig, ax = plt.subplots(figsize=(12, 6))

    for class_id in range(3):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) == 0:
            continue

        selected = np.random.choice(class_indices, min(n_samples, len(class_indices)), replace=False)

        cams = []
        for idx in selected:
            spectrum = X[idx]
            # 模型期望输入形状 (batch, features)，内部会自动添加通道维度
            x_input = torch.FloatTensor(spectrum).unsqueeze(0).to(device)

            try:
                cam = gradcam.generate(x_input, class_id)
                cam_interp = np.interp(
                    np.linspace(0, len(cam)-1, len(spectrum)),
                    np.arange(len(cam)),
                    cam
                )
                cams.append(cam_interp)
            except:
                continue

        if cams:
            mean_cam = np.mean(cams, axis=0)
            std_cam = np.std(cams, axis=0)

            if wavenumbers is not None and len(wavenumbers) == len(mean_cam):
                x_axis = wavenumbers
                x_label = 'Raman Shift (cm⁻¹)'
            else:
                x_axis = np.arange(len(mean_cam))
                x_label = 'Spectral Band'
            ax.plot(x_axis, mean_cam, color=colors[class_id], linewidth=2, label=class_names[class_id])
            ax.fill_between(x_axis, mean_cam - std_cam, mean_cam + std_cam,
                           color=colors[class_id], alpha=0.2)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Average Grad-CAM Activation', fontsize=12)
    ax.set_title(f'Average Grad-CAM by Class ({task} Task)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'gradcam_by_class_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_gradcam_visualizations(config, output_dir: Path):
    """生成Grad-CAM可视化"""
    logger = get_logger('gradcam')

    output_dir.mkdir(parents=True, exist_ok=True)

    for task in ['PP', 'PE']:
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"生成 {task} 任务 Grad-CAM 可视化...")

        try:
            model, X, y, device, wavenumbers = load_model_and_data(config, task)

            if X is not None and y is not None:
                plot_gradcam_samples(model, X, y, device, task_dir, task, wavenumbers)
                plot_gradcam_by_class(model, X, y, device, task_dir, task, wavenumbers)
                logger.info(f"{task} Grad-CAM 已保存到: {task_dir}")
            else:
                logger.warning(f"未找到 {task} 数据")

        except Exception as e:
            logger.error(f"{task} Grad-CAM 生成失败: {e}")


if __name__ == '__main__':
    config = get_config()
    logger = get_logger('gradcam')

    logger.info("=" * 60)
    logger.info("生成 Grad-CAM 可视化")
    logger.info("=" * 60)

    output_dir = Path(config.paths['output_dir']) / 'gradcam'
    generate_gradcam_visualizations(config, output_dir)

    logger.info("\nGrad-CAM 可视化完成！")
