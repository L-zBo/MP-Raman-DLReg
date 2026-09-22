"""
基础模型定义
包含: RADAR-Net backbone, Simple1DCNN, ResNet1D, SpectralTransformer1D, PLSDA
"""

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import PLSRegression


class RamanCNNLSTM(nn.Module):
    """
    RADAR-Net backbone模型 (完整模型)

    结构:
    Input → CNN Block → BiLSTM → Attention → FC → Output

    Args:
        input_len: 输入光谱长度 (默认1024)
        num_classes: 分类类别数 (默认3)
        return_attention: 是否返回注意力权重 (默认False)
    """

    def __init__(self, input_len: int = 1024, num_classes: int = 3, return_attention: bool = False):
        super().__init__()
        self.return_attention = return_attention

        # CNN特征提取
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # BiLSTM时序建模
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=True)

        # 注意力机制
        self.attn = nn.Linear(128, 1)

        # 全连接分类器
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv(x)     # (batch, 64, features/16)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 64)
        x, _ = self.lstm(x)  # (batch, seq_len, 128)

        # 注意力加权
        attn_w = F.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
        x = (x * attn_w).sum(dim=1)  # (batch, 128)

        out = self.fc(x)

        if self.return_attention:
            return out, attn_w.squeeze(-1)
        return out


class Simple1DCNN(nn.Module):
    """
    保留的单头卷积基线模型。
    """

    def __init__(self, input_len: int = 1024, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


class ResidualBlock1D(nn.Module):
    """1D残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    """
    1D ResNet模型

    结构:
    Input → Conv → ResBlock×6 → AvgPool → FC → Output
    """

    def __init__(self, input_len: int = 1024, num_classes: int = 3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock1D(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.squeeze(-1)
        return self.fc(x)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        width = out_channels
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class StandardResNet1D(nn.Module):
    """
    Standardized 1D ResNet family for fairer comparison with classical ResNet-18/50 style designs.
    """

    def __init__(self, block, layers, input_len: int = 1024, num_classes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, blocks: int, stride: int):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def ResNet18_1D(input_len: int = 1024, num_classes: int = 3, dropout: float = 0.0) -> StandardResNet1D:
    return StandardResNet1D(BasicBlock1D, [2, 2, 2, 2], input_len=input_len, num_classes=num_classes, dropout=dropout)


def ResNet50_1D(input_len: int = 1024, num_classes: int = 3, dropout: float = 0.0) -> StandardResNet1D:
    return StandardResNet1D(Bottleneck1D, [3, 4, 6, 3], input_len=input_len, num_classes=num_classes, dropout=dropout)


class SpectralTransformer1D(nn.Module):
    """
    1D Spectral Transformer模型

    结构:
    Input → Linear Projection → Positional Encoding → TransformerEncoder × N
          → Global Average Pooling → FC → Output

    适用于1D光谱数据的轻量级Transformer架构

    Args:
        input_len: 输入光谱长度 (默认1024)
        num_classes: 分类类别数 (默认3)
        d_model: Transformer嵌入维度 (默认64)
        nhead: 多头注意力头数 (默认4)
        num_layers: TransformerEncoder层数 (默认2)
        dim_feedforward: FFN隐藏层维度 (默认128)
        dropout: Dropout比率 (默认0.1)
        patch_size: 将光谱分割为patch的大小 (默认16)
    """

    def __init__(self, input_len: int = 1024, num_classes: int = 3,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1,
                 patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        num_patches = input_len // patch_size

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.unfold(1, self.patch_size, self.patch_size)
        x = self.patch_embed(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)


class KANLinear(nn.Module):
    """Lightweight KAN-inspired linear layer with radial basis expansion."""

    def __init__(self, in_features: int, out_features: int, n_basis: int = 8):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.n_basis = n_basis
        self.register_buffer('grid', torch.linspace(-1.0, 1.0, steps=n_basis))
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)
        sigma = torch.exp(self.log_sigma) + 1e-6
        basis = torch.exp(-((x_norm.unsqueeze(-1) - self.grid) ** 2) / (2 * sigma ** 2))
        spline_out = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        return self.base(x) + spline_out


class KANClassifierHead(nn.Module):
    """Compact KAN-inspired classifier head used by SMART-NIR adaptation."""

    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            KANLinear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualMLPTransformerBlock(nn.Module):
    """Transformer block with dual feed-forward branches inspired by SMART-NIR."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        mlp_in = self.norm2(x)
        x = x + 0.5 * (self.mlp1(mlp_in) + self.mlp2(mlp_in))
        return x


class SMARTNIRClassifier(nn.Module):
    """
    SMART-NIR 单头模型。
    """

    def __init__(
        self,
        input_len: int = 1024,
        num_classes: int = 3,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        patch_size: int = 16,
        branch_channels: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_len = input_len
        self.patch_size = patch_size
        self.num_patches = math.ceil(input_len / patch_size)

        kernel_sizes = [3, 5, 7, 9]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, branch_channels, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        fused_channels = branch_channels * len(kernel_sizes)
        self.fuse = nn.Sequential(
            nn.Conv1d(fused_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.patch_embed = nn.Conv1d(d_model, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DualMLPTransformerBlock(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = KANClassifierHead(d_model, d_model, num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        multi_scale = [branch(x) for branch in self.branches]
        x = torch.cat(multi_scale, dim=1)
        x = self.fuse(x)
        x = self.patch_embed(x).transpose(1, 2)
        if x.size(1) != self.num_patches:
            x = F.pad(x, (0, 0, 0, self.num_patches - x.size(1)))
        x = self.dropout(x + self.pos_embedding[:, :x.size(1), :])
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class DilatedConvBlock(nn.Module):
    """多尺度膨胀卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MixerBlock1D(nn.Module):
    """MLP-Mixer style block for token and channel mixing."""

    def __init__(self, num_tokens: int, num_channels: int, token_dim: int = 128, channel_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_tokens),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(num_channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_channels, channel_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, num_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        y = self.channel_mlp(self.norm2(x))
        return x + y


class RSMLPClassifier(nn.Module):
    """单头 MLP 风格基线模型。"""

    def __init__(
        self,
        input_len: int = 1024,
        num_classes: int = 3,
        reference_spectra: Optional[torch.Tensor] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        reference_scale: float = 0.2
    ):
        super().__init__()
        self.input_len = input_len
        self.num_segments = 64
        self.feature_dim = 128

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.multi_scale = nn.ModuleList([
            DilatedConvBlock(64, 64, dilation=d) for d in (1, 2, 4)
        ])
        self.downsample = nn.Sequential(
            nn.Conv1d(64 * 3, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.segment_pool = nn.AdaptiveAvgPool1d(self.num_segments)

        self.pos_attn = nn.MultiheadAttention(self.feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.intensity_attn = nn.MultiheadAttention(self.feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.feature_dim)

        self.register_buffer('reference_spectra', torch.zeros(3, input_len))
        if reference_spectra is not None:
            if not isinstance(reference_spectra, torch.Tensor):
                reference_spectra = torch.tensor(reference_spectra, dtype=torch.float32)
            self.reference_spectra.copy_(reference_spectra.float())

        self.reference_proj = nn.Linear(self.reference_spectra.size(0), self.feature_dim)
        self.reference_scale = nn.Parameter(torch.tensor(float(reference_scale)))
        self.mixers = nn.ModuleList([
            MixerBlock1D(self.num_segments, self.feature_dim, token_dim=128, channel_dim=256, dropout=dropout)
            for _ in range(2)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _encode_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.cat([block(x) for block in self.multi_scale], dim=1)
        x = self.downsample(x)
        x = self.segment_pool(x)
        return x.transpose(1, 2)  # (B, 64, 128)

    def _encode_reference_library(self, device: torch.device) -> torch.Tensor:
        refs = self.reference_spectra.to(device).unsqueeze(1)
        ref_features = self._encode_spectrum(refs)
        return ref_features.mean(dim=1)  # (R, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        features = self._encode_spectrum(x)
        norm_features = self.norm(features)

        pos_out, _ = self.pos_attn(norm_features, norm_features, norm_features, need_weights=False)
        intensity_input = norm_features * torch.sigmoid(norm_features)
        intensity_out, _ = self.intensity_attn(intensity_input, intensity_input, intensity_input, need_weights=False)
        features = features + 0.5 * (pos_out + intensity_out)

        ref_library = self._encode_reference_library(features.device)
        ref_library = F.normalize(ref_library, dim=-1)
        similarity = torch.einsum(
            'btd,rd->btr',
            F.normalize(features, dim=-1),
            ref_library
        )
        features = features + self.reference_scale * self.reference_proj(similarity)

        for mixer in self.mixers:
            features = mixer(features)

        pooled = features.mean(dim=1)
        return self.classifier(pooled)


class MambaStateSpaceBlock(nn.Module):
    """Lightweight bidirectional state-space block inspired by Mamba."""

    def __init__(self, dim: int, expand_ratio: float = 2.0, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.dwconv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim
        )
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.a_log = nn.Parameter(torch.zeros(hidden_dim))
        self.skip = nn.Parameter(torch.ones(hidden_dim))

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        # Vectorized EMA-style scan:
        # y_t = a * y_{t-1} + (1-a) * x_t, y_{-1}=0
        # This removes the Python loop that dominated MambaHSI inference time.
        decay = torch.sigmoid(self.a_log).clamp(1e-4, 1 - 1e-4).view(1, 1, -1).to(x.dtype)
        steps = x.size(1)
        powers = torch.arange(steps, device=x.device, dtype=x.dtype).view(1, steps, 1)
        decay_pow = torch.pow(decay, powers)
        weighted_x = (1.0 - decay) * x / decay_pow
        return decay_pow * torch.cumsum(weighted_x, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x_proj, gate = self.in_proj(x).chunk(2, dim=-1)
        x_proj = self.dwconv(x_proj.transpose(1, 2)).transpose(1, 2)

        forward = self._scan(x_proj)
        backward = self._scan(torch.flip(x_proj, dims=[1]))
        backward = torch.flip(backward, dims=[1])

        y = 0.5 * (forward + backward) + self.skip.view(1, 1, -1) * x_proj
        y = torch.sigmoid(gate) * y
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


class SpectralGroupMambaBlock(nn.Module):
    """Grouped spectral relation block inspired by SpeMB in MambaHSI."""

    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.LayerNorm(dim)
        self.group_proj = nn.Linear(dim, dim)
        self.state_block = MambaStateSpaceBlock(dim, expand_ratio=1.5, kernel_size=3, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        group_size = max(1, t // self.num_groups)
        usable = group_size * self.num_groups
        x_main = x[:, :usable, :]
        x_rest = x[:, usable:, :]
        grouped = x_main.reshape(b, self.num_groups, group_size, d).mean(dim=2)
        grouped = self.state_block(self.group_proj(self.norm(grouped)))
        grouped = grouped.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(b, usable, d)
        out = torch.cat([grouped, x_rest], dim=1) if x_rest.numel() > 0 else grouped
        return x + out[:, :t, :]


class MambaHSIClassifier(nn.Module):
    """
    Spectral-only adaptation inspired by MambaHSI.

    Retained ideas:
    - full-sequence state-space modeling (SpaMB-inspired)
    - grouped spectral relation modeling (SpeMB-inspired)
    - adaptive fusion of dual spectral branches (SSFM-inspired)
    """

    def __init__(
        self,
        input_len: int = 1024,
        num_classes: int = 3,
        d_model: int = 96,
        patch_size: int = 16,
        num_layers: int = 3,
        num_groups: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = math.ceil(input_len / patch_size)

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.conv_embed = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.token_pool = nn.AdaptiveAvgPool1d(self.num_tokens)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.global_blocks = nn.ModuleList([
            MambaStateSpaceBlock(d_model, expand_ratio=2.0, kernel_size=3, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.group_blocks = nn.ModuleList([
            SpectralGroupMambaBlock(d_model, num_groups=num_groups, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        target_len = self.num_tokens * self.patch_size
        if x.size(1) < target_len:
            x = F.pad(x, (0, target_len - x.size(1)))
        elif x.size(1) > target_len:
            x = x[:, :target_len]
        x = x.reshape(x.size(0), self.num_tokens, self.patch_size)
        return self.patch_embed(x)

    def _conv_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_embed(x.unsqueeze(1))
        x = self.token_pool(x).transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self._patchify(x)
        conv_tokens = self._conv_tokens(x)

        global_feat = self.dropout(patch_tokens + self.pos_embedding)
        grouped_feat = self.dropout(conv_tokens + self.pos_embedding)

        for block in self.global_blocks:
            global_feat = block(global_feat)
        for block in self.group_blocks:
            grouped_feat = block(grouped_feat)

        fusion_input = torch.cat([global_feat, grouped_feat], dim=-1)
        gate = self.fusion_gate(fusion_input)
        fused = gate * global_feat + (1 - gate) * grouped_feat
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


class ConvTranTAPE1D(nn.Module):
    """time Absolute Position Encoding adapted from the official ConvTran repository."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024, scale_factor: float = 1.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        self.register_buffer('pe', scale_factor * pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class ConvTranERPEAttention1D(nn.Module):
    """efficient Relative Position Encoding adapted from the official ConvTran repository."""

    def __init__(self, emb_size: int, num_heads: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * seq_len - 1), num_heads))
        positions = torch.arange(seq_len)
        relative_index = (positions[None, :] - positions[:, None] + seq_len - 1).reshape(-1, 1)
        self.register_buffer('relative_index', relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = relative_bias.reshape(seq_len, seq_len, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        attn = self.dropout(attn + relative_bias)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out)


class ConvTranBlock1D(nn.Module):
    """Single ConvTran block: attention with eRPE followed by feed-forward projection."""

    def __init__(self, emb_size: int, num_heads: int, seq_len: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = ConvTranERPEAttention1D(emb_size, num_heads, seq_len, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(emb_size, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x))
        return self.norm2(x + self.ffn(x))


class ConvTranClassifier(nn.Module):
    """
    1D spectral adaptation of ConvTran.

    Keeps the ConvTran core ideas:
    - convolutional front-end
    - tAPE absolute positional encoding
    - eRPE relative positional bias after softmax
    """

    def __init__(
        self,
        input_len: int = 1024,
        num_classes: int = 3,
        emb_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_ff: int = 128,
        patch_size: int = 16,
        conv_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = math.ceil(input_len / patch_size)

        hidden_channels = emb_size * conv_expansion
        self.embed = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=8, padding='same', bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, emb_size, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(emb_size),
            nn.GELU(),
        )
        self.tokenizer = nn.Conv1d(emb_size, emb_size, kernel_size=patch_size, stride=patch_size, bias=False)
        self.position = ConvTranTAPE1D(emb_size, dropout=dropout, max_len=self.num_tokens)
        self.blocks = nn.ModuleList([
            ConvTranBlock1D(emb_size, num_heads, self.num_tokens, dim_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.embed(x)
        target_len = self.num_tokens * self.patch_size
        if x.size(-1) < target_len:
            x = F.pad(x, (0, target_len - x.size(-1)))
        elif x.size(-1) > target_len:
            x = x[..., :target_len]
        x = self.tokenizer(x).transpose(1, 2)
        x = self.position(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class DepthwiseSeparableConv1DBlock(nn.Module):
    """深度可分离卷积块。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding='same',
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HybridFixedFilterBank1D(nn.Module):
    """固定滤波器组。"""

    def __init__(self, kernel_sizes: list[int] | None = None):
        super().__init__()
        kernel_sizes = kernel_sizes or [2, 4, 8, 16, 32, 64]
        self.filters = nn.ModuleList()

        for kernel_size in kernel_sizes:
            weight = torch.ones(1, 1, kernel_size)
            weight[..., ::2] *= -1
            self.filters.append(self._make_fixed_conv(weight, kernel_size))

        for kernel_size in kernel_sizes:
            weight = torch.ones(1, 1, kernel_size)
            weight[..., 1::2] *= -1
            self.filters.append(self._make_fixed_conv(weight, kernel_size))

        for kernel_size in kernel_sizes[1:]:
            total_kernel = kernel_size + kernel_size // 2
            weight = torch.zeros(1, 1, total_kernel)
            xmash = torch.linspace(0, 1, steps=kernel_size // 4 + 1, dtype=torch.float32)[1:]
            left = xmash.square()
            right = torch.flip(left, dims=[0])
            weight[..., 0:kernel_size // 4] = -left.view(1, 1, -1)
            weight[..., kernel_size // 4:kernel_size // 2] = -right.view(1, 1, -1)
            weight[..., kernel_size // 2:3 * kernel_size // 4] = 2 * left.view(1, 1, -1)
            weight[..., 3 * kernel_size // 4:kernel_size] = 2 * right.view(1, 1, -1)
            weight[..., kernel_size:5 * kernel_size // 4] = -left.view(1, 1, -1)
            weight[..., 5 * kernel_size // 4:] = -right.view(1, 1, -1)
            self.filters.append(self._make_fixed_conv(weight, total_kernel))

    @staticmethod
    def _make_fixed_conv(weight: torch.Tensor, kernel_size: int) -> nn.Conv1d:
        conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding='same', bias=False)
        conv.weight = nn.Parameter(weight, requires_grad=False)
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [conv(x) for conv in self.filters]
        return F.relu(torch.cat(outputs, dim=1))


class LITEInceptionBlock1D(nn.Module):
    """Inception 风格特征块。"""

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int,
        dilation_rate: int = 1,
        use_custom_filters: bool = True,
        use_multiplexing: bool = True,
    ):
        super().__init__()
        self.use_custom_filters = use_custom_filters
        self.hybrid = HybridFixedFilterBank1D() if use_custom_filters else None

        if use_multiplexing:
            n_convs = 3
            out_filters = n_filters
        else:
            n_convs = 1
            out_filters = n_filters * 3

        kernel_sizes = [max(3, kernel_size // (2 ** i)) for i in range(n_convs)]
        self.branches = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                out_filters,
                kernel_size=k,
                padding='same',
                dilation=dilation_rate,
                bias=False,
            )
            for k in kernel_sizes
        ])

        branch_channels = out_filters * len(self.branches)
        if use_custom_filters:
            branch_channels += len(self.hybrid.filters)

        self.post = nn.Sequential(
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        if self.hybrid is not None:
            outputs.append(self.hybrid(x))
        return self.post(torch.cat(outputs, dim=1))


class LITEClassifier(nn.Module):
    """轻量单头基线模型。"""

    def __init__(
        self,
        input_len: int = 1024,
        num_classes: int = 3,
        n_filters: int = 32,
        kernel_size: int = 41,
        use_custom_filters: bool = True,
        use_dilation: bool = True,
        use_multiplexing: bool = True,
    ):
        super().__init__()
        base_kernel = max(8, kernel_size - 1)
        self.inception = LITEInceptionBlock1D(
            in_channels=1,
            n_filters=n_filters,
            kernel_size=base_kernel,
            dilation_rate=1,
            use_custom_filters=use_custom_filters,
            use_multiplexing=use_multiplexing,
        )

        inception_channels = n_filters * (3 if use_multiplexing else 1)
        if use_custom_filters:
            inception_channels += len(self.inception.hybrid.filters)

        dilation_rates = [2, 4] if use_dilation else [1, 1]
        self.fcn_blocks = nn.Sequential(
            DepthwiseSeparableConv1DBlock(
                inception_channels,
                kernel_size=max(5, base_kernel // 2),
                dilation=dilation_rates[0],
            ),
            DepthwiseSeparableConv1DBlock(
                inception_channels,
                kernel_size=max(3, base_kernel // 4),
                dilation=dilation_rates[1],
            ),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(inception_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.inception(x)
        x = self.fcn_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class PLSDA:
    """
    传统线性判别基线。
    """

    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1
        self.pls.fit(X, Y)
        return self

    def predict(self, X):
        Y_pred = self.pls.predict(X)
        return self.classes_[np.argmax(Y_pred, axis=1)]

    def predict_proba(self, X):
        """返回预测概率"""
        Y_pred = self.pls.predict(X)
        # 将PLS输出转换为概率
        Y_pred = np.clip(Y_pred, 0, None)
        row_sums = Y_pred.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return Y_pred / row_sums
