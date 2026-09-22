"""
RADAR-Net 双头模型
同时预测PP和PE的污染等级
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    专门针对类别不平衡问题设计，支持Label Smoothing

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha: 类别权重，可以是标量或张量
        gamma: 聚焦参数，γ越大对难分样本越关注（默认2.0）
        reduction: 'mean', 'sum', 或 'none'
        label_smoothing: 标签平滑系数（默认0.0，范围0-1）
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)

        # 应用Label Smoothing
        if self.label_smoothing > 0:
            # 将硬标签转换为软标签
            # 原标签位置: 1 - smoothing + smoothing/num_classes
            # 其他位置: smoothing/num_classes
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing + self.label_smoothing / num_classes)

            # 计算软标签的交叉熵
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            # 标准交叉熵
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 获取预测概率 p_t
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))

        # 计算focal weight: (1 - p_t)^γ
        focal_weight = (1 - pt) ** self.gamma

        # 应用类别权重 α
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            else:
                alpha = self.alpha
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DualHeadRamanCNNLSTM(nn.Module):
    """
    RADAR-Net 双头模型（独立注意力版本）

    结构:
    Input → CNN Block (共享) → BiLSTM (共享) → LSTM特征
                                                ├→ PP注意力 → PP加权特征 → PP分类头 → PP等级(0/1/2)
                                                └→ PE注意力 → PE加权特征 → PE分类头 → PE等级(0/1/2)

    Args:
        input_len: 输入光谱长度 (默认1024)
        num_classes: 每个头的分类类别数 (默认3)
        return_attention: 是否返回注意力权重
    """

    def __init__(
        self,
        input_len: int = 1024,
        num_classes: int = 3,
        return_attention: bool = False
    ):
        super().__init__()
        self.return_attention = return_attention
        self.num_classes = num_classes

        # 共享CNN特征提取器
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

        # 共享BiLSTM时序建模
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=True)

        # PP独立注意力机制
        self.pp_attn = nn.Linear(128, 1)

        # PE独立注意力机制
        self.pe_attn = nn.Linear(128, 1)

        # PP分类头
        self.pp_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

        # PE分类头
        self.pe_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def freeze_shared_layers(self) -> None:
        """冻结共享层（CNN + LSTM），只训练注意力和分类头"""
        for param in self.conv.parameters():
            param.requires_grad = False
        for param in self.lstm.parameters():
            param.requires_grad = False

    def unfreeze_shared_layers(self) -> None:
        """解冻共享层"""
        for param in self.conv.parameters():
            param.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True

    def freeze_pp_attention(self) -> None:
        """冻结PP注意力层"""
        for param in self.pp_attn.parameters():
            param.requires_grad = False

    def unfreeze_pp_attention(self) -> None:
        """解冻PP注意力层"""
        for param in self.pp_attn.parameters():
            param.requires_grad = True

    def freeze_pe_attention(self) -> None:
        """冻结PE注意力层"""
        for param in self.pe_attn.parameters():
            param.requires_grad = False

    def unfreeze_pe_attention(self) -> None:
        """解冻PE注意力层"""
        for param in self.pe_attn.parameters():
            param.requires_grad = True

    def freeze_pp_head(self) -> None:
        """冻结PP分类头"""
        for param in self.pp_head.parameters():
            param.requires_grad = False

    def unfreeze_pp_head(self) -> None:
        """解冻PP分类头"""
        for param in self.pp_head.parameters():
            param.requires_grad = True

    def freeze_pe_head(self) -> None:
        """冻结PE分类头"""
        for param in self.pe_head.parameters():
            param.requires_grad = False

    def unfreeze_pe_head(self) -> None:
        """解冻PE分类头"""
        for param in self.pe_head.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入光谱 (batch, features)

        Returns:
            pp_out: PP分类输出 (batch, num_classes)
            pe_out: PE分类输出 (batch, num_classes)
            pp_attn_w: PP注意力权重 (可选, batch, seq_len)
            pe_attn_w: PE注意力权重 (可选, batch, seq_len)
        """
        # 共享特征提取
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv(x)     # (batch, 64, features/16)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 64)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 128)

        # PP独立注意力加权
        pp_attn_w = F.softmax(self.pp_attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        pp_features = (lstm_out * pp_attn_w).sum(dim=1)  # (batch, 128)
        pp_out = self.pp_head(pp_features)

        # PE独立注意力加权
        pe_attn_w = F.softmax(self.pe_attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        pe_features = (lstm_out * pe_attn_w).sum(dim=1)  # (batch, 128)
        pe_out = self.pe_head(pe_features)

        if self.return_attention:
            return pp_out, pe_out, pp_attn_w.squeeze(-1), pe_attn_w.squeeze(-1)
        return pp_out, pe_out


class DualHeadLoss(nn.Module):
    """
    双头模型的联合损失函数
    支持Focal Loss、Label Smoothing和普通CrossEntropyLoss
    支持PP和PE使用不同的gamma值

    Args:
        pp_weight: PP损失的权重
        pe_weight: PE损失的权重
        class_weights_pp: PP分类的类别权重
        class_weights_pe: PE分类的类别权重
        use_focal: 是否使用Focal Loss（默认True）
        gamma_pp: PP的Focal Loss聚焦参数（默认2.0）
        gamma_pe: PE的Focal Loss聚焦参数（默认2.0）
        label_smoothing: 标签平滑系数（默认0.0）
    """

    def __init__(
        self,
        pp_weight: float = 1.0,
        pe_weight: float = 1.0,
        class_weights_pp: Optional[torch.Tensor] = None,
        class_weights_pe: Optional[torch.Tensor] = None,
        use_focal: bool = True,
        gamma: float = 2.0,
        gamma_pp: Optional[float] = None,
        gamma_pe: Optional[float] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.pp_weight = pp_weight
        self.pe_weight = pe_weight
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing

        # 如果没有单独指定，使用统一的gamma
        gamma_pp = gamma_pp if gamma_pp is not None else gamma
        gamma_pe = gamma_pe if gamma_pe is not None else gamma

        if use_focal:
            self.criterion_pp = FocalLoss(alpha=class_weights_pp, gamma=gamma_pp, label_smoothing=label_smoothing)
            self.criterion_pe = FocalLoss(alpha=class_weights_pe, gamma=gamma_pe, label_smoothing=label_smoothing)
        else:
            # nn.CrossEntropyLoss 从 PyTorch 1.10 起支持 label_smoothing
            try:
                self.criterion_pp = nn.CrossEntropyLoss(weight=class_weights_pp, label_smoothing=label_smoothing)
                self.criterion_pe = nn.CrossEntropyLoss(weight=class_weights_pe, label_smoothing=label_smoothing)
            except TypeError:
                import warnings
                warnings.warn(
                    f"当前PyTorch版本不支持CrossEntropyLoss的label_smoothing参数，"
                    f"label_smoothing={label_smoothing}将被忽略。请升级到PyTorch>=1.10"
                )
                self.criterion_pp = nn.CrossEntropyLoss(weight=class_weights_pp)
                self.criterion_pe = nn.CrossEntropyLoss(weight=class_weights_pe)

    def forward(
        self,
        pp_pred: torch.Tensor,
        pe_pred: torch.Tensor,
        pp_target: torch.Tensor,
        pe_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算联合损失

        Returns:
            total_loss: 总损失
            pp_loss: PP损失
            pe_loss: PE损失
        """
        pp_loss = self.criterion_pp(pp_pred, pp_target)
        pe_loss = self.criterion_pe(pe_pred, pe_target)
        total_loss = self.pp_weight * pp_loss + self.pe_weight * pe_loss

        return total_loss, pp_loss, pe_loss


RADARNet = DualHeadRamanCNNLSTM
ADMIC = DualHeadRamanCNNLSTM


if __name__ == '__main__':
    # 测试模型
    model = DualHeadRamanCNNLSTM(input_len=1024, num_classes=3)
    x = torch.randn(4, 1024)  # batch_size=4

    pp_out, pe_out = model(x)
    print(f"PP输出形状: {pp_out.shape}")  # (4, 3)
    print(f"PE输出形状: {pe_out.shape}")  # (4, 3)

    # 测试损失函数
    criterion = DualHeadLoss()
    pp_target = torch.randint(0, 3, (4,))
    pe_target = torch.randint(0, 3, (4,))

    total_loss, pp_loss, pe_loss = criterion(pp_out, pe_out, pp_target, pe_target)
    print(f"总损失: {total_loss.item():.4f}")
    print(f"PP损失: {pp_loss.item():.4f}")
    print(f"PE损失: {pe_loss.item():.4f}")
