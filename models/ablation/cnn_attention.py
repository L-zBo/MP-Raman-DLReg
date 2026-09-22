"""
消融模型2: CNN-Attention (移除BiLSTM)

结构:
Input → CNN Block → Self-Attention → FC → Output

与完整模型的区别:
- 移除了BiLSTM层
- CNN输出直接接Self-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RamanCNNAttention(nn.Module):
    """
    CNN-Attention模型 (无BiLSTM)

    消融实验: 验证BiLSTM模块的贡献

    Args:
        input_len: 输入光谱长度 (默认1024)
        num_classes: 分类类别数 (默认3)
        return_attention: 是否返回注意力权重 (默认False)
    """

    def __init__(self, input_len: int = 1024, num_classes: int = 3, return_attention: bool = False):
        super().__init__()
        self.return_attention = return_attention

        # CNN特征提取 (与完整模型相同)
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

        # 注意力机制 (直接作用于CNN输出)
        # CNN输出通道数为64
        self.attn = nn.Linear(64, 1)

        # 全连接分类器
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.conv(x)     # (batch, 64, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 64)

        # 注意力加权 (不经过LSTM)
        attn_w = F.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
        x = (x * attn_w).sum(dim=1)  # (batch, 64)

        out = self.fc(x)

        if self.return_attention:
            return out, attn_w.squeeze(-1)
        return out
