"""
消融模型3: CNN-only (移除Attention和BiLSTM)

结构:
Input → CNN Block → AdaptiveAvgPool → FC → Output

与完整模型的区别:
- 移除了BiLSTM层
- 移除了Attention机制
- 使用全局平均池化聚合特征
"""

import torch
import torch.nn as nn


class RamanCNNOnly(nn.Module):
    """
    CNN-only模型 (无BiLSTM和Attention)

    消融实验: 验证BiLSTM+Attention的联合贡献

    Args:
        input_len: 输入光谱长度 (默认1024)
        num_classes: 分类类别数 (默认3)
    """

    def __init__(self, input_len: int = 1024, num_classes: int = 3):
        super().__init__()

        # CNN特征提取 (与完整模型相同的前两层)
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

        # 全局平均池化 (替代LSTM+Attention)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接分类器
        # 输入维度为64 (CNN输出通道数)
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

        # 全局平均池化
        x = self.global_pool(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)

        return self.fc(x)
