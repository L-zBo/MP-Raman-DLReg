"""
消融模型1: CNN-LSTM (移除Attention)

结构:
Input → CNN Block → BiLSTM → 取最后时间步 → FC → Output

与完整模型的区别:
- 移除了注意力机制
- 使用BiLSTM最后时间步的输出作为特征
"""

import torch
import torch.nn as nn


class RamanCNNLSTMNoAttention(nn.Module):
    """
    CNN-LSTM模型 (无Attention)

    消融实验: 验证Attention模块的贡献

    Args:
        input_len: 输入光谱长度 (默认1024)
        num_classes: 分类类别数 (默认3)
    """

    def __init__(self, input_len: int = 1024, num_classes: int = 3):
        super().__init__()

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

        # BiLSTM时序建模 (与完整模型相同)
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=True)

        # 全连接分类器 (与完整模型相同)
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

        # 不使用注意力，直接取最后时间步
        x = x[:, -1, :]  # (batch, 128)

        return self.fc(x)
