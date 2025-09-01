import torch
import math
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import brevitas.nn as bnn
from brevitas.quant import Int8ActPerTensorFloat,Uint8ActPerTensorFloat
import itertools
import time
import os
import torch
import torch.nn as nn

class HarmonicEstimationLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, dropout=0.2):
        """
        LSTM谐波估计模型
        
        参数:
            input_size: 输入特征维度 (默认为64)
            hidden_size: LSTM隐藏层维度 (默认为128)
            num_layers: LSTM层数 (默认为2)
            dropout: Dropout概率 (默认为0.2)
        """
        super(HarmonicEstimationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, 4)
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.1)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, input_size)
            
        返回:
            输出张量，形状为(batch_size, 4)
        """
        # 如果输入是二维的，添加一个序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 形状变为(batch_size, 1, input_size)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 应用dropout
        dropped_out = self.dropout(last_output)
        
        # 全连接层
        output = self.fc(dropped_out)
        
        return output

lstm_weight_bit_width = 8
linear_weight_bit_width = 8
lstm_activation_bit_width = 6
linear_activation_bit_width = 6


class QLSTMHarmonic(nn.Module):
    def __init__(self, input_size=64, hidden_size=20, num_layers=1):
        super(QLSTMHarmonic, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 量化LSTM层
        self.qlstm = bnn.QuantLSTM(
            input_size=1,  # 每个时间步输入1个特征（幅度值）
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            weight_bit_width=lstm_weight_bit_width,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_bit_width=lstm_activation_bit_width,
            sigmoid_bit_width=lstm_activation_bit_width,
            tanh_bit_width=lstm_activation_bit_width,
            cell_state_bit_width=lstm_activation_bit_width,
            bias_quant=None
        )
        
        # 全连接层
        self.qfc1 = bnn.QuantLinear(hidden_size, 64, bias=True, weight_bit_width=linear_weight_bit_width)
        self.qfc2 = bnn.QuantLinear(64, 32, bias=True, weight_bit_width=linear_weight_bit_width)
        self.qfc3 = bnn.QuantLinear(32, 4, bias=True, weight_bit_width=linear_weight_bit_width)  # 输出4个幅度值
        
        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.qrelu = bnn.QuantReLU(bit_width=linear_activation_bit_width)

    def forward(self, x):
        # 确保输入数据类型正确
        x = x.float()  # 转换为float32
        
        # x形状: (batch_size, input_size) -> 需要重塑为 (batch_size, seq_len, features)
        x = x.view(-1, self.input_size, 1)  # 转换为(batch_size, seq_len, 1)
        
        # 让 QuantLSTM 处理隐藏状态初始化
        # 传递 None 作为隐藏状态，让 QuantLSTM 使用默认初始化
        out, _ = self.qlstm(x, None)
        
        # 取最后一个时间步的输出
        out = out[:, -1, :]  # 形状: (batch_size, hidden_size)
        
        # 全连接层
        out = self.qrelu(out)
        out = self.qfc1(out)
        out = self.qrelu(out)
        out = self.qfc2(out)
        out = self.qrelu(out)
        out = self.qfc3(out)
        
        # 使用ReLU确保输出为非负幅度值
        out = self.relu(out)
        
        return out  # 输出形状: (batch_size, 4)