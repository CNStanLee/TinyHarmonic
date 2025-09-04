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

# class HarmonicEstimationMLP(nn.Module):
#     def __init__(self, input_size=64, hidden_size=128, num_layers=2, dropout=0.2):
#         """
#         谐波估计MLP模型
        
#         参数:
#             input_size: 输入特征维度
#             hidden_size: 隐藏层维度
#             num_layers: 隐藏层数量
#             dropout: dropout概率
#         """
#         super(HarmonicEstimationMLP, self).__init__()
        
#         self.input_size = input_size
        
#         # 构建MLP层
#         layers = []
        
#         # 输入层
#         layers.append(nn.Linear(input_size, hidden_size))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout))
        
#         # 隐藏层
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(hidden_size, hidden_size))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))
        
#         # 输出层
#         layers.append(nn.Linear(hidden_size, 4))
        
#         # 组合所有层
#         self.mlp = nn.Sequential(*layers)
        
#         # 初始化权重
#         self.init_weights()
    
#     def init_weights(self):
#         """初始化模型权重"""
#         for layer in self.mlp:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 if layer.bias is not None:
#                     layer.bias.data.fill_(0.1)
    
#     def forward(self, x):
#         """
#         前向传播
        
#         参数:
#             x: 输入张量，形状为(batch_size, input_size)
            
#         返回:
#             输出张量，形状为(batch_size, 4)
#         """
#         # 确保输入是二维的 (batch_size, input_size)
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)
        
#         # MLP前向传播
#         output = self.mlp(x)
        
#         return output
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class MultiHeadHarmonicEstimationLSTM(nn.Module):
    def __init__(self, input_size=32, lstm_hidden_size=128, lstm_layers=2, 
                 mlp_hidden_size=256, mlp_layers=5, dropout=0.2):
        """
        带LSTM特征提取的多头谐波估计MLP模型
        
        参数:
            input_size: 输入特征维度
            lstm_hidden_size: LSTM隐藏层维度
            lstm_layers: LSTM层数
            mlp_hidden_size: MLP隐藏层维度
            mlp_layers: MLP隐藏层数量
            dropout: dropout概率
        """
        super(MultiHeadHarmonicEstimationLSTM, self).__init__()
        self.input_size = input_size
        
        # LSTM特征提取层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # LSTM输出后的全连接层，用于降维和特征提取
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, mlp_hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.Dropout(dropout)
        )
        
        # 共享的特征提取backbone (MLP部分)
        backbone_layers = []
        
        # 输入层
        backbone_layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
        backbone_layers.append(nn.LeakyReLU(negative_slope=0.01))
        backbone_layers.append(nn.BatchNorm1d(mlp_hidden_size))
        backbone_layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(mlp_layers - 1):
            backbone_layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
            backbone_layers.append(nn.LeakyReLU(negative_slope=0.01))
            backbone_layers.append(nn.BatchNorm1d(mlp_hidden_size))
            backbone_layers.append(nn.Dropout(dropout))
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 四个独立的输出头，每个对应一个谐波
        self.head1 = nn.Sequential(
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )

        self.head3 = nn.Sequential(
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )
        
        self.head5 = nn.Sequential(
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 2, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )
        
        self.head7 = nn.Sequential(
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 2, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, mlp_hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        # 初始化LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置遗忘门偏置为1，有助于梯度流动
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        # 初始化LSTM后的全连接层
        for layer in self.lstm_fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)
        
        # 初始化backbone
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)
        
        # 初始化各个头
        for head in [self.head1, self.head3, self.head5, self.head7]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                    if layer.bias is not None:
                        # 最后一层初始化为0，其他层初始化为0.1
                        if layer == head[-2]:  # 倒数第二层是线性层
                            nn.init.constant_(layer.bias, 0.0)
                        else:
                            nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, seq_len, input_size)
            
        返回:
            输出张量，形状为(batch_size, 4)
        """
        # 确保输入是三维的 (batch_size, seq_len, input_size)
        if x.dim() == 2:
            # 如果输入是二维的，添加序列维度
            x = x.unsqueeze(1)
        
        # 通过LSTM提取时序特征
        lstm_out, (hn, cn) = self.lstm(x)
        # 取最后一个时间步的输出
        lstm_features = lstm_out[:, -1, :]
        
        # 通过全连接层调整特征维度
        features = self.lstm_fc(lstm_features)
        
        # 通过共享backbone提取特征
        mlp_features = self.backbone(features)
        
        # 通过各个头预测不同谐波
        harmonic1 = self.head1(mlp_features)
        harmonic3 = self.head3(mlp_features)
        harmonic5 = self.head5(mlp_features)
        harmonic7 = self.head7(mlp_features)
        
        # 拼接所有谐波预测
        output = torch.cat([harmonic1, harmonic3, harmonic5, harmonic7], dim=1)
        
        return output
class MultiHeadHarmonicEstimationMLP(nn.Module):
    def __init__(self, input_size=32, hidden_size=256, num_layers=5, dropout=0.2):
        """
        多头谐波估计MLP模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: 隐藏层数量
            dropout: dropout概率
        """
        super(MultiHeadHarmonicEstimationMLP, self).__init__()
        self.input_size = input_size
        
        # 共享的特征提取backbone
        backbone_layers = []
        
        # 输入层
        backbone_layers.append(nn.Linear(input_size, hidden_size))
        backbone_layers.append(nn.LeakyReLU(negative_slope=0.01))
        backbone_layers.append(nn.BatchNorm1d(hidden_size))
        backbone_layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            backbone_layers.append(nn.Linear(hidden_size, hidden_size))
            backbone_layers.append(nn.LeakyReLU(negative_slope=0.01))
            backbone_layers.append(nn.BatchNorm1d(hidden_size))
            backbone_layers.append(nn.Dropout(dropout))
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 四个独立的输出头，每个对应一个谐波
        self.head1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )

        
        self.head3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )
        
        self.head5 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )
        
        self.head7 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # 限制输出在[-1, 1]范围内
        )
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        """初始化模型权重"""
        # 初始化backbone
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)
        
        # 初始化各个头
        for head in [self.head1, self.head3, self.head5, self.head7]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                    if layer.bias is not None:
                        # 最后一层初始化为0，其他层初始化为0.1
                        if layer == head[-2]:  # 倒数第二层是线性层
                            nn.init.constant_(layer.bias, 0.0)
                        else:
                            nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为(batch_size, input_size)
            
        返回:
            输出张量，形状为(batch_size, 4)
        """
        # 确保输入是二维的 (batch_size, input_size)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 通过共享backbone提取特征
        features = self.backbone(x)
        
        # 通过各个头预测不同谐波
        harmonic1 = self.head1(features)
        harmonic3 = self.head3(features)
        harmonic5 = self.head5(features)
        harmonic7 = self.head7(features)
        
        # 拼接所有谐波预测
        output = torch.cat([harmonic1, harmonic3, harmonic5, harmonic7], dim=1)
        
        return output
class HarmonicEstimationMLP(nn.Module):
    def __init__(self, input_size=32, hidden_size=256, num_layers=5, dropout=0.2):
        super(HarmonicEstimationMLP, self).__init__()
        self.input_size = input_size
        
        layers = []
        # 输入层
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_size, 4))
        
        self.mlp = nn.Sequential(*layers)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                # 使用He初始化
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                if layer.bias is not None:
                    # 输出层偏差初始化为0，其他层初始化为0.1
                    if layer == self.mlp[-1]:  # 输出层
                        nn.init.constant_(layer.bias, 0.0)
                    else:
                        nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.mlp(x)
    
class HarmonicEstimationLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, dropout=0.2):

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