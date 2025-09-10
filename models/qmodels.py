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
import torch.nn.functional as F

# brevitas imports
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat

# quant bit widths
cnn_weight_bit_width = 4
cnn_activation_bit_width = 4
lstm_weight_bit_width = 8
lstm_activation_bit_width = 6 # change to 6 need tot retrain the model
linear_weight_bit_width = 4
linear_activation_bit_width = 4

class QCNNLSTM(nn.Module):
    def __init__(self, input_size=32, cnn_channels=64, kernel_size=3, 
                lstm_hidden_size=128, lstm_num_layers=1, 
                mlp_hidden_size=256, mlp_num_layers=3, dropout=0.2, num_heads=4):
        super(QCNNLSTM, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.input_quant = qnn.QuantIdentity(bit_width=8)
        
        self.qcnn = nn.Sequential(
            qnn.QuantConv1d(in_channels=1, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2,
                            weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            qnn.QuantConv1d(cnn_channels, cnn_channels*2, kernel_size, padding=kernel_size//2, weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels*2),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.qlstm = bnn.QuantLSTM(input_size=cnn_channels*2, hidden_size=lstm_hidden_size,num_layers=lstm_num_layers,batch_first=True,
            weight_bit_width=lstm_weight_bit_width,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_bit_width=lstm_activation_bit_width,
            sigmoid_bit_width=lstm_activation_bit_width,
            tanh_bit_width=lstm_activation_bit_width,
            cell_state_bit_width=lstm_activation_bit_width,
            bias_quant=None)
        
        self.qmlp_heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []

            layers.append(qnn.QuantLinear(lstm_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
            layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
            layers.append(nn.BatchNorm1d(mlp_hidden_size))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(mlp_num_layers - 1):
                layers.append(qnn.QuantLinear(mlp_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
                layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
                layers.append(nn.BatchNorm1d(mlp_hidden_size))
                layers.append(nn.Dropout(dropout))

            layers.append(qnn.QuantLinear(mlp_hidden_size, 1, bias=True, weight_bit_width=linear_weight_bit_width))

            self.qmlp_heads.append(nn.Sequential(*layers))
        
    def forward(self, x):
        # if x.dim() == 2:
        #     x = x.unsqueeze(1)  # **Reshape input to (batch_size, 1, input_size)
        
        # 应用输入量化
        x = self.input_quant(x)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)

        qcnn_out = self.qcnn(x)
        qcnn_out = qcnn_out.permute(0, 2, 1)  # (batch_size, seq_len, features)
        qlstmout, (hn, cn) = self.qlstm(qcnn_out, (h0.detach(), c0.detach()))
        qlstmout = hn[-1, :, :]  # (batch_size, lstm_hidden_size)
        qlstmout = qlstmout.view(batch_size, -1)
        outputs = []
        for head in self.qmlp_heads:
            out = head(qlstmout)
            outputs.append(out)
        out = torch.cat(outputs, dim=1)  # (batch_size, num_heads)

        return out
    
class QCNNLSTM_subCNN(nn.Module):
    def __init__(self, input_size=32, cnn_channels=64, kernel_size=3, 
                lstm_hidden_size=128, lstm_num_layers=1, 
                mlp_hidden_size=256, mlp_num_layers=3, dropout=0.2, num_heads=4):
        super(QCNNLSTM_subCNN, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        # 添加8bit量化输入层
        self.input_quant = qnn.QuantIdentity(bit_width=8)
        
        self.qcnn = nn.Sequential(
            qnn.QuantConv1d(in_channels=1, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2,
                            weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            qnn.QuantConv1d(cnn_channels, cnn_channels*2, kernel_size, padding=kernel_size//2, weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels*2),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.qlstm = bnn.QuantLSTM(input_size=cnn_channels*2, hidden_size=lstm_hidden_size,num_layers=lstm_num_layers,batch_first=True,
            weight_bit_width=lstm_weight_bit_width,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_bit_width=lstm_activation_bit_width,
            sigmoid_bit_width=lstm_activation_bit_width,
            tanh_bit_width=lstm_activation_bit_width,
            cell_state_bit_width=lstm_activation_bit_width,
            bias_quant=None)
        
        self.qmlp_heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []

            layers.append(qnn.QuantLinear(lstm_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
            layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
            layers.append(nn.BatchNorm1d(mlp_hidden_size))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(mlp_num_layers - 1):
                layers.append(qnn.QuantLinear(mlp_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
                layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
                layers.append(nn.BatchNorm1d(mlp_hidden_size))
                layers.append(nn.Dropout(dropout))

            layers.append(qnn.QuantLinear(mlp_hidden_size, 1, bias=True, weight_bit_width=linear_weight_bit_width))

            self.qmlp_heads.append(nn.Sequential(*layers))
        
    def forward(self, x):
        # if x.dim() == 2:
        #     x = x.unsqueeze(1)  # **Reshape input to (batch_size, 1, input_size)
        
        # 应用输入量化
        x = self.input_quant(x)
        # print("forward in subCNN")
        # print(x.shape)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)

        qcnn_out = self.qcnn(x)
        qcnn_out = qcnn_out.permute(0, 2, 1)

        return qcnn_out

class QCNNLSTM_subLSTM(nn.Module):
    def __init__(self, input_size=32, cnn_channels=64, kernel_size=3, 
                lstm_hidden_size=128, lstm_num_layers=1, 
                mlp_hidden_size=256, mlp_num_layers=3, dropout=0.2, num_heads=4):
        super(QCNNLSTM_subLSTM, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        # 添加8bit量化输入层
        self.input_quant = qnn.QuantIdentity(bit_width=8)
        
        self.qcnn = nn.Sequential(
            qnn.QuantConv1d(in_channels=1, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2,
                            weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            qnn.QuantConv1d(cnn_channels, cnn_channels*2, kernel_size, padding=kernel_size//2, weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels*2),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.qlstm = bnn.QuantLSTM(input_size=cnn_channels*2, hidden_size=lstm_hidden_size,num_layers=lstm_num_layers,batch_first=True,
            weight_bit_width=lstm_weight_bit_width,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_bit_width=lstm_activation_bit_width,
            sigmoid_bit_width=lstm_activation_bit_width,
            tanh_bit_width=lstm_activation_bit_width,
            cell_state_bit_width=lstm_activation_bit_width,
            bias_quant=None)
        
        self.qmlp_heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []

            layers.append(qnn.QuantLinear(lstm_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
            layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
            layers.append(nn.BatchNorm1d(mlp_hidden_size))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(mlp_num_layers - 1):
                layers.append(qnn.QuantLinear(mlp_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
                layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
                layers.append(nn.BatchNorm1d(mlp_hidden_size))
                layers.append(nn.Dropout(dropout))

            layers.append(qnn.QuantLinear(mlp_hidden_size, 1, bias=True, weight_bit_width=linear_weight_bit_width))

            self.qmlp_heads.append(nn.Sequential(*layers))
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)

        qcnn_out = x
        qlstmout, (hn, cn) = self.qlstm(qcnn_out, (h0.detach(), c0.detach()))
        qlstmout = hn[-1, :, :]  # (batch_size, lstm_hidden_size)
        qlstmout = qlstmout.view(batch_size, -1)

        return qlstmout

class QCNNLSTM_subMLP(nn.Module):
    def __init__(self, input_size=32, cnn_channels=64, kernel_size=3, 
                lstm_hidden_size=128, lstm_num_layers=1, 
                mlp_hidden_size=256, mlp_num_layers=3, dropout=0.2, num_heads=4):
        super(QCNNLSTM_subMLP, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.input_quant = qnn.QuantIdentity(bit_width=8)
        
        self.qcnn = nn.Sequential(
            qnn.QuantConv1d(in_channels=1, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2,
                            weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            qnn.QuantConv1d(cnn_channels, cnn_channels*2, kernel_size, padding=kernel_size//2, weight_bit_width=cnn_weight_bit_width),
            qnn.QuantReLU(bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels*2),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.qlstm = bnn.QuantLSTM(input_size=cnn_channels*2, hidden_size=lstm_hidden_size,num_layers=lstm_num_layers,batch_first=True,
            weight_bit_width=lstm_weight_bit_width,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_bit_width=lstm_activation_bit_width,
            sigmoid_bit_width=lstm_activation_bit_width,
            tanh_bit_width=lstm_activation_bit_width,
            cell_state_bit_width=lstm_activation_bit_width,
            bias_quant=None)
        
        self.qmlp_heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []

            layers.append(qnn.QuantLinear(lstm_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
            layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
            layers.append(nn.BatchNorm1d(mlp_hidden_size))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(mlp_num_layers - 1):
                layers.append(qnn.QuantLinear(mlp_hidden_size, mlp_hidden_size, bias=True, weight_bit_width=linear_weight_bit_width))
                layers.append(qnn.QuantReLU(bit_width=linear_activation_bit_width))
                layers.append(nn.BatchNorm1d(mlp_hidden_size))
                layers.append(nn.Dropout(dropout))

            layers.append(qnn.QuantLinear(mlp_hidden_size, 1, bias=True, weight_bit_width=linear_weight_bit_width))

            self.qmlp_heads.append(nn.Sequential(*layers))
        
    def forward(self, x):
        qlstmout = x
        outputs = []
        for head in self.qmlp_heads:
            out = head(qlstmout)
            outputs.append(out)
        out = torch.cat(outputs, dim=1)  # (batch_size, num_heads)

        return out