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

cnn_weight_bit_width = 8
lstm_weight_bit_width = 8
linear_weight_bit_width = 8
cnn_activation_bit_width = 6
lstm_activation_bit_width = 6
linear_activation_bit_width = 6



class CNNLSTM_f32(nn.Module):
    def __init__(self, input_size=32, cnn_channels=64, kernel_size=3, 
                 lstm_hidden_size=128, lstm_num_layers=2, 
                 mlp_hidden_size=256, mlp_num_layers=3, dropout=0.2, num_heads=4):
        super(CNNLSTM_f32, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels*2, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels*2),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
 
        self.lstm = nn.LSTM(
            input_size=cnn_channels*2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        

        self.mlp_heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []

            layers.append(nn.Linear(lstm_hidden_size, mlp_hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(mlp_hidden_size))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(mlp_num_layers - 1):
                layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(mlp_hidden_size))
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(mlp_hidden_size, 1))
            
            self.mlp_heads.append(nn.Sequential(*layers))
        
        self.init_weights()
    
    def init_weights(self):
        for layer in self.cnn:
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu', a=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
        
        for head in self.mlp_heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu', a=0.01)
                    if layer.bias is not None:
                        if layer == head[-1]: 
                            nn.init.constant_(layer.bias, 0.0)
                        else:
                            nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]

        outputs = []
        for head in self.mlp_heads:
            outputs.append(head(lstm_out))
        
        return torch.cat(outputs, dim=1) 
    


from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

class QCNNLSTM(nn.Module):
    def __init__(self, 
                 input_size=32, 
                 cnn_channels=64, 
                 kernel_size=3, 
                 lstm_hidden_size=128, 
                 lstm_num_layers=2, 
                 mlp_hidden_size=256, 
                 mlp_num_layers=3, 
                 dropout=0.2, 
                 num_heads=4,
                 # Add bit-width parameters with sensible defaults
                 cnn_weight_bit_width=8,
                 cnn_activation_bit_width=8,
                 lstm_weight_bit_width=8,
                 lstm_activation_bit_width=8,
                 linear_weight_bit_width=8,
                 linear_activation_bit_width=8):
        
        super(QCNNLSTM, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        # Input quantization layer
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        self.cnn = nn.Sequential(
            qnn.QuantConv1d(1, cnn_channels, kernel_size, padding=kernel_size//2,
                           weight_quant=Int8WeightPerTensorFloat,
                           weight_bit_width=cnn_weight_bit_width,
                           bias=True,
                           input_quant=Int8ActPerTensorFloat,
                           return_quant_tensor=True),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat, 
                         bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            
            qnn.QuantConv1d(cnn_channels, cnn_channels*2, kernel_size, padding=kernel_size//2,
                           weight_quant=Int8WeightPerTensorFloat,
                           weight_bit_width=cnn_weight_bit_width,
                           bias=True,
                           input_quant=Int8ActPerTensorFloat,
                           return_quant_tensor=True),
            qnn.QuantReLU(act_quant=Int8ActPerTensorFloat,
                         bit_width=cnn_activation_bit_width),
            nn.BatchNorm1d(cnn_channels*2),
            nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool1d(16)
        )
        
        self.qlstm = qnn.QuantLSTM(
            input_size=cnn_channels*2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=lstm_weight_bit_width,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_bit_width=lstm_activation_bit_width,
            sigmoid_bit_width=lstm_activation_bit_width,
            tanh_bit_width=lstm_activation_bit_width,
            cell_state_bit_width=lstm_activation_bit_width,
            bias_quant=None)
        
        self.mlp_heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []
            
            layers.append(qnn.QuantLinear(lstm_hidden_size, mlp_hidden_size,
                                        weight_quant=Int8WeightPerTensorFloat,
                                        weight_bit_width=linear_weight_bit_width,
                                        bias=True,
                                        input_quant=Int8ActPerTensorFloat,
                                        return_quant_tensor=True))
            layers.append(qnn.QuantReLU(act_quant=Int8ActPerTensorFloat,
                                      bit_width=linear_activation_bit_width))
            layers.append(nn.BatchNorm1d(mlp_hidden_size))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(mlp_num_layers - 1):
                layers.append(qnn.QuantLinear(mlp_hidden_size, mlp_hidden_size,
                                            weight_quant=Int8WeightPerTensorFloat,
                                            weight_bit_width=linear_weight_bit_width,
                                            bias=True,
                                            input_quant=Int8ActPerTensorFloat,
                                            return_quant_tensor=True))
                layers.append(qnn.QuantReLU(act_quant=Int8ActPerTensorFloat,
                                          bit_width=linear_activation_bit_width))
                layers.append(nn.BatchNorm1d(mlp_hidden_size))
                layers.append(nn.Dropout(dropout))
            
            layers.append(qnn.QuantLinear(mlp_hidden_size, 1,
                                        weight_quant=Int8WeightPerTensorFloat,
                                        weight_bit_width=linear_weight_bit_width,
                                        bias=True))
            
            self.mlp_heads.append(nn.Sequential(*layers))
        
        self.init_weights()
    
    def init_weights(self):
        for layer in self.cnn:
            if isinstance(layer, qnn.QuantConv1d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)
        
        for name, param in self.qlstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
        
        for head in self.mlp_heads:
            for layer in head:
                if isinstance(layer, qnn.QuantLinear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.01)
                    if layer.bias is not None:
                        if layer == head[-1]: 
                            nn.init.constant_(layer.bias, 0.0)
                        else:
                            nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        # Quantize input
        x = self.input_quant(x)
        
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).requires_grad_().to(x.device)

        lstm_out, _ = self.qlstm(cnn_out, (h0.detach(), c0.detach()))
        lstm_out = lstm_out[:, -1, :]
        
        outputs = []
        for head in self.mlp_heads:
            outputs.append(head(lstm_out))
        
        return torch.cat(outputs, dim=1)