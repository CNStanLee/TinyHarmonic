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


lstm_weight_bit_width = 8
linear_weight_bit_width = 8
lstm_activation_bit_width = 6
linear_activation_bit_width = 6


class QLSTMIDS(nn.Module):
    def __init__(self):
        super(QLSTMIDS, self).__init__()
        # To be uncommented while training an LSTM model.
        self.qlstm = bnn.QuantLSTM(input_size=10, hidden_size=20,num_layers=1,batch_first=True,
            weight_bit_width=lstm_weight_bit_width,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_bit_width=lstm_activation_bit_width,
            sigmoid_bit_width=lstm_activation_bit_width,
            tanh_bit_width=lstm_activation_bit_width,
            cell_state_bit_width=lstm_activation_bit_width,
            bias_quant=None)#Setting batch_first to "True" changed everything, Need to investigate why it worked.
        self.qfc1 = bnn.QuantLinear(20, 64,bias=True, weight_bit_width=linear_weight_bit_width)
        self.qfc2 = bnn.QuantLinear(64, 32,bias=True, weight_bit_width=linear_weight_bit_width)
        self.qfc3 = bnn.QuantLinear(32, 5,bias=True, weight_bit_width=linear_weight_bit_width)
        self.relu = nn.ReLU()
        self.qrelu = bnn.QuantReLU(bit_width=linear_activation_bit_width)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,batch_size):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1,batch_size, 20).requires_grad_().to("cuda:0")
        # Initialize cell state
        c0 = torch.zeros(1,batch_size, 20).requires_grad_().to("cuda:0")    
        #Start model definition
        out,(hn,cn) = self.qlstm(x,(h0.detach(),c0.detach())) 
        out = hn[-1, :, :]
        out = self.qrelu(out)
        out = self.qfc1(out)
        out = self.qrelu(out)
        out = self.qfc2(out)
        out = self.qrelu(out)
        out = self.qfc3(out)
        return out