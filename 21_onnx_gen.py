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
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import tqdm
# ---------------------------------------------------------
from brevitas.export import export_onnx_qcdq, export_qonnx
# ---------------------------------------------------------
from models.models import CNNLSTM_f32
from models.qmodels import QCNNLSTM, QCNNLSTM_subCNN, QCNNLSTM_subLSTM, QCNNLSTM_subMLP
from utils.trainer_qlstm_harmonic import TrainerQLSTMHarmonic
#from utils.data_init import data_init
# --------------------------------------------------------
# set seed
# --------------------------------------------------------
torch.manual_seed(1998)
np.random.seed(1998)
os.environ['PYTHONHASHSEED'] = '1998'
# --------------------------------------------------------
# model and data def
# --------------------------------------------------------
input_cycle_fraction = 0.5
input_size = int(input_cycle_fraction * 64)

cnn_channels= input_size
kernel_size=3
lstm_hidden_size=128 # was 128
lstm_num_layers=1
mlp_hidden_size=512 # was 256 broader is much better
mlp_num_layers=1 # was 3
dropout=0.2
num_heads=4
# --------------------------------------------------------
# hyperparameters
# --------------------------------------------------------
num_epochs = 500
lr = 0.001
batch_size = 128
weight_decay=1e-5
early_stopping_patience=50
delete_temp_files=False
model_name = f"cnn_lstm_real_c{input_cycle_fraction}"
fmodel_name= f"f{model_name}"
qmodel_name= f"q{model_name}"

qmodel_pt_path = f"./models/{qmodel_name}/final_model.pt"
qmodel_pth_path = f"./models/{qmodel_name}/final_model.pth"

export_path = f"./models/{qmodel_name}/final_model.onnx"

sub_lstm_path = f"./models/{qmodel_name}/sublstm.onnx"
sub_cnn_path = f"./models/{qmodel_name}/subcnn.onnx"
sub_mlp_path = f"./models/{qmodel_name}/submlp.onnx"
# --------------------------------------------------------
# model definition
# --------------------------------------------------------
qmodel_def = QCNNLSTM(
                 input_size=input_size, 
                 cnn_channels=cnn_channels, 
                 kernel_size=kernel_size, 
                 lstm_hidden_size=lstm_hidden_size, 
                 lstm_num_layers=lstm_num_layers, 
                 mlp_hidden_size=mlp_hidden_size, 
                 mlp_num_layers=mlp_num_layers, 
                 dropout=dropout, 
                 num_heads=num_heads
                    )
subcnn_def = QCNNLSTM_subCNN(
                 input_size=input_size, 
                 cnn_channels=cnn_channels, 
                 kernel_size=kernel_size, 
                 lstm_hidden_size=lstm_hidden_size, 
                 lstm_num_layers=lstm_num_layers, 
                 mlp_hidden_size=mlp_hidden_size, 
                 mlp_num_layers=mlp_num_layers, 
                 dropout=dropout, 
                 num_heads=num_heads
                    )
sublstm_def = QCNNLSTM_subLSTM(
                 input_size=input_size, 
                 cnn_channels=cnn_channels, 
                 kernel_size=kernel_size, 
                 lstm_hidden_size=lstm_hidden_size, 
                 lstm_num_layers=lstm_num_layers, 
                 mlp_hidden_size=mlp_hidden_size, 
                 mlp_num_layers=mlp_num_layers, 
                 dropout=dropout, 
                 num_heads=num_heads
                    )
submlp_def = QCNNLSTM_subMLP(
                 input_size=input_size, 
                 cnn_channels=cnn_channels, 
                 kernel_size=kernel_size, 
                 lstm_hidden_size=lstm_hidden_size, 
                 lstm_num_layers=lstm_num_layers, 
                 mlp_hidden_size=mlp_hidden_size, 
                 mlp_num_layers=mlp_num_layers, 
                 dropout=dropout, 
                 num_heads=num_heads
                    )
def convert_pth():
    model = qmodel_def
    model.load_state_dict(torch.load(qmodel_pt_path),strict=False)
    # save the model as pth
    torch.save(model.state_dict(), qmodel_pth_path, _use_new_zipfile_serialization=False)
    print(f"Model saved to {qmodel_pth_path}")

def onnx_gen():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = qmodel_def
    model.load_state_dict(torch.load(qmodel_pth_path),strict=False)
    print("Model loaded")
    model.eval()
    random_input = torch.randn(batch_size, 1, input_size)
    output = model.forward(random_input)
    print(f"Q Output shape: {output.size()}")
    export_onnx_qcdq(model,random_input, opset_version=14, export_path=export_path)
    print(f"ONNX model exported to {export_path}")

    # gen sub1 model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = subcnn_def
    model.load_state_dict(torch.load(qmodel_pth_path),strict=False)
    print("Model loaded")
    model.eval()
    random_input = torch.randn(batch_size, 1, input_size)
    output = model.forward(random_input)
    print(f"Q Output shape: {output.size()}")
    export_onnx_qcdq(model,random_input, opset_version=14, export_path=sub_cnn_path)
    print(f"ONNX model exported to {sub_cnn_path}")

    # gen sub2 model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = sublstm_def
    model.load_state_dict(torch.load(qmodel_pth_path),strict=False)
    print("Model loaded")
    model.eval()
    random_input = torch.randn(output.size(0), output.size(1), output.size(2))
    output = model.forward(random_input)
    print(f"Q Output shape: {output.size()}")
    export_onnx_qcdq(model,random_input, opset_version=14, export_path=sub_lstm_path)
    print(f"ONNX model exported to {sub_lstm_path}")

    # gen sub3 model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = submlp_def
    model.load_state_dict(torch.load(qmodel_pth_path),strict=False)
    print("Model loaded")
    model.eval()
    random_input = torch.randn(output.size(0), output.size(1))
    print(f"Random Input shape: {random_input.size()}")
    output = model.forward(random_input)
    print(f"Q Output shape: {output.size()}")
    export_onnx_qcdq(model,random_input, opset_version=14, export_path=sub_mlp_path)
    print(f"ONNX model exported to {sub_mlp_path}")

def main():
    convert_pth()
    onnx_gen()


if __name__ == "__main__":
    main()