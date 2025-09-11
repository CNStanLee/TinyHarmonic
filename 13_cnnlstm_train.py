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
from models.models import CNNLSTM_f32
from models.qmodels import QCNNLSTM
from utils.trainer_qlstm_harmonic import TrainerQLSTMHarmonic
from utils.data_init import data_init
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
dropout=0.1
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
fmodel_pt_path = f"./models/{fmodel_name}/final_model.pt"
qmodel_pt_path = f"./models/{qmodel_name}/final_model.pt"
# --------------------------------------------------------
# model definition
# --------------------------------------------------------
fmodel_def = CNNLSTM_f32(input_size=input_size,
                                           cnn_channels=cnn_channels,
                                           kernel_size=kernel_size,
                                           lstm_hidden_size=lstm_hidden_size,
                                           lstm_num_layers=lstm_num_layers,
                                           mlp_hidden_size=mlp_hidden_size,
                                           mlp_num_layers=mlp_num_layers,
                                           dropout=dropout,
                                           num_heads=num_heads)


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

    

def float_training(fmodel_name, fmodel_def):

    # init the dataset
    voltage_loaders, current_loaders, simulation_loaders = data_init(batch_size=batch_size, input_cycle_fraction=input_cycle_fraction, delete_temp_files=delete_temp_files)
    use_loader = current_loaders
    train_loader = use_loader['train']
    val_loader = use_loader['val']
    test_loader = use_loader['test']
    test_input_scale = use_loader['test_input_scale']
    test_output_scale = use_loader['test_output_scale']
    # init the model and training stuffs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_f32 = fmodel_def.to(device)
    optimizer = optim.Adam(model_f32.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    trainer = TrainerQLSTMHarmonic(
        model=model_f32,
        trainloader=train_loader,
        validationloader=val_loader,
        test_loader=test_loader,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer=optimizer,
        model_folder=f"./models/{fmodel_name}",
        device=device,
        epsilon=5e-3,
        loss_type="harmonic_log_mse",
        weights_arrange=torch.tensor([1.0, 1.0, 5.0, 5.0]),
        lr_scheduler=scheduler,
        grad_clip=1.0,
        early_stopping_patience=early_stopping_patience,
        error_metric="smape",
        test_input_scale=test_input_scale,
        test_output_scale=test_output_scale
    )
    trainer.train()
    trainer.test(test_loader)
    #trainer.test_fft(test_loader)

def QAT(qmodel_name, qmodel_def, fmodel_pt_path):
    # init the dataset
    voltage_loaders, current_loaders, simulation_loaders = data_init(batch_size=batch_size, input_cycle_fraction=input_cycle_fraction, delete_temp_files=delete_temp_files)
    use_loader = current_loaders
    train_loader = use_loader['train']
    val_loader = use_loader['val']
    test_loader = use_loader['test']
    # init the model and training stuffs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create the quantized model
    
    qmodel = qmodel_def.to(device)

    random_input = torch.randn(batch_size, 1, input_size).to(device)
    print(f"Input shape: {random_input.size()}")
    output = qmodel.forward(random_input)
    # #print(f"QAT model output: {output}")
    print(f"Q Output shape: {output.size()}")


    optimizer = optim.Adam(qmodel.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )


    trainer = TrainerQLSTMHarmonic(
        model=qmodel,
        trainloader=train_loader,
        validationloader=val_loader,
        test_loader=test_loader,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer=optimizer,
        model_folder=f"./models/{qmodel_name}",
        device=device,
        epsilon=5e-3,
        loss_type="harmonic_log_mse",
        weights_arrange=torch.tensor([1.0, 1.0, 5.0, 5.0]),
        lr_scheduler=scheduler,
        grad_clip=1.0,
        early_stopping_patience=early_stopping_patience,
        error_metric="smape"
    )
    trainer.train()
    trainer.test(test_loader)
    
    
    # there is no need to test fft cuz 0.5 cycle wont work for fft
    #trainer.test_fft(test_loader)

def main():
    retrain_f32 = True

    # if not os.path.exists(fmodel_pt_path):
    #     float_training(fmodel_name, fmodel_def)
    # elif retrain_f32:
    #     float_training(fmodel_name, fmodel_def)
    # else:
    #     print(f"Float model {fmodel_name} already trained, skip float training.")
    
    # start QAT
    #export env : BREVITAS_JIT=1
#     (py310) changhong@changhong-Alienware-m16-R2:~/prj/finn_dev/finn$ export BREVITAS_JIT=1
# (   (py310) changhong@changhong-Alienware-m16-R2:~/prj/finn_dev/finn$ export -p
    os.environ["BREVITAS_JIT"] = "1"

    #
    QAT(qmodel_name, qmodel_def, fmodel_pt_path)

if __name__ == "__main__":
    main()