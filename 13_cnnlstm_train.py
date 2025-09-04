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
# ---------------------------------------------------------
from models.models import CNNLSTM_f32, QCNNLSTM
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
input_cycle_fraction = 1
input_size = int(input_cycle_fraction * 64)
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
                                           cnn_channels=64,
                                           kernel_size=3,
                                           lstm_hidden_size=128,
                                           lstm_num_layers=2,
                                           mlp_hidden_size=256,
                                           mlp_num_layers=3,
                                           dropout=0.2,
                                           num_heads=4)
# qmodel_def = QCNNLSTM(input_size=input_size,
#                                            cnn_channels=64,
#                                            kernel_size=3,
#                                            lstm_hidden_size=128,
#                                            lstm_num_layers=2,
#                                            mlp_hidden_size=256,
#                                            mlp_num_layers=3,
#                                            dropout=0.2,
#                                            num_heads=4)

qmodel_def = QCNNLSTM(
                 input_size=input_size, 
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
                 linear_activation_bit_width=8)

    

def float_training(fmodel_name, fmodel_def):

    # init the dataset
    voltage_loaders, current_loaders, simulation_loaders = data_init(batch_size=batch_size, input_cycle_fraction=input_cycle_fraction, delete_temp_files=delete_temp_files)
    train_loader = voltage_loaders['train']
    val_loader = voltage_loaders['val']
    test_loader = voltage_loaders['test']
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
        error_metric="smape"
    )
    trainer.train()
    trainer.test(test_loader)

def QAT(qmodel_name, qmodel_def, fmodel_pt_path):
    # init the dataset
    voltage_loaders, current_loaders, simulation_loaders = data_init(batch_size=batch_size, input_cycle_fraction=input_cycle_fraction, delete_temp_files=delete_temp_files)
    train_loader = voltage_loaders['train']
    val_loader = voltage_loaders['val']
    test_loader = voltage_loaders['test']
    # init the model and training stuffs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create the quantized model
    qmodel = qmodel_def.to(device)
    # load the float model weights
    #qmodel.load_state_dict(torch.load(fmodel_pt_path))
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

def main():
    retrain_f32 = False
    # if model_pt_path this pt file exists, skip the float training
    if not os.path.exists(fmodel_pt_path):
        float_training(fmodel_name, fmodel_def)
    elif retrain_f32:
        float_training(fmodel_name, fmodel_def)
    else:
        print(f"Float model {fmodel_name} already trained, skip float training.")
    # start QAT
    QAT(qmodel_name, qmodel_def, fmodel_pt_path)

if __name__ == "__main__":
    main()