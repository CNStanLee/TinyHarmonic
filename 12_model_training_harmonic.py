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
from models.qlstm_harmonic import QLSTMHarmonic, HarmonicEstimationLSTM, HarmonicEstimationMLP, MultiHeadHarmonicEstimationLSTM, MultiHeadHarmonicEstimationMLP, HarmonicEstimationCNNLSTM
from utils.trainer_qlstm_harmonic import TrainerQLSTMHarmonic
from utils.data_init import data_init
# --------------------------------------------------------
# set seed
# --------------------------------------------------------
torch.manual_seed(1998)
np.random.seed(1998)
os.environ['PYTHONHASHSEED'] = '1998'


def case_A1():
    # --------------------------------------------------------
    # model and data def
    # --------------------------------------------------------
    input_cycle_fraction = 0.25
    input_size = int(input_cycle_fraction * 64)
    hidden_size = 128
    num_layers = 2
    # --------------------------------------------------------
    # hyperparameters
    # --------------------------------------------------------
    num_epochs = 100
    lr = 0.001
    batch_size = 64
    weight_decay=1e-5
    early_stopping_patience=20
    # init the dataset
    voltage_loaders, current_loaders, simulation_loaders = data_init(batch_size=batch_size, input_cycle_fraction=input_cycle_fraction)
    train_loader = simulation_loaders['train']
    val_loader = simulation_loaders['val']
    test_loader = simulation_loaders['test']
    # init the model and training stuffs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_f32 = HarmonicEstimationLSTM(input_size=input_size
    #                                 , hidden_size=hidden_size,
    #                                   num_layers=num_layers).to(device)
    model_f32 = HarmonicEstimationMLP(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers).to(device)

    optimizer = optim.Adam(model_f32.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    trainer = TrainerQLSTMHarmonic(
        model=model_f32,
        trainloader=train_loader,
        validationloader=val_loader,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer=optimizer,
        model_folder="./models",
        device=device,
        loss_type="mse",
        lr_scheduler=scheduler,
        grad_clip=1.0,
        early_stopping_patience=early_stopping_patience
    )
    # train the model and test
    trainer.train()
    trainer.test(test_loader)
def case_real():
    # --------------------------------------------------------
    # model and data def
    # --------------------------------------------------------
    input_cycle_fraction = 1
    input_size = int(input_cycle_fraction * 64)
    hidden_size = 1024
    num_layers = 2
    # --------------------------------------------------------
    # hyperparameters
    # --------------------------------------------------------
    num_epochs = 500
    lr = 0.001
    batch_size = 128
    weight_decay=1e-5
    early_stopping_patience=50
    delete_temp_files=False
    # init the dataset
    voltage_loaders, current_loaders, simulation_loaders = data_init(batch_size=batch_size, input_cycle_fraction=input_cycle_fraction, delete_temp_files=delete_temp_files)
    train_loader = voltage_loaders['train']
    val_loader = voltage_loaders['val']
    test_loader = voltage_loaders['test']
    # init the model and training stuffs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_f32 = HarmonicEstimationLSTM(input_size=input_size
    #                                 , hidden_size=hidden_size,
    #                                   num_layers=num_layers).to(device)
    # model_f32 = MultiHeadHarmonicEstimationMLP(input_size=input_size,
    #                                 hidden_size=hidden_size,
    #                                 num_layers=num_layers).to(device)
    #batch_size = 16       # 一批样本数
    #seq_len = 50          # 序列长度 (时序长度)
    #input_size = 32       # 每个时间步输入特征维度
    #input_size = 64
    #hidden_size = 512     # MLP隐藏层维度
    lstm_hidden = 128     # LSTM隐藏层维度
    lstm_layers = 2       # LSTM层数
    model_f32 = HarmonicEstimationCNNLSTM(input_size=input_size,
                                           cnn_channels=64,
                                           kernel_size=3,
                                           lstm_hidden_size=128,
                                           lstm_num_layers=2,
                                           mlp_hidden_size=256,
                                           mlp_num_layers=3,
                                           dropout=0.2,
                                           num_heads=4).to(device)
    
    optimizer = optim.Adam(model_f32.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
    trainer = TrainerQLSTMHarmonic(
        model=model_f32,
        trainloader=train_loader,
        validationloader=val_loader,
        test_loader=test_loader,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer=optimizer,
        model_folder="./models",
        device=device,
        # loss_type="combined",
        epsilon=5e-3,
        loss_type="harmonic_log_mse",
        weights_arrange=torch.tensor([1.0, 1.0, 5.0, 5.0]),
        lr_scheduler=scheduler,
        grad_clip=1.0,
        early_stopping_patience=early_stopping_patience,
        error_metric="smape"
    )
    # train the model and test
    trainer.train()
    trainer.test(test_loader)


def main():
    #case_A1()
    case_real()



if __name__ == "__main__":
    main()