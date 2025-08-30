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
# ---------------------------------------------------------
from utils.ids_dataloader import IDSTrainDataset, IDSValDataset, IDSTestDataset, download_ids_data, get_ids_loaders
from models.qlstmids import QLSTMIDS
from utils.trainer_qlstm_ids import TrainerQLSTMIDS
# --------------------------------------------------------
# Setting basic environment
# --------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(1998)
np.random.seed(1998)



# --------------------------------------------------------
# hyperparameters
# --------------------------------------------------------
batch_size = 2000
epochs = 50
lr = 0.0001



def main():
    download_ids_data()
    train_loader, val_loader, test_loader = get_ids_loaders(batch_size)

    model = QLSTMIDS().to(device)
    print(model)
    print("No. of parameters in the model = ", sum(p.numel() for p in model.parameters() if p.requires_grad))


    model = model.float()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = TrainerQLSTMIDS(model=model,
                            trainloader=train_loader,
                            validationloader=val_loader,
                            train_batch_size=batch_size,
                            val_batch_size=batch_size,
                            num_epochs=epochs,
                            criterion=criterion,
                            optimizer=optimizer,
                            model_folder='checkpoints/ids_qlstm',
                            device=device)
    trainer.train()

if __name__ == "__main__":
    main()