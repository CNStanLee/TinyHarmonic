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
import gdown
import os
import zipfile

train_x_path = './data/ids/ids/train_x.txt'
train_y_path = './data/ids/ids/train_y_multiclass.txt'
val_x_path = './data/ids/ids/val_x.txt'
val_y_path = './data/ids/ids/val_y_multiclass.txt'
test_x_path = './data/ids/ids/test_x.txt'
test_y_path = './data/ids/ids/test_y_multiclass.txt'

class IDSTrainDataset(Dataset):
    def __init__(self):
        x_load = np.loadtxt(train_x_path, delimiter=",", dtype=np.float32)
        y_load = np.loadtxt(train_y_path, delimiter=",", dtype=np.float32)
        #x_load = x_load - 128
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # Will allow us to get the length of our dataset
        return self.n_samples


class IDSValDataset(Dataset):
    def __init__(self):
        x_load = np.loadtxt(val_x_path, delimiter=",", dtype=np.float32)
        y_load = np.loadtxt(val_y_path, delimiter=",", dtype=np.float32)
        #x_load = x_load - 128
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]
        print("X_Load shape is = ",len(x_load))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # Will allow us to get the length of our dataset
        return self.n_samples
    
class IDSTestDataset(Dataset):
    def __init__(self):
        # will be mostly used for dataloading
        x_load = np.loadtxt(test_x_path, delimiter=",", dtype=np.float32)
        y_load = np.loadtxt(test_y_path, delimiter=",", dtype=np.float32)
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_ids_data():
# cyber_data
    file_id = "1CdLTcBdjL95zlAgB-onMuQqPJ6HJM4bA"
    output_path = "./data/ids/ids.zip"
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # if the file does not exist, download it
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
        unzip_file(output_path, "./data/ids/")
    print("IDS dataset downloaded and unzipped.")

def get_ids_loaders(batch_size):
    train_dataset = IDSTrainDataset()
    val_dataset = IDSValDataset()
    test_dataset = IDSTestDataset()
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
