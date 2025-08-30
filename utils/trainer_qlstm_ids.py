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
from tqdm import tqdm  # 添加进度条库

seq_len = 2
input_length = 10
num_outputs = 5


class TrainerQLSTMIDS:
    def __init__(self, 
                 model, 
                 trainloader, 
                 validationloader, 
                 train_batch_size, 
                 val_batch_size, 
                 num_epochs, 
                 criterion, 
                 optimizer, 
                 model_folder,
                 device):
        self.model = model
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_folder = model_folder
        self.device = device
        self.epoch_loss = np.zeros(num_epochs)
        self.avg_epoch_loss_per_batch = np.zeros(num_epochs)
        self.total_samples = len(trainloader.dataset)
        self.batches = math.ceil(self.total_samples/self.train_batch_size)
        # 添加验证损失和准确度记录
        self.valid_loss = np.zeros(num_epochs)
        self.avg_valid_loss = np.zeros(num_epochs)
        self.train_acc = np.zeros(num_epochs)
        self.valid_acc = np.zeros(num_epochs)

    def compute_accuracy(self, outputs, labels):
        """计算准确度"""
        # 根据你的任务类型调整准确度计算方式
        # 这里假设是多分类问题，使用argmax获取预测类别
        _, preds = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        correct = (preds == labels).float().sum()
        accuracy = correct / labels.shape[0]
        return accuracy.item()

    def train(self):
        for j in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            running_val_loss = 0.0
            running_acc = 0.0
            running_val_acc = 0.0
            count = 0
            count_val = 0  # 修复未定义变量
            
            # 使用tqdm添加进度条
            train_pbar = tqdm(self.trainloader, desc=f'Epoch {j+1}/{self.num_epochs} [Train]')
            
            epoch_time_start = time.time()
            
            for k, (inputs, labels) in enumerate(train_pbar):
                inputs = inputs.reshape([self.train_batch_size, seq_len, input_length])
                inputs = inputs.to(self.device)
                outputs = self.model(inputs, self.train_batch_size)
                outputs = outputs.cpu()
                labels = labels.reshape([self.train_batch_size, num_outputs])
                outputs = outputs.reshape([self.train_batch_size, num_outputs])
                
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                
                # 计算准确度
                acc = self.compute_accuracy(outputs, labels)
                running_acc += acc
                running_loss += loss.item()
                count += 1
                
                # 更新进度条信息
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc:.4f}',
                    'AvgLoss': f'{running_loss/count:.4f}',
                    'AvgAcc': f'{running_acc/count:.4f}'
                })
            
            # 验证阶段
            self.model.eval()
            val_pbar = tqdm(self.validationloader, desc=f'Epoch {j+1}/{self.num_epochs} [Val]')
            
            with torch.no_grad():
                for l, (val_inputs, val_labels) in enumerate(val_pbar):
                    val_inputs = val_inputs.reshape([self.val_batch_size, seq_len, input_length])
                    val_inputs = val_inputs.to(self.device)
                    val_outputs = self.model(val_inputs, self.val_batch_size)
                    val_outputs = val_outputs.cpu()
                    val_labels = val_labels.reshape([self.val_batch_size, num_outputs])
                    val_outputs = val_outputs.reshape([self.val_batch_size, num_outputs])
                    
                    val_loss = self.criterion(val_outputs, val_labels)
                    val_acc = self.compute_accuracy(val_outputs, val_labels)
                    
                    running_val_loss += val_loss.item()
                    running_val_acc += val_acc
                    count_val += 1
                    
                    # 更新验证进度条
                    val_pbar.set_postfix({
                        'ValLoss': f'{val_loss.item():.4f}',
                        'ValAcc': f'{val_acc:.4f}',
                        'AvgValLoss': f'{running_val_loss/count_val:.4f}',
                        'AvgValAcc': f'{running_val_acc/count_val:.4f}'
                    })
            
            # 记录 epoch 统计信息
            self.epoch_loss[j] = running_loss
            self.avg_epoch_loss_per_batch[j] = running_loss / count
            self.valid_loss[j] = running_val_loss
            self.avg_valid_loss[j] = running_val_loss / count_val
            self.train_acc[j] = running_acc / count
            self.valid_acc[j] = running_val_acc / count_val
            
            epoch_time_end = time.time()
            total_epoch_time = epoch_time_end - epoch_time_start
            
            print(f'Epoch {j+1}/{self.num_epochs} Summary:')
            print(f'Train - Loss: {self.avg_epoch_loss_per_batch[j]:.4f}, Acc: {self.train_acc[j]:.4f}')
            print(f'Val - Loss: {self.avg_valid_loss[j]:.4f}, Acc: {self.valid_acc[j]:.4f}')
            print(f'Time: {total_epoch_time:.2f}s')
            print('-' * 50)
            
            # 保存模型
            path = os.path.join(self.model_folder, f'model_{j}.pt')
            if not os.path.exists(self.model_folder):
                os.makedirs(self.model_folder)
            torch.save(self.model.state_dict(), path)