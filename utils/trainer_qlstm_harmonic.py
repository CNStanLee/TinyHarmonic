import torch
import math
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import brevitas.nn as bnn
from brevitas.quant import Int8ActPerTensorFloat, Uint8ActPerTensorFloat
import itertools
import time
import os
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 自定义 TMAPE 损失函数
class TMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(TMAPELoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, outputs, targets):
        """
        计算 Targeted Mean Absolute Percentage Error (TMAPE)
        
        参数:
        outputs: 模型预测值
        targets: 真实值
        epsilon: 避免除以零的小常数
        
        返回:
        loss: TMAPE 损失值
        """
        # 计算绝对误差
        absolute_error = torch.abs(outputs - targets)
        
        # 计算分母，避免除以零
        denominator = torch.abs(targets) + self.epsilon
        
        # 计算每个元素的百分比误差
        percentage_error = absolute_error / denominator
        
        # 计算均值
        loss = torch.mean(percentage_error) * 100  # 转换为百分比
        
        return loss

class TrainerQLSTMHarmonic:
    def __init__(self, 
                 model, 
                 trainloader, 
                 validationloader, 
                 train_batch_size, 
                 val_batch_size, 
                 num_epochs, 
                 optimizer, 
                 model_folder,
                 device,
                 input_length=64,
                 epsilon=1e-8):
        self.model = model
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.model_folder = model_folder
        self.device = device
        self.input_length = input_length
        self.epsilon = epsilon
        
        # 使用 TMAPE 作为损失函数
        self.criterion = TMAPELoss(epsilon=epsilon)
        
        # 记录训练和验证指标
        self.train_loss = np.zeros(num_epochs)  # 存储 MSE
        self.val_loss = np.zeros(num_epochs)    # 存储 MSE
        self.train_mae = np.zeros(num_epochs)
        self.val_mae = np.zeros(num_epochs)
        self.train_r2 = np.zeros(num_epochs)
        self.val_r2 = np.zeros(num_epochs)
        self.train_tmape = np.zeros(num_epochs)  # 存储 TMAPE
        self.val_tmape = np.zeros(num_epochs)    # 存储 TMAPE
        
        # 创建模型文件夹
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

    def compute_metrics(self, outputs, labels):
        """计算回归任务的评估指标"""
        outputs_np = outputs.detach().numpy()
        labels_np = labels.detach().numpy()
        
        # 添加调试信息
        if np.random.rand() < 0.01:  # 随机选择1%的批次打印调试信息
            print(f"\nDebug Info:")
            print(f"Outputs range: [{np.min(outputs_np):.6f}, {np.max(outputs_np):.6f}]")
            print(f"Labels range: [{np.min(labels_np):.6f}, {np.max(labels_np):.6f}]")
            for i in range(4):
                print(f"Channel {i+1} - First 5 outputs: {outputs_np[:5, i]}")
                print(f"Channel {i+1} - First 5 labels: {labels_np[:5, i]}")
                abs_errors = np.abs(outputs_np[:5, i] - labels_np[:5, i])
                abs_labels = np.abs(labels_np[:5, i])
                denominator = np.where(abs_labels == 0, np.abs(outputs_np[:5, i]), abs_labels)
                denominator[denominator == 0] = 1e-8
                pe_values = abs_errors / denominator * 100
                print(f"Channel {i+1} - PE values: {pe_values}")
        
        mse = mean_squared_error(labels_np, outputs_np)
        mae = mean_absolute_error(labels_np, outputs_np)
        r2 = r2_score(labels_np, outputs_np)
        
        # 计算 TMAPE
        absolute_error = np.abs(outputs_np - labels_np)
        denominator = np.abs(labels_np) + self.epsilon
        tmape = np.mean(absolute_error / denominator) * 100
        
        # 计算每个通道的相对误差百分比
        percentage_errors = []
        for i in range(4):  # 四个通道
            # 计算每个样本的相对误差
            abs_errors = np.abs(outputs_np[:, i] - labels_np[:, i])
            abs_labels = np.abs(labels_np[:, i])
            
            # 避免除以零 - 对于真实值为零的情况，使用预测值的绝对值作为分母
            denominator = np.where(abs_labels == 0, np.abs(outputs_np[:, i]), abs_labels)
            denominator[denominator == 0] = 1e-8  # 确保分母不为零
            
            # 计算相对误差百分比
            pe = np.mean(abs_errors / denominator) * 100
            percentage_errors.append(pe)
        
        return mse, mae, r2, tmape, percentage_errors

    def train(self):
        best_val_tmape = float('inf')
        
        for epoch in range(self.num_epochs):
            # 训练阶段
            self.model.train()
            train_mse, train_mae, train_r2, train_tmape = 0.0, 0.0, 0.0, 0.0
            train_percentage_errors = [0.0, 0.0, 0.0, 0.0]  # 四个通道的相对误差百分比
            train_batches = 0
            
            train_pbar = tqdm(self.trainloader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
            
            for signals, targets, _ in train_pbar:
                # 获取当前批次大小
                batch_size = signals.size(0)
                
                # 将信号数据移动到设备
                signals = signals.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(signals)
                
                # 计算 TMAPE 损失
                loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                
                # 计算指标
                mse, mae, r2, tmape, percentage_errors = self.compute_metrics(outputs.cpu(), targets.cpu())
                train_mse += mse
                train_mae += mae
                train_r2 += r2
                train_tmape += tmape
                
                # 累加每个通道的相对误差百分比
                for i in range(4):
                    train_percentage_errors[i] += percentage_errors[i]
                    
                train_batches += 1
                
                # 更新进度条
                train_pbar.set_postfix({
                    'TMAPE': f'{tmape:.2f}%',
                    'MSE': f'{mse:.4f}',
                    'MAE': f'{mae:.4f}',
                    'R2': f'{r2:.4f}',
                    'PE1': f'{percentage_errors[0]:.2f}%',
                    'PE2': f'{percentage_errors[1]:.2f}%',
                    'PE3': f'{percentage_errors[2]:.2f}%',
                    'PE4': f'{percentage_errors[3]:.2f}%'
                })
            
            # 计算平均训练指标
            self.train_loss[epoch] = train_mse / train_batches
            self.train_mae[epoch] = train_mae / train_batches
            self.train_r2[epoch] = train_r2 / train_batches
            self.train_tmape[epoch] = train_tmape / train_batches
            
            # 计算平均相对误差百分比
            avg_train_percentage_errors = [pe / train_batches for pe in train_percentage_errors]
            
            # 验证阶段
            self.model.eval()
            val_mse, val_mae, val_r2, val_tmape = 0.0, 0.0, 0.0, 0.0
            val_percentage_errors = [0.0, 0.0, 0.0, 0.0]  # 四个通道的相对误差百分比
            val_batches = 0
            
            val_pbar = tqdm(self.validationloader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            
            with torch.no_grad():
                for signals, targets, _ in val_pbar:
                    # 获取当前批次大小
                    batch_size = signals.size(0)
                    
                    # 将信号数据移动到设备
                    signals = signals.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(signals)
                    
                    # 计算指标
                    mse, mae, r2, tmape, percentage_errors = self.compute_metrics(outputs.cpu(), targets.cpu())
                    val_mse += mse
                    val_mae += mae
                    val_r2 += r2
                    val_tmape += tmape
                    
                    # 累加每个通道的相对误差百分比
                    for i in range(4):
                        val_percentage_errors[i] += percentage_errors[i]
                        
                    val_batches += 1
                    
                    # 更新进度条
                    val_pbar.set_postfix({
                        'TMAPE': f'{tmape:.2f}%',
                        'MSE': f'{mse:.4f}',
                        'MAE': f'{mae:.4f}',
                        'R2': f'{r2:.4f}',
                        'PE1': f'{percentage_errors[0]:.2f}%',
                        'PE2': f'{percentage_errors[1]:.2f}%',
                        'PE3': f'{percentage_errors[2]:.2f}%',
                        'PE4': f'{percentage_errors[3]:.2f}%'
                    })
            
            # 计算平均验证指标
            self.val_loss[epoch] = val_mse / val_batches
            self.val_mae[epoch] = val_mae / val_batches
            self.val_r2[epoch] = val_r2 / val_batches
            self.val_tmape[epoch] = val_tmape / val_batches
            
            # 计算平均相对误差百分比
            avg_val_percentage_errors = [pe / val_batches for pe in val_percentage_errors]
            
            # 打印epoch摘要
            print(f'Epoch {epoch+1}/{self.num_epochs} Summary:')
            print(f'Train - TMAPE: {self.train_tmape[epoch]:.2f}%, MSE: {self.train_loss[epoch]:.4f}, '
                  f'MAE: {self.train_mae[epoch]:.4f}, R²: {self.train_r2[epoch]:.4f}')
            print(f'Train - PE1: {avg_train_percentage_errors[0]:.2f}%, PE2: {avg_train_percentage_errors[1]:.2f}%, '
                  f'PE3: {avg_train_percentage_errors[2]:.2f}%, PE4: {avg_train_percentage_errors[3]:.2f}%')
            print(f'Val - TMAPE: {self.val_tmape[epoch]:.2f}%, MSE: {self.val_loss[epoch]:.4f}, '
                  f'MAE: {self.val_mae[epoch]:.4f}, R²: {self.val_r2[epoch]:.4f}')
            print(f'Val - PE1: {avg_val_percentage_errors[0]:.2f}%, PE2: {avg_val_percentage_errors[1]:.2f}%, '
                  f'PE3: {avg_val_percentage_errors[2]:.2f}%, PE4: {avg_val_percentage_errors[3]:.2f}%')
            print('-' * 50)
            
            # 保存最佳模型（基于验证 TMAPE）
            if self.val_tmape[epoch] < best_val_tmape:
                best_val_tmape = self.val_tmape[epoch]
                torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'best_model.pt'))
                print(f'New best model saved with validation TMAPE: {best_val_tmape:.2f}%')
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_folder, f'model_epoch_{epoch+1}.pt'))
        
        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'final_model.pt'))
        
        # 保存训练历史
        np.savez(os.path.join(self.model_folder, 'training_history.npz'),
                train_loss=self.train_loss,
                val_loss=self.val_loss,
                train_mae=self.train_mae,
                val_mae=self.val_mae,
                train_r2=self.train_r2,
                val_r2=self.val_r2,
                train_tmape=self.train_tmape,
                val_tmape=self.val_tmape)

    def test(self, testloader):
        self.model.eval()
        test_mse, test_mae, test_r2, test_tmape = 0.0, 0.0, 0.0, 0.0
        test_percentage_errors = [0.0, 0.0, 0.0, 0.0]  # 四个通道的相对误差百分比
        test_batches = 0
        
        all_outputs = []
        all_targets = []
        
        test_pbar = tqdm(testloader, desc='Testing')
        
        with torch.no_grad():
            for signals, targets, _ in test_pbar:
                # 获取当前批次大小
                batch_size = signals.size(0)
                
                # 将信号数据移动到设备
                signals = signals.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(signals)
                
                # 计算指标
                mse, mae, r2, tmape, percentage_errors = self.compute_metrics(outputs.cpu(), targets.cpu())
                test_mse += mse
                test_mae += mae
                test_r2 += r2
                test_tmape += tmape
                
                # 累加每个通道的相对误差百分比
                for i in range(4):
                    test_percentage_errors[i] += percentage_errors[i]
                    
                test_batches += 1
                
                # 保存所有预测和真实值
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 更新进度条
                test_pbar.set_postfix({
                    'TMAPE': f'{tmape:.2f}%',
                    'MSE': f'{mse:.4f}',
                    'MAE': f'{mae:.4f}',
                    'R2': f'{r2:.4f}',
                    'PE1': f'{percentage_errors[0]:.2f}%',
                    'PE2': f'{percentage_errors[1]:.2f}%',
                    'PE3': f'{percentage_errors[2]:.2f}%',
                    'PE4': f'{percentage_errors[3]:.2f}%'
                })
        
        # 计算平均测试指标
        avg_mse = test_mse / test_batches
        avg_mae = test_mae / test_batches
        avg_r2 = test_r2 / test_batches
        avg_tmape = test_tmape / test_batches
        
        # 计算平均相对误差百分比
        avg_percentage_errors = [pe / test_batches for pe in test_percentage_errors]
        
        # 合并所有预测和真实值
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        
        # 计算整体指标
        overall_mse = mean_squared_error(all_targets, all_outputs)
        overall_mae = mean_absolute_error(all_targets, all_outputs)
        overall_r2 = r2_score(all_targets, all_outputs)
        
        # 计算整体 TMAPE
        absolute_error = np.abs(all_outputs - all_targets)
        denominator = np.abs(all_targets) + self.epsilon
        overall_tmape = np.mean(absolute_error / denominator) * 100
        
        # 计算整体相对误差百分比
        overall_percentage_errors = []
        for i in range(4):  # 四个通道
            # 计算每个样本的相对误差
            abs_errors = np.abs(all_outputs[:, i] - all_targets[:, i])
            abs_labels = np.abs(all_targets[:, i])
            
            # 避免除以零 - 对于真实值为零的情况，使用预测值的绝对值作为分母
            denominator = np.where(abs_labels == 0, np.abs(all_outputs[:, i]), abs_labels)
            denominator[denominator == 0] = 1e-8  # 确保分母不为零
            
            # 计算相对误差百分比
            pe = np.mean(abs_errors / denominator) * 100
            overall_percentage_errors.append(pe)
        
        print(f'Test Summary:')
        print(f'Average TMAPE: {avg_tmape:.2f}%, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}')
        print(f'Average PE1: {avg_percentage_errors[0]:.2f}%, PE2: {avg_percentage_errors[1]:.2f}%, '
              f'PE3: {avg_percentage_errors[2]:.2f}%, PE4: {avg_percentage_errors[3]:.2f}%')
        print(f'Overall TMAPE: {overall_tmape:.2f}%, MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}')
        print(f'Overall PE1: {overall_percentage_errors[0]:.2f}%, PE2: {overall_percentage_errors[1]:.2f}%, '
              f'PE3: {overall_percentage_errors[2]:.2f}%, PE4: {overall_percentage_errors[3]:.2f}%')
        
        # 保存测试结果
        np.savez(os.path.join(self.model_folder, 'test_results.npz'),
                outputs=all_outputs,
                targets=all_targets,
                avg_tmape=avg_tmape,
                avg_mse=avg_mse,
                avg_mae=avg_mae,
                avg_r2=avg_r2,
                avg_pe=avg_percentage_errors,
                overall_tmape=overall_tmape,
                overall_mse=overall_mse,
                overall_mae=overall_mae,
                overall_r2=overall_r2,
                overall_pe=overall_percentage_errors)
        
        return all_outputs, all_targets