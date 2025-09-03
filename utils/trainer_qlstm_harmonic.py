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
import matplotlib.pyplot as plt

# 自定义组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, epsilon=1e-8):
        """
        组合损失函数：MSE + MAE
        
        参数:
        alpha: MSE权重
        beta: MAE权重
        epsilon: 避免除以零的小常数
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        mae_loss = self.mae(outputs, targets)
        
        return self.alpha * mse_loss + self.beta * mae_loss

# 自定义谐波感知损失函数
class HarmonicAwareLoss(nn.Module):
    def __init__(self, fundamental_weight=1.0, harmonic_decay=0.8, epsilon=1e-8):
        """
        谐波感知损失函数，对高阶谐波使用递减权重
        
        参数:
        fundamental_weight: 基波(第一通道)的权重
        harmonic_decay: 谐波权重衰减因子
        epsilon: 避免除以零的小常数
        """
        super(HarmonicAwareLoss, self).__init__()
        self.epsilon = epsilon
        
        # 创建权重向量 [基波, 二次谐波, 三次谐波, 四次谐波]
        self.weights = torch.tensor([
            fundamental_weight,
            fundamental_weight * harmonic_decay,
            fundamental_weight * harmonic_decay**2,
            fundamental_weight * harmonic_decay**3
        ])
        
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, input, target):
        se = (input - target) ** 2
        weights = self.weights.to(se.device)
        weighted_se = se * weights
        return weighted_se.mean()
    def forward(self, outputs, targets):
        # 计算每个通道的MSE
        channel_errors = (outputs - targets) ** 2
        
        # 应用权重
        weighted_errors = channel_errors * self.weights.to(outputs.device)
        
        # 计算加权平均
        loss = torch.mean(weighted_errors)
        
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
                 epsilon=1e-8,
                 loss_type="combined",
                 lr_scheduler=None,
                 grad_clip=1.0,
                 early_stopping_patience=20):
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
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        
        # 选择损失函数
        if loss_type == "combined":
            self.criterion = CombinedLoss(alpha=0.7, beta=0.3, epsilon=epsilon)
        elif loss_type == "harmonic":
            self.criterion = HarmonicAwareLoss(fundamental_weight=1.0, harmonic_decay=0.8, epsilon=epsilon)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "weighted_mse":
            self.criterion = WeightedMSELoss(weights=torch.tensor([1.0, 3.0, 5.0, 7.0]))
        else:
            self.criterion = nn.MSELoss()
        
        # 学习率调度器
        self.lr_scheduler = lr_scheduler
        
        # 记录训练和验证指标
        self.train_loss = np.zeros(num_epochs)
        self.val_loss = np.zeros(num_epochs)
        self.train_mae = np.zeros(num_epochs)
        self.val_mae = np.zeros(num_epochs)
        self.train_mse = np.zeros(num_epochs)
        self.val_mse = np.zeros(num_epochs)
        self.train_r2 = np.zeros(num_epochs)
        self.val_r2 = np.zeros(num_epochs)
        self.learning_rates = np.zeros(num_epochs)
        
        # 记录每个通道的指标
        self.train_channel_mae = np.zeros((num_epochs, 4))
        self.val_channel_mae = np.zeros((num_epochs, 4))
        self.train_channel_mse = np.zeros((num_epochs, 4))
        self.val_channel_mse = np.zeros((num_epochs, 4))
        self.train_channel_pe = np.zeros((num_epochs, 4))  # 相对误差百分比
        self.val_channel_pe = np.zeros((num_epochs, 4))    # 相对误差百分比
        
        # 梯度统计
        self.grad_norms = []
        
        # 创建模型文件夹
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            
        # 将模型移动到设备
        self.model.to(self.device)

    def compute_metrics(self, outputs, labels):
        """计算回归任务的评估指标，包括相对误差百分比"""
        outputs_np = outputs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 计算整体指标
        mse = mean_squared_error(labels_np, outputs_np)
        mae = mean_absolute_error(labels_np, outputs_np)
        r2 = r2_score(labels_np, outputs_np)
        
        # 计算每个通道的指标
        channel_mae = []
        channel_mse = []
        channel_pe = []  # 相对误差百分比
        
        for i in range(4):
            channel_mse.append(mean_squared_error(labels_np[:, i], outputs_np[:, i]))
            channel_mae.append(mean_absolute_error(labels_np[:, i], outputs_np[:, i]))
            
            # 计算相对误差百分比
            abs_errors = np.abs(outputs_np[:, i] - labels_np[:, i])
            abs_labels = np.abs(labels_np[:, i])
            
            # 避免除以零 - 对于真实值为零的情况，使用预测值的绝对值作为分母
            denominator = np.where(abs_labels == 0, np.abs(outputs_np[:, i]), abs_labels)
            denominator[denominator == 0] = self.epsilon  # 确保分母不为零
            
            # 计算相对误差百分比
            pe = np.mean(abs_errors / denominator) * 100
            channel_pe.append(pe)
        
        return mse, mae, r2, channel_mae, channel_mse, channel_pe

    def compute_gradient_norms(self):
        """计算模型参数的梯度范数"""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def train(self):
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(self.num_epochs):
            # 记录当前学习率
            if self.lr_scheduler is not None:
                self.learning_rates[epoch] = self.optimizer.param_groups[0]['lr']
            else:
                self.learning_rates[epoch] = self.optimizer.param_groups[0]['lr']
            
            # 训练阶段
            self.model.train()
            train_loss, train_mae, train_mse, train_r2 = 0.0, 0.0, 0.0, 0.0
            train_channel_mae = [0.0, 0.0, 0.0, 0.0]
            train_channel_mse = [0.0, 0.0, 0.0, 0.0]
            train_channel_pe = [0.0, 0.0, 0.0, 0.0]  # 相对误差百分比
            train_batches = 0
            epoch_grad_norms = []
            
            train_pbar = tqdm(self.trainloader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
            
            for signals, targets, _ in train_pbar:
                # 将数据移动到设备
                signals = signals.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(signals)
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                
                # 计算梯度范数
                grad_norm = self.compute_gradient_norms()
                epoch_grad_norms.append(grad_norm)
                
                # 应用梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
                
                # 计算指标
                mse, mae, r2, channel_mae, channel_mse, channel_pe = self.compute_metrics(outputs, targets)
                train_loss += loss.item()
                train_mse += mse
                train_mae += mae
                train_r2 += r2
                
                # 累加每个通道的指标
                for i in range(4):
                    train_channel_mae[i] += channel_mae[i]
                    train_channel_mse[i] += channel_mse[i]
                    train_channel_pe[i] += channel_pe[i]
                    
                train_batches += 1
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MSE': f'{mse:.6f}',
                    'MAE': f'{mae:.6f}',
                    'R2': f'{r2:.4f}',
                    'Grad': f'{grad_norm:.4f}'
                })
            
            # 记录平均梯度范数
            self.grad_norms.append(np.mean(epoch_grad_norms))
            
            # 计算平均训练指标
            self.train_loss[epoch] = train_loss / train_batches
            self.train_mae[epoch] = train_mae / train_batches
            self.train_mse[epoch] = train_mse / train_batches
            self.train_r2[epoch] = train_r2 / train_batches
            
            # 计算每个通道的平均指标
            for i in range(4):
                self.train_channel_mae[epoch, i] = train_channel_mae[i] / train_batches
                self.train_channel_mse[epoch, i] = train_channel_mse[i] / train_batches
                self.train_channel_pe[epoch, i] = train_channel_pe[i] / train_batches
            
            # 验证阶段
            self.model.eval()
            val_loss, val_mae, val_mse, val_r2 = 0.0, 0.0, 0.0, 0.0
            val_channel_mae = [0.0, 0.0, 0.0, 0.0]
            val_channel_mse = [0.0, 0.0, 0.0, 0.0]
            val_channel_pe = [0.0, 0.0, 0.0, 0.0]  # 相对误差百分比
            val_batches = 0
            
            val_pbar = tqdm(self.validationloader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
            
            with torch.no_grad():
                for signals, targets, _ in val_pbar:
                    # 将数据移动到设备
                    signals = signals.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(signals)
                    
                    # 计算损失
                    loss = self.criterion(outputs, targets)
                    
                    # 计算指标
                    mse, mae, r2, channel_mae, channel_mse, channel_pe = self.compute_metrics(outputs, targets)
                    val_loss += loss.item()
                    val_mse += mse
                    val_mae += mae
                    val_r2 += r2
                    
                    # 累加每个通道的指标
                    for i in range(4):
                        val_channel_mae[i] += channel_mae[i]
                        val_channel_mse[i] += channel_mse[i]
                        val_channel_pe[i] += channel_pe[i]
                        
                    val_batches += 1
                    
                    # 更新进度条
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'MSE': f'{mse:.6f}',
                        'MAE': f'{mae:.6f}',
                        'R2': f'{r2:.4f}'
                    })
            
            # 计算平均验证指标
            self.val_loss[epoch] = val_loss / val_batches
            self.val_mae[epoch] = val_mae / val_batches
            self.val_mse[epoch] = val_mse / val_batches
            self.val_r2[epoch] = val_r2 / val_batches
            
            # 计算每个通道的平均指标
            for i in range(4):
                self.val_channel_mae[epoch, i] = val_channel_mae[i] / val_batches
                self.val_channel_mse[epoch, i] = val_channel_mse[i] / val_batches
                self.val_channel_pe[epoch, i] = val_channel_pe[i] / val_batches
            
            # 更新学习率
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(self.val_loss[epoch])
                else:
                    self.lr_scheduler.step()
            
            # 打印epoch摘要
            print(f'Epoch {epoch+1}/{self.num_epochs} Summary:')
            print(f'LR: {self.learning_rates[epoch]:.8f}, Grad Norm: {np.mean(epoch_grad_norms):.4f}')
            print(f'Train - Loss: {self.train_loss[epoch]:.6f}, MSE: {self.train_mse[epoch]:.6f}, '
                  f'MAE: {self.train_mae[epoch]:.6f}, R²: {self.train_r2[epoch]:.4f}')
            print(f'Val - Loss: {self.val_loss[epoch]:.6f}, MSE: {self.val_mse[epoch]:.6f}, '
                  f'MAE: {self.val_mae[epoch]:.6f}, R²: {self.val_r2[epoch]:.4f}')
            
            # 打印每个通道的指标
            for i in range(4):
                print(f'Channel {i+1} - Train MSE: {self.train_channel_mse[epoch, i]:.6f}, '
                      f'Val MSE: {self.val_channel_mse[epoch, i]:.6f}')
                # print(f'Channel {i+1} - Train PE: {self.train_channel_pe[epoch, i]:.2f}%, '
                #       f'Val PE: {self.val_channel_pe[epoch, i]:.2f}%')
            
            print('-' * 60)
            
            # 早停检查
            if self.val_loss[epoch] < best_val_loss:
                best_val_loss = self.val_loss[epoch]
                torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'best_model.pt'))
                print(f'New best model saved with validation loss: {best_val_loss:.6f}')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f'No improvement for {epochs_without_improvement} epochs')
                
                # 检查是否应该早停
                if epochs_without_improvement >= self.early_stopping_patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            # 定期保存检查点
            # if (epoch + 1) % 10 == 0:
            #     torch.save(self.model.state_dict(), os.path.join(self.model_folder, f'model_epoch_{epoch+1}.pt'))
                
                # 绘制训练曲线
                #self.plot_training_progress()
        
        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'final_model.pt'))
        
        # 绘制最终训练曲线
        #self.plot_training_progress()
        
        # 保存训练历史
        np.savez(os.path.join(self.model_folder, 'training_history.npz'),
                train_loss=self.train_loss,
                val_loss=self.val_loss,
                train_mae=self.train_mae,
                val_mae=self.val_mae,
                train_mse=self.train_mse,
                val_mse=self.val_mse,
                train_r2=self.train_r2,
                val_r2=self.val_r2,
                train_channel_mae=self.train_channel_mae,
                val_channel_mae=self.val_channel_mae,
                train_channel_mse=self.train_channel_mse,
                val_channel_mse=self.val_channel_mse,
                train_channel_pe=self.train_channel_pe,
                val_channel_pe=self.val_channel_pe,
                learning_rates=self.learning_rates,
                grad_norms=self.grad_norms)

    def plot_training_progress(self):
        """绘制训练进度图"""
        epochs = range(1, len(self.train_loss) + 1)
        
        # 创建2x3的子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 绘制损失曲线
        axes[0, 0].plot(epochs, self.train_loss, 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.val_loss, 'r-', label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 绘制MSE曲线
        axes[0, 1].plot(epochs, self.train_mse, 'b-', label='Training MSE')
        axes[0, 1].plot(epochs, self.val_mse, 'r-', label='Validation MSE')
        axes[0, 1].set_title('MSE')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 绘制相对误差百分比曲线
        colors = ['blue', 'green', 'red', 'purple']
        for i in range(4):
            axes[0, 2].plot(epochs, self.train_channel_pe[:, i], f'{colors[i]}-', alpha=0.7, label=f'Channel {i+1} Train PE')
            axes[0, 2].plot(epochs, self.val_channel_pe[:, i], f'{colors[i]}--', alpha=0.7, label=f'Channel {i+1} Val PE')
        axes[0, 2].set_title('Percentage Error by Channel')
        axes[0, 2].set_xlabel('Epochs')
        axes[0, 2].set_ylabel('Percentage Error (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 绘制学习率曲线
        axes[1, 0].plot(epochs, self.learning_rates, 'g-')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # 绘制梯度范数曲线
        if self.grad_norms:
            axes[1, 1].plot(range(1, len(self.grad_norms) + 1), self.grad_norms, 'm-')
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        # 绘制R²曲线
        axes[1, 2].plot(epochs, self.train_r2, 'b-', label='Training R²')
        axes[1, 2].plot(epochs, self.val_r2, 'r-', label='Validation R²')
        axes[1, 2].set_title('R² Score')
        axes[1, 2].set_xlabel('Epochs')
        axes[1, 2].set_ylabel('R²')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_folder, 'training_progress.png'))
        plt.close()

    def test(self, testloader):
        """在测试集上评估模型性能，包括相对误差百分比，并打印3组输入、GT和模型输出"""
        self.model.eval()
        test_loss, test_mae, test_mse, test_r2 = 0.0, 0.0, 0.0, 0.0
        test_channel_mae = [0.0, 0.0, 0.0, 0.0]
        test_channel_mse = [0.0, 0.0, 0.0, 0.0]
        test_channel_pe = [0.0, 0.0, 0.0, 0.0]  # 相对误差百分比
        test_batches = 0
        
        all_outputs = []
        all_targets = []
        
        # 用于存储前3组输入、GT和模型输出
        sample_inputs = []
        sample_targets = []
        sample_outputs = []
        
        test_pbar = tqdm(testloader, desc='Testing')
        
        with torch.no_grad():
            for batch_idx, (signals, targets, _) in enumerate(test_pbar):
                # 将数据移动到设备
                signals = signals.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(signals)
                
                # 保存前3组数据
                if batch_idx < 3:
                    sample_inputs.append(signals.cpu().numpy())
                    sample_targets.append(targets.cpu().numpy())
                    sample_outputs.append(outputs.cpu().numpy())
                
                # 计算损失
                loss = self.criterion(outputs, targets)
                
                # 计算指标
                mse, mae, r2, channel_mae, channel_mse, channel_pe = self.compute_metrics(outputs, targets)
                test_loss += loss.item()
                test_mse += mse
                test_mae += mae
                test_r2 += r2
                
                # 累加每个通道的指标
                for i in range(4):
                    test_channel_mae[i] += channel_mae[i]
                    test_channel_mse[i] += channel_mse[i]
                    test_channel_pe[i] += channel_pe[i]
                    
                test_batches += 1
                
                # 保存所有预测和真实值
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 更新进度条
                test_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MSE': f'{mse:.6f}',
                    'MAE': f'{mae:.6f}',
                    'R2': f'{r2:.4f}'
                })
        
        # 打印前3组输入、GT和模型输出
        print("\n=== 前3组样本的输入、GT和模型输出 ===")
        for i in range(min(3, len(sample_inputs))):
            print(f"\n样本 {i+1}:")
            print(f"输入信号形状: {sample_inputs[i].shape}")
            print(f"GT (真实值): {sample_targets[i].flatten()}")
            print(f"模型输出: {sample_outputs[i].flatten()}")
            
            # 计算并打印每个通道的相对误差百分比
            gt = sample_targets[i].flatten()
            pred = sample_outputs[i].flatten()
            for ch in range(4):
                abs_error = np.abs(pred[ch] - gt[ch])
                denominator = gt[ch] if gt[ch] != 0 else np.abs(pred[ch])
                if denominator == 0:
                    pe = 0.0
                else:
                    pe = (abs_error / denominator) * 100
                print(f"通道 {ch+1} 相对误差: {pe:.2f}%")
        
        # 计算平均测试指标
        avg_loss = test_loss / test_batches
        avg_mse = test_mse / test_batches
        avg_mae = test_mae / test_batches
        avg_r2 = test_r2 / test_batches
        
        # 计算每个通道的平均指标
        avg_channel_mae = [mae / test_batches for mae in test_channel_mae]
        avg_channel_mse = [mse / test_batches for mse in test_channel_mse]
        avg_channel_pe = [pe / test_batches for pe in test_channel_pe]  # 平均相对误差百分比
        
        # 合并所有预测和真实值
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        
        # 计算整体指标
        overall_mse = mean_squared_error(all_targets, all_outputs)
        overall_mae = mean_absolute_error(all_targets, all_outputs)
        overall_r2 = r2_score(all_targets, all_outputs)
        
        # 计算整体相对误差百分比
        abs_errors = np.abs(all_outputs - all_targets)
        abs_labels = np.abs(all_targets)
        
        # 避免除以零 - 对于真实值为零的情况，使用预测值的绝对值作为分母
        denominator = np.where(abs_labels == 0, np.abs(all_outputs), abs_labels)
        denominator[denominator == 0] = self.epsilon  # 确保分母不为零
        
        # 计算整体相对误差百分比
        overall_pe = np.mean(abs_errors / denominator, axis=0) * 100
        
        # 计算每个通道的整体指标
        overall_channel_mae = []
        overall_channel_mse = []
        for i in range(4):
            overall_channel_mse.append(mean_squared_error(all_targets[:, i], all_outputs[:, i]))
            overall_channel_mae.append(mean_absolute_error(all_targets[:, i], all_outputs[:, i]))
        
        print(f'\nTest Summary:')
        print(f'Average Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, R²: {avg_r2:.4f}')
        print(f'Overall - MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}, R²: {overall_r2:.4f}')
        
        # 打印每个通道的指标
        for i in range(4):
            print(f'Channel {i+1}:')
            print(f'  Average - MSE: {avg_channel_mse[i]:.6f}, MAE: {avg_channel_mae[i]:.6f}, PE: {avg_channel_pe[i]:.2f}%')
            print(f'  Overall - MSE: {overall_channel_mse[i]:.6f}, MAE: {overall_channel_mae[i]:.6f}, PE: {overall_pe[i]:.2f}%')
        
        # 保存测试结果
        np.savez(os.path.join(self.model_folder, 'test_results.npz'),
                outputs=all_outputs,
                targets=all_targets,
                avg_loss=avg_loss,
                avg_mse=avg_mse,
                avg_mae=avg_mae,
                avg_r2=avg_r2,
                avg_channel_mae=avg_channel_mae,
                avg_channel_mse=avg_channel_mse,
                avg_channel_pe=avg_channel_pe,
                overall_mse=overall_mse,
                overall_mae=overall_mae,
                overall_r2=overall_r2,
                overall_pe=overall_pe,
                overall_channel_mae=overall_channel_mae,
                overall_channel_mse=overall_channel_mse)
        
        # 绘制预测结果散点图
        self.plot_predictions(all_outputs, all_targets)
        
        # 绘制相对误差百分比直方图
        self.plot_percentage_error_histogram(all_outputs, all_targets)
        

        return all_outputs, all_targets

    def plot_predictions(self, outputs, targets):
        """绘制预测值与真实值的散点图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i in range(4):
            axes[i].scatter(targets[:, i], outputs[:, i], alpha=0.5)
            axes[i].plot([targets[:, i].min(), targets[:, i].max()], 
                        [targets[:, i].min(), targets[:, i].max()], 'r--')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predictions')
            axes[i].set_title(f'Channel {i+1}')
            
            # 计算R²
            r2 = r2_score(targets[:, i], outputs[:, i])
            axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_folder, 'predictions_scatter.png'))
        plt.close()

    def plot_percentage_error_histogram(self, outputs, targets):
        """绘制相对误差百分比直方图"""
        # 计算相对误差百分比
        abs_errors = np.abs(outputs - targets)
        abs_labels = np.abs(targets)
        
        # 避免除以零
        denominator = np.where(abs_labels == 0, np.abs(outputs), abs_labels)
        denominator[denominator == 0] = self.epsilon
        
        percentage_errors = (abs_errors / denominator) * 100
        
        # 创建直方图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i in range(4):
            axes[i].hist(percentage_errors[:, i], bins=50, alpha=0.7)
            axes[i].set_xlabel('Percentage Error (%)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Channel {i+1} - Percentage Error Distribution')
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_pe = np.mean(percentage_errors[:, i])
            median_pe = np.median(percentage_errors[:, i])
            std_pe = np.std(percentage_errors[:, i])
            
            axes[i].axvline(mean_pe, color='r', linestyle='--', label=f'Mean: {mean_pe:.2f}%')
            axes[i].axvline(median_pe, color='g', linestyle='--', label=f'Median: {median_pe:.2f}%')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_folder, 'percentage_error_histogram.png'))
        plt.close()
        
        # 保存详细的相对误差百分比统计
        pe_stats = {}
        for i in range(4):
            pe_stats[f'channel_{i+1}'] = {
                'mean': np.mean(percentage_errors[:, i]),
                'median': np.median(percentage_errors[:, i]),
                'std': np.std(percentage_errors[:, i]),
                'min': np.min(percentage_errors[:, i]),
                'max': np.max(percentage_errors[:, i]),
                'q25': np.percentile(percentage_errors[:, i], 25),
                'q75': np.percentile(percentage_errors[:, i], 75)
            }
        
        np.savez(os.path.join(self.model_folder, 'percentage_error_stats.npz'), **pe_stats)
        


        print("\nPercentage Error Statistics:")
        for i in range(4):
            stats = pe_stats[f'channel_{i+1}']
            print(f"Channel {i+1}:")
            print(f"  Mean: {stats['mean']:.2f}%, Median: {stats['median']:.2f}%, Std: {stats['std']:.2f}%")
            print(f"  Min: {stats['min']:.2f}%, Max: {stats['max']:.2f}%")
            print(f"  25th percentile: {stats['q25']:.2f}%, 75th percentile: {stats['q75']:.2f}%")