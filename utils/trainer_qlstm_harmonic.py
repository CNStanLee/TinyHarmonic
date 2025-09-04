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
from utils.loss import CombinedLoss, HarmonicAwareLoss, WeightedMSELoss, WeightedLogMSELoss


class TrainerQLSTMHarmonic:
    def __init__(self, 
                 model, 
                 trainloader, 
                 validationloader, 
                 test_loader,
                 train_batch_size, 
                 val_batch_size, 
                 num_epochs, 
                 optimizer, 
                 model_folder,
                 device,
                 input_length=64,
                 epsilon=1e-8,
                 loss_type="combined",
                 weights_arrange=torch.tensor([1.0, 7.0, 7.0, 7.0]),
                 lr_scheduler=None,
                 grad_clip=1.0,
                 early_stopping_patience=20,
                 error_metric="pe"):  # 新增参数：误差度量指标，可选"pe"或"smape"
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
        self.weights_arrange = weights_arrange
        self.test_loader = test_loader
        self.error_metric = error_metric  # 保存误差度量指标

        self.num_features = input_length
        self.test_length = len(test_loader.dataset)

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
            self.criterion = WeightedMSELoss(weights=self.weights_arrange)
        elif loss_type == "harmonic_log_mse":
            self.criterion = WeightedLogMSELoss(weights=self.weights_arrange)
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
        self.train_adjusted_r2 = np.zeros(num_epochs)  # 新增：调整后R²
        self.val_adjusted_r2 = np.zeros(num_epochs)    # 新增：调整后R²
        self.learning_rates = np.zeros(num_epochs)
        
        # 记录每个通道的指标
        self.train_channel_mae = np.zeros((num_epochs, 4))
        self.val_channel_mae = np.zeros((num_epochs, 4))
        self.train_channel_mse = np.zeros((num_epochs, 4))
        self.val_channel_mse = np.zeros((num_epochs, 4))
        self.train_channel_error = np.zeros((num_epochs, 4))  # 改为通用误差指标
        self.val_channel_error = np.zeros((num_epochs, 4))    # 改为通用误差指标
        
        # 梯度统计
        self.grad_norms = []
        
        # 创建模型文件夹
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 计算模型参数数量（用于Adjusted R²）
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def compute_adjusted_r2(self, r2, n, p):
        """计算调整后R²"""
        p = self.num_features
        n = self.test_length
        if n - p - 1 <= 0:
            return r2  # 避免除零错误
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def compute_error_metric(self, outputs, labels):
        """计算选择的误差度量指标（相对误差或sMAPE）"""
        outputs_np = outputs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        if self.error_metric == "pe":
            # 计算相对误差百分比
            abs_errors = np.abs(outputs_np - labels_np)
            abs_labels = np.abs(labels_np)
            
            # 避免除以零 - 对于真实值为零的情况，使用预测值的绝对值作为分母
            denominator = np.where(abs_labels == 0, np.abs(outputs_np), abs_labels)
            denominator[denominator <= self.epsilon] = self.epsilon
            
            # 计算相对误差百分比
            error = np.mean(abs_errors / denominator, axis=0) * 100
            
        elif self.error_metric == "smape":
            # 计算sMAPE (Symmetric Mean Absolute Percentage Error)
            abs_errors = np.abs(outputs_np - labels_np)
            sum_abs = np.abs(outputs_np) + np.abs(labels_np)
            
            # 避免除以零
            sum_abs[sum_abs <= self.epsilon] = self.epsilon
            
            # 计算sMAPE
            smape = 2.0 * np.mean(abs_errors / sum_abs, axis=0) * 100
            error = smape
            
        return error

    def compute_tmape(self, outputs, labels):
        """计算整体TMAPE (Total Mean Absolute Percentage Error)"""
        outputs_np = outputs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 计算绝对误差
        abs_errors = np.abs(outputs_np - labels_np)
        abs_labels = np.abs(labels_np)
        
        # 避免除以零
        denominator = np.where(abs_labels == 0, np.abs(outputs_np), abs_labels)
        denominator[denominator <= self.epsilon] = self.epsilon
        
        # 计算每个样本的相对误差
        per_sample_error = np.sum(abs_errors, axis=1) / np.sum(denominator, axis=1)
        
        # 计算平均TMAPE
        tmape = np.mean(per_sample_error) * 100
        
        return tmape

    def compute_metrics(self, outputs, labels):
        """计算回归任务的评估指标"""
        outputs_np = outputs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 计算整体指标
        mse = mean_squared_error(labels_np, outputs_np)
        mae = mean_absolute_error(labels_np, outputs_np)
        r2 = r2_score(labels_np, outputs_np)
        
        # 计算调整后R²
        n = len(labels_np)  # 样本数量
        adjusted_r2 = self.compute_adjusted_r2(r2, n, self.num_params)
        
        # 计算每个通道的指标
        channel_mae = []
        channel_mse = []
        channel_error = []  # 改为通用误差指标
        
        for i in range(4):
            channel_mse.append(mean_squared_error(labels_np[:, i], outputs_np[:, i]))
            channel_mae.append(mean_absolute_error(labels_np[:, i], outputs_np[:, i]))
            
        # 计算选择的误差度量指标
        error_metric = self.compute_error_metric(outputs, labels)
        channel_error = error_metric
        
        return mse, mae, r2, adjusted_r2, channel_mae, channel_mse, channel_error

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
            train_loss, train_mae, train_mse, train_r2, train_adjusted_r2 = 0.0, 0.0, 0.0, 0.0, 0.0
            train_channel_mae = [0.0, 0.0, 0.0, 0.0]
            train_channel_mse = [0.0, 0.0, 0.0, 0.0]
            train_channel_error = [0.0, 0.0, 0.0, 0.0]  # 改为通用误差指标
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
                mse, mae, r2, adjusted_r2, channel_mae, channel_mse, channel_error = self.compute_metrics(outputs, targets)
                train_loss += loss.item()
                train_mse += mse
                train_mae += mae
                train_r2 += r2
                train_adjusted_r2 += adjusted_r2
                
                # 累加每个通道的指标
                for i in range(4):
                    train_channel_mae[i] += channel_mae[i]
                    train_channel_mse[i] += channel_mse[i]
                    train_channel_error[i] += channel_error[i]
                    
                train_batches += 1
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MSE': f'{mse:.6f}',
                    'MAE': f'{mae:.6f}',
                    'R2': f'{r2:.4f}',
                    'Adj R2': f'{adjusted_r2:.4f}',
                    'Grad': f'{grad_norm:.4f}'
                })
            
            # 记录平均梯度范数
            self.grad_norms.append(np.mean(epoch_grad_norms))
            
            # 计算平均训练指标
            self.train_loss[epoch] = train_loss / train_batches
            self.train_mae[epoch] = train_mae / train_batches
            self.train_mse[epoch] = train_mse / train_batches
            self.train_r2[epoch] = train_r2 / train_batches
            self.train_adjusted_r2[epoch] = train_adjusted_r2 / train_batches
            
            # 计算每个通道的平均指标
            for i in range(4):
                self.train_channel_mae[epoch, i] = train_channel_mae[i] / train_batches
                self.train_channel_mse[epoch, i] = train_channel_mse[i] / train_batches
                self.train_channel_error[epoch, i] = train_channel_error[i] / train_batches
            
            # 验证阶段
            self.model.eval()
            val_loss, val_mae, val_mse, val_r2, val_adjusted_r2 = 0.0, 0.0, 0.0, 0.0, 0.0
            val_channel_mae = [0.0, 0.0, 0.0, 0.0]
            val_channel_mse = [0.0, 0.0, 0.0, 0.0]
            val_channel_error = [0.0, 0.0, 0.0, 0.0]  # 改为通用误差指标
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
                    mse, mae, r2, adjusted_r2, channel_mae, channel_mse, channel_error = self.compute_metrics(outputs, targets)
                    val_loss += loss.item()
                    val_mse += mse
                    val_mae += mae
                    val_r2 += r2
                    val_adjusted_r2 += adjusted_r2
                    
                    # 累加每个通道的指标
                    for i in range(4):
                        val_channel_mae[i] += channel_mae[i]
                        val_channel_mse[i] += channel_mse[i]
                        val_channel_error[i] += channel_error[i]
                        
                    val_batches += 1
                    
                    # 更新进度条
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'MSE': f'{mse:.6f}',
                        'MAE': f'{mae:.6f}',
                        'R2': f'{r2:.4f}',
                        'Adj R2': f'{adjusted_r2:.4f}'
                    })
            
            # 计算平均验证指标
            self.val_loss[epoch] = val_loss / val_batches
            self.val_mae[epoch] = val_mae / val_batches
            self.val_mse[epoch] = val_mse / val_batches
            self.val_r2[epoch] = val_r2 / val_batches
            self.val_adjusted_r2[epoch] = val_adjusted_r2 / val_batches
            
            # 计算每个通道的平均指标
            for i in range(4):
                self.val_channel_mae[epoch, i] = val_channel_mae[i] / val_batches
                self.val_channel_mse[epoch, i] = val_channel_mse[i] / val_batches
                self.val_channel_error[epoch, i] = val_channel_error[i] / val_batches
            
            # 更新学习率
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(self.val_loss[epoch])
                else:
                    self.lr_scheduler.step()
            
            # 打印epoch摘要
            error_metric_name = "PE" if self.error_metric == "pe" else "sMAPE"
            print(f'Epoch {epoch+1}/{self.num_epochs} Summary:')
            print(f'LR: {self.learning_rates[epoch]:.8f}, Grad Norm: {np.mean(epoch_grad_norms):.4f}')
            print(f'Train - Loss: {self.train_loss[epoch]:.6f}, MSE: {self.train_mse[epoch]:.6f}, '
                  f'MAE: {self.train_mae[epoch]:.6f}, R²: {self.train_r2[epoch]:.4f}, Adj R²: {self.train_adjusted_r2[epoch]:.4f}')
            print(f'Val - Loss: {self.val_loss[epoch]:.6f}, MSE: {self.val_mse[epoch]:.6f}, '
                  f'MAE: {self.val_mae[epoch]:.6f}, R²: {self.val_r2[epoch]:.4f}, Adj R²: {self.val_adjusted_r2[epoch]:.4f}')
            
            # 打印每个通道的指标
            for i in range(4):
                print(f'Channel {i+1} - Train MSE: {self.train_channel_mse[epoch, i]:.6f}, '
                      f'Val MSE: {self.val_channel_mse[epoch, i]:.6f}')
                print(f'Channel {i+1} - Train {error_metric_name}: {self.train_channel_error[epoch, i]:.2f}%, '
                      f'Val {error_metric_name}: {self.val_channel_error[epoch, i]:.2f}%')
            
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
            if (epoch + 1) % 20 == 0:
                #torch.save(self.model.state_dict(), os.path.join(self.model_folder, f'model_epoch_{epoch+1}.pt'))
                
                #绘制训练曲线
                self.plot_training_progress()
                self.test(self.test_loader)
        
        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(self.model_folder, 'final_model.pt'))
        
        # 绘制最终训练曲线
        self.plot_training_progress()
        
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
                train_adjusted_r2=self.train_adjusted_r2,  # 新增
                val_adjusted_r2=self.val_adjusted_r2,      # 新增
                train_channel_mae=self.train_channel_mae,
                val_channel_mae=self.val_channel_mae,
                train_channel_mse=self.train_channel_mse,
                val_channel_mse=self.val_channel_mse,
                train_channel_error=self.train_channel_error,
                val_channel_error=self.val_channel_error,
                learning_rates=self.learning_rates,
                grad_norms=self.grad_norms)

    def plot_training_progress(self):
        """绘制训练进度图"""
        epochs = range(1, len(self.train_loss) + 1)
        
        # 创建2x3的子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 增加高度以容纳更多内容
        
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
        
        # 绘制误差指标曲线
        error_metric_name = "PE" if self.error_metric == "pe" else "sMAPE"
        colors = ['blue', 'green', 'red', 'purple']
        for i in range(4):
            axes[0, 2].plot(epochs, self.train_channel_error[:, i], color=colors[i], linestyle='-', alpha=0.7, 
                           label=f'Channel {i+1} Train {error_metric_name}')
            axes[0, 2].plot(epochs, self.val_channel_error[:, i], color=colors[i], linestyle='--', alpha=0.7, 
                           label=f'Channel {i+1} Val {error_metric_name}')

        axes[0, 2].set_title(f'{error_metric_name} by Channel')
        axes[0, 2].set_xlabel('Epochs')
        axes[0, 2].set_ylabel(f'{error_metric_name} (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 绘制R²和调整后R²曲线
        axes[1, 0].plot(epochs, self.train_r2, 'b-', label='Training R²')
        axes[1, 0].plot(epochs, self.val_r2, 'r-', label='Validation R²')
        axes[1, 0].plot(epochs, self.train_adjusted_r2, 'b--', label='Training Adj R²')
        axes[1, 0].plot(epochs, self.val_adjusted_r2, 'r--', label='Validation Adj R²')
        axes[1, 0].set_title('R² and Adjusted R²')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('R² / Adj R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 绘制学习率曲线
        axes[1, 1].plot(epochs, self.learning_rates, 'g-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        # 绘制梯度范数曲线
        if self.grad_norms:
            axes[1, 2].plot(range(1, len(self.grad_norms) + 1), self.grad_norms, 'm-')
            axes[1, 2].set_title('Gradient Norms')
            axes[1, 2].set_xlabel('Epochs')
            axes[1, 2].set_ylabel('Gradient Norm')
            axes[1, 2].grid(True)
            axes[1, 2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_folder, 'training_progress.png'))
        plt.close()

    def test(self, testloader):
        """在测试集上评估模型性能，包括选择的误差指标和TMAPE，并打印3组输入、GT和模型输出"""
        self.model.eval()
        test_loss, test_mae, test_mse, test_r2, test_adjusted_r2 = 0.0, 0.0, 0.0, 0.0, 0.0
        test_channel_mae = [0.0, 0.0, 0.0, 0.0]
        test_channel_mse = [0.0, 0.0, 0.0, 0.0]
        test_channel_error = [0.0, 0.0, 0.0, 0.0]  # 改为通用误差指标
        test_tmape = 0.0  # 整体TMAPE
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
                mse, mae, r2, adjusted_r2, channel_mae, channel_mse, channel_error = self.compute_metrics(outputs, targets)
                test_loss += loss.item()
                test_mse += mse
                test_mae += mae
                test_r2 += r2
                test_adjusted_r2 += adjusted_r2
                
                # 计算TMAPE
                tmape = self.compute_tmape(outputs, targets)
                test_tmape += tmape
                
                # 累加每个通道的指标
                for i in range(4):
                    test_channel_mae[i] += channel_mae[i]
                    test_channel_mse[i] += channel_mse[i]
                    test_channel_error[i] += channel_error[i]
                    
                test_batches += 1
                
                # 保存所有预测和真实值
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # 更新进度条
                test_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MSE': f'{mse:.6f}',
                    'MAE': f'{mae:.6f}',
                    'R2': f'{r2:.4f}',
                    'Adj R2': f'{adjusted_r2:.4f}'
                })
        
        # 计算平均测试指标
        avg_loss = test_loss / test_batches
        avg_mse = test_mse / test_batches
        avg_mae = test_mae / test_batches
        avg_r2 = test_r2 / test_batches
        avg_adjusted_r2 = test_adjusted_r2 / test_batches
        avg_tmape = test_tmape / test_batches
        
        # 计算每个通道的平均指标
        avg_channel_mae = [mae / test_batches for mae in test_channel_mae]
        avg_channel_mse = [mse / test_batches for mse in test_channel_mse]
        avg_channel_error = [error / test_batches for error in test_channel_error]
        
        # 合并所有预测和真实值
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        
        # 计算整体指标
        overall_mse = mean_squared_error(all_targets, all_outputs)
        overall_mae = mean_absolute_error(all_targets, all_outputs)
        overall_r2 = r2_score(all_targets, all_outputs)
        overall_adjusted_r2 = self.compute_adjusted_r2(overall_r2, len(all_targets), self.num_params)
        overall_tmape = self.compute_tmape(torch.tensor(all_outputs), torch.tensor(all_targets))
        
        # 计算选择的误差度量指标
        overall_error = self.compute_error_metric(torch.tensor(all_outputs), torch.tensor(all_targets))
        
        # 计算每个通道的整体指标
        overall_channel_mae = []
        overall_channel_mse = []
        for i in range(4):
            overall_channel_mse.append(mean_squared_error(all_targets[:, i], all_outputs[:, i]))
            overall_channel_mae.append(mean_absolute_error(all_targets[:, i], all_outputs[:, i]))
        
        # 打印测试结果
        error_metric_name = "PE" if self.error_metric == "pe" else "sMAPE"
        print(f'\nTest Summary:')
        print(f'Average Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, R²: {avg_r2:.4f}, Adj R²: {avg_adjusted_r2:.4f}')
        print(f'Average TMAPE: {avg_tmape:.2f}%')
        print(f'Overall - MSE: {overall_mse:.6f}, MAE: {overall_mae:.6f}, R²: {overall_r2:.4f}, Adj R²: {overall_adjusted_r2:.4f}, TMAPE: {overall_tmape:.2f}%')
        
        # 打印每个通道的指标
        for i in range(4):
            print(f'Channel {i+1}:')
            print(f'  Average - MSE: {avg_channel_mse[i]:.6f}, MAE: {avg_channel_mae[i]:.6f}, {error_metric_name}: {avg_channel_error[i]:.2f}%')
            print(f'  Overall - MSE: {overall_channel_mse[i]:.6f}, MAE: {overall_channel_mae[i]:.6f}, {error_metric_name}: {overall_error[i]:.2f}%')
        
        # 保存测试结果
        np.savez(os.path.join(self.model_folder, 'test_results.npz'),
                outputs=all_outputs,
                targets=all_targets,
                avg_loss=avg_loss,
                avg_mse=avg_mse,
                avg_mae=avg_mae,
                avg_r2=avg_r2,
                avg_adjusted_r2=avg_adjusted_r2,  # 新增
                avg_tmape=avg_tmape,
                avg_channel_mae=avg_channel_mae,
                avg_channel_mse=avg_channel_mse,
                avg_channel_error=avg_channel_error,
                overall_mse=overall_mse,
                overall_mae=overall_mae,
                overall_r2=overall_r2,
                overall_adjusted_r2=overall_adjusted_r2,  # 新增
                overall_tmape=overall_tmape,
                overall_error=overall_error,
                overall_channel_mae=overall_channel_mae,
                overall_channel_mse=overall_channel_mse)
        
        # 绘制预测结果散点图
        self.plot_predictions(all_outputs, all_targets)
        
        # 绘制误差指标直方图
        self.plot_error_histogram(all_outputs, all_targets)
        
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
            
            # 计算R²和调整后R²
            r2 = r2_score(targets[:, i], outputs[:, i])
            n = len(targets[:, i])
            adjusted_r2 = self.compute_adjusted_r2(r2, n, self.num_params)
            
            axes[i].text(0.05, 0.95, f'R² = {r2:.4f}\nAdj R² = {adjusted_r2:.4f}', transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_folder, 'predictions_scatter.png'))
        plt.close()

    def plot_error_histogram(self, outputs, targets):
        """绘制误差指标直方图"""
        # 计算选择的误差指标
        if self.error_metric == "pe":
            # 计算相对误差百分比
            abs_errors = np.abs(outputs - targets)
            abs_labels = np.abs(targets)
            
            # 避免除以零
            denominator = np.where(abs_labels == 0, np.abs(outputs), abs_labels)
            denominator[denominator <= self.epsilon] = self.epsilon
            
            errors = (abs_errors / denominator) * 100
            error_name = "Percentage Error"
            
        elif self.error_metric == "smape":
            # 计算sMAPE
            abs_errors = np.abs(outputs - targets)
            sum_abs = np.abs(outputs) + np.abs(targets)
            
            # 避免除以零
            sum_abs[sum_abs <= self.epsilon] = self.epsilon
            
            errors = 2.0 * (abs_errors / sum_abs) * 100
            error_name = "sMAPE"
        
        # 创建直方图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i in range(4):
            axes[i].hist(errors[:, i], bins=50, alpha=0.7)
            axes[i].set_xlabel(f'{error_name} (%)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Channel {i+1} - {error_name} Distribution')
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_error = np.mean(errors[:, i])
            median_error = np.median(errors[:, i])
            std_error = np.std(errors[:, i])
            
            axes[i].axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}%')
            axes[i].axvline(median_error, color='g', linestyle='--', label=f'Median: {median_error:.2f}%')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_folder, f'{self.error_metric}_histogram.png'))
        plt.close()
        
        # 保存详细的误差统计
        error_stats = {}
        for i in range(4):
            error_stats[f'channel_{i+1}'] = {
                'mean': np.mean(errors[:, i]),
                'median': np.median(errors[:, i]),
                'std': np.std(errors[:, i]),
                'min': np.min(errors[:, i]),
                'max': np.max(errors[:, i]),
                'q25': np.percentile(errors[:, i], 25),
                'q75': np.percentile(errors[:, i], 75)
            }
        
        np.savez(os.path.join(self.model_folder, f'{self.error_metric}_stats.npz'), **error_stats)
        
        # 打印误差统计
        print(f"\n{error_name} Statistics:")
        for i in range(4):
            stats = error_stats[f'channel_{i+1}']
            print(f"Channel {i+1}:")
            print(f"  Mean: {stats['mean']:.2f}%, Median: {stats['median']:.2f}%, Std: {stats['std']:.2f}%")
            print(f"  Min: {stats['min']:.2f}%, Max: {stats['max']:.2f}%")
            print(f"  25th percentile: {stats['q25']:.2f}%, 75th percentile: {stats['q75']:.2f}%")