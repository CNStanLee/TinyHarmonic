import torch
from torch import nn

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
class WeightedLogMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedLogMSELoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        # 计算对数MSE：log(1 + (outputs - targets)^2)
        log_mse = torch.log1p((outputs - targets) ** 2)
        
        # 应用权重
        weighted_errors = log_mse * self.weights.to(outputs.device)
        
        # 计算加权平均
        loss = torch.mean(weighted_errors)
        
        return loss

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

class HarmonicWeightedLogMSELoss(nn.Module):
    def __init__(self, base_weight=1.0, harmonic_weights=None, frequency_penalty=0.1, epsilon=1e-6):
        super(HarmonicWeightedLogMSELoss, self).__init__()
        self.base_weight = base_weight
        self.harmonic_weights = harmonic_weights if harmonic_weights is not None else [1.0, 0.8, 0.6, 0.4]
        self.frequency_penalty = frequency_penalty
        self.epsilon = epsilon  # 避免log(0)的情况
    
    def forward(self, outputs, targets):
        # 计算对数MSE
        # 首先对目标和输出取对数（加上一个小的常数避免log(0)）
        log_outputs = torch.log(outputs + self.epsilon)
        log_targets = torch.log(targets + self.epsilon)
        log_mse = (log_outputs - log_targets) ** 2
        
        # 应用谐波权重
        weighted_loss = log_mse * self.base_weight
        
        n_harmonics = min(len(self.harmonic_weights), outputs.shape[1])
        harmonic_weights_tensor = torch.tensor(self.harmonic_weights[:n_harmonics], 
                                              device=outputs.device).view(1, n_harmonics, 1)
        
        weighted_loss[:, :n_harmonics, :] *= harmonic_weights_tensor
        
        # 频率稳定性惩罚项（可选，如果输出是频率相关的话）
        if self.frequency_penalty > 0 and outputs.shape[0] > 1:
            # 注意：这里我们使用原始输出计算频率变化，因为对数变换后尺度变了
            freq_variation = torch.mean((outputs[1:] - outputs[:-1]) ** 2)
            penalty = self.frequency_penalty * freq_variation
        else:
            penalty = 0
        
        return torch.mean(weighted_loss) + penalty

