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
from scipy.fft import fft, fftfreq
import pywt # pip install PyWavelets
from scipy.signal import find_peaks, hilbert
# ---------------------------------------------------------
# from deps.qonnx.src.qonnx.util import config
from utils.harmonic_dataloader import download_harmonic_data, Config, init_dataloaders, process_all_data, create_data_loaders, visualize_sample, init_dataloaders
from models.qlstm_harmonic import QLSTMHarmonic
from utils.trainer_qlstm_harmonic import TrainerQLSTMHarmonic
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

epochs = 50#
lr = 0.0001
batch_size = 64

def fft_harmonic_analysis(input_signal, target_samples_per_cycle, input_cycle_fraction):
    """
    使用FFT进行谐波分析
    
    参数:
    - input_signal: 输入信号
    - target_samples_per_cycle: 每个周期的目标采样点数
    - input_cycle_fraction: 输入数据包含的周期数
    
    返回:
    - 估计的谐波幅度 [A1, A3, A5, A7]
    """
    N = len(input_signal)
    T = 1.0 / (target_samples_per_cycle * input_cycle_fraction)  # 采样间隔
    yf = fft(input_signal)
    xf = fftfreq(N, T)[:N//2]
    
    # 提取幅度
    amplitudes = 2.0/N * np.abs(yf[:N//2])
    
    # 找到基波频率（最大幅度对应的频率）
    fundamental_idx = np.argmax(amplitudes)
    fundamental_freq = xf[fundamental_idx]
    
    # 提取1,3,5,7次谐波
    harmonic_indices = []
    for harmonic in [1, 3, 5, 7]:
        # 计算谐波频率
        harmonic_freq = harmonic * fundamental_freq
        
        # 找到最接近的频率索引
        idx = np.argmin(np.abs(xf - harmonic_freq))
        harmonic_indices.append(idx)
    
    # 提取估计的谐波幅度
    estimated_amplitudes = amplitudes[harmonic_indices]
    
    return estimated_amplitudes

def wavelet_harmonic_analysis(input_signal, target_samples_per_cycle, input_cycle_fraction):
    # 进行小波分解
    wavelet='db30'
    level=4
    coeffs = pywt.wavedec(input_signal, wavelet, level=level)
    
    # 计算每个分解层级的能量（近似谐波幅度）
    harmonic_amplitudes = []
    for i, coeff in enumerate(coeffs[1:]):  # 跳过近似系数，只关注细节系数
        # 计算该层级的能量
        energy = np.sqrt(np.mean(coeff**2))
        harmonic_amplitudes.append(energy)
    
    # 确保有4个谐波幅度值
  
    if len(harmonic_amplitudes) < 4:
        # 如果不足4个，用0填充
        harmonic_amplitudes.extend([0] * (4 - len(harmonic_amplitudes)))
    elif len(harmonic_amplitudes) > 4:
        # 如果超过4个，取前4个
        harmonic_amplitudes = harmonic_amplitudes[:4]
    
    return np.array(harmonic_amplitudes)

def calculate_harmonic_errors(data_loader, input_cycle_fraction, target_samples_per_cycle, method='fft'):
    """
    计算谐波分析误差指标
    
    参数:
    - data_loader: 数据加载器
    - input_cycle_fraction: 输入数据包含的周期数
    - target_samples_per_cycle: 每个周期的目标采样点数
    - method: 分析方法，'fft' 或 'wavelet'
    
    返回:
    - 字典包含各次谐波的平均/最大相对误差以及整体的TMAPE
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 选择分析方法
    if method == 'fft':
        analysis_func = fft_harmonic_analysis
    elif method == 'wavelet':
        analysis_func = wavelet_harmonic_analysis
    else:
        raise ValueError("method must be 'fft' or 'wavelet'")
    
    # 初始化误差列表，分别为1、3、5、7次谐波
    harmonic_errors = {
        1: [],  # 1次谐波误差
        3: [],  # 3次谐波误差
        5: [],  # 5次谐波误差
        7: []   # 7次谐波误差
    }
    
    # 遍历数据加载器
    for batch_idx, (inputs, outputs, _) in enumerate(data_loader):
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        
        # 将批次数据转换为numpy数组
        inputs_np = inputs.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        
        # 对每个样本进行处理
        for i in range(inputs_np.shape[0]):
            input_signal = inputs_np[i]
            output_harmonics = outputs_np[i]
            
            # 检查输出长度是否足够
            if len(output_harmonics) < 4:
                print(f"Warning: Output length {len(output_harmonics)} is less than 4. Skipping sample.")
                continue
            
            # 使用选择的分析方法估计谐波幅度
            estimated_amplitudes = analysis_func(
                input_signal, 
                target_samples_per_cycle, 
                input_cycle_fraction
            )
            
            # 提取真实的谐波幅度
            true_amplitudes = output_harmonics[:4]
            print("True amplitudes:", true_amplitudes)
            print("Estimated amplitudes:", estimated_amplitudes)
            
            # 计算各次谐波的相对误差
            for j, harmonic in enumerate([1, 3, 5, 7]):
                if true_amplitudes[j] > 0:  # 避免除以零
                    error = np.abs(estimated_amplitudes[j] - true_amplitudes[j]) / true_amplitudes[j]
                    harmonic_errors[harmonic].append(error)
    
    # 计算各次谐波的平均和最大相对误差
    results = {}
    for harmonic in [1, 3, 5, 7]:
        if harmonic_errors[harmonic]:
            results[f'harmonic_{harmonic}_mean_error'] = np.mean(harmonic_errors[harmonic])
            results[f'harmonic_{harmonic}_max_error'] = np.max(harmonic_errors[harmonic])
        else:
            results[f'harmonic_{harmonic}_mean_error'] = 0
            results[f'harmonic_{harmonic}_max_error'] = 0
    
    # 计算整体的TMAPE（所有谐波误差的平均值）
    all_errors = []
    for harmonic in [1, 3, 5, 7]:
        all_errors.extend(harmonic_errors[harmonic])
    
    if all_errors:
        results['TMAPE'] = np.mean(all_errors)
    else:
        results['TMAPE'] = 0
    
    # 打印结果
    print(f"Harmonic Analysis Results ({method.upper()} method):")
    for harmonic in [1, 3, 5, 7]:
        print(f"  Harmonic {harmonic}:")
        print(f"    Mean Relative Error(%): {results[f'harmonic_{harmonic}_mean_error']*100:.4f}")
        print(f"    Max Relative Error(%): {results[f'harmonic_{harmonic}_max_error']*100:.4f}")
    print(f"  Overall TMAPE(%): {results['TMAPE']*100:.4f}")

    return results


def main():
    download_harmonic_data()
    config = Config()
    if not (config.TEMP_DATA_DIR / 'voltage_data.csv').exists():
        process_all_data(config, include_simulation=True)

    input_cycle_fraction = 1
    data_loaders = init_dataloaders(config,
                      batch_size=batch_size,
                        shuffle=True,
                          train_ratio=0.7,
                            val_ratio=0.1, 
                                sim_test_ratio=0.2, 
                                    input_cycle_fraction=input_cycle_fraction, 
                                        show_figs=False)
    #voltage_train_loader = data_loaders['voltage_train_loader']
    current_test_sim_loader = data_loaders['current_test_sim_loader']
    #calculate_harmonic_errors(current_test_sim_loader, input_cycle_fraction=input_cycle_fraction, target_samples_per_cycle=128)
    current_train_loader = data_loaders['current_train_loader']
    current_val_loader = data_loaders['current_val_loader']

    # plot one signal
    signal, target, vehicle_id = next(iter(current_test_sim_loader))
    signal = signal[0].numpy()
    target = target[0].numpy()
    vehicle_id = vehicle_id[0].item()
    print("Vehicle ID:", vehicle_id)
    print("Target Harmonics:", target)
    print('Input Signal:', signal)
    print('input shape:', signal.shape)

    
    # init the training env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 64 * input_cycle_fraction
    hidden_size = input_size // 2
    num_layers = 2

    model = QLSTMHarmonic(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    #criterion = nn.MSELoss()  # 使用均方误差损失



    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)


    model_folder = "./harmonic_models"

    epsilon = 1e-8  # TMAPE 的小常数

    # 设置训练参数
    num_epochs = 100
    #batch_size = 32
    model_folder = "./harmonic_models"
    epsilon = 1e-8  # TMAPE 的小常数

    # 创建Trainer实例，使用TMAPE作为损失函数
    trainer = TrainerQLSTMHarmonic(
        model=model,
        trainloader=current_train_loader,  # 使用电流训练数据
        validationloader=current_val_loader,  # 使用电流验证数据
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer=optimizer,
        model_folder=model_folder,
        device=device,
        input_length=input_size,
        epsilon=epsilon
    )

    # 开始训练
    trainer.train()

    # 测试模型
    test_outputs, test_targets = trainer.test(current_test_sim_loader)  
if __name__ == "__main__":
    main()