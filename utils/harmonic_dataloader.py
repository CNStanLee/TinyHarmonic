import os
import gdown
import zipfile
import numpy as np
import torch

import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

# 配置参数
class Config:
    # 数据路径
    DATA_ROOT = Path('data/real_data/EV-CPW Dataset')
    
    # 采样参数
    ORIGINAL_SAMPLES_PER_CYCLE = 512
    TARGET_SAMPLING_RATE = 3840  # Hz
    SYSTEM_FREQUENCY = 60  # Hz (从数据推断)
    TARGET_SAMPLES_PER_CYCLE = TARGET_SAMPLING_RATE // SYSTEM_FREQUENCY  # 64
    
    # 数据处理参数
    INPUT_CYCLE_FRACTION = 0.5  # 1/2个周期作为输入
    OUTPUT_HARMONICS = [1, 3, 5, 7]  # 输出谐波次数
    EXTENSION_CYCLES = 8  # 复制延拓的周期数
    SLIDING_STEP = 8  # 滑动窗口步长
    
    # 输出参数
    TEMP_DATA_DIR = Path('processed_data')
    
    # 计算派生参数
    INPUT_LENGTH = int(TARGET_SAMPLES_PER_CYCLE * INPUT_CYCLE_FRACTION)  # 32
    OUTPUT_CYCLES = 12  # 符合IEC标准的12周期FFT
    OUTPUT_LENGTH = TARGET_SAMPLES_PER_CYCLE * OUTPUT_CYCLES  # 768
    
    # 仿真参数
    SIMULATION_VEHICLE_ID = 1000  # 为仿真数据分配的特殊车型ID
    NUM_SIMULATION_SAMPLES = 2500  # 仿真样本数量

# 创建车型到数字的映射
def create_vehicle_mapping(data_root):
    vehicle_folders = [f for f in data_root.iterdir() if f.is_dir()]
    return {vehicle.name: i for i, vehicle in enumerate(vehicle_folders)}

# 下采样函数
def downsample_data(data, original_rate, target_rate):
    """
    将数据下采样到目标采样率
    """
    # 计算降采样因子并确保是整数
    downsample_factor = original_rate / target_rate
    if not downsample_factor.is_integer():
        # 如果不是整数，使用重采样而不是降采样
        num_samples = int(len(data) * target_rate / original_rate)
        return signal.resample(data, num_samples)
    else:
        downsample_factor = int(downsample_factor)
        return signal.decimate(data, downsample_factor, zero_phase=True)

# 计算谐波幅值
def calculate_harmonics(signal_data, sampling_rate, system_frequency, harmonic_orders):
    """
    计算信号的谐波幅值
    """
    n = len(signal_data)
    fft_result = fft(signal_data)
    fft_freqs = fftfreq(n, 1/sampling_rate)
    
    harmonics = []
    for order in harmonic_orders:
        target_freq = order * system_frequency
        # 找到最接近目标频率的FFT频点
        idx = np.argmin(np.abs(fft_freqs - target_freq))
        # 计算幅值 (乘以2/N并取绝对值，忽略负频率)
        magnitude = 2 * np.abs(fft_result[idx]) / n
        harmonics.append(magnitude)
    
    return np.array(harmonics)

# 处理单个CSV文件
def process_csv_file(file_path, vehicle_id, config):
    """
    处理单个CSV文件并返回处理后的数据
    """
    # 读取CSV文件
    with open(file_path, 'r') as f:
        # 读取元数据
        trigger_date = f.readline().strip().split(',')[1]
        trigger_time = f.readline().strip().split(',')[1]
        samples_per_cycle = int(f.readline().strip().split(',')[1])
        microseconds_per_sample = float(f.readline().strip().split(',')[1])
        
        # 读取数据
        df = pd.read_csv(f)
    
    # 计算原始采样率
    original_sampling_rate = 1e6 / microseconds_per_sample
    
    # 提取电压和电流数据
    time_ms = df['Time (ms)'].values
    voltage = df['Voltage (V)'].values
    current = df['Current (A)'].values
    
    # 下采样到目标采样率
    voltage_ds = downsample_data(voltage, original_sampling_rate, config.TARGET_SAMPLING_RATE)
    current_ds = downsample_data(current, original_sampling_rate, config.TARGET_SAMPLING_RATE)
    
    # 复制数据两次以增加长度
    voltage_extended = np.concatenate([voltage_ds, voltage_ds])
    current_extended = np.concatenate([current_ds, current_ds])
    
    # 使用扩展周期进行前后填充
    extension_length = config.EXTENSION_CYCLES * config.TARGET_SAMPLES_PER_CYCLE
    voltage_padded = np.pad(
        voltage_extended, 
        (extension_length, extension_length), 
        mode='reflect'
    )
    current_padded = np.pad(
        current_extended, 
        (extension_length, extension_length), 
        mode='reflect'
    )
    
    # 准备存储输入输出对
    voltage_inputs = []
    voltage_outputs = []
    current_inputs = []
    current_outputs = []
    
    # 使用滑动窗口创建样本
    total_length = len(voltage_padded)
    for start_idx in range(extension_length, total_length - extension_length - config.OUTPUT_LENGTH, config.SLIDING_STEP):
        # 计算输出窗口的起始位置
        output_start = start_idx + config.INPUT_LENGTH
        
        # 提取输入和输出窗口
        # 这里我们存储4个周期的数据，以便在数据加载器中灵活选择
        voltage_input = voltage_padded[start_idx:start_idx + 4 * config.TARGET_SAMPLES_PER_CYCLE]
        voltage_output_window = voltage_padded[output_start:output_start + config.OUTPUT_LENGTH]
        
        current_input = current_padded[start_idx:start_idx + 4 * config.TARGET_SAMPLES_PER_CYCLE]
        current_output_window = current_padded[output_start:output_start + config.OUTPUT_LENGTH]
        
        # 计算谐波幅值
        voltage_harmonics = calculate_harmonics(
            voltage_output_window, 
            config.TARGET_SAMPLING_RATE, 
            config.SYSTEM_FREQUENCY, 
            config.OUTPUT_HARMONICS
        )
        
        current_harmonics = calculate_harmonics(
            current_output_window, 
            config.TARGET_SAMPLING_RATE, 
            config.SYSTEM_FREQUENCY, 
            config.OUTPUT_HARMONICS
        )
        
        # 存储结果
        voltage_inputs.append(voltage_input)
        voltage_outputs.append(voltage_harmonics)
        current_inputs.append(current_input)
        current_outputs.append(current_harmonics)
    
    return {
        'voltage_inputs': np.array(voltage_inputs),
        'voltage_outputs': np.array(voltage_outputs),
        'current_inputs': np.array(current_inputs),
        'current_outputs': np.array(current_outputs),
        'vehicle_id': vehicle_id,
        'file_path': str(file_path)
    }

# 生成仿真信号
def generate_simulation_data(config):
    """
    生成仿真信号数据
    """
    # 仿真参数
    f0 = 60  # 基波频率
    fs = 3840  # 采样频率
    
    # 谐波配置
    amplitude_gt = [100, 30, 20, 15]  # 基波和谐波幅值
    frequency_gt = [f0, 3*f0, 5*f0, 7*f0]  # 基波和谐波频率
    phi_true = np.deg2rad([152, 35, 0, 0])  # 相位角（弧度）
    harmonics = [1, 3, 5, 7]  # 谐波次数
    
    # 实验参数
    num_trials = config.NUM_SIMULATION_SAMPLES
    frequency_range = 0.005  # 频率变化范围
    amplitude_range = 0.01  # 幅值变化范围
    noise_snr = 26  # 信噪比 (dB)
    
    # 准备存储数据
    voltage_inputs = []
    voltage_outputs = []
    current_inputs = []
    current_outputs = []
    
    # 生成仿真数据
    for i in range(num_trials):
        # 添加随机变化
        freq_variation = f0 * frequency_range * (2 * np.random.rand() - 1)
        current_f0 = f0 + freq_variation
        
        amp_variation = [a * amplitude_range * (2 * np.random.rand() - 1) for a in amplitude_gt]
        current_amplitudes = [a + v for a, v in zip(amplitude_gt, amp_variation)]
        
        # 生成时间序列
        t = np.arange(4 * config.TARGET_SAMPLES_PER_CYCLE) / fs
        
        # 生成电压信号 (假设电压只有基波)
        voltage_signal = current_amplitudes[0] * np.sin(2 * np.pi * current_f0 * t + phi_true[0])
        
        # 生成电流信号 (包含所有谐波)
        current_signal = np.zeros_like(t)
        for j, (amp, freq, phi) in enumerate(zip(current_amplitudes, frequency_gt, phi_true)):
            current_signal += amp * np.sin(2 * np.pi * freq * t + phi)
        
        # 添加噪声
        voltage_power = np.mean(voltage_signal ** 2)
        voltage_noise_power = voltage_power / (10 ** (noise_snr / 10))
        voltage_noise = np.random.normal(0, np.sqrt(voltage_noise_power), voltage_signal.shape)
        voltage_signal += voltage_noise
        
        current_power = np.mean(current_signal ** 2)
        current_noise_power = current_power / (10 ** (noise_snr / 10))
        current_noise = np.random.normal(0, np.sqrt(current_noise_power), current_signal.shape)
        current_signal += current_noise
        
        # 计算谐波幅值 (使用12个周期的数据)
        t_output = np.arange(config.OUTPUT_LENGTH) / fs
        current_output_signal = np.zeros_like(t_output)
        for j, (amp, freq, phi) in enumerate(zip(current_amplitudes, frequency_gt, phi_true)):
            current_output_signal += amp * np.sin(2 * np.pi * freq * t_output + phi)
        
        # 添加噪声到输出信号
        current_output_noise = np.random.normal(0, np.sqrt(current_noise_power), current_output_signal.shape)
        current_output_signal += current_output_noise
        
        # 计算谐波
        current_harmonics = calculate_harmonics(
            current_output_signal, 
            fs, 
            f0, 
            harmonics
        )
        
        # 存储数据
        voltage_inputs.append(voltage_signal)
        voltage_outputs.append(np.array([current_amplitudes[0], 0, 0, 0]))  # 电压只有基波
        current_inputs.append(current_signal)
        current_outputs.append(current_harmonics)
    
    return {
        'voltage_inputs': np.array(voltage_inputs),
        'voltage_outputs': np.array(voltage_outputs),
        'current_inputs': np.array(current_inputs),
        'current_outputs': np.array(current_outputs),
        'vehicle_id': config.SIMULATION_VEHICLE_ID
    }

# 主处理函数
def process_all_data(config, include_simulation=True):
    """
    处理所有数据并保存到CSV文件
    """
    # 创建输出目录
    config.TEMP_DATA_DIR.mkdir(exist_ok=True)
    
    # 创建车型映射
    vehicle_mapping = create_vehicle_mapping(config.DATA_ROOT)
    
    # 准备存储所有数据
    all_voltage_inputs = []
    all_voltage_outputs = []
    all_current_inputs = []
    all_current_outputs = []
    all_vehicle_ids = []
    
    # 遍历所有车型文件夹
    for vehicle_name, vehicle_id in vehicle_mapping.items():
        vehicle_path = config.DATA_ROOT / vehicle_name / 'Waveforms'
        
        if not vehicle_path.exists():
            print(f"警告: {vehicle_path} 不存在，跳过")
            continue
        
        # 处理该车型下的所有CSV文件
        csv_files = list(vehicle_path.glob('*.csv'))
        print(f"处理车型 {vehicle_name} (ID: {vehicle_id}), 找到 {len(csv_files)} 个文件")
        
        for i, csv_file in enumerate(csv_files):
            print(f"  处理文件 {i+1}/{len(csv_files)}: {csv_file.name}")
            
            try:
                processed_data = process_csv_file(csv_file, vehicle_id, config)
                
                # 收集数据
                all_voltage_inputs.append(processed_data['voltage_inputs'])
                all_voltage_outputs.append(processed_data['voltage_outputs'])
                all_current_inputs.append(processed_data['current_inputs'])
                all_current_outputs.append(processed_data['current_outputs'])
                
                # 为每个样本添加车型ID
                n_samples = processed_data['voltage_inputs'].shape[0]
                all_vehicle_ids.extend([vehicle_id] * n_samples)
                
            except Exception as e:
                print(f"  处理文件 {csv_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 检查是否有数据
    if not all_voltage_inputs:
        print("警告: 没有处理任何真实数据!")
    else:
        # 合并所有真实数据
        voltage_inputs = np.vstack(all_voltage_inputs)
        voltage_outputs = np.vstack(all_voltage_outputs)
        current_inputs = np.vstack(all_current_inputs)
        current_outputs = np.vstack(all_current_outputs)
        vehicle_ids = np.array(all_vehicle_ids)
    
    # 添加仿真数据
    if include_simulation:
        print("生成仿真数据...")
        simulation_data = generate_simulation_data(config)
        
        if not all_voltage_inputs:
            # 如果没有真实数据，只使用仿真数据
            voltage_inputs = simulation_data['voltage_inputs']
            voltage_outputs = simulation_data['voltage_outputs']
            current_inputs = simulation_data['current_inputs']
            current_outputs = simulation_data['current_outputs']
            vehicle_ids = np.array([config.SIMULATION_VEHICLE_ID] * config.NUM_SIMULATION_SAMPLES)
        else:
            # 合并真实数据和仿真数据
            voltage_inputs = np.vstack([voltage_inputs, simulation_data['voltage_inputs']])
            voltage_outputs = np.vstack([voltage_outputs, simulation_data['voltage_outputs']])
            current_inputs = np.vstack([current_inputs, simulation_data['current_inputs']])
            current_outputs = np.vstack([current_outputs, simulation_data['current_outputs']])
            vehicle_ids = np.concatenate([vehicle_ids, 
                                         np.array([config.SIMULATION_VEHICLE_ID] * config.NUM_SIMULATION_SAMPLES)])
    
    # 保存到CSV文件
    # 电压数据
    voltage_df = pd.DataFrame(voltage_inputs)
    voltage_df.columns = [f'voltage_input_{i}' for i in range(voltage_inputs.shape[1])]
    for i, harmonic in enumerate(config.OUTPUT_HARMONICS):
        voltage_df[f'voltage_harmonic_{harmonic}'] = voltage_outputs[:, i]
    voltage_df['vehicle_id'] = vehicle_ids
    voltage_df.to_csv(config.TEMP_DATA_DIR / 'voltage_data.csv', index=False)
    
    # 电流数据
    current_df = pd.DataFrame(current_inputs)
    current_df.columns = [f'current_input_{i}' for i in range(current_inputs.shape[1])]
    for i, harmonic in enumerate(config.OUTPUT_HARMONICS):
        current_df[f'current_harmonic_{harmonic}'] = current_outputs[:, i]
    current_df['vehicle_id'] = vehicle_ids
    current_df.to_csv(config.TEMP_DATA_DIR / 'current_data.csv', index=False)
    
    # 保存配置信息
    config_df = pd.DataFrame({
        'parameter': [
            'input_cycle_fraction', 
            'output_harmonics', 
            'target_sampling_rate', 
            'system_frequency',
            'target_samples_per_cycle',
            'simulation_vehicle_id'
        ],
        'value': [
            str(config.INPUT_CYCLE_FRACTION),
            str(config.OUTPUT_HARMONICS),
            str(config.TARGET_SAMPLING_RATE),
            str(config.SYSTEM_FREQUENCY),
            str(config.TARGET_SAMPLES_PER_CYCLE),
            str(config.SIMULATION_VEHICLE_ID)
        ]
    })
    config_df.to_csv(config.TEMP_DATA_DIR / 'config.csv', index=False)
    
    print(f"处理完成! 共生成 {voltage_inputs.shape[0]} 个样本")
    print(f"数据已保存到 {config.TEMP_DATA_DIR}")
    
    return voltage_inputs, voltage_outputs, current_inputs, current_outputs, vehicle_ids

# 数据加载器
class HarmonicDataset:
    def __init__(self, inputs, outputs, vehicle_ids=None, input_cycle_fraction=0.5, target_samples_per_cycle=64):
        self.full_inputs = inputs  # 存储完整的4个周期数据
        self.outputs = outputs
        self.vehicle_ids = vehicle_ids
        self.n_samples = inputs.shape[0]
        self.input_cycle_fraction = input_cycle_fraction
        self.target_samples_per_cycle = target_samples_per_cycle
        self.update_input_length()
    
    def update_input_length(self, input_cycle_fraction=None):
        """更新输入长度"""
        if input_cycle_fraction is not None:
            self.input_cycle_fraction = input_cycle_fraction
        
        # 计算输入长度（绝对值）
        self.input_length = int(self.target_samples_per_cycle * abs(self.input_cycle_fraction))
    
    def set_input_cycle_fraction(self, input_cycle_fraction):
        """设置输入周期比例"""
        self.update_input_length(input_cycle_fraction)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 从完整的4个周期数据中提取指定长度的输入
        # 这里可以根据需要选择不同的部分
        
        # 计算起始索引
        if self.input_cycle_fraction >= 0:
            # 正数表示从当前周期开始
            start_idx = self.target_samples_per_cycle  # 当前周期的起始索引
            start_idx += int(self.target_samples_per_cycle * (1 - self.input_cycle_fraction))
        else:
            # 负数表示从上一个周期开始
            start_idx = int(self.target_samples_per_cycle * (1 + self.input_cycle_fraction))
        
        # 确保索引在有效范围内
        start_idx = max(0, min(start_idx, self.full_inputs.shape[1] - self.input_length))
        end_idx = start_idx + self.input_length
        
        input_data = self.full_inputs[idx, start_idx:end_idx]
        
        if self.vehicle_ids is not None:
            return input_data, self.outputs[idx], self.vehicle_ids[idx]
        return input_data, self.outputs[idx]

# 创建数据加载器
# def create_data_loaders(config, batch_size=32, shuffle=True, train_ratio=0.8, input_cycle_fraction=None):
#     """
#     从CSV文件创建数据加载器
#     """
#     # 从CSV文件加载数据
#     voltage_df = pd.read_csv(config.TEMP_DATA_DIR / 'voltage_data.csv')
#     current_df = pd.read_csv(config.TEMP_DATA_DIR / 'current_data.csv')
    
#     # 提取输入和输出数据
#     voltage_input_cols = [col for col in voltage_df.columns if col.startswith('voltage_input_')]
#     voltage_output_cols = [col for col in voltage_df.columns if col.startswith('voltage_harmonic_')]
    
#     current_input_cols = [col for col in current_df.columns if col.startswith('current_input_')]
#     current_output_cols = [col for col in current_df.columns if col.startswith('current_harmonic_')]
    
#     voltage_inputs = voltage_df[voltage_input_cols].values
#     voltage_outputs = voltage_df[voltage_output_cols].values
#     voltage_vehicle_ids = voltage_df['vehicle_id'].values
    
#     current_inputs = current_df[current_input_cols].values
#     current_outputs = current_df[current_output_cols].values
#     current_vehicle_ids = current_df['vehicle_id'].values
    
#     # 创建电压和电流数据集
#     voltage_dataset = HarmonicDataset(
#         voltage_inputs, 
#         voltage_outputs, 
#         voltage_vehicle_ids,
#         input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
#         config.TARGET_SAMPLES_PER_CYCLE
#     )
    
#     current_dataset = HarmonicDataset(
#         current_inputs, 
#         current_outputs, 
#         current_vehicle_ids,
#         input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
#         config.TARGET_SAMPLES_PER_CYCLE
#     )
    
#     # 分割训练集和测试集
#     n_total = voltage_inputs.shape[0]
#     n_train = int(n_total * train_ratio)
#     indices = np.random.permutation(n_total)
    
#     train_indices = indices[:n_train]
#     test_indices = indices[n_train:]
    
#     # 创建电压数据加载器
#     voltage_train_dataset = HarmonicDataset(
#         voltage_inputs[train_indices], 
#         voltage_outputs[train_indices],
#         voltage_vehicle_ids[train_indices] if voltage_vehicle_ids is not None else None,
#         input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
#         config.TARGET_SAMPLES_PER_CYCLE
#     )
    
#     voltage_test_dataset = HarmonicDataset(
#         voltage_inputs[test_indices], 
#         voltage_outputs[test_indices],
#         voltage_vehicle_ids[test_indices] if voltage_vehicle_ids is not None else None,
#         input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
#         config.TARGET_SAMPLES_PER_CYCLE
#     )
    
#     # 创建电流数据加载器
#     current_train_dataset = HarmonicDataset(
#         current_inputs[train_indices], 
#         current_outputs[train_indices],
#         current_vehicle_ids[train_indices] if current_vehicle_ids is not None else None,
#         input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
#         config.TARGET_SAMPLES_PER_CYCLE
#     )
    
#     current_test_dataset = HarmonicDataset(
#         current_inputs[test_indices], 
#         current_outputs[test_indices],
#         current_vehicle_ids[test_indices] if current_vehicle_ids is not None else None,
#         input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
#         config.TARGET_SAMPLES_PER_CYCLE
#     )
    
#     return {
#         'voltage': {
#             'train': voltage_train_dataset,
#             'test': voltage_test_dataset
#         },
#         'current': {
#             'train': current_train_dataset,
#             'test': current_test_dataset
#         }
#     }
def create_data_loaders(config, batch_size=32, shuffle=True, train_ratio=0.7, val_ratio=0.1, 
                        sim_test_ratio=0.2, input_cycle_fraction=None):
    """
    从CSV文件创建数据加载器，确保没有数据泄漏
    训练集和验证集来自真实数据和部分仿真数据
    测试集分为仿真数据测试集和真实数据测试集
    """
    # 从CSV文件加载数据
    voltage_df = pd.read_csv(config.TEMP_DATA_DIR / 'voltage_data.csv')
    current_df = pd.read_csv(config.TEMP_DATA_DIR / 'current_data.csv')
    
    # 提取输入和输出数据
    voltage_input_cols = [col for col in voltage_df.columns if col.startswith('voltage_input_')]
    voltage_output_cols = [col for col in voltage_df.columns if col.startswith('voltage_harmonic_')]
    
    current_input_cols = [col for col in current_df.columns if col.startswith('current_input_')]
    current_output_cols = [col for col in current_df.columns if col.startswith('current_harmonic_')]
    
    voltage_inputs = voltage_df[voltage_input_cols].values
    voltage_outputs = voltage_df[voltage_output_cols].values
    voltage_vehicle_ids = voltage_df['vehicle_id'].values
    
    current_inputs = current_df[current_input_cols].values
    current_outputs = current_df[current_output_cols].values
    current_vehicle_ids = current_df['vehicle_id'].values
    
    # 分离仿真数据 (vehicle_id = 1000) 和真实数据 (vehicle_id != 1000)
    voltage_sim_mask = (voltage_vehicle_ids == 1000)
    voltage_real_mask = (voltage_vehicle_ids != 1000)
    
    current_sim_mask = (current_vehicle_ids == 1000)
    current_real_mask = (current_vehicle_ids != 1000)
    
    # 获取仿真数据和真实数据
    voltage_sim_inputs = voltage_inputs[voltage_sim_mask]
    voltage_sim_outputs = voltage_outputs[voltage_sim_mask]
    voltage_sim_ids = voltage_vehicle_ids[voltage_sim_mask]
    
    voltage_real_inputs = voltage_inputs[voltage_real_mask]
    voltage_real_outputs = voltage_outputs[voltage_real_mask]
    voltage_real_ids = voltage_vehicle_ids[voltage_real_mask]
    
    current_sim_inputs = current_inputs[current_sim_mask]
    current_sim_outputs = current_outputs[current_sim_mask]
    current_sim_ids = current_vehicle_ids[current_sim_mask]
    
    current_real_inputs = current_inputs[current_real_mask]
    current_real_outputs = current_outputs[current_real_mask]
    current_real_ids = current_vehicle_ids[current_real_mask]
    
    # 划分仿真数据为训练/验证和测试部分
    n_sim = voltage_sim_inputs.shape[0]
    n_sim_test = int(n_sim * sim_test_ratio)
    
    sim_indices = np.random.permutation(n_sim)
    sim_test_indices = sim_indices[:n_sim_test]
    sim_train_val_indices = sim_indices[n_sim_test:]
    
    # 划分真实数据为训练/验证和测试部分
    n_real = voltage_real_inputs.shape[0]
    n_real_test = int(n_real * sim_test_ratio)  # 使用相同的测试比例
    
    real_indices = np.random.permutation(n_real)
    real_test_indices = real_indices[:n_real_test]
    real_train_val_indices = real_indices[n_real_test:]
    
    # 合并仿真和真实的训练/验证数据
    voltage_train_val_inputs = np.concatenate([
        voltage_sim_inputs[sim_train_val_indices],
        voltage_real_inputs[real_train_val_indices]
    ], axis=0)
    
    voltage_train_val_outputs = np.concatenate([
        voltage_sim_outputs[sim_train_val_indices],
        voltage_real_outputs[real_train_val_indices]
    ], axis=0)
    
    voltage_train_val_ids = np.concatenate([
        voltage_sim_ids[sim_train_val_indices],
        voltage_real_ids[real_train_val_indices]
    ], axis=0)
    
    current_train_val_inputs = np.concatenate([
        current_sim_inputs[sim_train_val_indices],
        current_real_inputs[real_train_val_indices]
    ], axis=0)
    
    current_train_val_outputs = np.concatenate([
        current_sim_outputs[sim_train_val_indices],
        current_real_outputs[real_train_val_indices]
    ], axis=0)
    
    current_train_val_ids = np.concatenate([
        current_sim_ids[sim_train_val_indices],
        current_real_ids[real_train_val_indices]
    ], axis=0)
    
    # 划分训练/验证数据为训练集和验证集
    n_train_val = voltage_train_val_inputs.shape[0]
    n_train = int(n_train_val * train_ratio)
    n_val = n_train_val - n_train
    
    train_val_indices = np.random.permutation(n_train_val)
    train_indices = train_val_indices[:n_train]
    val_indices = train_val_indices[n_train:n_train+n_val]
    
    # 创建电压数据集
    voltage_train_dataset = HarmonicDataset(
        voltage_train_val_inputs[train_indices].astype(np.float32),  # 确保输入是 float32
        voltage_train_val_outputs[train_indices].astype(np.float32),  # 确保输出是 float32
        voltage_train_val_ids[train_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )

    voltage_val_dataset = HarmonicDataset(
        voltage_train_val_inputs[val_indices].astype(np.float32),  # 确保输入是 float32
        voltage_train_val_outputs[val_indices].astype(np.float32),  # 确保输出是 float32
        voltage_train_val_ids[val_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )

    # 创建电压仿真测试集
    voltage_test_sim_dataset = HarmonicDataset(
        voltage_sim_inputs[sim_test_indices].astype(np.float32),  # 确保输入是 float32
        voltage_sim_outputs[sim_test_indices].astype(np.float32),  # 确保输出是 float32
        voltage_sim_ids[sim_test_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )

    # 创建电压真实测试集
    voltage_test_real_dataset = HarmonicDataset(
        voltage_real_inputs[real_test_indices].astype(np.float32),  # 确保输入是 float32
        voltage_real_outputs[real_test_indices].astype(np.float32),  # 确保输出是 float32
        voltage_real_ids[real_test_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )

    # 创建电流数据集
    current_train_dataset = HarmonicDataset(
        current_train_val_inputs[train_indices].astype(np.float32),  # 确保输入是 float32
        current_train_val_outputs[train_indices].astype(np.float32),  # 确保输出是 float32
        current_train_val_ids[train_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )

    current_val_dataset = HarmonicDataset(
        current_train_val_inputs[val_indices].astype(np.float32),  # 确保输入是 float32
        current_train_val_outputs[val_indices].astype(np.float32),  # 确保输出是 float32
        current_train_val_ids[val_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )

    # 创建电流仿真测试集
    current_test_sim_dataset = HarmonicDataset(
        current_sim_inputs[sim_test_indices].astype(np.float32),  # 确保输入是 float32
        current_sim_outputs[sim_test_indices].astype(np.float32),  # 确保输出是 float32
        current_sim_ids[sim_test_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )

    # 创建电流真实测试集
    current_test_real_dataset = HarmonicDataset(
        current_real_inputs[real_test_indices].astype(np.float32),  # 确保输入是 float32
        current_real_outputs[real_test_indices].astype(np.float32),  # 确保输出是 float32
        current_real_ids[real_test_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )
    
    return {
        'voltage': {
            'train': voltage_train_dataset,
            'val': voltage_val_dataset,
            'test_sim': voltage_test_sim_dataset,
            'test_real': voltage_test_real_dataset
        },
        'current': {
            'train': current_train_dataset,
            'val': current_val_dataset,
            'test_sim': current_test_sim_dataset,
            'test_real': current_test_real_dataset
        }
    }

# 可视化函数
def visualize_sample(input_data, output_data, vehicle_id, signal_type='current', config=None, name='test', show_fig=False):
    """
    可视化样本的波形和谐波直方图
    """
    # 获取样本数据
    if signal_type == 'current':
        #input_data, output_data, vehicle_id = dataset['current']['train'][sample_idx]
        signal_label = 'Current (A)'
    else:
        #input_data, output_data, vehicle_id = dataset['voltage']['train'][sample_idx]
        signal_label = 'Voltage (V)'
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制输入波形
    time_ms = np.arange(len(input_data)) * (1000 / config.TARGET_SAMPLING_RATE)
    ax1.plot(time_ms, input_data)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel(signal_label)
    ax1.set_title(f'{signal_type.capitalize()} Waveform (Vehicle {vehicle_id})')
    ax1.grid(True)
    
    # 绘制谐波直方图
    harmonics = config.OUTPUT_HARMONICS
    ax2.bar(range(len(harmonics)), output_data, tick_label=[f'H{h}' for h in harmonics])
    ax2.set_xlabel('Harmonic Order')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f'{signal_type.capitalize()} Harmonic Magnitudes')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.TEMP_DATA_DIR / f'{signal_type}_{name}_sample.png')
    if show_fig:
        plt.show()
    plt.close()
    return fig

# 示例使用
if __name__ == "__main__":
    config = Config()
    
    # 处理所有数据（只需要运行一次）
    if not (config.TEMP_DATA_DIR / 'voltage_data.csv').exists():
        process_all_data(config, include_simulation=True)
    
    # 创建数据加载器，可以指定输入周期比例
    loaders = create_data_loaders(config, batch_size=32, input_cycle_fraction=0.5)
    
    # 示例：获取一批电压数据
    voltage_train_loader = loaders['voltage']['train']
    sample_input, sample_output, vehicle_id = voltage_train_loader[0]
    
    print(f"输入形状: {sample_input.shape}")
    print(f"输出形状: {sample_output.shape}")
    print(f"车型ID: {vehicle_id}")
    
    # 示例：获取一批电流数据
    current_train_loader = loaders['current']['train']
    sample_input, sample_output, vehicle_id = current_train_loader[0]
    
    print(f"输入形状: {sample_input.shape}")
    print(f"输出形状: {sample_output.shape}")
    print(f"车型ID: {vehicle_id}")
    
    # 可视化一个样本
    visualize_sample(loaders, sample_idx=0, signal_type='current', config=config)
    visualize_sample(loaders, sample_idx=0, signal_type='voltage', config=config)
    
    # 示例：更改输入周期比例
    print("\n更改输入周期比例为-0.5（使用上一个周期的后半部分）")
    current_train_loader.set_input_cycle_fraction(-0.5)
    sample_input, sample_output, vehicle_id = current_train_loader[0]
    print(f"新的输入形状: {sample_input.shape}")
    
    # 绘制新的输入波形
    plt.figure(figsize=(10, 5))
    time_ms = np.arange(len(sample_input)) * (1000 / config.TARGET_SAMPLING_RATE)
    plt.plot(time_ms, sample_input)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (A)')
    plt.title('Current Waveform with -0.5 Cycle Fraction')
    plt.grid(True)
    plt.savefig(config.TEMP_DATA_DIR / 'current_waveform_-0.5_cycle.png')
    plt.show()

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def download_harmonic_data():
    # harmonic data
    file_id = '1B10CmlWIx_n1kC4WmvQhYLzA-tgpT3Kp' 
    output_path = './data/real_data/EV_CPW.zip'
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # if the file does not exist, download it
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
        unzip_file(output_path, "./data/real_data/")
    print("Harmonic dataset downloaded and unzipped.")

def init_dataloaders(config, batch_size=32, shuffle=True, train_ratio=0.7, val_ratio=0.1, 
                        sim_test_ratio=0.2, input_cycle_fraction=3, show_figs=False):
    loaders = create_data_loaders(config, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  train_ratio=train_ratio, 
                                  val_ratio=val_ratio, 
                                  sim_test_ratio=sim_test_ratio,
                                  input_cycle_fraction=input_cycle_fraction)

    voltage_train_loader = loaders['voltage']['train']
    voltage_val_loader = loaders['voltage']['val']
    voltage_test_sim_loader = loaders['voltage']['test_sim']
    voltage_test_real_loader = loaders['voltage']['test_real']
    current_train_loader = loaders['current']['train']
    current_val_loader = loaders['current']['val']
    current_test_sim_loader = loaders['current']['test_sim']
    current_test_real_loader = loaders['current']['test_real']

    print("voltage dataset info:")
    print(f"training set size: {len(voltage_train_loader)}")
    print(f"validation set size: {len(voltage_val_loader)}")
    print(f"test set (simulation) size: {len(voltage_test_sim_loader)}")
    print(f"test set (real) size: {len(voltage_test_real_loader)}")
    print(f"input cycle fraction: {voltage_train_loader.input_cycle_fraction}")

    print("\ncurrent dataset info:")
    print(f"training set size: {len(current_train_loader)}")
    print(f"validation set size: {len(current_val_loader)}")
    print(f"test set (simulation) size: {len(current_test_sim_loader)}")
    print(f"test set (real) size: {len(current_test_real_loader)}")
    print(f"input cycle fraction: {current_train_loader.input_cycle_fraction}")

    sample_input, sample_output, vehicle_id = current_train_loader[15]
    visualize_sample(sample_input, sample_output, vehicle_id, signal_type='current', config=config, name=f'train_{input_cycle_fraction}', show_fig=show_figs)
    sample_input, sample_output, vehicle_id = current_val_loader[15]
    visualize_sample(sample_input, sample_output, vehicle_id, signal_type='current', config=config, name=f'val_{input_cycle_fraction}', show_fig=show_figs)
    sample_input, sample_output, vehicle_id = current_test_sim_loader[15]
    visualize_sample(sample_input, sample_output, vehicle_id, signal_type='current', config=config, name=f'test_sim_{input_cycle_fraction}', show_fig=show_figs)
    sample_input, sample_output, vehicle_id = current_test_real_loader[15]
    visualize_sample(sample_input, sample_output, vehicle_id, signal_type='current', config=config, name=f'test_real_{input_cycle_fraction}', show_fig=show_figs)
    # print data shape
    print(f"current sample input shape: {sample_input.shape}")
    print(f"current sample output shape: {sample_output.shape}")

    # return voltage_train_loader, voltage_val_loader, voltage_test_sim_loader, voltage_test_real_loader, \
    #        current_train_loader, current_val_loader, current_test_sim_loader, current_test_real_loader
    # convert to loaders with batch size
    voltage_train_loader = DataLoader(voltage_train_loader, batch_size=batch_size, shuffle=shuffle)
    voltage_val_loader = DataLoader(voltage_val_loader, batch_size=batch_size, shuffle=shuffle)
    voltage_test_sim_loader = DataLoader(voltage_test_sim_loader, batch_size=batch_size, shuffle=shuffle)
    voltage_test_real_loader = DataLoader(voltage_test_real_loader, batch_size=batch_size, shuffle=shuffle)
    current_train_loader = DataLoader(current_train_loader, batch_size=batch_size, shuffle=shuffle)
    current_val_loader = DataLoader(current_val_loader, batch_size=batch_size, shuffle=shuffle)
    current_test_sim_loader = DataLoader(current_test_sim_loader, batch_size=batch_size, shuffle=shuffle)
    current_test_real_loader = DataLoader(current_test_real_loader, batch_size=batch_size, shuffle=shuffle)

    return {
        'voltage_train_loader': voltage_train_loader,
        'voltage_val_loader': voltage_val_loader,
        'voltage_test_sim_loader': voltage_test_sim_loader,
        'voltage_test_real_loader': voltage_test_real_loader,
        'current_train_loader': current_train_loader,
        'current_val_loader': current_val_loader,
        'current_test_sim_loader': current_test_sim_loader,
        'current_test_real_loader': current_test_real_loader
    }