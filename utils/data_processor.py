import os
import gdown
import zipfile
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, config):
        self.config = config
        
    def unzip_file(self, zip_path, extract_to):
        """解压文件"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    def download_harmonic_data(self):
        """下载谐波数据集"""
        file_id = '1B10CmlWIx_n1kC4WmvQhYLzA-tgpT3Kp' 
        output_path = './data/real_data/EV_CPW.zip'
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 如果文件不存在，则下载
        if not os.path.exists(output_path):
            print("下载谐波数据集...")
            gdown.download(url, output_path, quiet=False)
            print("解压文件...")
            self.unzip_file(output_path, "./data/real_data/")
        else:
            print("谐波数据集已存在，跳过下载")
            
        print("谐波数据集准备完成")

    def create_vehicle_mapping(self, data_root):
        """创建车型到数字的映射"""
        vehicle_folders = [f for f in data_root.iterdir() if f.is_dir()]
        return {vehicle.name: i for i, vehicle in enumerate(vehicle_folders)}

    def downsample_data(self, data, original_rate, target_rate):
        """将数据下采样到目标采样率"""
        # 计算降采样因子并确保是整数
        downsample_factor = original_rate / target_rate
        if not downsample_factor.is_integer():
            # 如果不是整数，使用重采样而不是降采样
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        else:
            downsample_factor = int(downsample_factor)
            return signal.decimate(data, downsample_factor, zero_phase=True)

    def calculate_harmonics(self, signal_data, sampling_rate, system_frequency, harmonic_orders):
        """计算信号的谐波幅值"""
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

    def process_csv_file(self, file_path, vehicle_id):
        """处理单个CSV文件并返回处理后的数据"""
        config = self.config
        
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
        voltage_ds = self.downsample_data(voltage, original_sampling_rate, config.TARGET_SAMPLING_RATE)
        current_ds = self.downsample_data(current, original_sampling_rate, config.TARGET_SAMPLING_RATE)
        
        # # 复制数据两次以增加长度
        # voltage_extended = np.concatenate([voltage_ds, voltage_ds])
        # current_extended = np.concatenate([current_ds, current_ds])
        
        # # 使用扩展周期进行前后填充
        # extension_length = config.EXTENSION_CYCLES * config.TARGET_SAMPLES_PER_CYCLE
        # voltage_padded = np.pad(
        #     voltage_extended, 
        #     (extension_length, extension_length), 
        #     mode='reflect'
        # )
        # current_padded = np.pad(
        #     current_extended, 
        #     (extension_length, extension_length), 
        #     mode='reflect'
        # )
        # 将原始波形重复延展多次
        extension_times = config.EXTENSION_TIMES
        voltage_extended = np.concatenate([voltage_ds] * (extension_times + 1))
        current_extended = np.concatenate([current_ds] * (extension_times + 1))

        # 直接使用延展后的数据
        voltage_padded = voltage_extended
        current_padded = current_extended

        # 计算扩展长度（用于后续的滑动窗口起始位置）
        extension_length = config.EXTENSION_TIMES * config.TARGET_SAMPLES_PER_CYCLE

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
            
            # 计算谐波幅值 - 真实数据使用12个周期的FFT作为GT
            voltage_harmonics = self.calculate_harmonics(
                voltage_output_window, 
                config.TARGET_SAMPLING_RATE, 
                config.SYSTEM_FREQUENCY, 
                config.OUTPUT_HARMONICS
            )
            
            current_harmonics = self.calculate_harmonics(
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

    def generate_simulation_data(self):
        """生成仿真信号数据 - 仿真数据直接使用实际数值作为GT，添加随机移位增加数据多样性"""
        config = self.config
        
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
            
            # 添加随机移位 - 在0到1个周期范围内随机偏移
            random_shift = np.random.rand() * (1 / current_f0)  # 随机时间偏移
            t_shifted = t + random_shift
            
            # 生成电压信号 (假设电压只有基波)
            voltage_signal = current_amplitudes[0] * np.sin(2 * np.pi * current_f0 * t_shifted + phi_true[0])
            
            # 生成电流信号 (包含所有谐波)
            current_signal = np.zeros_like(t_shifted)
            for j, (amp, freq, phi) in enumerate(zip(current_amplitudes, frequency_gt, phi_true)):
                current_signal += amp * np.sin(2 * np.pi * freq * t_shifted + phi)
            
            # 添加噪声
            voltage_power = np.mean(voltage_signal ** 2)
            voltage_noise_power = voltage_power / (10 ** (noise_snr / 10))
            voltage_noise = np.random.normal(0, np.sqrt(voltage_noise_power), voltage_signal.shape)
            voltage_signal += voltage_noise
            
            current_power = np.mean(current_signal ** 2)
            current_noise_power = current_power / (10 ** (noise_snr / 10))
            current_noise = np.random.normal(0, np.sqrt(current_noise_power), current_signal.shape)
            current_signal += current_noise
            
            # 仿真数据直接使用实际数值作为GT，而不是通过FFT计算
            # 电压只有基波，其他谐波为0
            voltage_harmonics_gt = np.array([current_amplitudes[0]] + [0]*(len(config.OUTPUT_HARMONICS)-1))
            
            # 电流包含所有谐波
            current_harmonics_gt = np.array(current_amplitudes)
            
            # 存储数据
            voltage_inputs.append(voltage_signal)
            voltage_outputs.append(voltage_harmonics_gt)
            current_inputs.append(current_signal)
            current_outputs.append(current_harmonics_gt)
        
        return {
            'voltage_inputs': np.array(voltage_inputs),
            'voltage_outputs': np.array(voltage_outputs),
            'current_inputs': np.array(current_inputs),
            'current_outputs': np.array(current_outputs),
            'vehicle_id': config.SIMULATION_VEHICLE_ID
        }

    def process_all_data(self, include_simulation=True):
        """处理所有数据并保存到CSV文件"""
        config = self.config
        
        # 创建输出目录
        config.TEMP_DATA_DIR.mkdir(exist_ok=True)
        
        # 下载数据（如果不存在）
        self.download_harmonic_data()
        
        # 创建车型映射
        vehicle_mapping = self.create_vehicle_mapping(config.DATA_ROOT)
        
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
                    processed_data = self.process_csv_file(csv_file, vehicle_id)
                    
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
            simulation_data = self.generate_simulation_data()
            
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

    def visualize_sample(self, input_data, output_data, vehicle_id, signal_type='current', name='test', show_fig=False):
        """
        可视化样本的波形和谐波直方图
        """
        config = self.config
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制输入波形
        time_ms = np.arange(len(input_data)) * (1000 / config.TARGET_SAMPLING_RATE)
        ax1.plot(time_ms, input_data)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (V)' if signal_type == 'voltage' else 'Current (A)')
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
        return fig  # 修复了这里缺少的括号