import numpy as np
import pywt
import matplotlib.pyplot as plt

# 参数设置
fs = 3840  # 采样率 (Hz)
duration = 1.0  # 信号时长 (秒)
f0 = 60  # 基波频率 (Hz)
harmonics = [3, 5, 7]  # 谐波次数
amplitudes = [1.0, 0.1, 0.05, 0.03]  # 基波+谐波幅值 [A0, A3, A5, A7]

# 生成时间序列
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 生成信号：基波 + 3/5/7次谐波
signal = amplitudes[0] * np.sin(2 * np.pi * f0 * t)
for i, h in enumerate(harmonics):
    freq = f0 * h
    signal += amplitudes[i+1] * np.sin(2 * np.pi * freq * t)

# 添加高斯白噪声 (SNR=60dB)
noise_power = 10**(-60/10) * np.var(signal)
signal += np.random.normal(0, np.sqrt(noise_power), len(signal))

# 小波包变换参数
wavelet = 'db20'
level = 4  # 分解层数

# 执行小波包分解
wp = pywt.WaveletPacket(signal, wavelet, mode='symmetric', maxlevel=level)

# 获取频率带对应的节点
freq_bands = []
node_names = [node.path for node in wp.get_level(level, 'freq')]
bandwidth = fs / 2 / (2**level)  # 每个频带的宽度

# 计算各频带中心频率
center_freqs = [(i + 0.5) * bandwidth for i in range(2**level)]

# 目标频率列表
target_freqs = [f0 * h for h in [1] + harmonics]  # [60, 180, 300, 420] Hz

# 幅值估计
estimated_amps = []
for freq in target_freqs:
    # 找到最接近目标频率的频带索引
    band_idx = int(freq / bandwidth)
    node_name = node_names[band_idx]
    
    # 重构该节点信号
    node_signal = wp[node_name].reconstruct(update=False)
    
    # 使用希尔伯特变换计算包络
    analytic_signal = np.abs(pywt.hilbert(node_signal))
    
    # 去除边界效应 (去掉首尾10%)
    n_remove = int(0.1 * len(analytic_signal))
    clean_envelope = analytic_signal[n_remove:-n_remove]
    
    # 取包络中值作为幅值估计 (比峰值更鲁棒)
    amp_est = np.median(clean_envelope)
    estimated_amps.append(amp_est)

# 计算相对误差
true_amps = amplitudes
errors = []
for i in range(len(true_amps)):
    rel_error = 100 * abs(estimated_amps[i] - true_amps[i]) / true_amps[i]
    errors.append(rel_error)

# 输出结果
print("频率(Hz)\t真实幅值\t估计幅值\t相对误差(%)")
print("------------------------------------------------")
for i, freq in enumerate(target_freqs):
    print(f"{freq:.1f}\t\t{true_amps[i]:.4f}\t\t{estimated_amps[i]:.4f}\t\t{errors[i]:.2f}%")

# 绘制结果
plt.figure(figsize=(12, 8))

# 原始信号
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('原始信号 (含60Hz基波+3/5/7次谐波)')
plt.xlabel('时间 (s)')
plt.ylabel('幅值')
plt.xlim(0, 0.1)  # 显示前0.1秒

# 幅值比较
plt.subplot(2, 1, 2)
x = np.arange(len(target_freqs))
width = 0.35
plt.bar(x - width/2, true_amps, width, label='真实幅值')
plt.bar(x + width/2, estimated_amps, width, label='估计幅值')
plt.xticks(x, [f'{freq}Hz' for freq in target_freqs])
plt.xlabel('频率分量')
plt.ylabel('幅值')
plt.title('谐波幅值估计对比')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
