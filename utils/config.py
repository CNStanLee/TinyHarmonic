from pathlib import Path

class Config:
    DATA_ROOT = Path('data/real_data/EV-CPW Dataset')
    
    ORIGINAL_SAMPLES_PER_CYCLE = 512
    TARGET_SAMPLING_RATE = 3840 # 7680 Hz  
    SYSTEM_FREQUENCY = 60 
    TARGET_SAMPLES_PER_CYCLE = TARGET_SAMPLING_RATE // SYSTEM_FREQUENCY  # 64
    
    INPUT_CYCLE_FRACTION = 0.5  # 1/2个周期作为输入
    OUTPUT_HARMONICS = [1, 3, 5, 7]  # 输出谐波次数
    EXTENSION_TIMES = 8
    SLIDING_STEP = 6  # 滑动窗口步长
    
    # 输出参数
    TEMP_DATA_DIR = Path('processed_data')
    
    # 计算派生参数
    INPUT_LENGTH = int(TARGET_SAMPLES_PER_CYCLE * INPUT_CYCLE_FRACTION)  # 32
    OUTPUT_CYCLES = 12  # 符合IEC标准的12周期FFT
    OUTPUT_LENGTH = TARGET_SAMPLES_PER_CYCLE * OUTPUT_CYCLES  # 768
    
    # 仿真参数
    SIMULATION_VEHICLE_ID = 1000  # 为仿真数据分配的特殊车型ID
    NUM_SIMULATION_SAMPLES = 25000  # 仿真样本数量