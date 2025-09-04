import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from utils.harmonic_dataset import HarmonicDataset

def create_real_voltage_datasets(config, shuffle=True, train_ratio=0.7, val_ratio=0.1, 
                                test_ratio=0.2, input_cycle_fraction=None):
    """
    创建真实电压数据集，确保没有数据泄漏
    """
    # 从CSV文件加载数据
    voltage_df = pd.read_csv(config.TEMP_DATA_DIR / 'voltage_data.csv')
    
    # 只保留真实数据 (vehicle_id != 1000)
    voltage_df = voltage_df[voltage_df['vehicle_id'] != config.SIMULATION_VEHICLE_ID]
    
    # 提取输入和输出数据
    voltage_input_cols = [col for col in voltage_df.columns if col.startswith('voltage_input_')]
    voltage_output_cols = [col for col in voltage_df.columns if col.startswith('voltage_harmonic_')]
    
    voltage_inputs = voltage_df[voltage_input_cols].values
    voltage_outputs = voltage_df[voltage_output_cols].values
    voltage_vehicle_ids = voltage_df['vehicle_id'].values
    
    # 划分数据集
    n_samples = voltage_inputs.shape[0]
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_test - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # 创建数据集
    train_dataset = HarmonicDataset(
        voltage_inputs[train_indices].astype(np.float32),
        voltage_outputs[train_indices].astype(np.float32),
        voltage_vehicle_ids[train_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )
    
    val_dataset = HarmonicDataset(
        voltage_inputs[val_indices].astype(np.float32),
        voltage_outputs[val_indices].astype(np.float32),
        voltage_vehicle_ids[val_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )
    
    test_dataset = HarmonicDataset(
        voltage_inputs[test_indices].astype(np.float32),
        voltage_outputs[test_indices].astype(np.float32),
        voltage_vehicle_ids[test_indices],
        input_cycle_fraction or config.INPUT_CYCLE_FRACTION,
        config.TARGET_SAMPLES_PER_CYCLE
    )
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

def init_voltage_dataloaders(config, batch_size=32, shuffle=True, train_ratio=0.7, 
                            val_ratio=0.1, test_ratio=0.2, input_cycle_fraction=0.5):
    """
    初始化电压数据加载器
    """
    datasets = create_real_voltage_datasets(
        config, shuffle=shuffle, train_ratio=train_ratio, 
        val_ratio=val_ratio, test_ratio=test_ratio, 
        input_cycle_fraction=input_cycle_fraction
    )
    
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    test_dataset = datasets['test']
    
    print("Voltage dataset info:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # 归一化
    train_input_scale, train_output_scale = train_dataset.normalize()
    val_input_scale, val_output_scale = val_dataset.normalize()
    test_input_scale, test_output_scale = test_dataset.normalize()

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_input_scale': train_input_scale,
        'train_output_scale': train_output_scale,
        'val_input_scale': val_input_scale,
        'val_output_scale': val_output_scale,
        'test_input_scale': test_input_scale,
        'test_output_scale': test_output_scale
    }