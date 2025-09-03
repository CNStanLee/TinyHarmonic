
from utils.config import Config
from utils.data_processor import DataProcessor
from utils.voltage_loader import init_voltage_dataloaders
from utils.current_loader import init_current_dataloaders
from utils.simulation_loader import init_simulation_dataloaders
import os
import shutil
def clear_directory(directory_path):

    if not os.path.exists(directory_path):
        print(f"directory {directory_path} not exists")
        return
    
    # 检查是否是目录
    if not os.path.isdir(directory_path):
        print(f"{directory_path} not is a directory")
        return
    
    try:
        # 删除目录及其所有内容
        shutil.rmtree(directory_path)
        print(f"deleted previous preprocessing: {directory_path}")
    except Exception as e:
        print(f"error occurred while clearing directory: {e}")
# ---------------------------------------------------------
def data_init(batch_size=32, input_cycle_fraction=1, delete_temp_files=False):
    config = Config()
    processor = DataProcessor(config)
    
    if delete_temp_files:
        # remove temporary files folder
        clear_directory(config.TEMP_DATA_DIR)

    if not (config.TEMP_DATA_DIR / 'voltage_data.csv').exists():
        processor.process_all_data(include_simulation=True)

    print("Initializing data loaders...")
    print('voltage')
    voltage_loaders = init_voltage_dataloaders(config, batch_size=batch_size, input_cycle_fraction=input_cycle_fraction)
    print('current')
    current_loaders = init_current_dataloaders(config, batch_size=batch_size, input_cycle_fraction=input_cycle_fraction)
    print('simulation')
    simulation_loaders = init_simulation_dataloaders(config, batch_size=batch_size, input_cycle_fraction=input_cycle_fraction)

    sample_input, sample_output, vehicle_id = current_loaders['train'].dataset[0]
    processor.visualize_sample(sample_input, sample_output, vehicle_id, 
                              signal_type='current', name=f'current_real_{input_cycle_fraction}', show_fig=False)
    sample_input, sample_output, vehicle_id = voltage_loaders['train'].dataset[0]
    processor.visualize_sample(sample_input, sample_output, vehicle_id, 
                              signal_type='voltage', name=f'voltage_real_{input_cycle_fraction}', show_fig=False)
    sample_input, sample_output, vehicle_id = simulation_loaders['train'].dataset[0]
    processor.visualize_sample(sample_input, sample_output, vehicle_id, 
                              signal_type='current', name=f'current_simu_{input_cycle_fraction}', show_fig=False)

    for batch_idx, (inputs, targets, vehicle_ids) in enumerate(voltage_loaders['train']):
        print(f"Voltage batch {batch_idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")
        if batch_idx >= 2: 
            break
    
    for batch_idx, (inputs, targets, vehicle_ids) in enumerate(current_loaders['train']):
        print(f"Current batch {batch_idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")
        if batch_idx >= 2:
            break
    
    for batch_idx, (inputs, targets, vehicle_ids) in enumerate(simulation_loaders['train']):
        print(f"Simulation batch {batch_idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")
        if batch_idx >= 2:
            break
    return voltage_loaders, current_loaders, simulation_loaders