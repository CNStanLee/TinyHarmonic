import numpy as np

class HarmonicDataset:
    def __init__(self, inputs, outputs, vehicle_ids=None, 
                 input_cycle_fraction=0.5, target_samples_per_cycle=64):
        self.full_inputs = inputs.astype(np.float32)
        self.outputs = outputs.astype(np.float32)
        self.vehicle_ids = vehicle_ids
        self.n_samples = inputs.shape[0]
        self.input_cycle_fraction = input_cycle_fraction
        self.target_samples_per_cycle = target_samples_per_cycle
        self.update_input_length()
        
        # 归一化参数（初始化为None，将在normalize方法中计算）
        self.normalized = False
        self.input_scale = None
        self.output_scale = None  # 现在这将是一个数组，每个通道一个缩放因子
        
        # 数据增强相关（初始化时不启用）
        self.augment = False
        self.augment_config = {
            "noise_std": 0.01,       # 高斯噪声标准差
            "time_jitter": 2,        # 时间抖动最大偏移点数
            "scaling_range": (0.9,1.1) # 振幅缩放
        }

    def update_input_length(self, input_cycle_fraction=None):
        """更新输入长度"""
        if input_cycle_fraction is not None:
            self.input_cycle_fraction = input_cycle_fraction
        self.input_length = int(self.target_samples_per_cycle * abs(self.input_cycle_fraction))
    
    def set_input_cycle_fraction(self, input_cycle_fraction):
        """设置输入周期比例"""
        self.update_input_length(input_cycle_fraction)
    
    def normalize(self):
        """将 full_inputs 归一化到 [-1,1]，将 outputs 的四个通道分别归一化到 [-1,1]"""
        # 计算输入数据的最大绝对值
        input_max_abs = np.max(np.abs(self.full_inputs))
        
        # 计算输出数据每个通道的最大绝对值
        if self.outputs.ndim == 1:
            # 如果输出是一维的（单通道）
            output_max_abs = np.array([np.max(np.abs(self.outputs))])
        else:
            # 如果输出是多维的（多通道）
            output_max_abs = np.max(np.abs(self.outputs), axis=0)
        
        # 归一化输入
        self.full_inputs = self.full_inputs / input_max_abs
        
        # 分别归一化输出的每个通道
        if self.outputs.ndim == 1:
            # 单通道情况
            self.outputs = self.outputs / output_max_abs[0]
        else:
            # 多通道情况
            for i in range(self.outputs.shape[1]):
                self.outputs[:, i] = self.outputs[:, i] / output_max_abs[i]
        
        # 保存归一化参数
        self.input_scale = input_max_abs
        self.output_scale = output_max_abs  # 现在这是一个数组
        
        self.normalized = True
        print(f"Inputs normalized: mapped from [-{input_max_abs:.2f}, {input_max_abs:.2f}] to [-1, 1]")
        print(f"Outputs normalized per channel:")
        for i, scale in enumerate(output_max_abs):
            print(f"  Channel {i}: mapped from [-{scale:.2f}, {scale:.2f}] to [-1, 1]")
        
        return self.input_scale, self.output_scale
    
    def denormalize_outputs(self, y_norm):
        """将归一化后的输出反归一化回原始范围（支持多通道）"""
        if not self.normalized:
            raise ValueError("Dataset has not been normalized yet.")
        
        if isinstance(self.output_scale, (int, float)):
            # 单通道情况
            return y_norm * self.output_scale
        else:
            # 多通道情况
            if y_norm.ndim == 1:
                # 单个样本
                return y_norm * self.output_scale
            else:
                # 批量样本
                return y_norm * self.output_scale[np.newaxis, :]
    
    def denormalize_inputs(self, x_norm):
        """将归一化后的输入反归一化回原始范围"""
        if not self.normalized:
            raise ValueError("Dataset has not been normalized yet.")
        return x_norm * self.input_scale

    def enable_augmentation(self, config=None):
        """开启数据增强"""
        self.augment = True
        if config:
            self.augment_config.update(config)
        print("Data augmentation enabled with config:", self.augment_config)

    def disable_augmentation(self):
        """关闭数据增强"""
        self.augment = False
        print("Data augmentation disabled.")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 计算起始索引
        if self.input_cycle_fraction >= 0:
            start_idx = self.target_samples_per_cycle
            start_idx += int(self.target_samples_per_cycle * (1 - self.input_cycle_fraction))
        else:
            start_idx = int(self.target_samples_per_cycle * (1 + self.input_cycle_fraction))
        
        # 限制范围
        start_idx = max(0, min(start_idx, self.full_inputs.shape[1] - self.input_length))
        end_idx = start_idx + self.input_length
        
        input_data = self.full_inputs[idx, start_idx:end_idx].copy()
        target_data = self.outputs[idx]
        
        # ==== 数据增强（仅当启用时） ====
        if self.augment:
            input_data = self.apply_augmentation(input_data)
        
        if self.vehicle_ids is not None:
            return input_data, target_data, self.vehicle_ids[idx]
        return input_data, target_data
    
    def apply_augmentation(self, data):
        """对单个样本应用增强"""
        cfg = self.augment_config

        # 1. 高斯噪声
        if cfg.get("noise_std", 0) > 0:
            noise = np.random.normal(0, cfg["noise_std"], size=data.shape).astype(np.float32)
            data = data + noise

        # 2. 时间抖动（相当于随机平移序列）
        max_jitter = cfg.get("time_jitter", 0)
        if max_jitter > 0:
            shift = np.random.randint(-max_jitter, max_jitter + 1)
            data = np.roll(data, shift)

        # 3. 振幅缩放
        scale_min, scale_max = cfg.get("scaling_range", (1.0, 1.0))
        if scale_min != 1.0 or scale_max != 1.0:
            scale = np.random.uniform(scale_min, scale_max)
            data = data * scale

        return data.astype(np.float32)