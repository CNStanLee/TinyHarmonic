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
# ---------------------------------------------------------
from utils.ids_dataloader import IDSTrainDataset, IDSValDataset, IDSTestDataset, download_ids_data, get_ids_loaders
from models.qlstmids import QLSTMIDS
from utils.trainer_qlstm_ids import TrainerQLSTMIDS
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
batch_size = 2000
epochs = 2#50
lr = 0.0001
seq_len = 2
input_length = 10
num_outputs = 5


def export_model_to_onnx(model, batch_size, seq_len, input_length, device):
        # 导出模型为ONNX格式
    print("Exporting model to ONNX format...")
    model.eval()  # 设置为评估模式
    
    # 创建一个包装函数来处理batch_size参数
    class ModelWrapper(nn.Module):
        def __init__(self, original_model):
            super(ModelWrapper, self).__init__()
            self.original_model = original_model
            
        def forward(self, x):
            # 从输入张量的形状中获取batch_size
            batch_size = x.size(0)
            return self.original_model(x, batch_size)
    
    # 创建包装后的模型
    wrapped_model = ModelWrapper(model)
    
    # 创建一个示例输入
    dummy_input = torch.randn(batch_size, seq_len, input_length).to(device)
    
    # 定义ONNX文件路径
    onnx_path = os.path.join('checkpoints/ids_qlstm', 'model.onnx')
    
    # 确保目录存在
    os.makedirs('checkpoints/ids_qlstm', exist_ok=True)
    
    # 导出模型
    torch.onnx.export(
        wrapped_model,          # 使用包装后的模型
        dummy_input,            # 模型输入（示例）
        onnx_path,             # 输出文件路径
        export_params=True,     # 导出训练好的参数
        opset_version=11,       # ONNX操作集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],   # 输入名称
        output_names=['output'], # 输出名称
        dynamic_axes={          # 动态轴配置（处理可变长度输入）
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully exported to {onnx_path}")

def main():
    download_ids_data()
    train_loader, val_loader, test_loader = get_ids_loaders(batch_size)

    model = QLSTMIDS().to(device)
    print(model)
    print("No. of parameters in the model = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    export_model_to_onnx(model, batch_size, seq_len, input_length, device)

    model = model.float()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = TrainerQLSTMIDS(model=model,
                            trainloader=train_loader,
                            validationloader=val_loader,
                            train_batch_size=batch_size,
                            val_batch_size=batch_size,
                            num_epochs=epochs,
                            criterion=criterion,
                            optimizer=optimizer,
                            model_folder='checkpoints/ids_qlstm',
                            device=device)
    trainer.train()
    trainer.test(test_loader)

if __name__ == "__main__":
    main()