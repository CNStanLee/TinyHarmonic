import onnx
import numpy as np
from mqonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper
import os
from onnx import helper, shape_inference
import torch
# --------------------------------------------------------
from utils.l_onnx_utils import check_onnx_info
from models.lstm_qcdq_description import lstm_graph_def
from models.qmodels import QCNNLSTM, QCNNLSTM_subCNN, QCNNLSTM_subLSTM, QCNNLSTM_subMLP
# --------------------------------------------------------
# set random seed
np.random.seed(1998)
torch.manual_seed(1998)

# --------------------------------------------------------
input_cycle_fraction = 0.5
input_size = int(input_cycle_fraction * 64)
cnn_channels= input_size
kernel_size=3
lstm_hidden_size=128 # was 128
lstm_num_layers=1
mlp_hidden_size=512 # was 256 broader is much better
mlp_num_layers=1 # was 3
dropout=0.2
num_heads=4
# lstm input size after the cnn layer
qlstm_input_size = input_size * 2
hidden_size = 128
model_name = f"cnn_lstm_real_c{input_cycle_fraction}"
fmodel_name= f"f{model_name}"
qmodel_name= f"q{model_name}"
# --------------------------------------------------------
onnx_input_path = f"./models/{qmodel_name}/sublstm.onnx"
onnx_output_path = f"./models/{qmodel_name}/sublstm_qcdq.onnx"
log_path = f"./models/{qmodel_name}/sublstm_qcdq.log"
qmodel_pth_path = f"./models/{qmodel_name}/final_model.pth"
# --------------------------------------------------------

sublstm_def = QCNNLSTM_subLSTM(
                 input_size=input_size, 
                 cnn_channels=cnn_channels, 
                 kernel_size=kernel_size, 
                 lstm_hidden_size=lstm_hidden_size, 
                 lstm_num_layers=lstm_num_layers, 
                 mlp_hidden_size=mlp_hidden_size, 
                 mlp_num_layers=mlp_num_layers, 
                 dropout=dropout, 
                 num_heads=num_heads
                    )

def behavior_test():

    # Load the modified ONNX model
    onnx_model_qcdq = onnx.load(onnx_output_path)
    onnx_model_qcdq.opset_import[0].version = 14
    sess_qcdq = rt.InferenceSession(onnx_model_qcdq.SerializeToString())
    # init the input
    batch_size = 1
    seq_len = 10
    x = np.random.randn(qlstm_input_size, batch_size).astype(np.float32)
    h0 = np.zeros((hidden_size, batch_size), dtype=np.float32)
    c0 = np.zeros((hidden_size, batch_size), dtype=np.float32)
    # Get the input name for the ONNX model
    output = sess_qcdq.run(None, {"X": x, "h_t-1": h0, "c_t-1": c0})
    output = output[0].transpose(1,0)

    qmodel = sublstm_def
    qmodel.load_state_dict(torch.load(qmodel_pth_path))
    qmodel.eval()
    with torch.no_grad():
        print(x.shape)
        # pack x from (64, 1) to (1, 1, 64)
        x_torch = torch.from_numpy(x).permute(1,0).unsqueeze(0)
        print(x_torch.shape)
        with torch.no_grad():
            out_torch = qmodel.forward(x_torch).numpy()

    print('------------------------------')
    print('qcdq onnx output:')
    print(output.shape)
    print(output)
    print('------------------------------')

    print('------------------------------')
    print('qmodel output:')
    print(out_torch.shape)
    print(out_torch)
    print('------------------------------')

    mse = np.mean((output - out_torch) ** 2)
    print(f'Mean Squared Error between ONNX QCDQ model and PyTorch model: {mse}')
def main():
    weights = check_onnx_info(onnx_input_path, log_path)
    lstm_graph_def(input_size=qlstm_input_size, hidden_size=hidden_size, weights=weights, save_path=onnx_output_path)
    behavior_test()
if __name__ == "__main__":
    main()