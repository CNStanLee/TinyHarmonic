import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper
import os
from onnx import helper, shape_inference
# --------------------------------------------------------
from utils.l_onnx_utils import check_onnx_info
from models.lstm_qcdq_description import lstm_graph_def
# --------------------------------------------------------
input_cycle_fraction = 0.5
input_size = int(input_cycle_fraction * 64)
input_size = 64
hidden_size = 128
model_name = f"cnn_lstm_real_c{input_cycle_fraction}"
fmodel_name= f"f{model_name}"
qmodel_name= f"q{model_name}"
# --------------------------------------------------------
onnx_input_path = f"./models/{qmodel_name}/sublstm.onnx"
onnx_output_path = f"./models/{qmodel_name}/sublstm_qcdq.onnx"
log_path = f"./models/{qmodel_name}/sublstm_qcdq.log"
# --------------------------------------------------------


def main():
    weights = check_onnx_info(onnx_input_path, log_path)

    lstm_graph_def(input_size=input_size, hidden_size=hidden_size, weights=weights, save_path=onnx_output_path)
if __name__ == "__main__":
    main()