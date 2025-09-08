import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
from finn.util.visualization import showInNetron,showSrc
import onnxruntime as rt
from qonnx.util.basic import qonnx_make_model
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
import finn.core.onnx_exec as oxe
import torch
from brevitas.nn import QuantReLU, QuantIdentity
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline import RoundAndClipThresholds
from qonnx.core.datatype import DataType
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
import finn.core.onnx_exec as oxe

# --------------------------------------------------------
# set random seed
np.random.seed(1998)
torch.manual_seed(1998)

# --------------------------------------------------------
model_brevitas_path = "models/ids/ids_brevitas.onnx"
model_qcdq_path = "models/ids/ids_qcdq.onnx"
model_qonnx_quant_threshhold_transform_path = "models/ids/ids_qonnx_quant_threshold_transform.onnx"
model_quant_threshold_finn_path = "models/ids/ids_quant_threshold_finn.onnx"
def ids_flow():
    # --------------------------------------------------------
    # qcdq to qonnx
    # --------------------------------------------------------

    print("Load the ONNX model")
    model_qcdq = ModelWrapper(model_qcdq_path)
    model_qonnx_quant_threshhold_transform = model_qcdq.transform(QCDQToQuant())
    model_qonnx_quant_threshhold_transform = model_qonnx_quant_threshhold_transform.transform(InferShapes())
    model_qonnx_quant_threshhold_transform.save(model_qonnx_quant_threshhold_transform_path)

    # --------------------------------------------------------
    # qonnx behavior test
    # --------------------------------------------------------

    in_X = np.ones([10,1],dtype=np.float32).reshape([10,1])
    in_X[0][0] = 0
    in_X[1][0] = 1
    in_X[2][0] = 2
    in_X[3][0] = 3
    in_X[4][0] = 4
    in_X[5][0] = 5
    in_X[6][0] = 6
    in_X[7][0] = 7
    in_X[8][0] = 8
    in_X[9][0] = 9

    in_h_t_1 = np.ones([20,1],dtype=np.float32).reshape([20,1])
    in_c_t_1 = np.ones([20,1],dtype=np.float32).reshape([20,1])
    in_h_t_1[0][0] = 15
    in_c_t_1[0][0] = 12

    input_dict = {}
    input_dict["X"] = in_X
    input_dict["h_t-1"] = in_h_t_1
    input_dict["c_t-1"] = in_c_t_1

    output_dict_qonnx = oxe.execute_onnx(model_qonnx_quant_threshhold_transform, input_dict,return_full_exec_context=True)
    print(output_dict_qonnx)
    QONNX_out = np.array(output_dict_qonnx.get("dql_hidden_out"))
    print(QONNX_out)
    QONNX_out_ct = np.array(output_dict_qonnx.get("c_t_out"))
    print(QONNX_out_ct)

    # --------------------------------------------------------
    # convert to finn
    # --------------------------------------------------------

    model_finn = model_qonnx_quant_threshhold_transform.transform(ConvertQONNXtoFINN())
    model_finn.save(model_quant_threshold_finn_path)

if __name__ == "__main__":
    ids_flow()