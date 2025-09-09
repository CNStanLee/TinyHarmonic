import onnx
import numpy as np
import os
# --------------------------------------------------------
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
from qonnx.transformation.base import Transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps

from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    AbsorbTransposeIntoMultiThreshold
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.reorder import (
    MoveAddPastConv,
    MoveAddPastMul,
    MoveMulPastMaxPool,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
    MoveLinearPastEltwiseAdd,
    MoveLinearPastEltwiseMul
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
def export_weights(streamlined_onnx_model_path, save_path, submodel_name):
    os.makedirs(save_path, exist_ok=True)
    streamlined_graph = ModelWrapper(streamlined_onnx_model_path)
    parameters = streamlined_graph.graph.initializer
    print(f"Number of parameters: {len(parameters)}")

    header_path = os.path.join(save_path, f"{submodel_name}.h")

    with open(header_path, "w") as hfile:
        hfile.write(f"#ifndef {submodel_name.upper()}_H\n")
        hfile.write(f"#define {submodel_name.upper()}_H\n\n")

        for i, param in enumerate(parameters):
            w = numpy_helper.to_array(param)
            shape = w.shape
            name = param.name.replace(".", "_").replace("-", "_")  # 合法C变量名

            # 确保至少1D，避免0D报错
            w_flat = np.atleast_1d(w.flatten())

            # 保存扁平数据到txt（可选，方便调试）
            #np.savetxt(f"{save_path}/weight_{i}_{name}.txt", w_flat, delimiter=',', fmt="%d")

            # 转换成C数组
            c_array_str = np.array2string(
                w,
                separator=', ',
                threshold=np.inf,
                max_line_width=np.inf
            ).replace('[', '{').replace(']', '}')

            # C数组维度定义
            dims = "][".join(str(d) for d in shape) if shape else "1"
            c_def = f"int {name}[{']['.join(map(str, shape))}] = {c_array_str};"

            # 写入h文件
            hfile.write(f"// Weight {i}, Shape: {shape}\n")
            hfile.write(c_def + "\n\n")

            print(f"Weight {i} with name {param.name}, shape {shape} saved.")

        hfile.write(f"#endif // {submodel_name.upper()}_H\n")

    print(f"Header file saved: {header_path}")


def main():
    submodels = ["sublstm", "subcnn", "submlp"]
    qmodel_name = "qcnn_lstm_real_c0.5"
    for submodel_name in submodels:
        streamlined_onnx_model_path = f"./models/{qmodel_name}/{submodel_name}_finn_streamlined.onnx"
        weight_file_name = f"./models/{qmodel_name}/{submodel_name}_weights"
        print(f"Processing {submodel_name}...")
        export_weights(streamlined_onnx_model_path, save_path=weight_file_name, submodel_name=submodel_name)


if __name__ == "__main__":
    main()