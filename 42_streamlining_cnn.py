import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil
import os
# --------------------------------------------------------
import torch
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d, AvgPool2d
from torch.nn import Module
from torch.nn import ModuleList
# --------------------------------------------------------
from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
# --------------------------------------------------------
from brevitas_examples.bnn_pynq.models.common import CommonActQuant
from brevitas_examples.bnn_pynq.models.common import CommonWeightQuant
from brevitas_examples.bnn_pynq.models.tensor_norm import TensorNorm
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import os
import configparser
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
import numpy as np

from qonnx.core.datatype import DataType
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)
# --------------------------------------------------------
pynq_part_map = dict()
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Ultra96-V2"] = "xczu3eg-sbva484-1-i"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"
pynq_part_map["Pynq-Z2"] = "xc7z020clg400-1"
pynq_part_map["ZCU102"] = "xczu9eg-ffvb1156-2-e"
pynq_part_map["ZCU104"] = "xczu7ev-ffvc1156-2-e"
pynq_part_map["ZCU111"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC2x2"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC4x2"] = "xczu48dr-ffvg1517-2-e"
pynq_part_map["KV260_SOM"] = "xck26-sfvc784-2LV-c"
pynq_part_map["U50"] = "xcu50-fsvh2104-2L-e"
# --------------------------------------------------------
from models.qmodels import QCNNLSTM, QCNNLSTM_subCNN, QCNNLSTM_subLSTM, QCNNLSTM_subMLP
# --------------------------------------------------------
input_cycle_fraction = 0.5
model_name = f"cnn_lstm_real_c{input_cycle_fraction}"
fmodel_name= f"f{model_name}"
qmodel_name= f"q{model_name}"
qmodel_pth_path = f"./models/{qmodel_name}/final_model.pth"
input_size = int(input_cycle_fraction * 64)
qlstm_input_size = input_size * 2
hidden_size = 128
cnn_channels= input_size
kernel_size=3
lstm_hidden_size=128 # was 128
lstm_num_layers=1
mlp_hidden_size=512 # was 256 broader is much better
mlp_num_layers=1 # was 3
dropout=0.2
num_heads=4

# class step_3_to_4D(DataflowStep):
#     def __call__(self, model):
#         model = model.transform(Change3DTo4DTensors())
#         return model
    
# def step_3_to_4D(model: ModelWrapper, cfg: DataflowBuildConfig):
#     """Run the tidy-up step on given model. This includes shape and datatype
#     inference, constant folding, and giving nodes and tensors better names.
#     """

#     model = model.transform(Change3DTo4DTensors())

#     return model

my_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_pre_streamline",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
]


subcnn_def = QCNNLSTM_subCNN(
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

submlp_def = QCNNLSTM_subMLP(
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

def estimate_ip(model, random_input, ready_model_filename="model_ready.onnx", estimates_output_dir="./estimates_output"):
    model.cpu()
    export_qonnx(
        model, export_path=ready_model_filename, input_t=random_input
    )

    qonnx_model = ModelWrapper(ready_model_filename)
    #qonnx_model = qonnx_model.transform(Change3DTo4DTensors())
    qonnx_cleanup(ready_model_filename, out_file=ready_model_filename)
    #print("3->4 D transformation done")
    # save the modified model
    qonnx_model.save(ready_model_filename)
    print("Ready Model saved to %s" % ready_model_filename)
    cfg_estimates = build.DataflowBuildConfig(
        output_dir          = estimates_output_dir,
        target_fps          = 10000,
        mvau_wwidth_max     = 1000, 
        synth_clk_period_ns = 10.0,
        fpga_part           = pynq_part_map["ZCU104"],
        #steps               = build_cfg.estimate_only_dataflow_steps,
        steps               = my_steps,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        ]
    )
    build.build_dataflow_cfg(ready_model_filename, cfg_estimates)
    print("Estimation completed. Results saved to %s" % estimates_output_dir)

def main():
    batch_size = 1
    cnn_model_name = model_name+"subcnn"
    ready_model_filename = f"{cnn_model_name}_ready_model.onnx"
    estimates_output_dir = f"./estimates_output/{cnn_model_name}"
    model = subcnn_def
    random_input = torch.randn(batch_size, input_size)
    model.load_state_dict(torch.load(qmodel_pth_path))
    print("Model loaded from %s" % qmodel_pth_path)
    estimate_ip(model, random_input, ready_model_filename=ready_model_filename, estimates_output_dir=estimates_output_dir)
    export_qonnx(model, export_path=ready_model_filename, input_t=random_input.float())
    # qonnx_cleanup(ready_model_filename, out_file=ready_model_filename)
    # model = ModelWrapper(ready_model_filename)
    # model.set_tensor_datatype(model.graph.input[0].name, DataType["INT8"])
    # model = model.transform(ConvertQONNXtoFINN())
    # model.save(ready_model_filename)
    # print("Ready Model saved to %s" % ready_model_filename)
    # mlp_model_name = model_name+"submlp"
    # ready_model_filename = f"{mlp_model_name}_ready_model.onnx"
    # estimates_output_dir = f"./estimates_output/{mlp_model_name}"
    # model = submlp_def
    # input_size = 128
    # batch_size = 1
    # random_input = torch.randn(batch_size, input_size)
    # model.load_state_dict(torch.load(qmodel_pth_path))
    # print("Model loaded from %s" % qmodel_pth_path)
    # estimate_ip(model, random_input, ready_model_filename=ready_model_filename, estimates_output_dir=estimates_output_dir)


if __name__ == "__main__":
    main()