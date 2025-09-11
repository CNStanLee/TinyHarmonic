import numpy as np
import torch
import os
# --------------------------------------------------------
import onnxruntime as rt
import onnx
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper
# --------------------------------------------------------
from brevitas.nn import QuantReLU, QuantIdentity
from brevitas.export import export_qonnx
# --------------------------------------------------------
from finn.util.visualization import showInNetron,showSrc
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline import RoundAndClipThresholds
import finn.core.onnx_exec as oxe
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
    MoveLinearPastEltwiseMul,
    MoveTransposePastScalarMul,
    MoveTransposePastJoinAdd,
    MoveMulPastDWConv
)
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

# --------------------------------------------------------
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.base import Transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.core.datatype import DataType
from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.util.basic import qonnx_make_model
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from qonnx.util.cleanup import cleanup as qonnx_cleanup

from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors
# --------------------------------------------------------
from models.qmodels import QCNNLSTM, QCNNLSTM_subCNN, QCNNLSTM_subLSTM, QCNNLSTM_subMLP
# --------------------------------------------------------
# set random seed
np.random.seed(1998)
torch.manual_seed(1998)
# --------------------------------------------------------
input_cycle_fraction = 0.5
model_name = f"cnn_lstm_real_c{input_cycle_fraction}"
fmodel_name= f"f{model_name}"
qmodel_name= f"q{model_name}"
input_size = int(input_cycle_fraction * 64)

hidden_size = 128
cnn_channels= input_size
kernel_size=3
lstm_hidden_size=128 # was 128
lstm_num_layers=1
mlp_hidden_size=512 # was 256 broader is much better
mlp_num_layers=1 # was 3
dropout=0.1
num_heads=4
batch_size = 1
# --------------------------------------------------------
submodel_name = "subcnn"
cnn_input_size = input_size
qlstm_input_size = input_size * 2
submodel_input_size = cnn_input_size

qmodel_pth_path = f"./models/{qmodel_name}/final_model.pth"
model_brevitas_path = f"./models/{qmodel_name}/{submodel_name}.onnx"
model_qcdq_path = model_brevitas_path
model_qonnx_path = f"./models/{qmodel_name}/{submodel_name}_qonnx.onnx"
model_finn_path = f"./models/{qmodel_name}/{submodel_name}_finn.onnx"
model_finn_tidy_up_path = f"./models/{qmodel_name}/{submodel_name}_finn_tidy_up.onnx"
model_finn_streamlined_path = f"./models/{qmodel_name}/{submodel_name}_finn_streamlined.onnx"
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
# --------------------------------------------------------
x = np.random.randn(submodel_input_size,1).astype(np.float32).reshape([submodel_input_size,1])
print(x.shape)
x = np.expand_dims(x, axis=1) # shape to (batch, channel, feature)
print(x.shape)
h0 = np.zeros((hidden_size, batch_size), dtype=np.float32)
c0 = np.zeros((hidden_size, batch_size), dtype=np.float32)

input_dict = {}
input_dict["X"] = x
input_dict["h_t-1"] = h0
input_dict["c_t-1"] = c0
input_dict["global_in"] = x.transpose(1,2,0)

def convert_brevitas_to_qonnx(model, weight_path, random_input, model_qonnx_path):
    print("step 1: brevitas to qonnx")
    model.load_state_dict(torch.load(weight_path))
    export_qonnx(model, export_path=model_qonnx_path, input_t=random_input.float())
    qonnx_cleanup(model_qonnx_path, out_file=model_qonnx_path)
    model = ModelWrapper(model_qonnx_path)
    return model

def brevitas_behavior_test(qmodel, qmodel_pth_path):
    qmodel.load_state_dict(torch.load(qmodel_pth_path))
    qmodel.eval()
    with torch.no_grad():
        print(x.shape)
        x_torch = torch.from_numpy(x).permute(1,2,0)
        print(x_torch.shape)
        #print(f"brevitas_input: {x_torch}")
        #print(x_torch.shape)
        with torch.no_grad():
            out_torch = qmodel.forward(x_torch).numpy()
            print("Brevitas output shape:", out_torch.shape)
    return out_torch

def qonnx_behavior_test(model_qonnx):
    #print(f"input_dict: {input_dict}")
    output_dict_qonnx = oxe.execute_onnx(model_qonnx, input_dict,return_full_exec_context=True)
    QONNX_out = np.array(output_dict_qonnx.get("global_out")) #global_out
    print("QONNX output shape:", QONNX_out.shape)
    return QONNX_out

def convert_qonnx_to_finn(model_qonnx_tmp1, model_finn_path):
    model_finn = model_qonnx_tmp1.transform(ConvertQONNXtoFINN())
    model_finn.save(model_finn_path)
    print("finn model saved to ", model_finn_path)
    return model_finn

def finn_behavior_test(model_finn):
    output_dict_finn = oxe.execute_onnx(model_finn, input_dict,return_full_exec_context=True)#return_full_exec_context=True
    finn_onnx_output = np.array(output_dict_finn.get("global_out")) #i_t_dql1
    print("FINN output shape:", finn_onnx_output.shape)
    return finn_onnx_output

def finn_tidyup(model_finn, model_finn_tidy_up_path):
    tidy_finn = model_finn.transform(InferShapes())
    tidy_finn = tidy_finn.transform(FoldConstants())
    tidy_finn = tidy_finn.transform(GiveUniqueNodeNames())
    tidy_finn = tidy_finn.transform(GiveReadableTensorNames())
    tidy_finn = tidy_finn.transform(InferDataTypes())
    tidy_finn = tidy_finn.transform(RemoveStaticGraphInputs())
    tidy_finn.save(model_finn_tidy_up_path)
    return tidy_finn

def streamline_model_behavior_test(streamlined_model):
    input_dict["global_in"] = np.expand_dims(input_dict["global_in"], axis=-1)   # add batch dim
    output_dict_streamlined = oxe.execute_onnx(streamlined_model, input_dict,return_full_exec_context=True)
    streamlined_output = np.array(output_dict_streamlined.get("global_out")) 
    print("Streamlined output shape:", streamlined_output.shape)
    return streamlined_output

def finn_streamlining(model_finn, model_finn_streamlined_path):
    # streamline_transformations = [
    #         #add conv
    #         Change3DTo4DTensors(),
    #         #absorb.AbsorbScalarMulAddIntoTopK(),
    #         LowerConvsToMatMul(),
    #         MakeMaxPoolNHWC(),
    #         AbsorbTransposeIntoMultiThreshold(),
    #         MakeMaxPoolNHWC(),
    #         absorb.AbsorbConsecutiveTransposes(),
    #         #end add conv

    #         ConvertSubToAdd(),
    #         ConvertDivToMul(),     
    #         BatchNormToAffine(), 
    #         ConvertSignToThres(),  
    #         MoveMulPastMaxPool(),
    #         MoveScalarLinearPastInvariants(),  
    #         AbsorbSignBiasIntoMultiThreshold(),

    #         MoveAddPastMul(),     
    #         MoveScalarAddPastMatMul(), 
    #         MoveAddPastConv(),       
    #         MoveScalarMulPastConv(), 
    #         MoveAddPastMul(), 
    #         CollapseRepeatedAdd(),
    #         CollapseRepeatedMul(),   
    #         MoveMulPastMaxPool(),  
    #         AbsorbAddIntoMultiThreshold(), 
    #         FactorOutMulSignMagnitude(), # from shashwat
    #         AbsorbMulIntoMultiThreshold(), #This transformation absorbs the Scalar Mul nodes into the next Multithreshold nodes.
    #         MoveLinearPastEltwiseAdd(), #This transformation helps us get all the scalar mul nodes past the elstwiseadd. 
    #         MoveLinearPastEltwiseMul(),#This transformation helps us get scalar mul's past eltwisemuls. We can then absorb them into the multithrehsold opertion and remove them from the graph entirely.
    #         AbsorbMulIntoMultiThreshold(), #The scalar mul nodes passed in the previous step are now merged into the multithreshold node.
    #         RoundAndClipThresholds(),
    #         MoveScalarMulPastMatMul(), #To move activation scales im the dense part beyond dense layers.
    #         AbsorbMulIntoMultiThreshold(),
    #         MoveLinearPastEltwiseAdd(),
    #         AbsorbMulIntoMultiThreshold(), #For the last Multithreshold node in the graph
    #         RoundAndClipThresholds(),
    #         CollapseRepeatedMul(),
    #     ]
    path = f"./models/{qmodel_name}/{submodel_name}_debug/"
    # create path if not exist
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # clean the path
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    streamline_transformations = [
            # 1D CONV
            Change3DTo4DTensors(),
            # NEW
            AbsorbAddIntoMultiThreshold(),
            AbsorbMulIntoMultiThreshold(),
            MoveScalarMulPastConv(),
            AbsorbMulIntoMultiThreshold(),
            MoveAddPastConv(),
            AbsorbAddIntoMultiThreshold(),
            BatchNormToAffine(),
            CollapseRepeatedMul(), # 8
            MoveAddPastConv(),
            AbsorbAddIntoMultiThreshold(),
            # MoveScalarMulPastConv(),
            MoveMulPastDWConv(),
            AbsorbMulIntoMultiThreshold(),
            AbsorbAddIntoMultiThreshold(),
            AbsorbMulIntoMultiThreshold(),
            AbsorbAddIntoMultiThreshold(),
            # GENERAL
            # absorb.AbsorbSignBiasIntoMultiThreshold(),
            # MoveScalarLinearPastInvariants(),
            # Streamline(),
            # CONV
            # LowerConvsToMatMul(),
            # MakeMaxPoolNHWC(),
            # absorb.AbsorbTransposeIntoMultiThreshold(),
            # MakeMaxPoolNHWC(),
            # absorb.AbsorbConsecutiveTransposes(),
            # absorb.AbsorbTransposeIntoMultiThreshold(),
            # GENERAL
            # Streamline(),
            # InferDataLayouts(),
            # RemoveUnusedTensors(),
    ]
    i = 0
    for trn in streamline_transformations:
        print('Transformation = ',trn)
        model_finn = model_finn.transform(trn)
        model_finn = model_finn.transform(RemoveIdentityOps())
        model_finn = model_finn.transform(GiveUniqueNodeNames())
        model_finn = model_finn.transform(GiveReadableTensorNames())
        model_finn = model_finn.transform(InferDataTypes())
        # if path does not exist, create it
        if not os.path.exists(f"./models/{qmodel_name}/{submodel_name}_debug/{submodel_name}_finn_streamlined{i}.onnx"):
            os.makedirs(f"./models/{qmodel_name}/{submodel_name}_debug", exist_ok=True)
        model_finn.save(f"./models/{qmodel_name}/{submodel_name}_debug/{submodel_name}_finn_streamlined{i}.onnx")
        i = i+1
    model_finn.save(model_finn_streamlined_path)
    return model_finn

if __name__ == "__main__":
    # brevitas model
    sub_model = subcnn_def
    brevitas_behaviour = brevitas_behavior_test(sub_model, qmodel_pth_path)
    # qcdq -> qonnx
    random_input = torch.randn(batch_size, 1, input_size)
    model_qonnx = convert_brevitas_to_qonnx(sub_model, qmodel_pth_path, random_input, model_qonnx_path)
    
    # behavior test: brevitas vs qonnx
    qonnx_behaviour = qonnx_behavior_test(model_qonnx)
    mse_brevitas_qonnx = np.mean((brevitas_behaviour - qonnx_behaviour) ** 2)
    print(f'MSE(Brevitas model and QONNX model): {mse_brevitas_qonnx}')
    # qonnx -> finn
    model_finn = convert_qonnx_to_finn(model_qonnx, model_finn_path)
    finn_onnx_behaviour = finn_behavior_test(model_finn)
    mse_qonnx_finn = np.mean((qonnx_behaviour - finn_onnx_behaviour) ** 2)
    print(f'MSE(QONNX model and FINN model): {mse_qonnx_finn}')
    tidy_finn = finn_tidyup(model_finn, model_finn_tidy_up_path)
    finn_streamlining = finn_streamlining(tidy_finn, model_finn_streamlined_path)
    streamlined_behaviour = streamline_model_behavior_test(finn_streamlining)
    mse_finn_streamlined = np.mean((finn_onnx_behaviour - np.squeeze(streamlined_behaviour, axis=-1)) ** 2)
    print(f'MSE(FINN model and streamlined FINN model): {mse_finn_streamlined}')


