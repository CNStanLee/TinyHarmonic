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
    FactorOutMulSignMagnitude_shashwat,
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
    MoveTransposePastJoinAdd
)
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
qlstm_input_size = input_size * 2
hidden_size = 128
# --------------------------------------------------------
qmodel_pth_path = f"./models/{qmodel_name}/final_model.pth"
model_brevitas_path = f"./models/{qmodel_name}/sublstm.onnx"
model_qcdq_path = f"./models/{qmodel_name}/sublstm_qcdq.onnx"
model_qonnx_path = f"./models/{qmodel_name}/sublstm_qonnx.onnx"
model_finn_path = f"./models/{qmodel_name}/sublstm_finn.onnx"
model_finn_tidy_up_path = f"./models/{qmodel_name}/sublstm_finn_tidy_up.onnx"
model_finn_streamlined_path = f"./models/{qmodel_name}/sublstm_finn_streamlined.onnx"
# --------------------------------------------------------
input_cycle_fraction = 0.5
input_size = int(input_cycle_fraction * 64)
cnn_channels= input_size
kernel_size=3
lstm_hidden_size=128 # was 128
lstm_num_layers=1
mlp_hidden_size=512 # was 256 broader is much better
mlp_num_layers=1 # was 3
dropout=0.1
num_heads=4
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
submodel_name = "sublstm"
# --------------------------------------------------------
x = np.random.randn(qlstm_input_size,1).astype(np.float32).reshape([qlstm_input_size,1])
batch_size = 1
h0 = np.zeros((hidden_size, batch_size), dtype=np.float32)
c0 = np.zeros((hidden_size, batch_size), dtype=np.float32)

input_dict = {}
input_dict["X"] = x
input_dict["h_t-1"] = h0
input_dict["c_t-1"] = c0

def convert_qcdq_to_qonnx(model_qcdq_path, model_qonnx_path):
    print("step 1: qcdq to qonnx_tmp1")
    model_qcdq = ModelWrapper(model_qcdq_path)
    model_qonnx = model_qcdq.transform(QCDQToQuant())
    model_qonnx = model_qonnx.transform(InferShapes())
    model_qonnx.save(model_qonnx_path)
    print("qonnx model saved to ", model_qonnx_path)
    return model_qonnx

def brevitas_behavior_test(qmodel, qmodel_pth_path):
    qmodel.load_state_dict(torch.load(qmodel_pth_path))
    qmodel.eval()
    with torch.no_grad():
        print(x.shape)
        x_torch = torch.from_numpy(x).permute(1,0).unsqueeze(0)
        print(x_torch.shape)
        with torch.no_grad():
            out_torch = qmodel.forward(x_torch).numpy()
            print("Brevitas output shape:", out_torch.shape)
    return out_torch

def qonnx_behavior_test(model_qonnx_tmp1):
    output_dict_qonnx = oxe.execute_onnx(model_qonnx_tmp1, input_dict,return_full_exec_context=True)
    QONNX_out = np.array(output_dict_qonnx.get("dql_hidden_out"))
    QONNX_out = QONNX_out.transpose(1,0)
    print("QONNX output shape:", QONNX_out.shape)
    return QONNX_out

def convert_qonnx_to_finn(model_qonnx_tmp1, model_finn_path):
    model_finn = model_qonnx_tmp1.transform(ConvertQONNXtoFINN())
    model_finn.save(model_finn_path)
    print("finn model saved to ", model_finn_path)
    return model_finn

def finn_behavior_test(model_finn):
    output_dict_finn = oxe.execute_onnx(model_finn, input_dict,return_full_exec_context=True)#return_full_exec_context=True
    finn_onnx_output = np.array(output_dict_finn.get("dql_hidden_out")) #i_t_dql1
    finn_onnx_output = finn_onnx_output.transpose(1,0)
    print("FINN output shape:", finn_onnx_output.shape)
    return finn_onnx_output

def finn_streamline_model_behavior_test(streamlined_model):
    output_dict_streamlined = oxe.execute_onnx(streamlined_model, input_dict,return_full_exec_context=True)
    streamlined_output = np.array(output_dict_streamlined.get("dql_hidden_out")) 
    streamlined_output = streamlined_output.transpose(1,0)
    print("Streamlined output shape:", streamlined_output.shape)
    return streamlined_output

def finn_tidyup(model_finn, model_finn_tidy_up_path):
    tidy_finn = model_finn.transform(InferShapes())
    tidy_finn = tidy_finn.transform(FoldConstants())
    tidy_finn = tidy_finn.transform(GiveUniqueNodeNames())
    tidy_finn = tidy_finn.transform(GiveReadableTensorNames())
    tidy_finn = tidy_finn.transform(InferDataTypes())
    tidy_finn = tidy_finn.transform(RemoveStaticGraphInputs())
    tidy_finn.save(model_finn_tidy_up_path)
    return tidy_finn

def finn_streamlining(model_finn, model_finn_streamlined_path):
    streamline_transformations = [
            ConvertSubToAdd(),
            ConvertDivToMul(),     
            BatchNormToAffine(), 
            ConvertSignToThres(),  
            MoveMulPastMaxPool(),
            MoveScalarLinearPastInvariants(),  
            AbsorbSignBiasIntoMultiThreshold(),
            MoveAddPastMul(),     
            MoveScalarAddPastMatMul(), 
            MoveAddPastConv(),       
            MoveScalarMulPastConv(), 
            MoveAddPastMul(), 
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),   
            MoveMulPastMaxPool(),  
            AbsorbAddIntoMultiThreshold(), 
            FactorOutMulSignMagnitude_shashwat(), # from shashwat
            AbsorbMulIntoMultiThreshold(), #This transformation absorbs the Scalar Mul nodes into the next Multithreshold nodes.
            MoveLinearPastEltwiseAdd(), #This transformation helps us get all the scalar mul nodes past the elstwiseadd. 
            MoveLinearPastEltwiseMul(),#This transformation helps us get scalar mul's past eltwisemuls. We can then absorb them into the multithrehsold opertion and remove them from the graph entirely.
            AbsorbMulIntoMultiThreshold(), #The scalar mul nodes passed in the previous step are now merged into the multithreshold node.
            RoundAndClipThresholds(),
            MoveScalarMulPastMatMul(), #To move activation scales im the dense part beyond dense layers.
            AbsorbMulIntoMultiThreshold(),
            MoveLinearPastEltwiseAdd(),
            AbsorbMulIntoMultiThreshold(), #For the last Multithreshold node in the graph
            RoundAndClipThresholds(),
            CollapseRepeatedMul(),
        ]
    i = 0
    for trn in streamline_transformations:
        print('Transformation = ',trn)
        model_finn = model_finn.transform(trn)
        model_finn = model_finn.transform(RemoveIdentityOps())
        model_finn = model_finn.transform(GiveUniqueNodeNames())
        model_finn = model_finn.transform(GiveReadableTensorNames())
        model_finn = model_finn.transform(InferDataTypes())
        if not os.path.exists(f"./models/{qmodel_name}/{submodel_name}_debug/{submodel_name}_finn_streamlined{i}.onnx"):
            os.makedirs(f"./models/{qmodel_name}/{submodel_name}_debug", exist_ok=True)
        model_finn.save(f"./models/{qmodel_name}/{submodel_name}_debug/{submodel_name}_finn_streamlined{i}.onnx")
        i = i+1
    model_finn.save(model_finn_streamlined_path)
    return model_finn

if __name__ == "__main__":
    # brevitas model
    brevitas_behaviour = brevitas_behavior_test(sublstm_def, qmodel_pth_path)
    # qcdq -> qonnx
    model_qonnx = convert_qcdq_to_qonnx(model_qcdq_path, model_qonnx_path)
    # behavior test: qcdq vs qonnx
    qonnx_behaviour = qonnx_behavior_test(model_qonnx)
    mse_brevitas_qonnx = np.mean((brevitas_behaviour - qonnx_behaviour) ** 2)
    print(f'MSE(Brevitas model and QONNX model): {mse_brevitas_qonnx}')
    # qonnx -> finn
    model_finn = convert_qonnx_to_finn(model_qonnx, model_finn_path)
    finn_onnx_behaviour = finn_behavior_test(model_finn)
    mse_qonnx_finn = np.mean((qonnx_behaviour - finn_onnx_behaviour) ** 2)
    print(f'MSE(QONNX model and FINN model): {mse_qonnx_finn}')
    print("qonnx behaviour:")
    print(qonnx_behaviour)
    print("finn behaviour:")
    print(finn_onnx_behaviour)
    tidy_finn = finn_tidyup(model_finn, model_finn_tidy_up_path)
    finn_streamlining = finn_streamlining(tidy_finn, model_finn_streamlined_path)
    # streamlined_behaviour = finn_streamline_model_behavior_test(finn_streamlining)
    # mse_finn_streamlined = np.mean((finn_onnx_behaviour - streamlined_behaviour) ** 2)
    # print(f'MSE(FINN model and streamlined FINN model): {mse_finn_streamlined}')


    # # --------------------------------------------------------
    # # transformation
    # # --------------------------------------------------------
    # # model_finn = model_finn.transform(InferShapes())
    # # model_finn = model_finn.transform(FoldConstants())
    # # model_finn = model_finn.transform(GiveUniqueNodeNames())
    # # model_finn = model_finn.transform(GiveReadableTensorNames())
    # # model_finn = model_finn.transform(InferDataTypes())
    # # model_finn = model_finn.transform(RemoveStaticGraphInputs())
    # # model_finn.save(model_finn_tidy_up_path)

    # # --------------------------------------------------------
    # # behavior test
    # # --------------------------------------------------------
    # # input_dict = {}
    # # input_dict["global_in_2"] = in_X
    # # input_dict["global_in"] = in_h_t_1
    # # input_dict["global_in_1"] = in_c_t_1

    # # output_dict_finn_tidy = oxe.execute_onnx(model_finn, input_dict,return_full_exec_context=True) 
    # # x = np.array(output_dict_finn.get("dql_hidden_out")) #i_t_dql1
    # # print(x)
    # # x1 = np.array(output_dict_finn_tidy) #i_t_dql1
    # # print(x1)
    # # --------------------------------------------------------
    # # transformation 2
    # # --------------------------------------------------------
    # # streamline_transformations = [
    # #         ConvertSubToAdd(),
    # #         ConvertDivToMul(),     
    # #         BatchNormToAffine(), 
    # #         ConvertSignToThres(),  
    # #         MoveMulPastMaxPool(),
    # #         MoveScalarLinearPastInvariants(),  
    # #         AbsorbSignBiasIntoMultiThreshold(),
    # #         MoveAddPastMul(),     
    # #         MoveScalarAddPastMatMul(), 
    # #         MoveAddPastConv(),       
    # #         MoveScalarMulPastConv(), 
    # #         MoveAddPastMul(), 
    # #         CollapseRepeatedAdd(),
    # #         CollapseRepeatedMul(),   
    # #         MoveMulPastMaxPool(),  
    # #         AbsorbAddIntoMultiThreshold(), 
    # #         FactorOutMulSignMagnitude(), 
    # #         AbsorbMulIntoMultiThreshold(), #This transformation absorbs the Scalar Mul nodes into the next Multithreshold nodes.
    # #         MoveLinearPastEltwiseAdd(), #This transformation helps us get all the scalar mul nodes past the elstwiseadd. 
    # #         MoveLinearPastEltwiseMul(),#This transformation helps us get scalar mul's past eltwisemuls. We can then absorb them into the multithrehsold opertion and remove them from the graph entirely.
    # #         AbsorbMulIntoMultiThreshold(), #The scalar mul nodes passed in the previous step are now merged into the multithreshold node.
    # #         RoundAndClipThresholds(),
    # #         MoveScalarMulPastMatMul(), #To move activation scales im the dense part beyond dense layers.
    # #         AbsorbMulIntoMultiThreshold(),
    # #         MoveLinearPastEltwiseAdd(),
    # #         AbsorbMulIntoMultiThreshold(), #For the last Multithreshold node in the graph
    # #         RoundAndClipThresholds(),
    # #         CollapseRepeatedMul(),
    # #     ]
    # # i = 0
    # # for trn in streamline_transformations:
    # #     print('Transformation = ',trn)
    # #     model_finn = model_finn.transform(trn)
    # #     model_finn = model_finn.transform(RemoveIdentityOps())
    # #     model_finn = model_finn.transform(GiveUniqueNodeNames())
    # #     model_finn = model_finn.transform(GiveReadableTensorNames())
    # #     model_finn = model_finn.transform(InferDataTypes())
    # #     model_finn.save('streamline_'+str(i)+'.onnx')
    # #     i = i+1