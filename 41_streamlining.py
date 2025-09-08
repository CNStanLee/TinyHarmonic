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

from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants

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
    MoveLinearPastEltwiseMul,
    MoveTransposePastScalarMul,
    MoveTransposePastJoinAdd
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres

# --------------------------------------------------------
# set random seed
np.random.seed(1998)
torch.manual_seed(1998)

# --------------------------------------------------------
model_brevitas_path = "models/ids/ids_brevitas.onnx"
model_qcdq_path = "models/ids/ids_qcdq.onnx"
model_qonnx_quant_threshhold_transform_path = "models/ids/ids_qonnx_quant_threshold_transform.onnx"
model_quant_threshold_finn_path = "models/ids/ids_quant_threshold_finn.onnx"
model_finn_tidy_up_path = "models/ids/ids_finn_tidy_up.onnx"
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
    # QONNX_out = np.array(output_dict_qonnx.get("dql_hidden_out"))
    # print(QONNX_out)

    # --------------------------------------------------------
    # convert to finn
    # --------------------------------------------------------

    model_finn = model_qonnx_quant_threshhold_transform.transform(ConvertQONNXtoFINN())
    model_finn.save(model_quant_threshold_finn_path)
    # !Attention, here the bias of tanh and sigmoid is not given, and the range of threshholding is hard-coded
    # need to fig this out in qonnx_activation_handlers.py

    # --------------------------------------------------------
    # behavior test
    # --------------------------------------------------------
    output_dict_finn = oxe.execute_onnx(model_finn, input_dict,return_full_exec_context=True)#return_full_exec_context=True
    qonnx_output = np.array(output_dict_qonnx.get("dql_hidden_out")) #i_t_dql1
    #print(qonnx_output)
    finn_onnx_output = np.array(output_dict_finn.get("dql_hidden_out")) #i_t_dql1
    #print(finn_onnx_output)
    print(qonnx_output - finn_onnx_output)
    # behavior_test not passed yet, need to check the thresholding function again

    # --------------------------------------------------------
    # transformation
    # --------------------------------------------------------
    # model_finn = model_finn.transform(InferShapes())
    # model_finn = model_finn.transform(FoldConstants())
    # model_finn = model_finn.transform(GiveUniqueNodeNames())
    # model_finn = model_finn.transform(GiveReadableTensorNames())
    # model_finn = model_finn.transform(InferDataTypes())
    # model_finn = model_finn.transform(RemoveStaticGraphInputs())
    # model_finn.save(model_finn_tidy_up_path)

    # --------------------------------------------------------
    # behavior test
    # --------------------------------------------------------
    # input_dict = {}
    # input_dict["global_in_2"] = in_X
    # input_dict["global_in"] = in_h_t_1
    # input_dict["global_in_1"] = in_c_t_1

    # output_dict_finn_tidy = oxe.execute_onnx(model_finn, input_dict,return_full_exec_context=True) 
    # x = np.array(output_dict_finn.get("dql_hidden_out")) #i_t_dql1
    # print(x)
    # x1 = np.array(output_dict_finn_tidy) #i_t_dql1
    # print(x1)
    # --------------------------------------------------------
    # transformation 2
    # --------------------------------------------------------
    # streamline_transformations = [
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
    #         FactorOutMulSignMagnitude(), 
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
    # i = 0
    # for trn in streamline_transformations:
    #     print('Transformation = ',trn)
    #     model_finn = model_finn.transform(trn)
    #     model_finn = model_finn.transform(RemoveIdentityOps())
    #     model_finn = model_finn.transform(GiveUniqueNodeNames())
    #     model_finn = model_finn.transform(GiveReadableTensorNames())
    #     model_finn = model_finn.transform(InferDataTypes())
    #     model_finn.save('streamline_'+str(i)+'.onnx')
    #     i = i+1



if __name__ == "__main__":
    ids_flow()