import onnx
import numpy as np
from mqonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper

##Parameters for ONNX definition
hidden_size = 20
input_size = 10

# Defining the inputs and outputs of the graph we need to create for the graph of the scan body.
# ---------------------------------------------------
# Defining the inputs value info tensors for the compute to be executed for each input.
inp2_m2 = make_tensor_value_info("h_t-1",onnx.TensorProto.FLOAT, [hidden_size,1])
inp2_elm1 = make_tensor_value_info("c_t-1", onnx.TensorProto.FLOAT, [hidden_size,1])
inp2_m1 = make_tensor_value_info("X",onnx.TensorProto.FLOAT, [input_size,1])

# out_hidden_state = make_tensor_value_info("h_t", onnx.TensorProto.FLOAT, [hidden_size,1])
out_cell_state = make_tensor_value_info("c_t_out", onnx.TensorProto.FLOAT, [hidden_size,1])
out_hidden_state_float = make_tensor_value_info("dql_hidden_out", onnx.TensorProto.FLOAT, [hidden_size,1])  
out_sigmoid = make_tensor_value_info("out_sigmoid", onnx.TensorProto.FLOAT, [1,1])
#out_sigmoid = make_tensor_value_info("out_sigmoid", onnx.TensorProto.FLOAT, [1,1])
out_matmul_fc3 = make_tensor_value_info("out_matmul_fc3",onnx.TensorProto.FLOAT, [5,1])


#Also quantizing the hidden_state and the cell_state for quantizing the input graph more efficiently 
ql_input = make_node("QuantizeLinear", inputs=["X","scale_11","zero_point_all"], outputs=["ql_input_out"],name="ql_input")
# clp_input_8b = make_node("Clip", inputs=["ql_input_out","min_8b_unsigned","max_8b_unsigned"], outputs=["ql_input_out_clip"], name="clp_input_8b")
dql_input = make_node("DequantizeLinear", inputs=["ql_input_out", 'scale_11', "zero_point_all"], outputs=["dql_input_out"],name="dql_input")

input_ql_hidden_state = make_node("QuantizeLinear", inputs=["h_t-1","scale_11","zero_point_all"], outputs=["input_ql_hidden_out"],name="input_ql_hidden_state")
# clp_hidden_8b = make_node("Clip", inputs=["input_ql_hidden_out","min_8b_unsigned","max_8b_unsigned"], outputs=["input_ql_hidden_out_clip"], name="clp_hidden_8b")
input_dql_hidden_state = make_node("DequantizeLinear", inputs=["input_ql_hidden_out", 'scale_11', "zero_point_all"], outputs=["input_dql_hidden_out"],name="input_dql_hidden_state")

ql_cell_state = make_node("QuantizeLinear", inputs=["c_t-1","scale_9","zero_point_all"], outputs=["ql_cell_out"],name="ql_cell_state")
clp_cell_6b = make_node("Clip", inputs=["ql_cell_out","min_6b_2","max_6b"], outputs=["clp_cell_6b"], name="clp_cell_6b")
dql_cell_state = make_node("DequantizeLinear", inputs=["clp_cell_6b", 'scale_9', "zero_point_all"], outputs=["dql_cell_out"],name="dql_cell_state")

#Pushing the weights quantisation in the scan node.
ql_w1 = make_node("QuantizeLinear", inputs=["W_f","scale_f","zero_point_all"], outputs=["ql_wf_out"], name="ql_w1")
clp_w1 = make_node("Clip", inputs=["ql_wf_out","min","max"], outputs=["clp_wf"], name="clp_w1")
dql_w1 = make_node("DequantizeLinear", inputs=["clp_wf","scale_f","zero_point_all"], outputs=["dql_wf_out"], name="dql_w1")

ql_w2 = make_node("QuantizeLinear", inputs=["W_i","scale_i","zero_point_all"], outputs=["ql_wi_out"], name="ql_w2")
clp_w2 = make_node("Clip", inputs=["ql_wi_out","min","max"], outputs=["clp_wi"], name="clp_w2")
dql_w2 = make_node("DequantizeLinear", inputs=["clp_wi","scale_i","zero_point_all"], outputs=["dql_wi_out"], name="dql_w2")

ql_w3 = make_node("QuantizeLinear", inputs=["W_c","scale_c","zero_point_all"], outputs=["ql_wc_out"], name="ql_w3")
clp_w3 = make_node("Clip", inputs=["ql_wc_out","min","max"], outputs=["clp_wc"], name="clp_w3")
dql_w3 = make_node("DequantizeLinear", inputs=["clp_wc","scale_c","zero_point_all"], outputs=["dql_wc_out"], name="dql_w3")

ql_w4 = make_node("QuantizeLinear", inputs=["W_o","scale_o","zero_point_all"], outputs=["ql_wo_out"], name="ql_w4")
clp_w4 = make_node("Clip", inputs=["ql_wo_out","min","max"], outputs=["clp_wo"], name="clp_w4")
dql_w4 = make_node("DequantizeLinear", inputs=["clp_wo","scale_o","zero_point_all"], outputs=["dql_wo_out"], name="dql_w4")

#These are the quantizations for the recurrence weight matrices.
ql_u1 = make_node("QuantizeLinear", inputs=["U_f","scale_f","zero_point_all"], outputs=["ql_uf_out"], name="ql_u1")
clp_u1 = make_node("Clip", inputs=["ql_uf_out","min","max"], outputs=["clp_uf"], name="clp_u1")
dql_u1 = make_node("DequantizeLinear", inputs=["clp_uf","scale_f","zero_point_all"], outputs=["dql_uf_out"], name="dql_u1")

ql_u2 = make_node("QuantizeLinear", inputs=["U_i","scale_i","zero_point_all"], outputs=["ql_ui_out"], name="ql_u2")
clp_u2 = make_node("Clip", inputs=["ql_ui_out","min","max"], outputs=["clp_ui"], name="clp_u2")
dql_u2 = make_node("DequantizeLinear", inputs=["clp_ui","scale_i","zero_point_all"], outputs=["dql_ui_out"], name="dql_u2")

ql_u3 = make_node("QuantizeLinear", inputs=["U_c","scale_c","zero_point_all"], outputs=["ql_uc_out"], name="ql_u3")
clp_u3 = make_node("Clip", inputs=["ql_uc_out","min","max"], outputs=["clp_uc"], name="clp_u3")
dql_u3 = make_node("DequantizeLinear", inputs=["clp_uc","scale_c","zero_point_all"], outputs=["dql_uc_out"], name="dql_u3")

ql_u4 = make_node("QuantizeLinear", inputs=["U_o","scale_o","zero_point_all"], outputs=["ql_uo_out"], name="ql_u4")
clp_u4 = make_node("Clip", inputs=["ql_uo_out","min","max"], outputs=["clp_uo"], name="clp_u4")
dql_u4 = make_node("DequantizeLinear", inputs=["clp_uo","scale_o","zero_point_all"], outputs=["dql_uo_out"], name="dql_u4")

#1st Equation : Forget gate
mul_node1_e1 = make_node("MatMul", inputs=["dql_wf_out","dql_input_out"], outputs=["out_m1_e1"], name="mul_node1_e1") #As io_quant was none during training. Hence setting the input as original input. We know it's UINT8 so we set that in FINN.
mul_node2_e1 = make_node("MatMul", inputs=["dql_uf_out","input_dql_hidden_out"], outputs=["out_m2_e1"],name="mul_node2_e1")
add_node1_e1 = make_node("Add", inputs=["out_m1_e1","out_m2_e1"], outputs=["out_add1_e1"],name="add_node1_e1")
add_node2_e1 = make_node("Add", inputs=["out_add1_e1","b_f"], outputs=["f_t_ba"],name="add_node2_e1")
quant_linear1_e1 = make_node("QuantizeLinear", inputs=["f_t_ba","scale_3","zero_point_all"], outputs=["f_t_ql1"],name="quant_linear1_e1")
clp1_e1 = make_node("Clip", inputs=["f_t_ql1","min_6b","max_6b"], outputs=["clp_f_t_ql1"], name="clp1_e1")
dequant_linear1_e1 = make_node("DequantizeLinear", inputs=["clp_f_t_ql1", "scale_3", "zero_point_all"], outputs=["f_t_dql1"], name="dequant_linear1_e1")
sig_f_e1     = make_node("Sigmoid", inputs=["f_t_dql1"], outputs=["f_t"],name="sig_f_e1")
quant_linear2_e1 = make_node("QuantizeLinear", inputs=["f_t","scale_4","zero_point_unsigned"], outputs=["f_t_ql2"],name="quant_linear2_e1")
clp2_e1 = make_node("Clip", inputs=["f_t_ql2","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_f_t_ql2"], name="clp2_e1")
dequant_linear2_e1 = make_node("DequantizeLinear", inputs=["clp_f_t_ql2", "scale_4", "zero_point_unsigned"], outputs=["f_t_dql2"], name="dequant_linear2_e1")

#2nd Equation : Input gate
mul_node1_e2 = make_node("MatMul", inputs=["dql_wi_out","dql_input_out"], outputs=["out_m1_e2"], name="mul_node1_e2")
mul_node2_e2 = make_node("MatMul", inputs=["dql_ui_out","input_dql_hidden_out"], outputs=["out_m2_e2"],name="mul_node2_e2")
add_node1_e2 = make_node("Add", inputs=["out_m1_e2","out_m2_e2"], outputs=["out_add1_e2"],name="add_node1_e2")
add_node2_e2 = make_node("Add", inputs=["out_add1_e2","b_i"], outputs=["i_t_ba"],name="add_node2_e2")
quant_linear1_e2 = make_node("QuantizeLinear", inputs=["i_t_ba","scale_1","zero_point_all"], outputs=["i_t_ql1"],name="quant_linear1_e2")
clp1_e2 = make_node("Clip", inputs=["i_t_ql1","min_6b","max_6b"], outputs=["clp_i_t_ql1"], name="clp1_e2")
dequant_linear1_e2 = make_node("DequantizeLinear", inputs=["clp_i_t_ql1","scale_1", "zero_point_all"], outputs=["i_t_dql1"], name="dequant_linear1_e2")
sig_i_e2     = make_node("Sigmoid", inputs=["i_t_dql1"], outputs=["i_t"],name="sig_i_e2")
quant_linear2_e2 = make_node("QuantizeLinear", inputs=["i_t","scale_2","zero_point_unsigned"], outputs=["i_t_ql2"],name="quant_linear2_e2")
clp2_e2 = make_node("Clip", inputs=["i_t_ql2","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_i_t_ql2"], name="clp2_e2")
dequant_linear2_e2 = make_node("DequantizeLinear", inputs=["clp_i_t_ql2", "scale_2", "zero_point_unsigned"], outputs=["i_t_dql2"], name="dequant_linear2_e2")

#3rd Equation : Output gate
mul_node1_e3 = make_node("MatMul", inputs=["dql_wo_out","dql_input_out"], outputs=["out_m1_e3"], name="mul_node1_e3")
mul_node2_e3 = make_node("MatMul", inputs=["dql_uo_out","input_dql_hidden_out"], outputs=["out_m2_e3"],name="mul_node2_e3")
add_node1_e3 = make_node("Add", inputs=["out_m1_e3","out_m2_e3"], outputs=["out_add1_e3"],name="add_node1_e3")
add_node2_e3 = make_node("Add", inputs=["out_add1_e3","b_o"], outputs=["o_t_ba"],name="add_node2_e3" )
quant_linear1_e3 = make_node("QuantizeLinear", inputs=["o_t_ba","scale_7","zero_point_all"], outputs=["o_t_ql1"],name="quant_linear_e3")
clp1_e3 = make_node("Clip", inputs=["o_t_ql1","min_6b","max_6b"], outputs=["clp_o_t_ql1"], name="clp1_e3")
dequant_linear1_e3 = make_node("DequantizeLinear", inputs=["clp_o_t_ql1","scale_7", "zero_point_all"], outputs=["o_t_dql1"], name="dequant_linear_e3")
sig_o_e3     = make_node("Sigmoid", inputs=["o_t_dql1"], outputs=["o_t"],name="sig_o_e3")
quant_linear2_e3 = make_node("QuantizeLinear", inputs=["o_t","scale_8","zero_point_unsigned"], outputs=["o_t_ql2"],name="quant_linear2_e3")
clp2_e3 = make_node("Clip", inputs=["o_t_ql2","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_o_t_ql2"], name="clp2_e3")
dequant_linear2_e3 = make_node("DequantizeLinear", inputs=["clp_o_t_ql2", "scale_8", "zero_point_unsigned"], outputs=["o_t_dql2"], name="dequant_linear2_e3")

#4th Equation : Cell gate
mul_node1_e4 = make_node("MatMul", inputs=["dql_wc_out","dql_input_out"], outputs=["out_m1_e4"], name="mul_node1_e4")
mul_node2_e4 = make_node("MatMul", inputs=["dql_uc_out","input_dql_hidden_out"], outputs=["out_m2_e4"],name="mul_node2_e4")
add_node1_e4 = make_node("Add", inputs=["out_m1_e4","out_m2_e4"], outputs=["out_add1_e4"],name="add_node1_e4")
add_node2_e4 = make_node("Add", inputs=["out_add1_e4","b_c"], outputs=["c_t_ba"],name="add_node2_e4")
quant_linear1_e4 = make_node("QuantizeLinear", inputs=["c_t_ba","scale_5","zero_point_all"], outputs=["c_t_ql1"],name="quant_linear1_e4")
clp1_e4 = make_node("Clip", inputs=["c_t_ql1","min_6b","max_6b"], outputs=["clp_c_t_ql1"], name="clp1_e4")
dequant_linear1_e4 = make_node("DequantizeLinear", inputs=["clp_c_t_ql1","scale_5", "zero_point_all"], outputs=["c_t_dql1"], name="dequant_linear1_e4")
tanh_c_e4    = make_node("Tanh", inputs=["c_t_dql1"], outputs=["c_t_partial"],name="tanh_c_e4")
quant_linear2_e4 = make_node("QuantizeLinear", inputs=["c_t_partial","scale_6","zero_point_all"], outputs=["c_t_ql2"],name="quant_linear2_e4")
clp2_e4 = make_node("Clip", inputs=["c_t_ql2","min_6b","max_6b"], outputs=["clp_c_t_ql2"], name="clp2_e4")
dequant_linear2_e4 = make_node("DequantizeLinear", inputs=["clp_c_t_ql2", "scale_6", "zero_point_all"], outputs=["c_t_dql2"], name="dequant_linear2_e4")

#5th Equation : Cell state compute
el_mul_node1_e5 = make_node("Mul", inputs=["f_t_dql2","dql_cell_out"], outputs=["out_el_mul1_e5"],name="el_mul_node1_e5") #c_t-1
quant_linear1_e5 = make_node("QuantizeLinear", inputs=["out_el_mul1_e5","scale_9","zero_point_all"], outputs=["fifth_ql1"],name="quant_linear1_e5")
clp1_e5 = make_node("Clip", inputs=["fifth_ql1","min_6b","max_6b"], outputs=["clp_fifth_ql1"], name="clp1_e5")
dequant_linear1_e5 = make_node("DequantizeLinear", inputs=["clp_fifth_ql1","scale_9", "zero_point_all"], outputs=["fifth_dql1"], name="dequant_linear1_e5")
el_mul_node2_e5 = make_node("Mul", inputs=["i_t_dql2","c_t_dql2"], outputs=["out_el_mul2_e5"], name="el_mul_node2_e5") 
quant_linear2_e5 = make_node("QuantizeLinear", inputs=["out_el_mul2_e5","scale_9","zero_point_all"], outputs=["fifth_ql2"],name="quant_linear2_e5")
clp2_e5 = make_node("Clip", inputs=["fifth_ql2","min_6b","max_6b"], outputs=["clp_fifth_ql2"], name="clp2_e5")
dequant_linear2_e5 = make_node("DequantizeLinear", inputs=["clp_fifth_ql2","scale_9", "zero_point_all"], outputs=["fifth_dql2"], name="dequant_linear2_e5")
out_add1_e5     = make_node("Add", inputs=["fifth_dql1","fifth_dql2"], outputs=["c_t"], name="out_add1_e5")
#Branch that gives the output
quant_linear3_e5 = make_node("QuantizeLinear", inputs=["c_t","scale_9","zero_point_all"], outputs=["h_t_ql"], name="quant_linear3_e5")
clp3_e5 = make_node("Clip", inputs=["h_t_ql","min_6b","max_6b"], outputs=["clp_h_t_ql"], name="clp3_e5")
dequant_linear3_e5 = make_node("DequantizeLinear", inputs=["clp_h_t_ql","scale_9","zero_point_all"], outputs=["c_t_out"], name="dequant_linear3_e5")
#Branch that carries it forward
quant_linear3_e5_v2 = make_node("QuantizeLinear", inputs=["c_t","scale_9","zero_point_all"], outputs=["c_t_carry_ql"], name="quant_linear3_e5_v2")
clp3_e5_v2 = make_node("Clip", inputs=["c_t_carry_ql","min_6b","max_6b"], outputs=["clp_c_t_carry_ql"], name="clp3_e5_v2")
dequant_linear3_e5_v2 = make_node("DequantizeLinear", inputs=["clp_c_t_carry_ql","scale_9","zero_point_all"], outputs=["c_t_carry_dql"], name="dequant_linear3_e5_v2")


#6th Equation : Hidden state compute
tanh_node_e6    = make_node("Tanh", inputs=["c_t_carry_dql"], outputs=["out_tanh_e6"], name="tanh_node_e6") 
quant_linear1_e6 = make_node("QuantizeLinear", inputs=["out_tanh_e6","scale_10","zero_point_all"], outputs=["sixth_ql1"], name="quant_linear1_e6")
clp1_e6 = make_node("Clip", inputs=["sixth_ql1","min_6b","max_6b"], outputs=["clp_sixth_ql1"], name="clp1_e6")
dequant_linear1_e6 = make_node("DequantizeLinear", inputs=["clp_sixth_ql1","scale_10","zero_point_all"], outputs=["sixth_dql1"], name="dequant_linear1_e6")
el_mul_node1_e6 = make_node("Mul", inputs=["sixth_dql1","o_t_dql2"], outputs=["h_t_inter"], name="el_mul_node1_e6")#h_t_inter : Curent Hidden State output
#Branch that gives the output
ql_hidden_state = make_node("QuantizeLinear", inputs=["h_t_inter","scale_11","zero_point_all"], outputs=["ql_hidden_out"],name="ql_hidden_state")
dql_hidden_state = make_node("DequantizeLinear", inputs=["ql_hidden_out", 'scale_11', "zero_point_all"], outputs=["dql_hidden_out"],name="dql_hidden_state")
#Branch that carries it forward
ql_hidden_state_v2 = make_node("QuantizeLinear", inputs=["h_t_inter","scale_11","zero_point_all"], outputs=["ql_hidden_out_carry"],name="ql_hidden_state_v2")
dql_hidden_state_v2 = make_node("DequantizeLinear", inputs=["ql_hidden_out_carry", 'scale_11', "zero_point_all"], outputs=["dql_hidden_out_carry"],name="dql_hidden_state_v2")


#LSTM operation complete : Now adding the dense layers computation graph below
#FC1
relu_act_l3 = make_node("Relu",inputs=["dql_hidden_out_carry"],outputs=["output_relu1"],name="relu_act_l3")
quant_linear_l3 = make_node("QuantizeLinear",inputs=["output_relu1","scale_l3","zero_point_l3"],outputs=["relu_ql1"],name="quant_linear_l3")
clp_fc1_act = make_node("Clip", inputs=["relu_ql1","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_relu_ql1"], name="clp_fc1_act")
dequant_linear_l3 = make_node("DequantizeLinear",inputs=["clp_relu_ql1","scale_l3","zero_point_l3"],outputs=["relu_dql1"],name="dequant_linear_l3")
quant_linear_w_fc1 = make_node("QuantizeLinear",inputs=["weights_fc1","scale_fc1","zero_point_fc1"],outputs=["weights_fc1_ql"],name="quant_linear_w_fc1")
clp_w_fc1 = make_node("Clip", inputs=["weights_fc1_ql","min","max"], outputs=["clp_w_fc1"], name="clp_w_fc1")
dequant_linear_w_fc1 = make_node("DequantizeLinear",inputs=["clp_w_fc1","scale_fc1","zero_point_fc1"],outputs=["weights_fc1_dql"],name="dequant_linear_w_fc1")
matmul_fc1 = make_node("Gemm", inputs=["weights_fc1_dql","relu_dql1","bias_fc1"], outputs=["out_matmul_fc1"], name="matmul_fc1") #(128,20)x(20,1)
# matmul_fc1 = make_node("MatMul", inputs=["weights_fc1_dql",""], outputs=["out_matmul_fc1"], name="matmul_fc1")
# add_fc1 = make_node("Add",inputs=["out_matmul_fc1","bias_fc1"],outputs=["out_add_fc1"],name="add_fc1")

#FC2
relu_act_l6 = make_node("Relu",inputs=["out_matmul_fc1"],outputs=["output_relu2"],name="relu_act_l6")
quant_linear_l6 = make_node("QuantizeLinear",inputs=["output_relu2","scale_l6","zero_point_l6"],outputs=["relu_ql2"],name="quant_linear_l6")
clp_fc2_act = make_node("Clip", inputs=["relu_ql2","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_relu_ql2"], name="clp_fc2_act")
dequant_linear_l6 = make_node("DequantizeLinear",inputs=["clp_relu_ql2","scale_l6","zero_point_l6"],outputs=["relu_dql2"],name="dequant_linear_l6")
quant_linear_w_fc2 = make_node("QuantizeLinear",inputs=["weights_fc2","scale_fc2","zero_point_fc2"],outputs=["weights_fc2_ql"],name="quant_linear_w_fc2")
clp_w_fc2 = make_node("Clip", inputs=["weights_fc2_ql","min","max"], outputs=["clp_w_fc2"], name="clp_w_fc2")
dequant_linear_w_fc2 = make_node("DequantizeLinear",inputs=["clp_w_fc2","scale_fc2","zero_point_fc2"],outputs=["weights_fc2_dql"],name="dequant_linear_w_fc2")
matmul_fc2 = make_node("Gemm", inputs=["weights_fc2_dql","relu_dql2","bias_fc2"], outputs=["out_matmul_fc2"], name="matmul_fc2") #(64,128)x(128,1) = (31x1)
# matmul_fc2 = make_node("MatMul", inputs=["weights_fc2_dql","out_add_fc1"], outputs=["out_matmul_fc2"], name="matmul_fc2")
# add_fc2 = make_node("Add",inputs=["out_matmul_fc2","bias_fc2"],outputs=["out_add_fc2"],name="add_fc2")

#FC3
relu_act_l8 = make_node("Relu",inputs=["out_matmul_fc2"],outputs=["output_relu3"],name="relu_act_l8")
quant_linear_l8 = make_node("QuantizeLinear",inputs=["output_relu3","scale_l8","zero_point_l8"],outputs=["relu_ql3"],name="quant_linear_l8")
clp_fc3_act = make_node("Clip", inputs=["relu_ql3","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_relu_ql3"], name="clp_fc3_act")
dequant_linear_l8 = make_node("DequantizeLinear",inputs=["clp_relu_ql3","scale_l8","zero_point_l8"],outputs=["relu_dql3"],name="dequant_linear_l8")
quant_linear_w_fc3 = make_node("QuantizeLinear",inputs=["weights_fc3","scale_fc3","zero_point_fc3"],outputs=["weights_fc3_ql"],name="quant_linear_w_fc3")
clp_w_fc3 = make_node("Clip", inputs=["weights_fc3_ql","min","max"], outputs=["clp_w_fc3"], name="clp_w_fc3")
dequant_linear_w_fc3 = make_node("DequantizeLinear",inputs=["clp_w_fc3","scale_fc3","zero_point_fc3"],outputs=["weights_fc3_dql"],name="dequant_linear_w_fc3")
matmul_fc3 = make_node("Gemm", inputs=["weights_fc3_dql","relu_dql3","bias_fc3"], outputs=["out_matmul_fc3"], name="matmul_fc3") #(1,64)x(64,1)
# matmul_fc3 = make_node("MatMul", inputs=["weights_fc3_dql","out_add_fc2"], outputs=["out_matmul_fc3"], name="matmul_fc3")
# add_fc3 = make_node("Add",inputs=["out_matmul_fc3","bias_fc3"],outputs=["out_add_fc3"],name="add_fc3")

#Final sigmoid activation
# sigmoid_output = make_node("Sigmoid",inputs=["out_matmul_fc3"],outputs=["out_sigmoid"],name="sigmoid_output")

################---------------------- Final Layer Definition of the IDS model ends here ---------------####################3

ids_model_brevitas_export_load = onnx.load("./acm-transactions-cps-model.onnx")
weights = ids_model_brevitas_export_load.graph.initializer
print(len(weights))
for i in range(len(weights)):
    w = numpy_helper.to_array(weights[i])
    print (ids_model_brevitas_export_load.graph.initializer[i].name)
    print(w.shape)
    print(w,',',i)
    print("-------------------------")

# exit()# : Activate this exit function to observe weights positions of the quantizers in the abvove exported brevitas graph.
    
bi_val = numpy_helper.to_array(weights[0]).reshape([20,1])#[20,]
Wi_val = numpy_helper.to_array(weights[1])#[20,5]
Ui_val = numpy_helper.to_array(weights[2])#[20,20]
bf_val = numpy_helper.to_array(weights[3]).reshape([20,1])#[20,]
Wf_val = numpy_helper.to_array(weights[4])#[20,5]
Uf_val = numpy_helper.to_array(weights[5])#[20,20]
bc_val = numpy_helper.to_array(weights[6]).reshape([20,1])#[20,]
Wc_val = numpy_helper.to_array(weights[7])#[20,5]
Uc_val = numpy_helper.to_array(weights[8])#[20,20]
bo_val = numpy_helper.to_array(weights[9]).reshape([20,1])#[20,]
Wo_val = numpy_helper.to_array(weights[10])#[20,5]
Uo_val = numpy_helper.to_array(weights[11])#[20,20]

#Parameters for the dense part of the IDS are defined below
weights_fc1_val = numpy_helper.to_array(weights[12])#[128,20]
weights_fc2_val = numpy_helper.to_array(weights[14])#[64,128]
weights_fc3_val = numpy_helper.to_array(weights[16])#[1,64]
bias_fc1_val = numpy_helper.to_array(weights[13]).reshape([64,1])
bias_fc2_val = numpy_helper.to_array(weights[15]).reshape([32,1])
bias_fc3_val = numpy_helper.to_array(weights[17]).reshape([5,1])


scale_l3 =0.0076811728067696095
zero_point_l3 = 0
scale_l3 = np.array(scale_l3).reshape([1,1])
zero_point_l3 = np.array(zero_point_l3).reshape([1,1])
scale_fc1 = 0.03191493824124336
zero_point_fc1 = 0
scale_fc1 = np.array(scale_fc1).reshape([1,1])
zero_point_fc1 = np.array(zero_point_fc1).reshape([1,1])

scale_l6 = 0.008241023868322372
zero_point_l6 = 0
scale_l6 = np.array(scale_l6).reshape([1,1])
zero_point_l6 = np.array(zero_point_l6).reshape([1,1])
scale_fc2 = 0.03191493824124336
zero_point_fc2 = 0
scale_fc2 = np.array(scale_fc2).reshape([1,1])
zero_point_fc2 = np.array(zero_point_fc2).reshape([1,1])

scale_l8 = 0.021023383364081383
zero_point_l8 = 0
scale_l8 = np.array(scale_l8).reshape([1,1])
zero_point_l8 = np.array(zero_point_l8).reshape([1,1])
scale_fc3 = 0.03191493824124336
zero_point_fc3 = 0
scale_fc3 = np.array(scale_fc3).reshape([1,1])
zero_point_fc3 = np.array(zero_point_fc3).reshape([1,1])

lstm_scan = make_graph(
    nodes=[  
    	   ql_input,
    	   # clp_input_8b,
    	   dql_input,		
    	   input_ql_hidden_state,
    	   # clp_hidden_8b,
    	   input_dql_hidden_state,		
           ql_cell_state,
           clp_cell_6b,
           dql_cell_state,
           ql_w1,
           clp_w1, 
           dql_w1,
           ql_w2,
           clp_w2, 
           dql_w2,
           ql_w3,
           clp_w3, 
           dql_w3,
           ql_w4,
           clp_w4, 
           dql_w4,
           ql_u1,
           clp_u1, 
           dql_u1,
           ql_u2,
           clp_u2,
           dql_u2,    
           ql_u3,
           clp_u3,
           dql_u3,    
           ql_u4,
           clp_u4,
           dql_u4, 
           mul_node1_e1,
           mul_node2_e1, 
           add_node1_e1, 
           add_node2_e1,
           quant_linear1_e1,
           clp1_e1,
           dequant_linear1_e1,
           sig_f_e1,
           quant_linear2_e1,
           clp2_e1, 
           dequant_linear2_e1,
           mul_node1_e2, 
           mul_node2_e2, 
           add_node1_e2, 
           add_node2_e2,
           quant_linear1_e2,
           clp1_e2,
           dequant_linear1_e2,
           sig_i_e2,
           quant_linear2_e2,
           clp2_e2,
           dequant_linear2_e2,
           mul_node1_e3, 
           mul_node2_e3,
           add_node1_e3, 
           add_node2_e3,
           quant_linear1_e3,
           clp1_e3,
           dequant_linear1_e3,
           sig_o_e3,
           quant_linear2_e3,
           clp2_e3,
           dequant_linear2_e3,
           mul_node1_e4, 
           mul_node2_e4, 
           add_node1_e4, 
           add_node2_e4,
           quant_linear1_e4,
           clp1_e4,
           dequant_linear1_e4,
           tanh_c_e4,
           quant_linear2_e4,
           clp2_e4,
           dequant_linear2_e4,
           el_mul_node1_e5,
           quant_linear1_e5, 
           clp1_e5,
           dequant_linear1_e5,
           el_mul_node2_e5,
           quant_linear2_e5,
           clp2_e5,
           dequant_linear2_e5,
           out_add1_e5,
           quant_linear3_e5,
           clp3_e5, 
           dequant_linear3_e5,
           quant_linear3_e5_v2,
           clp3_e5_v2,
           dequant_linear3_e5_v2,
           tanh_node_e6,
           quant_linear1_e6,
           clp1_e6, 
           dequant_linear1_e6,
           el_mul_node1_e6,
           # id_node_e6,
           ql_hidden_state,
           # clp_hidden_out_8b,
           dql_hidden_state,
           ql_hidden_state_v2,
           dql_hidden_state_v2,
           relu_act_l3,
           quant_linear_l3,
           clp_fc1_act,
           dequant_linear_l3,
           quant_linear_w_fc1,
		   clp_w_fc1,
           dequant_linear_w_fc1,
           matmul_fc1,
           relu_act_l6,
           quant_linear_l6,
           clp_fc2_act,
           dequant_linear_l6,
           quant_linear_w_fc2,
		   clp_w_fc2,
           dequant_linear_w_fc2,
           matmul_fc2,
           relu_act_l8,
           quant_linear_l8,
           clp_fc3_act,
           dequant_linear_l8,
           quant_linear_w_fc3,
		   clp_w_fc3,
           dequant_linear_w_fc3,
           matmul_fc3,
           # sigmoid_output
          ],
    name = "QCDQ-LSTM-SCAN",
    inputs=[inp2_m2,inp2_elm1,inp2_m1], #The order in which the inputs are defined here should match the input order when the scan node is defined.
    outputs = [out_matmul_fc3,out_hidden_state_float,out_cell_state],
    value_info=[
            make_tensor_value_info("ql_input_out",onnx.TensorProto.INT8, [input_size,1]),
            # make_tensor_value_info("ql_input_out_clip",onnx.TensorProto.UINT8, [input_size,1]),	
            make_tensor_value_info("dql_input_out",onnx.TensorProto.FLOAT, [input_size,1]),
            make_tensor_value_info("input_ql_hidden_out",onnx.TensorProto.INT8, [hidden_size,1]),
            # make_tensor_value_info("input_ql_hidden_out_clip",onnx.TensorProto.UINT8, [hidden_size,1]),
            make_tensor_value_info("input_dql_hidden_out",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("ql_hidden_out",onnx.TensorProto.INT8, [hidden_size,1]),
            make_tensor_value_info("ql_hidden_out_carry",onnx.TensorProto.INT8, [hidden_size,1]),
            make_tensor_value_info("dql_hidden_out_carry", onnx.TensorProto.FLOAT, [hidden_size,1]),
            # make_tensor_value_info("ql_hidden_out_clip",onnx.TensorProto.UINT8, [hidden_size,1]),
            # make_tensor_value_info("dql_hidden_out",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("h_t_inter_identity",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("ql_cell_out",onnx.TensorProto.INT8, [hidden_size,1]),
            make_tensor_value_info("dql_cell_out",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_m1_e1",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_m2_e1",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_add1_e1",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("f_t_ba",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("f_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_f_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("f_t_dql1", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("f_t_ql2",onnx.TensorProto.UINT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_f_t_ql2",onnx.TensorProto.UINT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("f_t_dql2", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e2",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_m2_e2",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_add1_e2",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("i_t_ba",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("i_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_i_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("i_t_dql1", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("i_t_ql2",onnx.TensorProto.UINT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_i_t_ql2",onnx.TensorProto.UINT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("i_t_dql2", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e3",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_m2_e3",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_add1_e3",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("o_t_ba",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("o_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_o_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("o_t_dql1", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("o_t_ql2",onnx.TensorProto.UINT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_o_t_ql2",onnx.TensorProto.UINT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("o_t_dql2", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_m1_e4",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_m2_e4",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_add1_e4",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("c_t_ba",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("c_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_c_t_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("c_t_dql1", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("c_t_ql2",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_c_t_ql2",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("c_t_dql2", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("f_t",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("i_t",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("o_t",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("c_t_partial",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_el_mul1_e5",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("out_el_mul2_e5",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("fifth_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_fifth_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("fifth_dql1", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("fifth_ql2",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_fifth_ql2",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("fifth_dql2", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("h_t_ql",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_h_t_ql",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("c_t_carry_ql",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_c_t_carry_ql",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("c_t_carry_dql", onnx.TensorProto.FLOAT, [hidden_size,1]),
            #make_tensor_value_info("h_t_dql", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("c_t", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("out_tanh_e6",onnx.TensorProto.FLOAT, [hidden_size,1]),
            make_tensor_value_info("sixth_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("clp_sixth_ql1",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("sixth_dql1", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("sixth_ql2",onnx.TensorProto.INT8, [hidden_size,1]),#Output of the quantized linear layer. Therefore the datatype INT8.
            make_tensor_value_info("h_t_inter", onnx.TensorProto.FLOAT, [hidden_size,1]),#Output of the dequantized linear layer. Therefore float datatype as the input will now be processed differently.
            make_tensor_value_info("ql_wf_out", onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("dql_wf_out",onnx.TensorProto.FLOAT, [hidden_size,input_size]),
            make_tensor_value_info("ql_wi_out", onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("dql_wi_out",onnx.TensorProto.FLOAT, [hidden_size,input_size]),
            make_tensor_value_info("ql_wc_out", onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("dql_wc_out",onnx.TensorProto.FLOAT, [hidden_size,input_size]),
            make_tensor_value_info("ql_wo_out", onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("dql_wo_out",onnx.TensorProto.FLOAT, [hidden_size,input_size]),
            make_tensor_value_info("ql_uf_out",onnx.TensorProto.INT8, [hidden_size,hidden_size]),
            make_tensor_value_info("dql_uf_out",onnx.TensorProto.FLOAT, [hidden_size,hidden_size]),
            make_tensor_value_info("ql_ui_out",onnx.TensorProto.INT8, [hidden_size,hidden_size]),
            make_tensor_value_info("dql_ui_out",onnx.TensorProto.FLOAT, [hidden_size,hidden_size]),
            make_tensor_value_info("ql_uc_out",onnx.TensorProto.INT8, [hidden_size,hidden_size]),
            make_tensor_value_info("dql_uc_out",onnx.TensorProto.FLOAT, [hidden_size,hidden_size]),
            make_tensor_value_info("ql_uo_out",onnx.TensorProto.INT8, [hidden_size,hidden_size]),
            make_tensor_value_info("dql_uo_out",onnx.TensorProto.FLOAT, [hidden_size,hidden_size]),
            make_tensor_value_info("clp_wf",onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("clp_wi",onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("clp_wc",onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("clp_wo",onnx.TensorProto.INT8, [hidden_size,input_size]),
            make_tensor_value_info("clp_uf",onnx.TensorProto.INT8, [hidden_size,hidden_size]), 
            make_tensor_value_info("clp_ui",onnx.TensorProto.INT8, [hidden_size,hidden_size]),
            make_tensor_value_info("clp_uc",onnx.TensorProto.INT8, [hidden_size,hidden_size]),
            make_tensor_value_info("clp_uo",onnx.TensorProto.INT8, [hidden_size,hidden_size]),
            make_tensor_value_info("clp_cell_6b",onnx.TensorProto.INT8, [hidden_size,1]),
            #After LSTM intermediate tensors
            #Dense Layer 1
            make_tensor_value_info("output_relu1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("clp_relu_ql1",onnx.TensorProto.UINT8, [20,1]),
            make_tensor_value_info("relu_ql1",onnx.TensorProto.UINT8, [20,1]),
            make_tensor_value_info("relu_dql1",onnx.TensorProto.FLOAT, [20,1]),
            make_tensor_value_info("weights_fc1_ql",onnx.TensorProto.INT8, [64,20]),
            make_tensor_value_info("clp_w_fc1",onnx.TensorProto.INT8, [64,20]),    
            make_tensor_value_info("weights_fc1_dql",onnx.TensorProto.FLOAT, [64,20]),
            make_tensor_value_info("out_matmul_fc1",onnx.TensorProto.FLOAT, [64,1]),
            make_tensor_value_info("out_add_fc1",onnx.TensorProto.FLOAT, [64,1]),

            #Dense Layer 2
            make_tensor_value_info("output_relu2",onnx.TensorProto.FLOAT, [64,1]),
            make_tensor_value_info("clp_relu_ql2",onnx.TensorProto.UINT8, [64,1]),
            make_tensor_value_info("relu_ql2",onnx.TensorProto.UINT8, [64,1]),
            make_tensor_value_info("relu_dql2",onnx.TensorProto.FLOAT, [64,1]),
            make_tensor_value_info("weights_fc2_ql",onnx.TensorProto.INT8, [32,64]),
            make_tensor_value_info("clp_w_fc2",onnx.TensorProto.INT8, [32,64]),    
            make_tensor_value_info("weights_fc2_dql",onnx.TensorProto.FLOAT, [32,64]),
            make_tensor_value_info("out_matmul_fc2",onnx.TensorProto.FLOAT, [32,1]),
            make_tensor_value_info("out_add_fc2",onnx.TensorProto.FLOAT, [32,1]),

            #Dense Layer 3
            make_tensor_value_info("output_relu3",onnx.TensorProto.FLOAT, [32,1]),
            make_tensor_value_info("clp_relu_ql3",onnx.TensorProto.UINT8, [32,1]),
            make_tensor_value_info("relu_ql3",onnx.TensorProto.UINT8, [32,1]),
            make_tensor_value_info("relu_dql3",onnx.TensorProto.FLOAT, [32,1]),
            make_tensor_value_info("weights_fc3_ql",onnx.TensorProto.INT8, [5,32]),
            make_tensor_value_info("clp_w_fc3",onnx.TensorProto.INT8, [5,32]),    
            make_tensor_value_info("weights_fc3_dql",onnx.TensorProto.FLOAT, [5,32]),
            #make_tensor_value_info("out_matmul_fc3",onnx.TensorProto.FLOAT, [5,1]),
            make_tensor_value_info("out_add_fc3",onnx.TensorProto.FLOAT, [5,1]),
            
        ],
    initializer=[
                 make_tensor('W_f',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wf_val)),
                 make_tensor('U_f',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Uf_val)),
                 make_tensor('b_f',onnx.TensorProto.FLOAT, [hidden_size,1], (bf_val)),
                 #Scalars 'scale' and 'zero_point' should be defined as below. Converting them into numpy array based single values causes some errors and exceptions saying that these values should be scalar. The definition has to be like this.
                 # Scalars are tensors with undefined shapes.
               #   make_tensor('scale_all',onnx.TensorProto.FLOAT,[],[1]),
                 # make_tensor('hidden_scale',onnx.TensorProto.FLOAT, [],[0.0078125]),
                 # make_tensor('inp_scale',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[30]))]),
                 make_tensor('scale_i',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[22]))]),
                 make_tensor('scale_c',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[25]))]),
                 make_tensor('scale_o',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[26]))]),
                 make_tensor('scale_f',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[27]))]),
                 make_tensor('scale_1',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[32]))]), #0.0057...
                 make_tensor('scale_2',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[36]))]), #0034227842
                 make_tensor('scale_3',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[33]))]),
                 make_tensor('scale_4',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[38]))]),
                 make_tensor('scale_5',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[34]))]),
                 make_tensor('scale_6',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[39]))]),
                 make_tensor('scale_7',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[35]))]), #Approximate scale_7 value = 0.0085895785
                 make_tensor('scale_8',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[40]))]),#0.0026683041
                 make_tensor('scale_9',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[28]))]),
                 make_tensor('scale_10',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[41]))]),
                 make_tensor('scale_11',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[19]))]),#0.0036052174                 
                 make_tensor('zero_point_all',onnx.TensorProto.INT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
                 make_tensor('zero_point_unsigned',onnx.TensorProto.UINT8,[],[0]),#Zero-point datatype is int8 or unit8 and some advanced floating point datatypes.
                 #Introducing scalars for the clip operators.
                 make_tensor('min', onnx.TensorProto.INT8, [], [-127]),
                 make_tensor('max', onnx.TensorProto.INT8, [], [127]),
                 make_tensor('min_6b', onnx.TensorProto.INT8, [], [-31]),
                 make_tensor('min_6b_2', onnx.TensorProto.INT8, [], [-32]),
                 make_tensor('max_6b', onnx.TensorProto.INT8, [], [31]),
                 make_tensor('min_8b_unsigned', onnx.TensorProto.UINT8, [], [0]),
                 make_tensor('max_8b_unsigned', onnx.TensorProto.UINT8, [], [255]),
                 make_tensor('min_6b_unsigned', onnx.TensorProto.UINT8, [], [0]),
                 make_tensor('max_6b_unsigned', onnx.TensorProto.UINT8, [], [63]),
                 make_tensor('W_i',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wi_val)),
                 make_tensor('U_i',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Ui_val)),
                 make_tensor('b_i',onnx.TensorProto.FLOAT, [hidden_size,1], (bi_val)),
                 make_tensor('W_o',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wo_val)),
                 make_tensor('U_o',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Uo_val)),
                 make_tensor('b_o',onnx.TensorProto.FLOAT, [hidden_size,1], (bo_val)),
                 make_tensor('W_c',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wc_val)),
                 make_tensor('U_c',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Uc_val)),
                 make_tensor('b_c',onnx.TensorProto.FLOAT, [hidden_size,1], (bc_val)),
                 #Initializers for the dense layers start from here.
                 #Weights of the three dense layers will come below.
                 make_tensor('weights_fc1',onnx.TensorProto.FLOAT,[64,20],(weights_fc1_val)),
                 make_tensor('weights_fc2',onnx.TensorProto.FLOAT,[32,64],(weights_fc2_val)),
                 make_tensor('weights_fc3',onnx.TensorProto.FLOAT,[5,32],(weights_fc3_val)),   
                 #fc1
                 make_tensor('scale_l3',onnx.TensorProto.FLOAT,[],(scale_l3)),
                 make_tensor('zero_point_l3',onnx.TensorProto.UINT8,[],(zero_point_l3)),
                 make_tensor('scale_fc1',onnx.TensorProto.FLOAT,[],(scale_fc1)),
                 make_tensor('zero_point_fc1',onnx.TensorProto.INT8,[],(zero_point_fc1)),
                 make_tensor('bias_fc1',onnx.TensorProto.FLOAT,[64,1],(bias_fc1_val)),
                 #fc2
                 make_tensor('scale_l6',onnx.TensorProto.FLOAT,[],(scale_l6)),
                 make_tensor('zero_point_l6',onnx.TensorProto.UINT8,[],(zero_point_l6)),
                 make_tensor('scale_fc2',onnx.TensorProto.FLOAT,[],(scale_fc2)),
                 make_tensor('zero_point_fc2',onnx.TensorProto.INT8,[],(zero_point_fc2)),
                 make_tensor('bias_fc2',onnx.TensorProto.FLOAT,[32,1],(bias_fc2_val)),
                 #fc3
                 make_tensor('scale_l8',onnx.TensorProto.FLOAT,[],(scale_l8)),
                 make_tensor('zero_point_l8',onnx.TensorProto.UINT8,[],(zero_point_l8)),
                 make_tensor('scale_fc3',onnx.TensorProto.FLOAT,[],(scale_fc3)),
                 make_tensor('zero_point_fc3',onnx.TensorProto.INT8,[],(zero_point_fc3)),
                 make_tensor('bias_fc3',onnx.TensorProto.FLOAT,[5,1],(bias_fc3_val)), 
                 make_tensor('min_fc', onnx.TensorProto.INT8, [], [-127]),
                 make_tensor('max_fc', onnx.TensorProto.INT8, [], [127]),           
                 # make_tensor('weights_fc1',onnx.TensorProto.FLOAT,[64,20],(weights_fc1_val)),
                 # make_tensor('weights_fc2',onnx.TensorProto.FLOAT,[32,64],(weights_fc2_val)),
                 # make_tensor('weights_fc3',onnx.TensorProto.FLOAT,[1,32],(weights_fc3_val)),   
                 # make_tensor('scale_bn1',onnx.TensorProto.FLOAT,[20],((scale_bn1))),
                 # make_tensor('bias_bn1',onnx.TensorProto.FLOAT,[20],((bias_bn1))),
                 # make_tensor('running_mean_bn1',onnx.TensorProto.FLOAT,[20],((running_mean_bn1))),
                 # make_tensor('running_var_bn1',onnx.TensorProto.FLOAT,[20],((running_var_bn1))),
                 # make_tensor('scale_l3',onnx.TensorProto.FLOAT,[],(scale_l3)),
                 # make_tensor('zero_point_l3',onnx.TensorProto.UINT8,[],(zero_point_l3)),
                 # make_tensor('scale_fc1',onnx.TensorProto.FLOAT,[],(scale_fc1)),
                 # make_tensor('zero_point_fc1',onnx.TensorProto.INT8,[],(zero_point_fc1)),
                 # make_tensor('bias_fc1',onnx.TensorProto.FLOAT,[64,1],(bias_fc1_val)),
                 #fc2
                 # make_tensor('scale_bn2',onnx.TensorProto.FLOAT,[64],(scale_bn2)),
                 # make_tensor('bias_bn2',onnx.TensorProto.FLOAT,[64],(bias_bn2)),
                 # make_tensor('running_mean_bn2',onnx.TensorProto.FLOAT,[64],(running_mean_bn2)),
                 # make_tensor('running_var_bn2',onnx.TensorProto.FLOAT,[64],(running_var_bn2)),
                 # make_tensor('scale_l6',onnx.TensorProto.FLOAT,[],(scale_l6)),
                 # make_tensor('zero_point_l6',onnx.TensorProto.UINT8,[],(zero_point_l6)),
                 # make_tensor('scale_fc2',onnx.TensorProto.FLOAT,[],(scale_fc2)),
                 # make_tensor('zero_point_fc2',onnx.TensorProto.INT8,[],(zero_point_fc2)),
                 # make_tensor('bias_fc2',onnx.TensorProto.FLOAT,[32,1],(bias_fc2_val)),
                 #fc3
                 # make_tensor('scale_bn3',onnx.TensorProto.FLOAT,[32],(scale_bn3)),
                 # make_tensor('bias_bn3',onnx.TensorProto.FLOAT,[32],(bias_bn3)),
                 # make_tensor('running_mean_bn3',onnx.TensorProto.FLOAT,[32],(running_mean_bn3)),
                 # make_tensor('running_var_bn3',onnx.TensorProto.FLOAT,[32],(running_var_bn3)),
                 # make_tensor('scale_l8',onnx.TensorProto.FLOAT,[],(scale_l8)),
                 # make_tensor('zero_point_l8',onnx.TensorProto.UINT8,[],(zero_point_l8)),
                 # make_tensor('scale_fc3',onnx.TensorProto.FLOAT,[],(scale_fc3)),
                 # make_tensor('zero_point_fc3',onnx.TensorProto.INT8,[],(zero_point_fc3)),
                 # make_tensor('bias_fc3',onnx.TensorProto.FLOAT,[1,1],(bias_fc3_val)),
                 # make_tensor('min_fc', onnx.TensorProto.INT8, [], [-127]),
                 # make_tensor('max_fc', onnx.TensorProto.INT8, [], [127]),
                 #More batchnorm initializers
                 # make_tensor('epsilon_val1', onnx.TensorProto.FLOAT, [], [epsilon_val1]),
                 # make_tensor('epsilon_val2', onnx.TensorProto.FLOAT, [], [epsilon_val2]),
                 # make_tensor('epsilon_val3', onnx.TensorProto.FLOAT, [], [epsilon_val3]),
                 # make_tensor('momentum_val1', onnx.TensorProto.FLOAT, [], [momentum_val1]),
                 # make_tensor('momentum_val2', onnx.TensorProto.FLOAT, [], [momentum_val2]),
                 # make_tensor('momentum_val3', onnx.TensorProto.FLOAT, [], [momentum_val3]),
                 # make_tensor('training_mode_val', onnx.TensorProto.INT32, [], [training_mode_val]),
                ]
)

onnx_model = qonnx_make_model(lstm_scan, producer_name="QuantizeLSTM_scan")
onnx.save(onnx_model, './onnx_definition_acm_trans_cps.onnx')

#Converting to opset version '14' to accomodate clip nodes with INT8 and UINT8 input 
onnx_model.opset_import[0].version = 14
# print(onnx_model)

# Testing to check if the model is serializing without errors or warnings
#Even after converting to an opset version of 14 there was an error saying that the clip operator is tied to two different datatypes (int8 and float)
#That was because the MIN and the MAX values were defined as FLOAT tensors and the Clip operator constrains the input and output datatypes to be the same.
#Converting them to INT8 datatypes solved that error.
sess = rt.InferenceSession(onnx_model.SerializeToString())
# Defining the values of the varibales to test the execution of the onnx model
# in1lstm = np.zeros([10,1],dtype=np.float32).reshape([10,1])
#Input values have to be below 0. Values greater than 1 will be quantized with the same values in the quantizerlinear layer and there will be no chnage in the outputs.
#For vallues between 0 and 1 we get similar outputs from the brevitas layer and this graph. Clean and organize this code properly tomorrow.

# in1lstm = np.array([10,20,30,40,50,60,70,80,90,100],dtype=np.float32).reshape([10,1])
# in1lstm = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.float32).reshape([10,1])
# in1lstm = np.ones([1,10],dtype=np.float32).reshape([10,1])
in1lstm = np.zeros([10,1],dtype=np.float32).reshape([10,1])
in1lstm[0][0] = 0
in1lstm[1][0] = 1
in1lstm[2][0] = 2
in1lstm[3][0] = 3
in1lstm[4][0] = 4
in1lstm[5][0] = 5
in1lstm[6][0] = 6
in1lstm[7][0] = 7
in1lstm[8][0] = 8
in1lstm[9][0] = 9
# in1lstm.fill(0.5)
# print(in1lstm)
# in2lstm =  np.zeros((20, 1)).astype(np.float32)
# in3lstm =  np.zeros((20, 1)).astype(np.float32)
in2lstm =  np.ones((20, 1)).astype(np.float32)
in3lstm =  np.ones((20, 1)).astype(np.float32)
in2lstm[0][0] = 15
in3lstm[0][0] = 12

# in2lstm = np.array([0,0.8614,0,0,0,0.8614,-0.8614,-0.8614,0,0,0.8332,0,-0.8614,0.8332,0.8614,0.8614,0,0.8614,0,0.8614],dtype=np.float32).reshape([20,1])
# in3lstm = np.array([0,1.2486,1.2486,-1.2486,0,1.2486,-1.2486,-1.2486,0,0,1.2486,0,-1.2486,-1.2486,1.2486,1.2486,1.2486,1.2486,1.2486,1.2486],dtype=np.float32).reshape([20,1])

input_dict = {}
input_dict["X"] = in1lstm
input_dict["h_t-1"] = in2lstm
input_dict["c_t-1"] = in3lstm 
# print(input_dict)

#Executing the onnx model here.
sess = rt.InferenceSession(onnx_model.SerializeToString())
output = sess.run(None, input_dict)
print(output)
print('------------------------------')
exit()










































#Defining the input and output value info tensors for the scan_graph creation. These tensors act as the wrapper to the previously defined graph.

#Inputs
# scan_input = make_tensor_value_info("scan_input",onnx.TensorProto.FLOAT, [None,10,1])#X ; scan input; Here None defines the varibale number of inputs that can be supplied for input processing.
# inp_a      = make_tensor_value_info("inp_a",onnx.TensorProto.FLOAT, [20,1])# h_t-1
# inp_b      = make_tensor_value_info("inp_b",onnx.TensorProto.FLOAT, [20,1])# c_t-1

#Outputs
# out_a = make_tensor_value_info("out_a", onnx.TensorProto.FLOAT, [20,1])#h_t
# out_b = make_tensor_value_info("out_b", onnx.TensorProto.FLOAT, [20,1])#c_t
# out_c = make_tensor_value_info("out_c", onnx.TensorProto.FLOAT, [None,20,1])

# final_output = make_tensor_value_info("final_output", onnx.TensorProto.FLOAT, [1,1])
# out_bn1 = make_tensor_value_info("output_bn1",onnx.TensorProto.FLOAT, [1,20])
# out_relu1 = make_tensor_value_info("output_relu1",onnx.TensorProto.FLOAT, [1,20])
# out_matmul_fc1 = make_tensor_value_info("out_matmul_fc1",onnx.TensorProto.FLOAT, [64,1])
# out_transpose_1 = make_tensor_value_info("relu_dql1_transpose",onnx.TensorProto.FLOAT, [20,1])
# out_weights_fc1_dql = make_tensor_value_info("weights_fc1_dql",onnx.TensorProto.FLOAT, [64,20])

# Defining the scan node here now
# scan_node_lstm = make_node(
#     "Scan", 
#     inputs=["inp_a","inp_b","scan_input"], 
#     outputs=["out_a","out_b"],#out_c#,"out_d" 
#     num_scan_inputs=1,
#     body=lstm_scan, domain=''
# )

#LSTM operation complete : Now adding the dense layers computation graph below
#FC1
#Need to add a reshape layer here to make this a vector
# transpose_1 = make_node("Transpose",inputs=["out_a"],outputs=["out_a_transpose"],name="transpose_1")#"reshape_dim_1"
# batchnorm_l2 = make_node("BatchNormalization",inputs=["out_a_transpose","scale_bn1","bias_bn1","running_mean_bn1","running_var_bn1"], outputs=["output_bn1"],name="batchnorm_l2")#training_mode="training_mode_val",momentum="momentum_val1",epsilon="epsilon_val1",
# relu_act_l3 = make_node("Relu",inputs=["output_bn1"],outputs=["output_relu1"],name="relu_act_l3")
# quant_linear_l3 = make_node("QuantizeLinear",inputs=["output_relu1","scale_l3","zero_point_l3"],outputs=["relu_ql1"],name="quant_linear_l3")
# dequant_linear_l3 = make_node("DequantizeLinear",inputs=["relu_ql1","scale_l3","zero_point_l3"],outputs=["relu_dql1"],name="dequant_linear_l3")
# transpose_2 = make_node("Transpose",inputs=["relu_dql1"],outputs=["relu_dql1_transpose"],name="transpose_2")#,axes=[1]
# quant_linear_w_fc1 = make_node("QuantizeLinear",inputs=["weights_fc1","scale_fc1","zero_point_fc1"],outputs=["weights_fc1_ql"],name="quant_linear_w_fc1")
# clp_w_fc1 = make_node("Clip", inputs=["weights_fc1_ql","min_fc","max_fc"], outputs=["clp_w_fc1"], name="clp_w_fc1")
# dequant_linear_w_fc1 = make_node("DequantizeLinear",inputs=["clp_w_fc1","scale_fc1","zero_point_fc1"],outputs=["weights_fc1_dql"],name="dequant_linear_w_fc1")
# matmul_fc1 = make_node("Gemm", inputs=["weights_fc1_dql","relu_dql1_transpose","bias_fc1"], outputs=["out_matmul_fc1"], name="matmul_fc1") #(64,20)x(20,1)

# #FC2
# transpose_3 = make_node("Transpose",inputs=["out_matmul_fc1"],outputs=["out_matmul_fc1_transpose"],name="transpose_3")#"reshape_dim_1"
# batchnorm_l5 = make_node("BatchNormalization",inputs=["out_matmul_fc1_transpose","scale_bn2","bias_bn2","running_mean_bn2","running_var_bn2"], outputs=["output_bn2"], name="batchnorm_l5")#epsilon="epsilon_val2",momentum="momentum_val2",training_mode="training_mode_val"
# relu_act_l6 = make_node("Relu",inputs=["output_bn2"],outputs=["output_relu2"],name="relu_act_l6")
# quant_linear_l6 = make_node("QuantizeLinear",inputs=["output_relu2","scale_l6","zero_point_l6"],outputs=["relu_ql2"],name="quant_linear_l6")
# dequant_linear_l6 = make_node("DequantizeLinear",inputs=["relu_ql2","scale_l6","zero_point_l6"],outputs=["relu_dql2"],name="dequant_linear_l6")
# transpose_4 = make_node("Transpose",inputs=["relu_dql2"],outputs=["relu_dql2_transpose"],name="transpose_4")#,axes=[1]
# quant_linear_w_fc2 = make_node("QuantizeLinear",inputs=["weights_fc2","scale_fc2","zero_point_fc2"],outputs=["weights_fc2_ql"],name="quant_linear_w_fc2")
# clp_w_fc2 = make_node("Clip", inputs=["weights_fc2_ql","min_fc","max_fc"], outputs=["clp_w_fc2"], name="clp_w_fc2")
# dequant_linear_w_fc2 = make_node("DequantizeLinear",inputs=["clp_w_fc2","scale_fc2","zero_point_fc2"],outputs=["weights_fc2_dql"],name="dequant_linear_w_fc2")
# matmul_fc2 = make_node("Gemm", inputs=["weights_fc2_dql","relu_dql2_transpose","bias_fc2"], outputs=["out_matmul_fc2"], name="matmul_fc2") #(32,64)x(64,1) = (31x1)

# #FC3
# transpose_5 = make_node("Transpose",inputs=["out_matmul_fc2"],outputs=["out_matmul_fc2_transpose"],name="transpose_5")#"reshape_dim_1"
# batchnorm_l7 = make_node("BatchNormalization",inputs=["out_matmul_fc2_transpose","scale_bn3","bias_bn3","running_mean_bn3","running_var_bn3"], outputs=["output_bn3"],name="batchnorm_l7")#epsilon="epsilon_val3",momentum="momentum_val3",training_mode="training_mode_val",
# relu_act_l8 = make_node("Relu",inputs=["output_bn3"],outputs=["output_relu3"],name="relu_act_l8")
# quant_linear_l8 = make_node("QuantizeLinear",inputs=["output_relu3","scale_l8","zero_point_l8"],outputs=["relu_ql3"],name="quant_linear_l8")
# dequant_linear_l8 = make_node("DequantizeLinear",inputs=["relu_ql3","scale_l8","zero_point_l8"],outputs=["relu_dql3"],name="dequant_linear_l8")
# transpose_6 = make_node("Transpose",inputs=["relu_dql3"],outputs=["relu_dql3_transpose"],name="transpose_6")#"reshape_dim_1"
# quant_linear_w_fc3 = make_node("QuantizeLinear",inputs=["weights_fc3","scale_fc3","zero_point_fc3"],outputs=["weights_fc3_ql"],name="quant_linear_w_fc3")
# clp_w_fc3 = make_node("Clip", inputs=["weights_fc3_ql","min_fc","max_fc"], outputs=["clp_w_fc3"], name="clp_w_fc3")
# dequant_linear_w_fc3 = make_node("DequantizeLinear",inputs=["clp_w_fc3","scale_fc3","zero_point_fc3"],outputs=["weights_fc3_dql"],name="dequant_linear_w_fc3")
# matmul_fc3 = make_node("Gemm", inputs=["weights_fc3_dql","relu_dql3_transpose","bias_fc3"], outputs=["out_matmul_fc3"], name="matmul_fc3") #(1,32)x(32,1)

# #Final relu activation
# relu_l10 = make_node("Relu",inputs=["out_matmul_fc3"],outputs=["out_relu10"],name="relu_l10")
# quant_linear_l10 = make_node("QuantizeLinear",inputs=["out_relu10","scale_l8","zero_point_l8"],outputs=["relu_ql10"],name="quant_linear_l10")
# dequant_linear_l10 = make_node("DequantizeLinear",inputs=["relu_ql10","scale_l8","zero_point_l8"],outputs=["final_output"],name="dequant_linear_l10")

# #Parameters for the dense part of the IDS are defined below
# training_mode_val = 0
# training_mode_val = int(np.array(training_mode_val).reshape([1,1]))
# epsilon_val1 = 0.000009999999747378752
# momentum_val1 = 0.8999999761581421
# scale_bn1 = numpy_helper.to_array(weights[18])
# bias_bn1 = numpy_helper.to_array(weights[19])
# running_mean_bn1 = numpy_helper.to_array(weights[20]) 
# running_var_bn1 = numpy_helper.to_array(weights[21])
# scale_l3 = 0.014219683595001698
# zero_point_l3 = 0
# scale_fc1 =   0.004962614271789789
# zero_point_fc1 = 0
# epsilon_val1 = float(np.array(epsilon_val1).reshape([1,1]))
# momentum_val1 = float(np.array(momentum_val1).reshape([1,1]))
# scale_l3 = np.array(scale_l3).reshape([1,1])
# zero_point_l3 = np.array(zero_point_l3).reshape([1,1])
# scale_fc1 = np.array(scale_fc1).reshape([1,1])
# zero_point_fc1 = np.array(zero_point_fc1).reshape([1,1])


# epsilon_val2 = 0.000009999999747378752
# momentum_val2 = 0.8999999761581421
# scale_bn2 = numpy_helper.to_array(weights[22])
# bias_bn2 = numpy_helper.to_array(weights[23])
# running_mean_bn2 = numpy_helper.to_array(weights[24])
# running_var_bn2 = numpy_helper.to_array(weights[25])
# scale_l6 = 0.014219683595001698
# zero_point_l6 = 0
# scale_fc2 = 0.007548323832452297 
# zero_point_fc2 = 0
# epsilon_val2 = float(np.array(epsilon_val2).reshape([1,1]))
# momentum_val2 = float(np.array(momentum_val2).reshape([1,1]))
# scale_l6 = np.array(scale_l6).reshape([1,1])
# zero_point_l6 = np.array(zero_point_l6).reshape([1,1])
# scale_fc2 = np.array(scale_fc2).reshape([1,1])
# zero_point_fc2 = np.array(zero_point_fc2).reshape([1,1])

# epsilon_val3 = 0.000009999999747378752
# momentum_val3 = 0.8999999761581421
# scale_bn3 = numpy_helper.to_array(weights[26])
# bias_bn3 = numpy_helper.to_array(weights[27])
# running_mean_bn3 = numpy_helper.to_array(weights[28])
# running_var_bn3 = numpy_helper.to_array(weights[29])
# scale_l8 = 0.014219683595001698
# zero_point_l8 = 0
# scale_fc3 = 0.008073310367763042  
# zero_point_fc3 = 0
# epsilon_val3 = float(np.array(epsilon_val3).reshape([1,1]))
# momentum_val3 = float(np.array(momentum_val3).reshape([1,1]))
# scale_l8 = np.array(scale_l8).reshape([1,1])
# zero_point_l8 = np.array(zero_point_l8).reshape([1,1])
# scale_fc3 = np.array(scale_fc3).reshape([1,1])
# zero_point_fc3 = np.array(zero_point_fc3).reshape([1,1])

# Define the graph for the scan node to execute it with onnxruntime.
# print("Scan Node graph definition is here")
# scan_lstm_node_graph = make_graph(
#     nodes = [
           # scan_node_lstm,
           # transpose_1,
           # batchnorm_l2,
           # relu_act_l3,
           # quant_linear_l3,
           # dequant_linear_l3,
           # transpose_2,
           # quant_linear_w_fc1,
           # clp_w_fc1,
           # dequant_linear_w_fc1,
           # matmul_fc1,
           # transpose_3,
           # batchnorm_l5,
           # relu_act_l6,
           # quant_linear_l6,
           # dequant_linear_l6,
           # transpose_4,
           # quant_linear_w_fc2,
           # clp_w_fc2,
           # dequant_linear_w_fc2,
           # matmul_fc2,
           # transpose_5,
           # batchnorm_l7,
           # relu_act_l8,
           # quant_linear_l8,
           # dequant_linear_l8,
           # transpose_6,
           # quant_linear_w_fc3,
           # clp_w_fc3,
           # dequant_linear_w_fc3,
           # matmul_fc3,
           # relu_l10,
           # quant_linear_l10,
           # dequant_linear_l10
        # ],
    # name="ids_model_sw_functional_verification",
    # inputs=[inp_a,inp_b,scan_input],#h_t-1, c_t-1, X
    # # outputs=[out_a,out_b,out_c],#h_t,c_t,h_t_concat
    # outputs=[final_output],#out_bn1,out_relu1,out_transpose_1,,out_sigmoid
    # value_info=[
            # make_tensor_value_info("out_a", onnx.TensorProto.FLOAT, [20,1]),#h_t_inter_transpose : Float as io_quant set to none
            # make_tensor_value_info("out_a_transpose", onnx.TensorProto.FLOAT, [1,20]),#h_t_inter_transpose
            # make_tensor_value_info("out_b", onnx.TensorProto.FLOAT, [20,1]),#h_t
            # make_tensor_value_info("out_c", onnx.TensorProto.FLOAT, [20,1]),#c_t
            # make_tensor_value_info("out_d", onnx.TensorProto.FLOAT, [None,20,1]),
            # #Definfing intermediate tensors of the dense layer here. 
            # #Dense Layer 1
            # make_tensor_value_info("output_bn1",onnx.TensorProto.FLOAT, [1,20]),  
            # make_tensor_value_info("output_relu1",onnx.TensorProto.FLOAT, [1,20]),
            # make_tensor_value_info("relu_ql1",onnx.TensorProto.UINT8, [1,20]),
            # make_tensor_value_info("relu_dql1",onnx.TensorProto.FLOAT, [1,20]),
            # make_tensor_value_info("relu_dql1_transpose",onnx.TensorProto.FLOAT, [20,1]),
            # make_tensor_value_info("weights_fc1_ql",onnx.TensorProto.INT8, [64,20]),
            # make_tensor_value_info("clp_w_fc1",onnx.TensorProto.INT8, [64,20]),    
            # make_tensor_value_info("weights_fc1_dql",onnx.TensorProto.FLOAT, [64,20]),    
            # make_tensor_value_info("out_matmul_fc1",onnx.TensorProto.FLOAT, [64,1]),
            # make_tensor_value_info("out_matmul_fc1_transpose",onnx.TensorProto.FLOAT, [1,64]), 
            # #Continue with tensor definition here and verify functional correctness tomorow.
            # #Dense Layer 2
            # make_tensor_value_info("output_bn2",onnx.TensorProto.FLOAT, [1,64]),  
            # make_tensor_value_info("output_relu2",onnx.TensorProto.FLOAT, [1,64]),
            # make_tensor_value_info("relu_ql2",onnx.TensorProto.UINT8, [1,64]),
            # make_tensor_value_info("relu_dql2",onnx.TensorProto.FLOAT, [1,64]),
            # make_tensor_value_info("relu_dql2_transpose",onnx.TensorProto.FLOAT, [64,1]),
            # make_tensor_value_info("weights_fc2_ql",onnx.TensorProto.INT8, [32,64]),
            # make_tensor_value_info("clp_w_fc2",onnx.TensorProto.INT8, [32,64]),    
            # make_tensor_value_info("weights_fc2_dql",onnx.TensorProto.FLOAT, [32,64]),    
            # make_tensor_value_info("out_matmul_fc2",onnx.TensorProto.FLOAT, [32,1]),
            # #Dense Layer 3
            # make_tensor_value_info("out_matmul_fc2_transpose",onnx.TensorProto.FLOAT, [1,32]), 
            # make_tensor_value_info("output_bn3",onnx.TensorProto.FLOAT, [1,32]),  
            # make_tensor_value_info("output_relu3",onnx.TensorProto.FLOAT, [1,32]),
            # make_tensor_value_info("relu_ql3",onnx.TensorProto.UINT8, [1,32]),
            # make_tensor_value_info("relu_dql3",onnx.TensorProto.FLOAT, [1,32]),
            # make_tensor_value_info("relu_dql3_transpose",onnx.TensorProto.FLOAT, [32,1]),
            # make_tensor_value_info("weights_fc3_ql",onnx.TensorProto.INT8, [1,32]),
            # make_tensor_value_info("clp_w_fc3",onnx.TensorProto.INT8, [1,32]),    
            # make_tensor_value_info("weights_fc3_dql",onnx.TensorProto.FLOAT, [1,32]),    
            # make_tensor_value_info("out_matmul_fc3",onnx.TensorProto.FLOAT, [1,1]),
            # make_tensor_value_info("out_relu10",onnx.TensorProto.FLOAT, [1,1]),
            # make_tensor_value_info("relu_ql10",onnx.TensorProto.UINT8, [1,1]),
            #This should be all the intermediate tensors required for the graph. 
    # ],
    # initializer=[
                 #Initializers for the dense layers start from here.
                 #Weights of the three dense layers will come below.
                 # make_tensor('weights_fc1',onnx.TensorProto.FLOAT,[64,20],(weights_fc1_val)),
                 # make_tensor('weights_fc2',onnx.TensorProto.FLOAT,[32,64],(weights_fc2_val)),
                 # make_tensor('weights_fc3',onnx.TensorProto.FLOAT,[1,32],(weights_fc3_val)),   
                 # make_tensor('scale_bn1',onnx.TensorProto.FLOAT,[20],((scale_bn1))),
                 # make_tensor('bias_bn1',onnx.TensorProto.FLOAT,[20],((bias_bn1))),
                 # make_tensor('running_mean_bn1',onnx.TensorProto.FLOAT,[20],((running_mean_bn1))),
                 # make_tensor('running_var_bn1',onnx.TensorProto.FLOAT,[20],((running_var_bn1))),
                 # make_tensor('scale_l3',onnx.TensorProto.FLOAT,[],(scale_l3)),
                 # make_tensor('zero_point_l3',onnx.TensorProto.UINT8,[],(zero_point_l3)),
                 # make_tensor('scale_fc1',onnx.TensorProto.FLOAT,[],(scale_fc1)),
                 # make_tensor('zero_point_fc1',onnx.TensorProto.INT8,[],(zero_point_fc1)),
                 # make_tensor('bias_fc1',onnx.TensorProto.FLOAT,[64,1],(bias_fc1_val)),
                 # #fc2
                 # make_tensor('scale_bn2',onnx.TensorProto.FLOAT,[64],(scale_bn2)),
                 # make_tensor('bias_bn2',onnx.TensorProto.FLOAT,[64],(bias_bn2)),
                 # make_tensor('running_mean_bn2',onnx.TensorProto.FLOAT,[64],(running_mean_bn2)),
                 # make_tensor('running_var_bn2',onnx.TensorProto.FLOAT,[64],(running_var_bn2)),
                 # make_tensor('scale_l6',onnx.TensorProto.FLOAT,[],(scale_l6)),
                 # make_tensor('zero_point_l6',onnx.TensorProto.UINT8,[],(zero_point_l6)),
                 # make_tensor('scale_fc2',onnx.TensorProto.FLOAT,[],(scale_fc2)),
                 # make_tensor('zero_point_fc2',onnx.TensorProto.INT8,[],(zero_point_fc2)),
                 # make_tensor('bias_fc2',onnx.TensorProto.FLOAT,[32,1],(bias_fc2_val)),
                 # #fc3
                 # make_tensor('scale_bn3',onnx.TensorProto.FLOAT,[32],(scale_bn3)),
                 # make_tensor('bias_bn3',onnx.TensorProto.FLOAT,[32],(bias_bn3)),
                 # make_tensor('running_mean_bn3',onnx.TensorProto.FLOAT,[32],(running_mean_bn3)),
                 # make_tensor('running_var_bn3',onnx.TensorProto.FLOAT,[32],(running_var_bn3)),
                 # make_tensor('scale_l8',onnx.TensorProto.FLOAT,[],(scale_l8)),
                 # make_tensor('zero_point_l8',onnx.TensorProto.UINT8,[],(zero_point_l8)),
                 # make_tensor('scale_fc3',onnx.TensorProto.FLOAT,[],(scale_fc3)),
                 # make_tensor('zero_point_fc3',onnx.TensorProto.INT8,[],(zero_point_fc3)),
                 # make_tensor('bias_fc3',onnx.TensorProto.FLOAT,[1,1],(bias_fc3_val)),
                 # make_tensor('min_fc', onnx.TensorProto.INT8, [], [-127]),
                 # make_tensor('max_fc', onnx.TensorProto.INT8, [], [127]),
                 # #More batchnorm initializers
                 # make_tensor('epsilon_val1', onnx.TensorProto.FLOAT, [], [epsilon_val1]),
                 # make_tensor('epsilon_val2', onnx.TensorProto.FLOAT, [], [epsilon_val2]),
                 # make_tensor('epsilon_val3', onnx.TensorProto.FLOAT, [], [epsilon_val3]),
                 # make_tensor('momentum_val1', onnx.TensorProto.FLOAT, [], [momentum_val1]),
                 # make_tensor('momentum_val2', onnx.TensorProto.FLOAT, [], [momentum_val2]),
                 # make_tensor('momentum_val3', onnx.TensorProto.FLOAT, [], [momentum_val3]),
                 # make_tensor('training_mode_val', onnx.TensorProto.INT32, [], [training_mode_val]),
                 # make_tensor('reshape_dim_1', onnx.TensorProto.INT64, [], [reshape_dim_1_val]),
                 # make_tensor('reshape_dim_2', onnx.TensorProto.INT64, [], [1]),
                 # make_tensor('starts', onnx.TensorProto.INT64, [], [starts_val]),
                 # make_tensor('ends', onnx.TensorProto.INT64, [2,1], [ends_val]),
                 # make_tensor('axes', onnx.TensorProto.INT64, [], [1]), 
                 # starts=[0,0],ends=[20,1],axes=[1]    
#     ]
# )

# ids_fn_verification = qonnx_make_model(scan_lstm_node_graph, producer_name="IDS-Functional-Verification")
# onnx.save(ids_fn_verification, './ids_fn_verification.onnx')

# #Checking the model for any errors
# onnx.checker.check_model(ids_fn_verification)
# # print(ids_fn_verification.graph.value_info)

# #Have to convert the opset version of the graph here because the clip operator in the previous version did not allow for INT8 inputs.
# # It only allowed for FLOAT inputs.
# from onnx import version_converter, helper
# ids_fn_verification_14 = version_converter.convert_version(ids_fn_verification, 14)
# # print(lstm_scan_node_model_14)
# onnx.save(ids_fn_verification_14, './ids_fn_verification.onnx')
# # Defining the values of the varibales to test the execution of the onnx model
# in1_inpa =  np.zeros((20, 1)).astype(np.float32)#'h_t-1'
# in2_inpb = np.zeros((20, 1)).astype(np.float32)#'c_t-1'
# in1_inpa[0] = 0
# in2_inpb[0] = 0
# # in3_scan_input =  np.ones((5, 10, 1)).astype(np.float32)#'X' 10,1 : Because that is the way the shape of the model has been defined.
# in3_scan_input = np.zeros([1,10,1],dtype=np.float32).reshape([1,10,1])
# # in3_scan_input.fill(0.5)

# # in_X = np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.float32).reshape([10,1])
# # in_X = np.array([10,20,30,40,50,60,70,80,90,100],dtype=np.float32).reshape([10,1])
# # in_X = np.zeros([1,10],dtype=np.float32).reshape([10,1])
# in_X = np.ones([1,10],dtype=np.float32).reshape([10,1])
# for i in range(1):
#     in3_scan_input[i] = in_X

# input_dict_ids = {}
# input_dict_ids["inp_a"] = in1_inpa
# input_dict_ids["inp_b"] = in2_inpb
# input_dict_ids["scan_input"] = in3_scan_input

# #Executing the onnx model here.
# sess = rt.InferenceSession(ids_fn_verification_14.SerializeToString())
# output = sess.run(None, input_dict_ids)
# print("Input = ",in3_scan_input[0])
# print("Final Hidden State = ", output)
# print("Final Hidden State = ", output[0].reshape([1,20]))
# print("Final Cell State = ", output[1].reshape([1,20]))
# print("All Hidden States = ", output[2].reshape([25,1,20]))

# transpose_1 = make_node("Transpose",inputs=["h_t_inter"],outputs=["out_a_transpose"],name="transpose_1")#"reshape_dim_1"
# batchnorm_l2 = make_node("BatchNormalization",inputs=["out_a_transpose","scale_bn1","bias_bn1","running_mean_bn1","running_var_bn1"], outputs=["output_bn1"],name="batchnorm_l2")#training_mode="training_mode_val",momentum="momentum_val1",epsilon="epsilon_val1",
# transpose_2 = make_node("Transpose",inputs=["output_bn1"],outputs=["output_bn1_transpose"],name="transpose_2")#,axes=[1]
# relu_act_l3 = make_node("Relu",inputs=["output_bn1_transpose"],outputs=["output_relu1"],name="relu_act_l3")
# quant_linear_l3 = make_node("QuantizeLinear",inputs=["output_relu1","scale_l3","zero_point_l3"],outputs=["relu_ql1"],name="quant_linear_l3")
# dequant_linear_l3 = make_node("DequantizeLinear",inputs=["relu_ql1","scale_l3","zero_point_l3"],outputs=["relu_dql1"],name="dequant_linear_l3")
# quant_linear_w_fc1 = make_node("QuantizeLinear",inputs=["weights_fc1","scale_fc1","zero_point_fc1"],outputs=["weights_fc1_ql"],name="quant_linear_w_fc1")
# clp_w_fc1 = make_node("Clip", inputs=["weights_fc1_ql","min_fc","max_fc"], outputs=["clp_w_fc1"], name="clp_w_fc1")
# dequant_linear_w_fc1 = make_node("DequantizeLinear",inputs=["clp_w_fc1","scale_fc1","zero_point_fc1"],outputs=["weights_fc1_dql"],name="dequant_linear_w_fc1")
# matmul_fc1 = make_node("Gemm", inputs=["weights_fc1_dql","relu_dql1","bias_fc1"], outputs=["out_matmul_fc1"], name="matmul_fc1") #(64,20)x(20,1)

#FC2
# transpose_3 = make_node("Transpose",inputs=["out_matmul_fc1"],outputs=["out_matmul_fc1_transpose"],name="transpose_3")#"reshape_dim_1"
# batchnorm_l5 = make_node("BatchNormalization",inputs=["out_matmul_fc1_transpose","scale_bn2","bias_bn2","running_mean_bn2","running_var_bn2"], outputs=["output_bn2"], name="batchnorm_l5")#epsilon="epsilon_val2",momentum="momentum_val2",training_mode="training_mode_val"
# transpose_4 = make_node("Transpose",inputs=["output_bn2"],outputs=["output_bn2_transpose"],name="transpose_4")#,axes=[1]
# relu_act_l6 = make_node("Relu",inputs=["output_bn2_transpose"],outputs=["output_relu2"],name="relu_act_l6")
# quant_linear_l6 = make_node("QuantizeLinear",inputs=["output_relu2","scale_l6","zero_point_l6"],outputs=["relu_ql2"],name="quant_linear_l6")
# dequant_linear_l6 = make_node("DequantizeLinear",inputs=["relu_ql2","scale_l6","zero_point_l6"],outputs=["relu_dql2"],name="dequant_linear_l6")
# quant_linear_w_fc2 = make_node("QuantizeLinear",inputs=["weights_fc2","scale_fc2","zero_point_fc2"],outputs=["weights_fc2_ql"],name="quant_linear_w_fc2")
# clp_w_fc2 = make_node("Clip", inputs=["weights_fc2_ql","min_fc","max_fc"], outputs=["clp_w_fc2"], name="clp_w_fc2")
# dequant_linear_w_fc2 = make_node("DequantizeLinear",inputs=["clp_w_fc2","scale_fc2","zero_point_fc2"],outputs=["weights_fc2_dql"],name="dequant_linear_w_fc2")
# matmul_fc2 = make_node("Gemm", inputs=["weights_fc2_dql","relu_dql2","bias_fc2"], outputs=["out_matmul_fc2"], name="matmul_fc2") #(32,64)x(64,1) = (31x1)

#FC3
# transpose_5 = make_node("Transpose",inputs=["out_matmul_fc2"],outputs=["out_matmul_fc2_transpose"],name="transpose_5")#"reshape_dim_1"
# batchnorm_l7 = make_node("BatchNormalization",inputs=["out_matmul_fc2_transpose","scale_bn3","bias_bn3","running_mean_bn3","running_var_bn3"], outputs=["output_bn3"],name="batchnorm_l7")#epsilon="epsilon_val3",momentum="momentum_val3",training_mode="training_mode_val",
# transpose_6 = make_node("Transpose",inputs=["output_bn3"],outputs=["output_bn3_transpose"],name="transpose_6")#"reshape_dim_1"
# relu_act_l8 = make_node("Relu",inputs=["output_bn3_transpose"],outputs=["output_relu3"],name="relu_act_l8")
# quant_linear_l8 = make_node("QuantizeLinear",inputs=["output_relu3","scale_l8","zero_point_l8"],outputs=["relu_ql3"],name="quant_linear_l8")
# dequant_linear_l8 = make_node("DequantizeLinear",inputs=["relu_ql3","scale_l8","zero_point_l8"],outputs=["relu_dql3"],name="dequant_linear_l8")
# quant_linear_w_fc3 = make_node("QuantizeLinear",inputs=["weights_fc3","scale_fc3","zero_point_fc3"],outputs=["weights_fc3_ql"],name="quant_linear_w_fc3")
# clp_w_fc3 = make_node("Clip", inputs=["weights_fc3_ql","min_fc","max_fc"], outputs=["clp_w_fc3"], name="clp_w_fc3")
# dequant_linear_w_fc3 = make_node("DequantizeLinear",inputs=["clp_w_fc3","scale_fc3","zero_point_fc3"],outputs=["weights_fc3_dql"],name="dequant_linear_w_fc3")
# matmul_fc3 = make_node("Gemm", inputs=["weights_fc3_dql","relu_dql3","bias_fc3"], outputs=["out_matmul_fc3"], name="matmul_fc3") #(1,32)x(32,1)

#Final relu activation
# relu_l10 = make_node("Relu",inputs=["out_matmul_fc3"],outputs=["out_relu10"],name="relu_l10")
# quant_linear_l10 = make_node("QuantizeLinear",inputs=["out_relu10","scale_l8","zero_point_l8"],outputs=["relu_ql10"],name="quant_linear_l10")
# dequant_linear_l10 = make_node("DequantizeLinear",inputs=["relu_ql10","scale_l8","zero_point_l8"],outputs=["final_output"],name="dequant_linear_l10")

           # transpose_1,
           # batchnorm_l2,
           # transpose_2,
           # relu_act_l3,
           # quant_linear_l3,
           # dequant_linear_l3,
           # quant_linear_w_fc1,
           # clp_w_fc1,
           # dequant_linear_w_fc1,
           # matmul_fc1,
           # transpose_3,
           # batchnorm_l5,
           # transpose_4,
           # relu_act_l6,
           # quant_linear_l6,
           # dequant_linear_l6,
           # quant_linear_w_fc2,
           # clp_w_fc2,
           # dequant_linear_w_fc2,
           # matmul_fc2,
           # transpose_5,
           # batchnorm_l7,
           # transpose_6,
           # relu_act_l8,
           # quant_linear_l8,
           # dequant_linear_l8,
           # quant_linear_w_fc3,
           # clp_w_fc3,
           # dequant_linear_w_fc3,
           # matmul_fc3,
           # relu_l10,
           # quant_linear_l10,
           # dequant_linear_l10

