import onnx
import numpy as np
from qonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper



def lstm_weights_def(input_size=32, hidden_size=128, weights = None):
    initializer = {}
    print(weights[0].name)
    print(weights[0].dims)
    # input gate
    bi_val = numpy_helper.to_array(weights[0]).reshape([input_size*2,1])
    Wi_val = numpy_helper.to_array(weights[1])
    Ui_val = numpy_helper.to_array(weights[2])
    # forget gate
    bf_val = numpy_helper.to_array(weights[3]).reshape([input_size*2,1])
    Wf_val = numpy_helper.to_array(weights[4])
    Uf_val = numpy_helper.to_array(weights[5])
    # cell gate
    bc_val = numpy_helper.to_array(weights[6]).reshape([input_size*2,1])
    Wc_val = numpy_helper.to_array(weights[7])
    Uc_val = numpy_helper.to_array(weights[8])
    # output gate
    bo_val = numpy_helper.to_array(weights[9]).reshape([input_size*2,1])
    Wo_val = numpy_helper.to_array(weights[10])
    Uo_val = numpy_helper.to_array(weights[11])
    # clip


    # set initializer
    # lstm weights
    initializer["W_f"] = make_tensor('W_f',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wf_val))
    initializer["U_f"] = make_tensor('U_f',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Uf_val))
    initializer["b_f"] = make_tensor('b_f',onnx.TensorProto.FLOAT, [hidden_size,1], (bf_val))
    initializer["W_i"] = make_tensor('W_i',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wi_val))
    initializer["U_i"] = make_tensor('U_i',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Ui_val))
    initializer["b_i"] = make_tensor('b_i',onnx.TensorProto.FLOAT, [hidden_size,1], (bi_val))
    initializer["W_c"] = make_tensor('W_c',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wc_val))
    initializer["U_c"] = make_tensor('U_c',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Uc_val))
    initializer["b_c"] = make_tensor('b_c',onnx.TensorProto.FLOAT, [hidden_size,1], (bc_val))
    initializer["W_o"] = make_tensor('W_o',onnx.TensorProto.FLOAT, [hidden_size,input_size], (Wo_val))
    initializer["U_o"] = make_tensor('U_o',onnx.TensorProto.FLOAT, [hidden_size,hidden_size], (Uo_val))
    initializer["b_o"] = make_tensor('b_o',onnx.TensorProto.FLOAT, [hidden_size,1], (bo_val))
    # clip
    initializer["min"] = make_tensor('min', onnx.TensorProto.INT8, [], [-127])
    initializer["max"] = make_tensor('max', onnx.TensorProto.INT8, [], [127])
    initializer["min_6b"] = make_tensor('min_6b', onnx.TensorProto.INT8, [], [-31])
    initializer["min_6b_2"] = make_tensor('min_6b_2', onnx.TensorProto.INT8, [], [-32])
    initializer["max_6b"] = make_tensor('max_6b', onnx.TensorProto.INT8, [], [31])
    initializer["min_8b_unsigned"] = make_tensor('min_8b_unsigned', onnx.TensorProto.UINT8, [], [0])
    initializer["max_8b_unsigned"] = make_tensor('max_8b_unsigned', onnx.TensorProto.UINT8, [], [255])
    initializer["min_6b_unsigned"] = make_tensor('min_6b_unsigned', onnx.TensorProto.UINT8, [], [0])
    initializer["max_6b_unsigned"] = make_tensor('max_6b_unsigned', onnx.TensorProto.UINT8, [], [63])
    # scale and zero point for quantization
    initializer["zero_point_all"] =  make_tensor('zero_point_all',onnx.TensorProto.INT8,[],[0])
    initializer["zero_point_unsigned"] = make_tensor('zero_point_unsigned',onnx.TensorProto.UINT8,[],[0])


    initializer["scale_1"] = make_tensor('scale_1',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[25]))])
    initializer["scale_2"] = make_tensor('scale_2',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[33]))])
    initializer["scale_3"] = make_tensor('scale_3',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[26]))])
    initializer["scale_4"] = make_tensor('scale_4',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[31]))])
    initializer["scale_5"] = make_tensor('scale_5',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[27]))])
    initializer["scale_6"] = make_tensor('scale_6',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[32]))])
    initializer["scale_7"] = make_tensor('scale_7',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[28]))])
    initializer["scale_8"] = make_tensor('scale_8',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[29]))])
    initializer["scale_9"] = make_tensor('scale_9',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[23]))])
    initializer["scale_10"] = make_tensor('scale_10',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[34]))])
    initializer["scale_11"] = make_tensor('scale_11',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[13]))])
    
    initializer["scale_i"] = make_tensor('scale_i',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[16]))])
    initializer["scale_c"] = make_tensor('scale_c',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[19]))])
    initializer["scale_o"] = make_tensor('scale_o',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[20]))])
    initializer["scale_f"] = make_tensor('scale_f',onnx.TensorProto.FLOAT, [],[float(numpy_helper.to_array(weights[21]))])
   
    # Different scales for different quantization nodes
    out_initializers = [initializer["W_f"], initializer["W_i"], initializer["W_c"], initializer["W_o"],
                        initializer["U_f"], initializer["U_i"], initializer["U_c"], initializer["U_o"],
                        initializer["b_f"], initializer["b_i"], initializer["b_c"], initializer["b_o"],
                        initializer["min"], initializer["max"], initializer["min_6b"], initializer["max_6b"],
                        initializer["min_6b_2"], initializer["max_6b"], initializer["min_8b_unsigned"], initializer["max_8b_unsigned"],
                        initializer["min_6b_unsigned"], initializer["max_6b_unsigned"], initializer["zero_point_all"], initializer["zero_point_unsigned"],
                        initializer["scale_1"], initializer["scale_2"], initializer["scale_3"], initializer["scale_4"],
                        initializer["scale_5"], initializer["scale_6"], initializer["scale_7"], initializer["scale_8"], initializer["scale_9"], initializer["scale_10"], initializer["scale_11"],
                        initializer["scale_i"], initializer["scale_c"], initializer["scale_o"], initializer["scale_f"]]
    return out_initializers

def lstm_nodes_def(input_size=32, hidden_size=128):
    nodes = {}

    nodes["inp2_m2"] = make_tensor_value_info("h_t-1",onnx.TensorProto.FLOAT, [hidden_size,1])
    
    nodes["inp2_elm1"] = make_tensor_value_info("c_t-1", onnx.TensorProto.FLOAT, [hidden_size,1])
    nodes["inp2_m1"] = make_tensor_value_info("X",onnx.TensorProto.FLOAT, [input_size,1])


    nodes["out_cell_state"] = make_tensor_value_info("c_t_out", onnx.TensorProto.FLOAT, [hidden_size,1])
    nodes["out_hidden_state_float"] = make_tensor_value_info("dql_hidden_out", onnx.TensorProto.FLOAT, [hidden_size,1])  
    nodes["out_sigmoid"] = make_tensor_value_info("out_sigmoid", onnx.TensorProto.FLOAT, [1,1])
    nodes["out_matmul_fc3"] = make_tensor_value_info("out_matmul_fc3",onnx.TensorProto.FLOAT, [5,1])


    #Also quantizing the hidden_state and the cell_state for quantizing the input graph more efficiently 
    nodes["ql_input"] = make_node("QuantizeLinear", inputs=["X","scale_11","zero_point_all"], outputs=["ql_input_out"],name="ql_input")
    # clp_input_8b = make_node("Clip", inputs=["ql_input_out","min_8b_unsigned","max_8b_unsigned"], outputs=["ql_input_out_clip"], name="clp_input_8b")
    nodes["dql_input"] = make_node("DequantizeLinear", inputs=["ql_input_out", 'scale_11', "zero_point_all"], outputs=["dql_input_out"],name="dql_input")

    nodes["input_ql_hidden_state"] = make_node("QuantizeLinear", inputs=["h_t-1","scale_11","zero_point_all"], outputs=["input_ql_hidden_out"],name="input_ql_hidden_state")
    # clp_hidden_8b = make_node("Clip", inputs=["input_ql_hidden_out","min_8b_unsigned","max_8b_unsigned"], outputs=["input_ql_hidden_out_clip"], name="clp_hidden_8b")
    nodes["input_dql_hidden_state"] = make_node("DequantizeLinear", inputs=["input_ql_hidden_out", 'scale_11', "zero_point_all"], outputs=["input_dql_hidden_out"],name="input_dql_hidden_state")

    nodes["ql_cell_state"] = make_node("QuantizeLinear", inputs=["c_t-1","scale_9","zero_point_all"], outputs=["ql_cell_out"],name="ql_cell_state")
    nodes["clp_cell_6b"] = make_node("Clip", inputs=["ql_cell_out","min_6b_2","max_6b"], outputs=["clp_cell_6b"], name="clp_cell_6b")
    nodes["dql_cell_state"] = make_node("DequantizeLinear", inputs=["clp_cell_6b", 'scale_9', "zero_point_all"], outputs=["dql_cell_out"],name="dql_cell_state")

    #Pushing the weights quantisation in the scan node.
    nodes["ql_w1"] = make_node("QuantizeLinear", inputs=["W_f","scale_f","zero_point_all"], outputs=["ql_wf_out"], name="ql_w1")
    nodes["clp_w1"] = make_node("Clip", inputs=["ql_wf_out","min","max"], outputs=["clp_wf"], name="clp_w1")
    nodes["dql_w1"] = make_node("DequantizeLinear", inputs=["clp_wf","scale_f","zero_point_all"], outputs=["dql_wf_out"], name="dql_w1")

    nodes["ql_w2"] = make_node("QuantizeLinear", inputs=["W_i","scale_i","zero_point_all"], outputs=["ql_wi_out"], name="ql_w2")
    nodes["clp_w2"] = make_node("Clip", inputs=["ql_wi_out","min","max"], outputs=["clp_wi"], name="clp_w2")
    nodes["dql_w2"] = make_node("DequantizeLinear", inputs=["clp_wi","scale_i","zero_point_all"], outputs=["dql_wi_out"], name="dql_w2")

    nodes["ql_w3"] = make_node("QuantizeLinear", inputs=["W_c","scale_c","zero_point_all"], outputs=["ql_wc_out"], name="ql_w3")
    nodes["clp_w3"] = make_node("Clip", inputs=["ql_wc_out","min","max"], outputs=["clp_wc"], name="clp_w3")
    nodes["dql_w3"] = make_node("DequantizeLinear", inputs=["clp_wc","scale_c","zero_point_all"], outputs=["dql_wc_out"], name="dql_w3")

    nodes["ql_w4"] = make_node("QuantizeLinear", inputs=["W_o","scale_o","zero_point_all"], outputs=["ql_wo_out"], name="ql_w4")
    nodes["clp_w4"] = make_node("Clip", inputs=["ql_wo_out","min","max"], outputs=["clp_wo"], name="clp_w4")
    nodes["dql_w4"] = make_node("DequantizeLinear", inputs=["clp_wo","scale_o","zero_point_all"], outputs=["dql_wo_out"], name="dql_w4")

    #These are the quantizations for the recurrence weight matrices.
    nodes["ql_u1"] = make_node("QuantizeLinear", inputs=["U_f","scale_f","zero_point_all"], outputs=["ql_uf_out"], name="ql_u1")
    nodes["clp_u1"] = make_node("Clip", inputs=["ql_uf_out","min","max"], outputs=["clp_uf"], name="clp_u1")
    nodes["dql_u1"] = make_node("DequantizeLinear", inputs=["clp_uf","scale_f","zero_point_all"], outputs=["dql_uf_out"], name="dql_u1")

    nodes["ql_u2"] = make_node("QuantizeLinear", inputs=["U_i","scale_i","zero_point_all"], outputs=["ql_ui_out"], name="ql_u2")
    nodes["clp_u2"] = make_node("Clip", inputs=["ql_ui_out","min","max"], outputs=["clp_ui"], name="clp_u2")
    nodes["dql_u2"] = make_node("DequantizeLinear", inputs=["clp_ui","scale_i","zero_point_all"], outputs=["dql_ui_out"], name="dql_u2")

    nodes["ql_u3"] = make_node("QuantizeLinear", inputs=["U_c","scale_c","zero_point_all"], outputs=["ql_uc_out"], name="ql_u3")
    nodes["clp_u3"] = make_node("Clip", inputs=["ql_uc_out","min","max"], outputs=["clp_uc"], name="clp_u3")
    nodes["dql_u3"] = make_node("DequantizeLinear", inputs=["clp_uc","scale_c","zero_point_all"], outputs=["dql_uc_out"], name="dql_u3")

    nodes["ql_u4"] = make_node("QuantizeLinear", inputs=["U_o","scale_o","zero_point_all"], outputs=["ql_uo_out"], name="ql_u4")
    nodes["clp_u4"] = make_node("Clip", inputs=["ql_uo_out","min","max"], outputs=["clp_uo"], name="clp_u4")
    nodes["dql_u4"] = make_node("DequantizeLinear", inputs=["clp_uo","scale_o","zero_point_all"], outputs=["dql_uo_out"], name="dql_u4")

    #1st Equation : Forget gate
    nodes["mul_node1_e1"] = make_node("MatMul", inputs=["dql_wf_out","dql_input_out"], outputs=["out_m1_e1"], name="mul_node1_e1") #As io_quant was none during training. Hence setting the input as original input. We know it's UINT8 so we set that in FINN.
    nodes["mul_node2_e1"] = make_node("MatMul", inputs=["dql_uf_out","input_dql_hidden_out"], outputs=["out_m2_e1"],name="mul_node2_e1")
    nodes["add_node1_e1"] = make_node("Add", inputs=["out_m1_e1","out_m2_e1"], outputs=["out_add1_e1"],name="add_node1_e1")
    nodes["add_node2_e1"] = make_node("Add", inputs=["out_add1_e1","b_f"], outputs=["f_t_ba"],name="add_node2_e1")
    nodes["quant_linear1_e1"] = make_node("QuantizeLinear", inputs=["f_t_ba","scale_3","zero_point_all"], outputs=["f_t_ql1"],name="quant_linear1_e1")
    nodes["clp1_e1"] = make_node("Clip", inputs=["f_t_ql1","min_6b","max_6b"], outputs=["clp_f_t_ql1"], name="clp1_e1")
    nodes["dequant_linear1_e1"] = make_node("DequantizeLinear", inputs=["clp_f_t_ql1", "scale_3", "zero_point_all"], outputs=["f_t_dql1"], name="dequant_linear1_e1")
    nodes["sig_f_e1"] = make_node("Sigmoid", inputs=["f_t_dql1"], outputs=["f_t"],name="sig_f_e1")
    nodes["quant_linear2_e1"] = make_node("QuantizeLinear", inputs=["f_t","scale_4","zero_point_unsigned"], outputs=["f_t_ql2"],name="quant_linear2_e1")
    nodes["clp2_e1"] = make_node("Clip", inputs=["f_t_ql2","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_f_t_ql2"], name="clp2_e1")
    nodes["dequant_linear2_e1"] = make_node("DequantizeLinear", inputs=["clp_f_t_ql2", "scale_4", "zero_point_unsigned"], outputs=["f_t_dql2"], name="dequant_linear2_e1")

    #2nd Equation : Input gate
    nodes["mul_node1_e2"] = make_node("MatMul", inputs=["dql_wi_out","dql_input_out"], outputs=["out_m1_e2"], name="mul_node1_e2")
    nodes["mul_node2_e2"] = make_node("MatMul", inputs=["dql_ui_out","input_dql_hidden_out"], outputs=["out_m2_e2"],name="mul_node2_e2")
    nodes["add_node1_e2"] = make_node("Add", inputs=["out_m1_e2","out_m2_e2"], outputs=["out_add1_e2"],name="add_node1_e2")
    nodes["add_node2_e2"] = make_node("Add", inputs=["out_add1_e2","b_i"], outputs=["i_t_ba"],name="add_node2_e2")
    nodes["quant_linear1_e2"] = make_node("QuantizeLinear", inputs=["i_t_ba","scale_1","zero_point_all"], outputs=["i_t_ql1"],name="quant_linear1_e2")
    nodes["clp1_e2"] = make_node("Clip", inputs=["i_t_ql1","min_6b","max_6b"], outputs=["clp_i_t_ql1"], name="clp1_e2")
    nodes["dequant_linear1_e2"] = make_node("DequantizeLinear", inputs=["clp_i_t_ql1","scale_1", "zero_point_all"], outputs=["i_t_dql1"], name="dequant_linear1_e2")
    nodes["sig_i_e2"] = make_node("Sigmoid", inputs=["i_t_dql1"], outputs=["i_t"],name="sig_i_e2")
    nodes["quant_linear2_e2"] = make_node("QuantizeLinear", inputs=["i_t","scale_2","zero_point_unsigned"], outputs=["i_t_ql2"],name="quant_linear2_e2")
    nodes["clp2_e2"] = make_node("Clip", inputs=["i_t_ql2","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_i_t_ql2"], name="clp2_e2")
    nodes["dequant_linear2_e2"] = make_node("DequantizeLinear", inputs=["clp_i_t_ql2", "scale_2", "zero_point_unsigned"], outputs=["i_t_dql2"], name="dequant_linear2_e2")

    #3rd Equation : Output gate
    nodes["mul_node1_e3"] = make_node("MatMul", inputs=["dql_wo_out","dql_input_out"], outputs=["out_m1_e3"], name="mul_node1_e3")
    nodes["mul_node2_e3"] = make_node("MatMul", inputs=["dql_uo_out","input_dql_hidden_out"], outputs=["out_m2_e3"],name="mul_node2_e3")
    nodes["add_node1_e3"] = make_node("Add", inputs=["out_m1_e3","out_m2_e3"], outputs=["out_add1_e3"],name="add_node1_e3")
    nodes["add_node2_e3"] = make_node("Add", inputs=["out_add1_e3","b_o"], outputs=["o_t_ba"],name="add_node2_e3" )
    nodes["quant_linear1_e3"] = make_node("QuantizeLinear", inputs=["o_t_ba","scale_7","zero_point_all"], outputs=["o_t_ql1"],name="quant_linear_e3")
    nodes["clp1_e3"] = make_node("Clip", inputs=["o_t_ql1","min_6b","max_6b"], outputs=["clp_o_t_ql1"], name="clp1_e3")
    nodes["dequant_linear1_e3"] = make_node("DequantizeLinear", inputs=["clp_o_t_ql1","scale_7", "zero_point_all"], outputs=["o_t_dql1"], name="dequant_linear1_e3")
    nodes["sig_o_e3"] = make_node("Sigmoid", inputs=["o_t_dql1"], outputs=["o_t"],name="sig_o_e3")
    nodes["quant_linear2_e3"] = make_node("QuantizeLinear", inputs=["o_t","scale_8","zero_point_unsigned"], outputs=["o_t_ql2"],name="quant_linear2_e3")
    nodes["clp2_e3"] = make_node("Clip", inputs=["o_t_ql2","min_6b_unsigned","max_6b_unsigned"], outputs=["clp_o_t_ql2"], name="clp2_e3")
    nodes["dequant_linear2_e3"] = make_node("DequantizeLinear", inputs=["clp_o_t_ql2", "scale_8", "zero_point_unsigned"], outputs=["o_t_dql2"], name="dequant_linear2_e3")

    #4th Equation : Cell gate
    nodes["mul_node1_e4"] = make_node("MatMul", inputs=["dql_wc_out","dql_input_out"], outputs=["out_m1_e4"], name="mul_node1_e4")
    nodes["mul_node2_e4"] = make_node("MatMul", inputs=["dql_uc_out","input_dql_hidden_out"], outputs=["out_m2_e4"],name="mul_node2_e4")
    nodes["add_node1_e4"] = make_node("Add", inputs=["out_m1_e4","out_m2_e4"], outputs=["out_add1_e4"],name="add_node1_e4")
    nodes["add_node2_e4"] = make_node("Add", inputs=["out_add1_e4","b_c"], outputs=["c_t_ba"],name="add_node2_e4")
    nodes["quant_linear1_e4"] = make_node("QuantizeLinear", inputs=["c_t_ba","scale_5","zero_point_all"], outputs=["c_t_ql1"],name="quant_linear1_e4")
    nodes["clp1_e4"] = make_node("Clip", inputs=["c_t_ql1","min_6b","max_6b"], outputs=["clp_c_t_ql1"], name="clp1_e4")
    nodes["dequant_linear1_e4"] = make_node("DequantizeLinear", inputs=["clp_c_t_ql1","scale_5", "zero_point_all"], outputs=["c_t_dql1"], name="dequant_linear1_e4")
    nodes["tanh_c_e4"] = make_node("Tanh", inputs=["c_t_dql1"], outputs=["c_t_partial"],name="tanh_c_e4")
    nodes["quant_linear2_e4"] = make_node("QuantizeLinear", inputs=["c_t_partial","scale_6","zero_point_all"], outputs=["c_t_ql2"],name="quant_linear2_e4")
    nodes["clp2_e4"] = make_node("Clip", inputs=["c_t_ql2","min_6b","max_6b"], outputs=["clp_c_t_ql2"], name="clp2_e4")
    nodes["dequant_linear2_e4"] = make_node("DequantizeLinear", inputs=["clp_c_t_ql2", "scale_6", "zero_point_all"], outputs=["c_t_dql2"], name="dequant_linear2_e4")

    #5th Equation : Cell state compute
    nodes["el_mul_node1_e5"] = make_node("Mul", inputs=["f_t_dql2","dql_cell_out"], outputs=["out_el_mul1_e5"],name="el_mul_node1_e5") #c_t-1
    nodes["quant_linear1_e5"] = make_node("QuantizeLinear", inputs=["out_el_mul1_e5","scale_9","zero_point_all"], outputs=["fifth_ql1"],name="quant_linear1_e5")
    nodes["clp1_e5"] = make_node("Clip", inputs=["fifth_ql1","min_6b","max_6b"], outputs=["clp_fifth_ql1"], name="clp1_e5")
    nodes["dequant_linear1_e5"] = make_node("DequantizeLinear", inputs=["clp_fifth_ql1","scale_9", "zero_point_all"], outputs=["fifth_dql1"], name="dequant_linear1_e5")
    nodes["el_mul_node2_e5"] = make_node("Mul", inputs=["i_t_dql2","c_t_dql2"], outputs=["out_el_mul2_e5"], name="el_mul_node2_e5")
    nodes["quant_linear2_e5"] = make_node("QuantizeLinear", inputs=["out_el_mul2_e5","scale_9","zero_point_all"], outputs=["fifth_ql2"],name="quant_linear2_e5")
    nodes["clp2_e5"] = make_node("Clip", inputs=["fifth_ql2","min_6b","max_6b"], outputs=["clp_fifth_ql2"], name="clp2_e5")
    nodes["dequant_linear2_e5"] = make_node("DequantizeLinear", inputs=["clp_fifth_ql2","scale_9", "zero_point_all"], outputs=["fifth_dql2"], name="dequant_linear2_e5")
    nodes["out_add1_e5"] = make_node("Add", inputs=["fifth_dql1","fifth_dql2"], outputs=["c_t"], name="out_add1_e5")
    #Branch that gives the output
    nodes["quant_linear3_e5"] = make_node("QuantizeLinear", inputs=["c_t","scale_9","zero_point_all"], outputs=["h_t_ql"], name="quant_linear3_e5")
    nodes["clp3_e5"] = make_node("Clip", inputs=["h_t_ql","min_6b","max_6b"], outputs=["clp_h_t_ql"], name="clp3_e5")
    nodes["dequant_linear3_e5"] = make_node("DequantizeLinear", inputs=["clp_h_t_ql","scale_9","zero_point_all"], outputs=["c_t_out"], name="dequant_linear3_e5")
    #Branch that carries it forward
    nodes["quant_linear3_e5_v2"] = make_node("QuantizeLinear", inputs=["c_t","scale_9","zero_point_all"], outputs=["c_t_carry_ql"], name="quant_linear3_e5_v2")
    nodes["clp3_e5_v2"] = make_node("Clip", inputs=["c_t_carry_ql","min_6b","max_6b"], outputs=["clp_c_t_carry_ql"], name="clp3_e5_v2")
    nodes["dequant_linear3_e5_v2"] = make_node("DequantizeLinear", inputs=["clp_c_t_carry_ql","scale_9","zero_point_all"], outputs=["c_t_carry_dql"], name="dequant_linear3_e5_v2")


    #6th Equation : Hidden state compute
    nodes["tanh_node_e6"] = make_node("Tanh", inputs=["c_t_carry_dql"], outputs=["out_tanh_e6"], name="tanh_node_e6") 
    nodes["quant_linear1_e6"] = make_node("QuantizeLinear", inputs=["out_tanh_e6","scale_10","zero_point_all"], outputs=["sixth_ql1"], name="quant_linear1_e6")
    nodes["clp1_e6"] = make_node("Clip", inputs=["sixth_ql1","min_6b","max_6b"], outputs=["clp_sixth_ql1"], name="clp1_e6")
    nodes["dequant_linear1_e6"] = make_node("DequantizeLinear", inputs=["clp_sixth_ql1","scale_10","zero_point_all"], outputs=["sixth_dql1"], name="dequant_linear1_e6")
    nodes["el_mul_node1_e6"] = make_node("Mul", inputs=["sixth_dql1","o_t_dql2"], outputs=["h_t_inter"], name="el_mul_node1_e6")#h_t_inter : Curent Hidden State output
    #Branch that gives the output
    nodes["ql_hidden_state"] = make_node("QuantizeLinear", inputs=["h_t_inter","scale_11","zero_point_all"], outputs=["ql_hidden_out"],name="ql_hidden_state")
    nodes["dql_hidden_state"] = make_node("DequantizeLinear", inputs=["ql_hidden_out", 'scale_11', "zero_point_all"], outputs=["dql_hidden_out"],name="dql_hidden_state")
    #Branch that carries it forward
    nodes["ql_hidden_state_v2"] = make_node("QuantizeLinear", inputs=["h_t_inter","scale_11","zero_point_all"], outputs=["ql_hidden_out_carry"],name="ql_hidden_state_v2")
    nodes["dql_hidden_state_v2"] = make_node("DequantizeLinear", inputs=["ql_hidden_out_carry", 'scale_11', "zero_point_all"], outputs=["dql_hidden_out_carry"],name="dql_hidden_state_v2")
    nodes["dql_hidden_out_carry"] = make_tensor_value_info("dql_hidden_out_carry", onnx.TensorProto.FLOAT, [hidden_size,1])

    return nodes

def lstm_values_def(input_size=32, hidden_size=128):
    values = [
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
            
        ]
    return values

def lstm_graph_def(input_size=32, hidden_size=128, weights = None, save_path="lstm_qcdq.onnx"):
    lstm_nodes = lstm_nodes_def(input_size=input_size, hidden_size=hidden_size)
    nodes=[  
    	   lstm_nodes["ql_input"],
    	   lstm_nodes["dql_input"],		
    	   lstm_nodes["input_ql_hidden_state"],
    	   lstm_nodes["input_dql_hidden_state"],		
           lstm_nodes["ql_cell_state"],
           lstm_nodes["clp_cell_6b"],
           lstm_nodes["dql_cell_state"],
           lstm_nodes["ql_w1"],
           lstm_nodes["clp_w1"], 
           lstm_nodes["dql_w1"],
           lstm_nodes["ql_w2"],
           lstm_nodes["clp_w2"], 
           lstm_nodes["dql_w2"],
           lstm_nodes["ql_w3"],
           lstm_nodes["clp_w3"], 
           lstm_nodes["dql_w3"],
           lstm_nodes["ql_w4"],
           lstm_nodes["clp_w4"], 
           lstm_nodes["dql_w4"],
           lstm_nodes["ql_u1"],
           lstm_nodes["clp_u1"], 
           lstm_nodes["dql_u1"],
           lstm_nodes["ql_u2"],
           lstm_nodes["clp_u2"],
           lstm_nodes["dql_u2"],    
           lstm_nodes["ql_u3"],
           lstm_nodes["clp_u3"],
           lstm_nodes["dql_u3"],    
           lstm_nodes["ql_u4"],
           lstm_nodes["clp_u4"],
           lstm_nodes["dql_u4"],
           lstm_nodes["mul_node1_e1"],
           lstm_nodes["mul_node2_e1"],
           lstm_nodes["add_node1_e1"],
           lstm_nodes["add_node2_e1"],
           lstm_nodes["quant_linear1_e1"],
           lstm_nodes["clp1_e1"],
           lstm_nodes["dequant_linear1_e1"],
           lstm_nodes["sig_f_e1"],
           lstm_nodes["quant_linear2_e1"],
           lstm_nodes["clp2_e1"],
           lstm_nodes["dequant_linear2_e1"],
           lstm_nodes["mul_node1_e2"],
           lstm_nodes["mul_node2_e2"],
           lstm_nodes["add_node1_e2"],
           lstm_nodes["add_node2_e2"],
           lstm_nodes["quant_linear1_e2"],
           lstm_nodes["clp1_e2"],
           lstm_nodes["dequant_linear1_e2"],
           lstm_nodes["sig_i_e2"],
           lstm_nodes["quant_linear2_e2"],
           lstm_nodes["clp2_e2"],
           lstm_nodes["dequant_linear2_e2"],
           lstm_nodes["mul_node1_e3"], 
           lstm_nodes["mul_node2_e3"],
           lstm_nodes["add_node1_e3"], 
           lstm_nodes["add_node2_e3"],
           lstm_nodes["quant_linear1_e3"],
           lstm_nodes["clp1_e3"],
           lstm_nodes["dequant_linear1_e3"],
           lstm_nodes["sig_o_e3"],
           lstm_nodes["quant_linear2_e3"],
           lstm_nodes["clp2_e3"],
           lstm_nodes["dequant_linear2_e3"],
           lstm_nodes["mul_node1_e4"],
           lstm_nodes["mul_node2_e4"],
           lstm_nodes["add_node1_e4"],
           lstm_nodes["add_node2_e4"],
           lstm_nodes["quant_linear1_e4"],
           lstm_nodes["clp1_e4"],
           lstm_nodes["dequant_linear1_e4"],
           lstm_nodes["tanh_c_e4"],
           lstm_nodes["quant_linear2_e4"],
           lstm_nodes["clp2_e4"],
           lstm_nodes["dequant_linear2_e4"],
           lstm_nodes["el_mul_node1_e5"],
           lstm_nodes["quant_linear1_e5"],
           lstm_nodes["clp1_e5"],
           lstm_nodes["dequant_linear1_e5"],
           lstm_nodes["el_mul_node2_e5"],
           lstm_nodes["quant_linear2_e5"],
           lstm_nodes["clp2_e5"],
           lstm_nodes["dequant_linear2_e5"],
           lstm_nodes["out_add1_e5"],
           lstm_nodes["quant_linear3_e5"],
           lstm_nodes["clp3_e5"],
           lstm_nodes["dequant_linear3_e5"],
           lstm_nodes["quant_linear3_e5_v2"],
           lstm_nodes["clp3_e5_v2"],
           lstm_nodes["dequant_linear3_e5_v2"],
           lstm_nodes["tanh_node_e6"],
           lstm_nodes["quant_linear1_e6"],
           lstm_nodes["clp1_e6"],
           lstm_nodes["dequant_linear1_e6"],
           lstm_nodes["el_mul_node1_e6"],
           lstm_nodes["ql_hidden_state"],
           lstm_nodes["dql_hidden_state"],
           lstm_nodes["ql_hidden_state_v2"],
           lstm_nodes["dql_hidden_state_v2"],
          ]
    value_info = lstm_values_def(input_size=input_size, hidden_size=hidden_size)

    initializer = lstm_weights_def(input_size=input_size, hidden_size=hidden_size, weights=weights)
 
    lstm_scan = make_graph(nodes = nodes,
                            name="lstm_scan",
                            inputs=[lstm_nodes["inp2_m2"],lstm_nodes["inp2_elm1"],lstm_nodes["inp2_m1"]],
                            outputs=[lstm_nodes["dql_hidden_out_carry"],lstm_nodes["out_hidden_state_float"],lstm_nodes["out_cell_state"]],
                            value_info=value_info,
                            initializer=initializer
                           )
    onnx_model = qonnx_make_model(lstm_scan, producer_name="QuantizeLSTM_scan")
    onnx.save(onnx_model, save_path)