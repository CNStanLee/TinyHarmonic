#This file needs to be executed from within the qonnx environment.
# source /home/temporary/Desktop/qlstm_finn/qlstm_sw/qonnx/venv/bin/activate
#This file also contains the code to view whhich scale belongs to which quantizer. That part is commented.

import torch
from brevitas.nn import QuantLSTM,QuantLinear,QuantReLU
import brevitas.nn as bnn
from brevitas.export import export_onnx_qcdq
import numpy as np
import torch.nn as nn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Int4ActPerTensorFloat
from brevitas.quant import Int9ActPerTensorFloat
from brevitas.quant import Uint4ActPerTensorFloat,Uint8ActPerTensorFloat

# Setting seeds for reproducibility
torch.manual_seed(0)

a1 = 8
a2 = 8
a3 = 8
class LSTMIDS(nn.Module):
    def __init__(self):
        super(LSTMIDS, self).__init__()
        # To be uncommented while training an LSTM model.
        #io_quant set to 'Int8ActPerTensorFloat' while exporting the model in ONNX format. It is set to 'None' to test it in this file.
        self.qlstm = bnn.QuantLSTM(input_size=10, hidden_size=20,num_layers=1,batch_first=True,
                weight_bit_width=8,
                io_quant=Int8ActPerTensorFloat,#None,Int8ActPerTensorFloat
                gate_acc_bit_width=6,
                sigmoid_bit_width=6,
                tanh_bit_width=6,
                cell_state_bit_width=6,
                bias_quant=None)#Setting batch_first to "True" changed everything, Need to investigate why it worked.
        self.qfc1 = bnn.QuantLinear(20, 64,bias=True, weight_bit_width=8)
        self.qfc2 = bnn.QuantLinear(64, 32,bias=True, weight_bit_width=8)
        self.qfc3 = bnn.QuantLinear(32, 5,bias=True, weight_bit_width=8)
        self.qrelu = bnn.QuantReLU(bit_width=6)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #h0,c0 need to be zeros before exporting the graph otherwsie get weird patterens in the ONNX graph.
        # Initialize hidden state with zeros
        h0 = torch.zeros(1,1,20)
        # h0 = torch.ones(1,1, 20)
        # h0[0][0][0] = 15
        # Initialize cell state
        c0 = torch.zeros(1,1,20)
        # c0 = torch.ones(1,1, 20) 
        # c0[0][0][0] = 12
        # print(h0)   
        #Start model definition
        out,(hn,cn) = self.qlstm(x,h0,c0)#h0 and c0 are given as inputs like this. Not like in pytorch. 
        out = self.dropout(hn[-1,:,:])#Taking the last hidden state out of the 10 sequences
        out = self.qrelu(out)
        out = self.qfc1(out)
        out = self.qrelu(out)
        out = self.qfc2(out)
        out = self.qrelu(out)
        out = self.qfc3(out)
        #out = self.sigmoid(out)
        return out

model_lstm = LSTMIDS()
path = "./model_29.pt"
i = 1
model_lstm.load_state_dict(torch.load(path),strict=False)#, map_location=device
# Both the below for loops print the parameters of the weights file along with their names.
# for param in model_lstm.parameters():
#     print(param)

print("------------------------------------------------------------------------")
#This loop is used to figure out which quantizer has what scale value. These are later mapped to the ONNX implementation
for name, param in model_lstm.named_parameters():
    # print(i)
    i = i+1
    # print(name)
    # print(param)
# print(f'Weight scale after load_state_dict: {model_lstm.quant_weight().scale}')
# exit()
model_lstm.eval()
model_lstm(torch.randn(10, 1))
export_path = 'acm-transactions-cps-model.onnx'
export_onnx_qcdq(model_lstm,(torch.randn(1, 1, 10)), opset_version=14, export_path=export_path)#(torch.randn(25, 1, 10))
exit()
# 6,11,69,41,36,255,41,36,0,255


in_qcdq_node = np.zeros([1,2,10],dtype=np.float32).reshape([1,2,10])#[batch_size, seq_length, input_length]
# in_qcdq_node[0][0][0] = 0
# in_qcdq_node[0][0][1] = 1
# in_qcdq_node[0][0][2] = 2
# in_qcdq_node[0][0][3] = 3
# in_qcdq_node[0][0][4] = 4
# in_qcdq_node[0][0][5] = 5
# in_qcdq_node[0][0][6] = 6
# in_qcdq_node[0][0][7] = 7
# in_qcdq_node[0][0][8] = 8
# in_qcdq_node[0][0][9] = 9

# in_qcdq_node[0][1][0] = 0
# in_qcdq_node[0][1][1] = 1
# in_qcdq_node[0][1][2] = 2
# in_qcdq_node[0][1][3] = 3
# in_qcdq_node[0][1][4] = 4
# in_qcdq_node[0][1][5] = 5
# in_qcdq_node[0][1][6] = 6
# in_qcdq_node[0][1][7] = 7
# in_qcdq_node[0][1][8] = 8
# in_qcdq_node[0][1][9] = 9


# in_h_t_1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
# in_c_t_1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=np.float32)
# in_h_t_1 = np.array([0,0.8614,0,0,0,0.8614,-0.8614,-0.8614,0,0,0.8332,0,-0.8614,0.8332,0.8614,0.8614,0,0.8614,0,0.8614],dtype=np.float32)#.reshape([20,1])
# in_c_t_1 = np.array([0,1.2486,1.2486,-1.2486,0,1.2486,-1.2486,-1.2486,0,0,1.2486,0,-1.2486,-1.2486,1.2486,1.2486,1.2486,1.2486,1.2486,1.2486],dtype=np.float32)#.reshape([20,1])

# # in_h_t_1[0][0] = 0; in_h_t_1[1][0] = -0.8614; in_h_t_1[2][0] = 0; in_h_t_1[3][0] = 0; in_h_t_1[4][0] = 0; in_h_t_1[5][0] = 0; in_h_t_1[6][0] = -0.8614; in_h_t_1[7][0] = -0.8614; in_h_t_1[8][0] = 0; in_h_t_1[9][0] = 0;
# # in_h_t_1[10][0] = 0.8332; in_h_t_1[11][0] = 0; in_h_t_1[12][0] = -0.4378; in_h_t_1[13][0] = 0; in_h_t_1[14][0] = 0.8332; in_h_t_1[15][0] = 0.8332; in_h_t_1[16][0] = 0; in_h_t_1[17][0] = 0; in_h_t_1[18][0] = 0; in_h_t_1[19][0] = 0;
# # in_c_t_1[0][0] = 0; in_c_t_1[1][0] = -1.2486; in_c_t_1[2][0] = 1.2486; in_c_t_1[3][0] = -1.2486; in_c_t_1[4][0] = 0; in_c_t_1[5][0] = 0; in_c_t_1[6][0] = -1.2486; in_c_t_1[7][0] = -1.2486; in_c_t_1[8][0] = 0; in_c_t_1[9][0] = 0;
# # in_c_t_1[10][0] = 1.2486; in_c_t_1[11][0] = 0; in_c_t_1[12][0] = -1.2486; in_c_t_1[13][0] = -1.2486; in_c_t_1[14][0] = 1.2486; in_c_t_1[15][0] = 1.2486; in_c_t_1[16][0] = 0; in_c_t_1[17][0] = 0; in_c_t_1[18][0] = 1.2486; in_c_t_1[19][0] = 1.2486;

# print(in_h_t_1.shape)
# # in_h_t_1 = np.ones([20,],dtype=np.float32)
# print('Supplied Input = ',in_qcdq_node[0][0][0])
# h0 = torch.from_numpy(in_h_t_1)
# c0 = torch.from_numpy(in_c_t_1) 
input_test = torch.from_numpy(in_qcdq_node)
print("Input_test [0] = ",input_test.data[0][0][0])
output_lstm = model_lstm(input_test)
# output_lstm = model_lstm(input_test,(h0,c0))
# print(type(output_lstm))
print(output_lstm)
# # print(output_lstm1)