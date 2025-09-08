#Optimisations : 

#1.  Sigmoid and Tanh activations threshold implementation.

#Shashwat torch imports
import torch
from torch import nn

class QuantReluHandler(QuantActBaseHandler):
    """Class for converting a quantized relu operation expressed in the QONNX
    dialect to the FINN ONNX dialect."""

    @classmethod
    def valid_predecessor_op_types(self):
        return [
            "Relu",
            "Selu",
            "Sigmoid", #Shashwat
            "Tanh", #Shashwat
        ]

    if act_node.op_type == "Relu" :
        pass
    elif act_node.op_type == "Sigmoid" :
        q_inst = getCustomOp(self._q_node)
        flag = 0
        # max_range = 0
        # min_range = 0
        narrow = q_inst.get_nodeattr("narrow")
        if narrow:
            num_distinct_values = 2 ** bit_width - 1
            start_val = -(2 ** (bit_width - 1) - 1)
        else:
            num_distinct_values = 2 ** bit_width
            start_val = -(2 ** (bit_width - 1))
        
        if(bit_width == 8):
            max_range = (1/quant_scale) #(This I used for INT8 quantization with precomputed scales)
        if(bit_width == 6 and quant_scale == 0.015187746845185757 ):#Post-activation quantizer scale
            max_range = 31*0.10569057613611221             #Pre-activation quantizer scale
            min_range = -31*0.10569057613611221              #Pre-activation quantizer scale
            # max_range = round(max_range)
            # min_range = round(min_range)
            print("Sigmoid1",quant_scale)
        if(bit_width == 6 and quant_scale == 0.015358800999820232): #Post-activation quantizer scale
            max_range = 31*0.11701396107673645               #Pre-activation quantizer scale
            min_range = -31*0.11701396107673645             #Pre-activation quantizer scale
            # max_range = round(max_range)
            # min_range = round(min_range)
            print("Sigmoid2",quant_scale)
        if(bit_width == 6 and quant_scale == 0.014742395840585232): #Post-activation quantizer scale
            max_range = 31*0.09088636934757233            #Pre-activation quantizer scale
            min_range = -31*0.09088636934757233             #Pre-activation quantizer scale
            # max_range = round(max_range)
            # min_range = round(min_range)
            print("Sigmoid3",quant_scale)   

        # max_range = round(max_range)
        #Defining inputs that the sigmoid activation will see
        if(max_range == 128 and bit_width == 8):
            int_input_range = np.linspace(-1.002, 0.999, num=255)#Had to set this to -0.502 because the first input that the MT node got was -0.5019 which was less than -0.501 that I had set earlier.
        if(max_range == 255 and bit_width == 8):
            int_input_range = np.linspace(-0.502, 0.499, num=255)
        if(bit_width == 6):
            int_input_range = np.linspace(min_range,max_range,num=63) #63 thresholds for 6-bit quantization

        int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
        sigmoid_out = nn.Sigmoid()              #defining sigmoid activation
        output = sigmoid_out(int_input_range)   #output of the activation function)

        #QuantizeLinear operation
        zero_point = 0 #For 8 bit activation Since the output is unsigned INT
        output = torch.round(output/quant_scale)
        #Factoring in the zero point
        output = output + zero_point 

        #Need to clip the outputs here to repliacte the fucnctioning of the Quant Node.
        #Clamping operation
        min_int_val = 0
        max_int_val = 63
        output = np.where(output > max_int_val,max_int_val, output)
        output = np.where(output < min_int_val,min_int_val, output)

        unique_input_indices = np.unique(output, return_index=True)[1] #[1:] #These are inputs which cause a change in the levels of the integer outputs from the quantization function
        unique_inputs = np.zeros(len(unique_input_indices))
        unique_inputs = int_input_range[unique_input_indices]#[1:] #Did not need to use [1:] as I am computing thresholds for 255 inputs only #identifying these inputs from the input_range array and ignoring the first threshold
        acivation_bit_width = bit_width
        num_thresholds = int(2 ** acivation_bit_width - 1)
        thresholds = np.zeros(num_thresholds)

        threshold_index = 0
        output_index = 0
        index_pos = 0
        while threshold_index < 63 and output_index < 62 and index_pos < len(unique_input_indices):
            if output[output_index] == output[output_index+1]: 
                output_index += 1
            elif output[output_index] != output[output_index+1]:
                if(index_pos == 0):
                    diff = output[0]
                    # index_pos = index_pos+1
                    # continue
                elif(index_pos != 0):
                    diff = output[output_index+1] - output[output_index] # Calculating number of required repeats
                while diff > 0:  #Copying repeats into the threshold thershold matrix       
                    thresholds[threshold_index] = unique_inputs[index_pos] 
                    threshold_index += 1
                    diff -= 1
                output_index += 1
                index_pos += 1
        
        for i in range(threshold_index, 63):
            if(index_pos != len(unique_input_indices)):
                thresholds[i] = (int_input_range[unique_input_indices[index_pos]])
            else:
                thresholds[i] = max_range+1 

    elif act_node.op_type == "Tanh" :
        q_inst = getCustomOp(self._q_node)
        narrow = q_inst.get_nodeattr("narrow")
        if narrow:
            num_distinct_values = 2 ** bit_width - 1
            start_val = -(2 ** (bit_width - 1) - 1)
        else:
            num_distinct_values = 2 ** bit_width
            start_val = -(2 ** (bit_width - 1))
        
        if(bit_width == 8):
            max_range = (1/quant_scale)
            max_range = round(max_range)
        if(bit_width==6 and quant_scale == 0.03136685863137245):#Post-activation quantizer scale
            max_range = 31*0.0793924629688263   #Pre-activation quantizer scale
            min_range = -31*0.0793924629688263  #Pre-activation quantizer scale
            # max_range = round(max_range)
            # min_range = round(min_range)
            print("Tanh1",quant_scale)
        if(bit_width==6 and quant_scale == 0.02342350408434868): ##Post-activation quantizer scale
            max_range = 31* 0.029902491718530655                         #Pre-activation quantizer scale
            min_range = -31*0.029902491718530655                      #Pre-activation quantizer scale
            # max_range = round(max_range)
            # min_range = round(min_range)
            print("Tanh2",quant_scale)

        if(bit_width == 8 and max_range == 128):
            int_input_range = np.linspace(-1.002, 0.999, num=255)#Had to set this to -0.502 because the first input that the MT node got was -0.5019 which was less than -0.501 that I had set earlier.
        if(bit_width == 8 and max_range == 255):
            int_input_range = np.linspace(-0.502, 0.499, num=255)
        if(bit_width == 6):
            int_input_range = np.linspace(min_range,max_range,num=63) #63 thresholds for 6-bit quantization

            
        #int_input_range = np.linspace(-0.502, 0.499, num=255)#Had to set this to -0.502 because the first input that the MT node got was -0.5019 which was less than -0.501 that I had set earlier. Similariry had to set this from 0.498 to 0.499 as it got input 0.498012 which was greater than 0.498            int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
        int_input_range = torch.from_numpy(int_input_range) #conversion to torch tensor
        tanh_out = nn.Tanh()              #defining sigmoid activation
        output = tanh_out(int_input_range)   #output of the activation function

        #QuantizeLinear operation
        # zero_point = 128 #This has to be 128 for Tanh with the range [-1,1] for 8-bit quantization
        zero_point = 0 #This has to be 0 for Tanh with 6-bit quantization
        #We need to implement this operation exactly according to the parameters speicified in the QCDQ graph.
        output = torch.round(output/quant_scale)
        #Factoring in the zero point
        output = output + zero_point 

        #Need to clip the outputs here to repliacte the fucnctioning of the Quant Node.
        #Clamping operation
        min_int_val = -31
        max_int_val = 31
        output = np.where(output > max_int_val,max_int_val, output)
        output = np.where(output < min_int_val,min_int_val, output)

        unique_input_indices = np.unique(output, return_index=True)[1] #[1:] #These are inputs which cause a change in the levels of the integer outputs from the quantization function
        unique_inputs = np.zeros(len(unique_input_indices))
        unique_inputs = int_input_range[unique_input_indices]#[1:] #identifying these inputs from the input_range array and ignoring the first threshold
        acivation_bit_width = bit_width
        num_thresholds = int(2 ** acivation_bit_width - 1)
        thresholds = np.zeros(num_thresholds)

        threshold_index = 0
        output_index = 0
        index_pos = 0
        first_value_zero_flag = 0
        if output[0] != 0:
            first_value_zero_flag = 1
        while threshold_index < 63 and output_index < 62 and index_pos < len(unique_input_indices):
            if output[output_index] != output[output_index+1]:
                if(index_pos == 0):
                    diff = output[0]
                elif(index_pos != 0):
                    diff = output[output_index+1] - output[output_index] # Calculating number of required repeats
                while diff > 0:  #Copying repeats into the threshold thershold matrix       
                    thresholds[threshold_index] = unique_inputs[index_pos] 
                    threshold_index += 1
                    diff -= 1
                index_pos += 1
                if first_value_zero_flag == 1:
                    first_value_zero_flag = 0
                    continue
                output_index += 1
            if output[output_index] == output[output_index+1]: 
                output_index += 1
        
        #This for loop copies the last index of the unique index list to fill the remaining spaces in the thresholding matrix.
        for i in range(threshold_index, 63):
            if(index_pos != len(unique_input_indices)):
                thresholds[i] = (int_input_range[unique_input_indices[index_pos]])
            else:
                thresholds[i] = max_range+1

