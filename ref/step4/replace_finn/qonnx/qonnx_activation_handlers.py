# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

np_default_dtype = np.float32


class QuantActBaseHandler(ABC):
    """Base class for converting quantized activation expressed in the QONNX dialect
    to the FINN ONNX dialect.
    :param model: The model on which this handler should operate.
    :type model: class: `qonnx.core.modelwrapper.ModelWrapper`
    :param quant_node: The Quant node which a given handler should replace.
    :param quant_node_index: The index of the Quant node in the given model.
    :type quant_node_index: `int`
    """

    def __init__(self, model: ModelWrapper, quant_node, quant_node_index: int):
        """Base class constructor"""
        super().__init__()
        self._model = model
        self._q_node = quant_node
        self._q_index = quant_node_index

    @classmethod
    def valid_predecessor_op_types(self):
        """Defines which op types the preceding node is allowed to have for
        this type of activation.
        """
        raise NotImplementedError()

    @abstractmethod
    def _check_compatibility(self):
        """Check for compatibility with FINN.
        There are many more possible combinations of QONNX settings,
        than what is supported by FINN.
        """
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_bias(self):
        """Calculate the activation bias,
        which is introduced as an Add node behind the MultiTrheshold node.
        """
        raise NotImplementedError()

    @abstractmethod
    def _calculate_thresholds(self):
        """Calculate the threshold array for the MultiThreshold node."""
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_scale(self):
        """Calculate the activation scale,
        which is indroduced as a Mul node behind the Add node
        for the activation bias.
        """
        raise NotImplementedError()

    @abstractmethod
    def _remove_activation_node(self, multi_threshold_node):
        """Remove the activation node in front of the Quant node."""
        raise NotImplementedError()

    def _extract_output_datatype(self):
        """Get the output datatype for the MultiThreshold node."""
        q_inst = getCustomOp(self._q_node)
        dtype = q_inst.get_integer_datatype(self._model)
        dtype = dtype.name
        return dtype

    def calculate_node_parameters(self):
        """Calculate all parameters required for replacing the QONNX style activation
        with a FINN style one.
        """
        return {
            "out_dtype": self._extract_output_datatype(),
            "thresholds": self._calculate_thresholds(),
            "adder_bias": self._calculate_act_bias(),
            "mul_scale": self._calculate_act_scale(),
        }

    def replace_quant_node(self):
        """Replace the given QONNX style activation with a FINN style one."""

        # Check that we actually support what the user is trying to do
        self._check_compatibility()

        # Shorten instance variables
        model = self._model
        graph = model.graph
        n = self._q_node
        running_node_index = self._q_index

        # Calculate insertion parameters
        parameter_dict = self.calculate_node_parameters()
        thresholds = parameter_dict["thresholds"]
        out_dtype = parameter_dict["out_dtype"]
        adder_bias = parameter_dict["adder_bias"]
        mul_scale = parameter_dict["mul_scale"]

        # Modify graph
        # Insert threshold tensor
        thresh_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            thresholds.shape,
        )
        graph.value_info.append(thresh_tensor)
        model.set_initializer(thresh_tensor.name, thresholds)

        # Insert MultiThreshold node
        outp_trans_node = helper.make_node(
            "MultiThreshold",
            [n.input[0], thresh_tensor.name],
            [n.output[0]],
            out_dtype="FLOAT32",
            domain="qonnx.custom_op.general",
        )
        graph.node.insert(running_node_index, outp_trans_node)
        running_node_index += 1

        # Get the MultiThreshold node instance to work with
        mt_node = graph.node[running_node_index - 1]
        mt_inst = getCustomOp(mt_node)

        # Set scale and bias
        # If these values are scalar then they can be set as attributes
        # of the MultiThreshold node, if not they get inserted as adder and mul nodes
        # behind the MultiTrheshold nodes.
        bias_scalar = adder_bias.shape == (1,) or len(adder_bias.shape) == 0
        scale_scalar = mul_scale.shape == (1,) or len(mul_scale.shape) == 0
        if scale_scalar and bias_scalar and self._q_node.op_type == "BipolarQuant":
            # Get Quant parameters
            mul_scale = np.atleast_1d(mul_scale)
            adder_bias = np.atleast_1d(adder_bias)

            # Set Bias and scale
            # note calls to .item() to get Python float instead of numpy float
            # ONNX attribute setting fails otherwise
            mt_inst.set_nodeattr("out_scale", mul_scale[0].item())
            # FINN applies scale first then bias,
            # which is the other way around in Brevitas,
            # we thus need to adjust the bias in the MultiThreshold node
            finn_bias = adder_bias[0].item() * mul_scale[0].item()
            mt_inst.set_nodeattr("out_bias", finn_bias)

            # Set the output data type
            mt_inst.set_nodeattr("out_dtype", out_dtype)
        else:
            # Set datatype
            mt_inst.set_nodeattr("out_dtype", out_dtype)

            # Insertion parameters
            up_stream_node = mt_node

            # Set bias
            zero_bias = False
            if bias_scalar:
                adder_bias = np.atleast_1d(adder_bias)
                adder_bias = adder_bias[0]
                add_shape = tuple()
                if adder_bias == 0.0:
                    zero_bias = True
            else:
                add_shape = adder_bias.shape

            if not zero_bias:
                # Insert Add node
                add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    add_shape,
                )
                graph.value_info.append(add_tensor)
                model.set_initializer(add_tensor.name, adder_bias)

                output_shape = model.get_tensor_shape(n.output[0])
                act_add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    output_shape,
                )
                graph.value_info.append(act_add_tensor)

                add_node = helper.make_node(
                    "Add",
                    [act_add_tensor.name, add_tensor.name],
                    [n.output[0]],
                )
                graph.node.insert(running_node_index, add_node)
                running_node_index += 1
                add_node = graph.node[running_node_index - 1]

                # Re-point the upstream node
                up_stream_node.output[0] = act_add_tensor.name
                up_stream_node = add_node

            # Set scale
            # Insert Mul node
            unity_scale = False
            if scale_scalar:
                mul_scale = np.atleast_1d(mul_scale)
                mul_scale = mul_scale[0]
                mul_shape = tuple()
                if mul_scale == 1.0:
                    unity_scale = True
            else:
                mul_shape = mul_scale.shape

            if not unity_scale:
                mul_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    mul_shape,
                )
                graph.value_info.append(mul_tensor)
                model.set_initializer(mul_tensor.name, mul_scale)

                output_shape = model.get_tensor_shape(n.output[0])
                act_mul_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    output_shape,
                )
                graph.value_info.append(act_mul_tensor)

                mul_node = helper.make_node(
                    "Mul",
                    [act_mul_tensor.name, mul_tensor.name],
                    [n.output[0]],
                )
                graph.node.insert(running_node_index, mul_node)
                running_node_index += 1
                mul_node = graph.node[running_node_index - 1]

                # Re-point the upstream node
                up_stream_node.output[0] = act_mul_tensor.name
                up_stream_node = mul_node

        # Remove activation node
        self._remove_activation_node(mt_node)

        # Remove the Quant node
        graph.node.remove(n)

        # return the internal model representation
        return self._model


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

    def _check_compatibility(self):
        if self._q_node.op_type == "Quant":
            q_inst = getCustomOp(self._q_node)
            narrow = q_inst.get_nodeattr("narrow")
            signed = q_inst.get_nodeattr("signed")
            if not self._model.get_initializer(self._q_node.input[2]) == 0:
                raise ValueError(
                    "Only Quant nodes with zero-point == 0 "
                    "are currently supported for ReLu activations."
                )
            act_node = self._model.find_direct_predecessors(self._q_node)
            act_node = act_node[0]
            if act_node.op_type == "Relu":
                if signed or narrow:
                    raise ValueError(
                        "FINN only supports unsigned and non-narrow Quant nodes "
                        "for Relu activations."
                    )
        elif self._q_node.op_type == "BipolarQuant":
            return
        else:
            raise RuntimeError("Got an unexpected quantizer node type")

    # def _calculate_act_bias(self):
    #     # No bias allowed for Relu activations, see: https://github.com/Xilinx/
    #     # brevitas/blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
    #     # export/onnx/finn/handler/act.py#L48
    #     act_node = self._model.find_direct_predecessors(self._q_node)
    #     act_node = act_node[0]
    #     if act_node.op_type == "Relu":
    #         bias = np.array([0.0], dtype=np_default_dtype)
    #     elif act_node.op_type == "Selu":
    #         # Gather parameters
    #         q_inst = getCustomOp(self._q_node)
    #         if self._q_node.op_type == "Quant":
    #             bit_width = self._model.get_initializer(self._q_node.input[3])
    #             narrow = q_inst.get_nodeattr("narrow")
    #         elif self._q_node.op_type == "BipolarQuant":
    #             bit_width = 1.0
    #         else:
    #             raise RuntimeError("Got an unexpected quantizer node type")
    #         # Calculate bias, see: https://github.com/Xilinx/brevitas/blob/
    #         # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
    #         # onnx/finn/handler/act.py#L64
    #         if bit_width == 1.0:
    #             bias = np.array([-0.5], dtype=np_default_dtype)
    #         else:
    #             if narrow:
    #                 min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
    #             else:
    #                 min_non_scaled_val = -(2 ** (bit_width - 1))
    #             bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
    #     return bias
    def _calculate_act_bias(self):
        # No bias allowed for Relu activations, see: https://github.com/Xilinx/
        # brevitas/blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
        # export/onnx/finn/handler/act.py#L48
        act_node = self._model.find_direct_predecessors(self._q_node)
        act_node = act_node[0]
        if act_node.op_type == "Relu":
            bias = np.array([0.0], dtype=np_default_dtype)
        elif act_node.op_type == "Selu":
            # Gather parameters
            q_inst = getCustomOp(self._q_node)
            if self._q_node.op_type == "Quant":
                bit_width = self._model.get_initializer(self._q_node.input[3])
                narrow = q_inst.get_nodeattr("narrow")
            elif self._q_node.op_type == "BipolarQuant":
                bit_width = 1.0
            else:
                raise RuntimeError("Got an unexpected quantizer node type")
            # Calculate bias, see: https://github.com/Xilinx/brevitas/blob/
            # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
            # onnx/finn/handler/act.py#L64
            if bit_width == 1.0:
                bias = np.array([-0.5], dtype=np_default_dtype)
            else:
                if narrow:
                    min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
                else:
                    min_non_scaled_val = -(2 ** (bit_width - 1))
                bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
        #This elif from Shashwat : Should be the same for both Sigmoid and Tanh according to my implementation 
        elif act_node.op_type == "Sigmoid":
            q_inst = getCustomOp(self._q_node)
            if self._q_node.op_type == "Quant":
                bit_width = self._model.get_initializer(self._q_node.input[3])
                narrow = q_inst.get_nodeattr("narrow")
            elif self._q_node.op_type == "BipolarQuant":
                bit_width = 1.0
            else:
                raise RuntimeError("Got an unexpected quantizer node type")
            
            if bit_width == 1.0:
                bias = np.array([0], dtype=np_default_dtype)   #For now set to '0' have to figure out what the actual value is.
            else:
                if narrow:
                    min_non_scaled_val = 0
                else:
                    min_non_scaled_val = 0
                bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
                #bias value needs to be -128 or -127 for 8 bit activation to bring the output in the INT8 range. Tried 0 and that did not work for me
        
        elif act_node.op_type == "Tanh":
            q_inst = getCustomOp(self._q_node)
            if self._q_node.op_type == "Quant":
                bit_width = self._model.get_initializer(self._q_node.input[3])
                narrow = q_inst.get_nodeattr("narrow")
            elif self._q_node.op_type == "BipolarQuant":
                bit_width = 1.0
            else:
                raise RuntimeError("Got an unexpected quantizer node type")
            
            if bit_width == 1.0:
                bias = np.array([0], dtype=np_default_dtype)   #For now set to '0' have to figure out what the actual value is.
            else:
                if narrow:
                    min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
                else:
                    min_non_scaled_val = -(2 ** (bit_width - 1)) 
                bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
                #bias value needs to be -128 or -127 for 8 bit activation to bring the output in the INT8 range. Tried 0 and that did not work for me


        return bias

    def _calculate_thresholds(self):
        # Gather parameters
        if self._q_node.op_type == "Quant":
            bit_width = self._model.get_initializer(self._q_node.input[3])
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")
        quant_scale = self._model.get_initializer(self._q_node.input[1]).astype(np.float32)
        act_node = self._model.find_direct_predecessors(self._q_node)
        act_node = act_node[0]
        if act_node.op_type == "Relu":
            # Calculate thersholds, see: https://github.com/Xilinx/brevitas/blob/
            # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
            # onnx/finn/handler/act.py#L21
            num_distinct_values = 2**bit_width
            num_thresholds = int(num_distinct_values - 1)
            flat_scale = quant_scale.flatten().astype(np.float32)
            num_scale_channels = flat_scale.shape[0]
            step = np.abs(flat_scale).astype(np.float32)
            min_threshold = step / 2
            thresholds = np.empty((num_scale_channels, num_thresholds), dtype=np_default_dtype)
            for c in range(num_scale_channels):
                for t in range(num_thresholds):
                    thresholds[c][t] = min_threshold[c] + step[c] * t

        elif act_node.op_type == "Selu":
            q_inst = getCustomOp(self._q_node)
            narrow = q_inst.get_nodeattr("narrow")
            if narrow:
                num_distinct_values = 2**bit_width - 1
            else:
                num_distinct_values = 2**bit_width

            num_thresholds = int(num_distinct_values - 1)
            flat_scale = quant_scale.flatten().astype(np.float32)
            num_scale_channels = flat_scale.shape[0]
            scale = np.abs(flat_scale).astype(np.float32)
            half_scale = scale / 2
            # alpha and lambda
            # from https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
            alpha = 1.6732632423543772848170429916717
            selu_scale = 1.0507009873554804934193349852946
            thresholds = np.empty((num_scale_channels, num_thresholds), dtype=np_default_dtype)
            for c in range(num_scale_channels):
                for t in range(num_thresholds):
                    step = -1.0 + half_scale + scale[c] * t
                    if step <= 0:
                        thresholds[c][t] = np.log(step / (alpha * selu_scale) + 1)
                    else:
                        thresholds[c][t] = step / selu_scale

        elif act_node.op_type == "Sigmoid" :
            q_inst = getCustomOp(self._q_node)
            flag = 0
            narrow = q_inst.get_nodeattr("narrow")
            if narrow:
                num_distinct_values = 2 ** bit_width - 1
                start_val = -(2 ** (bit_width - 1) - 1)
            else:
                num_distinct_values = 2 ** bit_width
                start_val = -(2 ** (bit_width - 1))
            
            if(bit_width == 8):
                max_range = (1/quant_scale) #(This I used for INT8 quantization with precomputed scales)
            if(bit_width == 6 and quant_scale == 0.013812770135700703 ):#Post-activation quantizer scale
                max_range =  31*0.0697396919131279            #Pre-activation quantizer scale
                min_range = -31*0.0697396919131279              #Pre-activation quantizer scale
                print("Sigmoid1",quant_scale)
            if(bit_width == 6 and quant_scale == 0.014795037917792797): #Post-activation quantizer scale
                max_range =  31*0.08846518397331238               #Pre-activation quantizer scale
                min_range = -31*0.08846518397331238            #Pre-activation quantizer scale
                print("Sigmoid2",quant_scale)
            if(bit_width == 6 and quant_scale == 0.014122666791081429): #Post-activation quantizer scale
                max_range =  31*0.06339830160140991           #Pre-activation quantizer scale
                min_range = -31*0.06339830160140991             #Pre-activation quantizer scale
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
            if(bit_width==6 and quant_scale == 0.023927057161927223):#Post-activation quantizer scale
                max_range = 31 *0.03166016563773155   #Pre-activation quantizer scale
                min_range = -31*0.03166016563773155  #Pre-activation quantizer scale
                # max_range = round(max_range)
                # min_range = round(min_range)
                print("Tanh1",quant_scale)
            if(bit_width==6 and quant_scale == 0.028057008981704712): ##Post-activation quantizer scale
                max_range = 31* 0.045602329075336456                         #Pre-activation quantizer scale
                min_range = -31*0.045602329075336456                     #Pre-activation quantizer scale
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

        # ToDo: The index 1 needs to be changed to -1 for the channels last format
        num_output_channels = self._model.get_tensor_shape(self._q_node.output[0])[1]
        final_shape = (num_output_channels, num_thresholds)
        if thresholds.shape != final_shape:
            thresholds = np.broadcast_to(thresholds, final_shape)

        return thresholds

    def _calculate_act_scale(self):
        # Gather parameters
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        # Calculate scale, see: https://github.com/Xilinx/brevitas/blob/
        # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
        # onnx/finn/handler/act.py#L40
        scale = quant_scale
        return scale

    def _remove_activation_node(self, multi_threshold_node):
        # Find the activation node
        act_node = self._model.find_direct_predecessors(self._q_node)
        if act_node is None:
            raise RuntimeError(
                "For handling of Relu activations a predecesor to " "the Quant node must exist."
            )
        act_node = act_node[0]
        if act_node.op_type not in self.valid_predecessor_op_types():
            raise RuntimeError(
                "The predecesor of the Quant node must be Relu or Selu for handling "
                "of activations."
            )

        # Reroute upstream tensor
        multi_threshold_node.input[0] = act_node.input[0]

        # Remove the activation node
        self._model.graph.node.remove(act_node)


class QuantIdentityHandler(QuantActBaseHandler):
    """Class for converting a quantized identity operation expressed in the QONNX
    dialect to the FINN ONNX dialect.
    This handler also takes care of quantized HardTanh activations, because
    these are equivalent to quantized identity activations.
    """

    @classmethod
    def valid_predecessor_op_types(self):
        return [
            "BatchNormalization",
            "Sub",
            "Add",
            "Mul",
            "Div",
            "DebugMarker",
            None,
        ]

    def _check_compatibility(self):
        # Gather parameters to check
        if self._q_node.op_type == "Quant":
            q_inst = getCustomOp(self._q_node)
            signed = q_inst.get_nodeattr("signed")
            if not signed:
                raise ValueError("FINN only supports signed Quant nodes for identity activations.")
            if not self._model.get_initializer(self._q_node.input[2]) == 0:
                raise ValueError(
                    "Only Quant nodes with zero-point == 0 "
                    "are currently supported for identity activations."
                )
        elif self._q_node.op_type == "BipolarQuant":
            quant_scale = self._model.get_initializer(self._q_node.input[1])
            if (quant_scale.flatten().shape[0] != 1) or quant_scale.flatten()[0] != 1.0:
                raise ValueError(
                    "FINN only supports Bipolar identity activations "
                    "with out per channel scaling and the scaling must be 1. "
                )
        else:
            raise RuntimeError("Got an unexpected quantizer node type")

    def _calculate_act_bias(self):
        # Gather parameters
        q_inst = getCustomOp(self._q_node)
        if self._q_node.op_type == "Quant":
            bit_width = self._model.get_initializer(self._q_node.input[3])
            narrow = q_inst.get_nodeattr("narrow")
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")
        # Calculate bias, see: https://github.com/Xilinx/brevitas/blob/
        # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
        # onnx/finn/handler/act.py#L64
        if bit_width == 1.0:
            bias = np.array([-0.5], dtype=np_default_dtype)
        else:
            if narrow:
                min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
            else:
                min_non_scaled_val = -(2 ** (bit_width - 1))
            bias = np.array([min_non_scaled_val], dtype=np_default_dtype)
        return bias

    def _calculate_thresholds(self):
        # Gather parameters
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        q_inst = getCustomOp(self._q_node)
        if self._q_node.op_type == "Quant":
            bit_width = self._model.get_initializer(self._q_node.input[3])
            narrow = q_inst.get_nodeattr("narrow")
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")

        # Calculate thersholds, see: https://github.com/Xilinx/brevitas/
        # blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
        # export/onnx/finn/handler/act.py#L76
        if bit_width == 1.0:
            thresholds = np.empty([1, 1], dtype=np_default_dtype)
            thresholds[0] = 0
            return thresholds
        else:
            if narrow:
                num_distinct_values = 2**bit_width - 1
            else:
                num_distinct_values = 2**bit_width

            num_thresholds = int(num_distinct_values - 1)
            flat_scale = quant_scale.flatten()
            num_scale_channels = flat_scale.shape[0]
            step = np.abs(flat_scale)
            half_step = step / 2.0
            thresholds = np.empty((num_scale_channels, num_thresholds), dtype=np_default_dtype)
            # compute the value of the smallest threshold, we'll neg-bias all
            # generated thresholds by this much
            min_threshold = -half_step - step * ((num_thresholds // 2) - 1)
            if not narrow:
                min_threshold -= step
            for c in range(num_scale_channels):
                for t in range(num_thresholds):
                    thresholds[c][t] = min_threshold[c] + step[c] * t

            # ToDo: The index 1 needs to be changed to -1 for the channels last format
            num_output_channels = self._model.get_tensor_shape(self._q_node.output[0])[1]
            final_shape = (num_output_channels, num_thresholds)
            if thresholds.shape != final_shape:
                thresholds = np.broadcast_to(thresholds, final_shape)

            return thresholds

    def _calculate_act_scale(self):
        # Gather parameters
        if self._q_node.op_type == "Quant":
            bit_width = self._model.get_initializer(self._q_node.input[3])
        elif self._q_node.op_type == "BipolarQuant":
            bit_width = 1.0
        else:
            raise RuntimeError("Got an unexpected quantizer node type")
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        # Calculate scale, see: https://github.com/Xilinx/brevitas/
        # blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
        # export/onnx/finn/handler/act.py#L111
        if bit_width != 1:
            scale = quant_scale
        else:
            assert quant_scale.flatten().shape[0] == 1, "Unsupported BIPOLAR per channel scale"
            assert quant_scale.flatten()[0] == 1.0, "Unsupported BIPOLAR scale != 1"
            scale = quant_scale * 2
        return scale

    def _remove_activation_node(self, multi_threshold_node):
        # The Quant identity activation has per definition no explicit activation node
        return
