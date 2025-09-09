import onnx
import numpy as np
from mqonnx.util.basic import qonnx_make_model
import onnxruntime as rt
from onnx.helper import make_tensor_value_info, make_node, make_graph, make_model, make_tensor
from onnx import numpy_helper
import os
from onnx import helper, shape_inference

def check_onnx_info(onnx_input_path, log_path):
    onnx_model = onnx.load(onnx_input_path)
    os.makedirs("temp", exist_ok=True)

    onnx_model = onnx.load(onnx_input_path)
    weights = onnx_model.graph.initializer

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Total weights: {len(weights)}\n")
        for i in range(len(weights)):
            w = numpy_helper.to_array(weights[i])
            f.write(f"{onnx_model.graph.initializer[i].name}\n")
            f.write(f"Shape: {w.shape}\n")
            f.write(f"Values:\n{w}\n, index={i}\n")
            f.write("-------------------------\n")
        f.write("Weights extraction completed.\n")

    print(f"Weights saved to {log_path}")
    return weights