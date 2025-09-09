# model_wrapper
original file: /home/changhong/prj/finn/deps/qonnx/src/qonnx/core/modelwrapper.py
new file: ref/step4/replace_finn/modelwrapper.py
- changed functions: set_tensor_datatype, find_upstream, get_node_index
- purpose: allow node find multiple consumer
# qonnx_activation_hanlders
original file: /home/changhong/prj/finn/src/finn/transformation/qonnx/qonnx_activation_handlers.py
new file: ref/step4/replace_finn/qonnx/qonnx_activation_handlers.py
- changed functions: _calculate_act_bias, _calculate_thresholds
- purpose: support activation functions: Sigmoid, Tanh
- to do: obtain the pre-quant scale from the graph instead of constant value.
# absorb.py
original file: /home/changhong/prj/finn/src/finn/transformation/streamline/absorb.py
new file: ref/step4/replace_finn/streamline/absorb.py
- changed functions: AbsorbSignBiasIntoMultiThreshold, AbsorbMulIntoMultiThreshold, FactorOutMulSignMagnitude
- add functions: AbsorbSignBiasIntoMultiThreshold
# reorder.py
original file: /home/changhong/prj/finn/src/finn/transformation/streamline/reorder.py
new file: ref/step4/replace_finn/streamline/reorder.py
- changed functions: MoveScalarMulPastMatMul, MoveScalarLinearPastInvariants, MakeMaxPoolNHWC, MakeScaleResizeNHWC, MoveOpPastFork, MoveTransposePastFork
# round_thresholds.py
original file: /home/changhong/prj/finn/src/finn/transformation/streamline/round_thresholds.py
new file: ref/step4/replace_finn/streamline/round_thresholds.py
- changed functions: RoundAndClipThresholds