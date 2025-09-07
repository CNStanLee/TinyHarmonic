This file containts the ONNX QCDQ descrption of the QLSTM model we want to implement on hardware.

In this step we will load the ONNX file generated in the previous step (Step-2) as well to extact all the weights/scales/quantisers that will be loaded as part of this implementation. 

Once we have completed this implementation, we need to confirm the functional correctness of this implementation by comparing the output of the model for a sample input with the brevitas version.

Once we verify this we then move forward with Step-4 of the implementation i.e. Streamlining.
