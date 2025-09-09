readme
# step 1
This step is for model's QAT training.
- data download and preprocessing and train the model with 13_cnnlstm_train.py
- In this step, we use slide windows to split the harmonic current/voltage, then use 12 cycles FFT harmonic estimation as the ground truth (from IEC standards).
- The task is 0.5cycle input current/voltage -> 1, 3, 5, 7 harmonic amp
- We can change the cycle from 0 - 4 (say 0.5) to have different model and corresponding dataset
- Model strucute is like 1DCNN->LSTM->4-head broad MLP regression (1DCNN+LSTM for feature extration, broad MLP has been verified fit this regression better.)
- Inputs: waveform_cycles, model_structure.
- Outputs: quantised_model_weight.pth, model_def 
# step 2
This step is for brevitas model export.
- must use brevitas 0.9.1 to make the helper work
```bash
pip install onnx==1.17.0
pip install brevitas==0.9.1
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```
- update two files in ref/step2/4_quant_lstm_helper to
- ../venv/lib/python3.10/site-packages/brevitas/export/onnx/standard/
- ../venv/lib/python3.10/site-packages/brevitas/export/onnx/standard/qcdq/
- then use 21_onnx_gen.py
- Inputs: quantised_model_weight.pth, model_def
- Outputs: brevitas_export_model.onnx 
- reference: https://github.com/fastmachinelearning/qonnx/tree/main/notebooks/4_quant_lstm_helper
# step 3
```bash
pip install onnxruntime==1.14
```
This step is for qcdq model generation (model description).
- To make life easier, we split the model to 3 submodels -> 1.1DCNN 2.LSTM 3.4-head MLP
- Then we hand-craft the LSTM operators and replace its parameters, for part 1 and 3, we consider use standard flow, cuz their ops already supported in this version of FINN.
- then use 31_onnx_des.py
- In this script, it will rebuid the lstm model in a qcdq way. (onnx construct -> initializers setup)
- After we have the qcdq model, then we do behaviour test to ensure our implementation is doing its correct things.
- Inputs: brevitas_export_model.onnx
- Outputs: sublstm_qcdq.onnx, subcnn.onnx. submlp.onnx
# step 4
This step is for finn model generation (streamlining).
- In this step, all the script should be excuted in the FINN docker.
- We need to update the activation function and transformation to finn src
- Some operations and transformations need to be replaced in FINN. In ref/step4/replace_finn, there are two folders and one script which need to be replaced to your finn respo finn/src/finn, for these transformations' detail, please check the readme under ref/step4
- After the replacement, now the transformations we need to streamline the lstm finn-onnx are satisfied.
- Run 41_streamling.py to get the streamlined onnx file.
- Inputs: sublstm_qcdq.onnx
- Outputs: sublstm_finn_streamlined.onnx