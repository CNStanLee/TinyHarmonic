readme
# step 1
This step is for model's QAT training.
- data download and preprocessing and train the model with 13_cnnlstm_train.py
- In this step, we use slide windows to split the harmonic current/voltage, then use 12 cycles FFT harmonic estimation as the ground truth (from IEC standards).
- The task is 0.5cycle input current/voltage -> 1, 3, 5, 7 harmonic amp
- We can change the cycle from 0 - 4 to have different model and corresponding dataset
- Model strucute is like 1DCNN->LSTM->4-head MLP regression
- Inputs: waveform_cycles, model_structure.
- Outputs: quantised_model_weight.pth, model_def 
# step 2
This step is for brevitas model export.
- must use brevitas 0.9.1 to make the helper work
```bash
pip install brevitas==0.9.1
```
- update two files in ref/step2/4_quant_lstm_helper to
- ../venv/lib/python3.10/site-packages/brevitas/export/onnx/standard/
- ../venv/lib/python3.10/site-packages/brevitas/export/onnx/standard/qcdq/
- then use 21_onnx_gen.py
- Inputs: quantised_model_weight.pth, model_def
- Outputs: brevitas_export_model.onnx 
# step 3