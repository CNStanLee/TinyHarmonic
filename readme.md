readme
# for step 1
data download and preprocessing and train the model with 13_cnnlstm_train.py

# for step 2
use brevitas 0.9.1
update two files in ref/4_quant_lstm_helper to
!cp ./4_quant_lstm_helper/function.py ../venv/lib/python3.10/site-packages/brevitas/export/onnx/standard/
!cp ./4_quant_lstm_helper/handler.py  ../venv/lib/python3.10/site-packages/brevitas/export/onnx/standard/qcdq/
then use 21_onnx_gen.py