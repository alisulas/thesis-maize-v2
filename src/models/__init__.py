import importlib

_mod = importlib.import_module('src.models.04_cnn_lstm')
CropYieldCNNLSTM = _mod.CropYieldCNNLSTM
CropYieldLSTM = _mod.CropYieldLSTM
