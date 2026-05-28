import importlib

_mod_data = importlib.import_module('src.data.04_dataset')
MaizeDataset = _mod_data.MaizeDataset
get_dataloaders = _mod_data.get_dataloaders
