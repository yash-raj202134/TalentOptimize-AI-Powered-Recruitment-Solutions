# src/utils/save.py

import pickle
from src.constants.paths import MODEL_PATH, SCALER_PATH

def save_model(model, scaler):
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(SCALER_PATH, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
