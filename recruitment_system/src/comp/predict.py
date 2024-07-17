# src/utils/predict.py

import pandas as pd # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import pickle
from src.constants.paths import MODEL_PATH, SCALER_PATH

def prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    data = {
        'sl_no': [sl_no],
        'gender': [gender],
        'ssc_p': [ssc_p],
        'hsc_p': [hsc_p],
        'degree_p': [degree_p],
        'workex': [workex],
        'etest_p': [etest_p],
        'specialisation': [specialisation],
        'mba_p': [mba_p]
    }
    data = pd.DataFrame(data)
    data['gender'] = data['gender'].map({'M': 1, "F": 0})
    data['workex'] = data['workex'].map({"Yes": 1, "No": 0})
    data['specialisation'] = data['specialisation'].map({"Mkt&HR": 1, "Mkt&Fin": 0})

    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    scaled_df = scaler.transform(data)

    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    result = model.predict(scaled_df)
    return result[0]
