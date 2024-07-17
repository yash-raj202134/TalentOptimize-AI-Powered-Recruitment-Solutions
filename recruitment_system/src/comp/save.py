# src/utils/save.py

import pickle
from src.constants.paths import MODEL_PATH, SCALER_PATH,RESULT_PATH

def save_model(model, scaler):
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(SCALER_PATH, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)


def save_results(results, result_path=f'{RESULT_PATH}/model_results.txt'):
    with open(result_path, 'w') as f:
        for result in results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"Accuracy: {result['accuracy']}\n")
            f.write(f"Classification Report:\n{result['report']}\n")
            f.write("\n" + "-"*50 + "\n\n")