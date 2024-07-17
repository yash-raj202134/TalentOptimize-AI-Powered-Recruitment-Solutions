# main.py
# from src import logger
# from GetData.download import DataIngestion

# data = DataIngestion()
# file_path = data.download_datasets()

# extract = data.extract_datasets(file_path)

import os
import sys
sys.path.append(".")
from recruitment_system.src.comp.data_loader import load_data
from recruitment_system.src.comp.preprocess import preprocess_data
from recruitment_system.src.comp.train import (
    train_knn, train_decision_tree, train_svm, train_random_forest, train_gaussian_nb, train_logistic_regression
)
from recruitment_system.src.comp.evaluate import evaluate_model
from recruitment_system.src.comp.save import save_model
from recruitment_system.src.constants.paths import RESULT_PATH

def main():
    data = load_data()
    X_train, X_test, Y_train, Y_test, scaler = preprocess_data(data)

    models = {
        "knn": train_knn(X_train, Y_train),
        "decision_tree": train_decision_tree(X_train, Y_train),
        "svm": train_svm(X_train, Y_train),
        "random_forest": train_random_forest(X_train, Y_train),
        "gaussian_nb": train_gaussian_nb(X_train, Y_train),
        "logistic_regression": train_logistic_regression(X_train, Y_train)
    }

    
    os.makedirs(RESULT_PATH, exist_ok=True)

    best_model_name = None
    best_accuracy = 0

    for model_name, model in models.items():
        accuracy = evaluate_model(model, X_test, Y_test, model_name, RESULT_PATH)
        print(f"Model: {model_name}, Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
    
    # Save the best model
    if best_model_name:
        save_model(models[best_model_name], scaler)
    
    # for model_name, model in models.items():
    #     accuracy = evaluate_model(model, X_test, Y_test, model_name, RESULT_PATH)
    #     print(f"Model: {model_name}, Accuracy: {accuracy}")

    # # Save the best model
    # best_model_name = max(models, key=lambda k: evaluate_model(models[k], X_test, Y_test, k, RESULT_PATH))
    # save_model(models[best_model_name], scaler)

if __name__ == "__main__":
    main()
