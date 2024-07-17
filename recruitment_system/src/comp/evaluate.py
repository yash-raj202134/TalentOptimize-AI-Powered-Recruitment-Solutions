# src/utils/evaluate.py

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, Y_test, model_name, result_path):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    conf_mat = confusion_matrix(Y_test, Y_pred)
    
    fig = plt.figure(figsize=(10, 7))
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in conf_mat.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(conf_mat, annot=labels, annot_kws={"size": 16}, fmt='')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"{result_path}/confusion_matrix_{model_name}.png")
    # plt.show()
    
    return accuracy
