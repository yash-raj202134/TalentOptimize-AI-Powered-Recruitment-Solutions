# src/utils/preprocess.py

import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_data(data):
    data.drop(['ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'salary'], axis=1, inplace=True)

    data["gender"] = data.gender.map({"M": 0, "F": 1})
    data["workex"] = data.workex.map({"No": 0, "Yes": 1})
    data["status"] = data.status.map({"Not Placed": 0, "Placed": 1})
    data["specialisation"] = data.specialisation.map({"Mkt&HR": 0, "Mkt&Fin": 1})

    # Separate the majority and minority classes
    data_majority = data[data['status'] == 1]
    data_minority = data[data['status'] == 0]

    # Upsample minority class
    data_minority_upsampled = resample(data_minority,
                                       replace=True,
                                       n_samples=len(data_majority),
                                       random_state=42)

    # Combine majority class with upsampled minority class
    balanced_data = pd.concat([data_majority, data_minority_upsampled])

    X = balanced_data.copy().drop('status', axis=1)
    y = balanced_data['status']

    scaler = MinMaxScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test, scaler
