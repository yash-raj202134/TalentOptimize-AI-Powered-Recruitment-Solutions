# src/utils/train.py

from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore

def train_knn(X_train, Y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    return knn

def train_decision_tree(X_train, Y_train):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    return decision_tree

def train_svm(X_train, Y_train):
    svc = SVC()
    svc.fit(X_train, Y_train)
    return svc

def train_random_forest(X_train, Y_train):
    random_forest = RandomForestClassifier(n_estimators=1000)
    random_forest.fit(X_train, Y_train)
    return random_forest

def train_gaussian_nb(X_train, Y_train):
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    return gaussian

def train_logistic_regression(X_train, Y_train):
    lg = LogisticRegression()
    lg.fit(X_train, Y_train)
    return lg
