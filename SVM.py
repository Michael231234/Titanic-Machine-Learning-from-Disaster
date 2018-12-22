from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def data(path):
    data = pd.read_csv(path)
    X = np.asarray(data.drop(columns=['Survived']))
    y = np.asarray(data['Survived'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=33)
    return X_train, X_test, y_train, y_test


def SVM(X_train, X_test, y_train, y_test):
    svc_clfs = [
        SVC(
            gamma=0.1, C=1.5, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
            kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False
        ),
        SVC(
            gamma=0.1, C=1.5, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
            kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False
        ),
        SVC(
            gamma=0.1, C=10, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
            kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False
        )
    ]
    for svc_clf in svc_clfs:
        svc_clf.fit(X_train, y_train)
        y_true, y_pred = y_test, svc_clf.predict(X_test)
        print(svc_clf)
        print(svc_clf.score(X_test, y_test))
        print(classification_report(y_true, y_pred))


path = '/Users/konglingtong/PycharmProjects/kaggle/Titanic/data/data.csv'
X_train, X_test, y_train, y_test = data(path)
SVM(X_train, X_test, y_train, y_test)