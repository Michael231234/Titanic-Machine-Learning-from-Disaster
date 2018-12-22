from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def readData(path):
    data = pd.read_csv(path)
    # print(data)
    X = np.asarray(data.drop(columns=['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin']))
    # print(X)
    y = np.asarray(data['Survived'])
    return X, y


def readTestData(dataPath, layblePath):
    data_test = pd.read_csv(dataPath)
    X_test = np.asarray(data_test.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin']))
    y_test = np.asarray(pd.read_csv(
        layblePath)['Survived'])
    return X_test, y_test


def decision_tree(X, y, X_test, y_test):
    tree_clfs = [
        DecisionTreeClassifier(splitter='best', max_depth=7, max_features='sqrt'),
        DecisionTreeClassifier(
            splitter='best', max_depth=7, max_features='sqrt', min_samples_split=3,
            min_samples_leaf=2, min_weight_fraction_leaf=0.01
        ),
        DecisionTreeClassifier(
            splitter='best', max_depth=7, max_features='sqrt', min_samples_split=5,
            min_samples_leaf=3, min_weight_fraction_leaf=0.01
        ),
        DecisionTreeClassifier(splitter='random', max_features='log2', max_depth=7),
        DecisionTreeClassifier(splitter='random', max_features=0.5),
                 ]
    for tree_clf in tree_clfs:
        tree_clf.fit(X, y)
        scores = cross_val_score(tree_clf, X, y, cv=10, scoring='accuracy')
        print(tree_clf)
        print(np.mean(scores))
        print('Train Accuracy:', tree_clf.score(X, y))
        print('Test Accuracy:', tree_clf.score(X_test, y_test))

def random_forest(X, y, X_test, y_test):
    forest_clfs = [
        RandomForestClassifier(n_estimators=100, max_depth=7, max_features='sqrt'),
        RandomForestClassifier(n_estimators=100, max_features='log2', max_depth=7),
        RandomForestClassifier(n_estimators=100, max_features=2, max_depth=7)
    ]
    for forest_clf in forest_clfs:
        forest_clf.fit(X, y)
        scores = cross_val_score(forest_clf, X, y, cv=10, scoring='accuracy')
        print(forest_clf)
        print(np.mean(scores))
        print('Train Accuracy:', forest_clf.score(X, y))
        print('Test Accuracy:', forest_clf.score(X_test, y_test))

path = '/Users/konglingtong/PycharmProjects/kaggle/Titanic/data/train1.csv'
dataPath = '/Users/konglingtong/PycharmProjects/kaggle/Titanic/data/test1.csv'
layblePath = '/Users/konglingtong/PycharmProjects/kaggle/Titanic/data/gender_submission.csv'
X_test, y_test = readTestData(dataPath, layblePath)
X, y = readData(path)
decision_tree(X, y, X_test, y_test)
random_forest(X, y, X_test, y_test)
