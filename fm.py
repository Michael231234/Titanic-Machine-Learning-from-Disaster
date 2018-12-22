from __future__ import division
from math import exp
import numpy as np
from numpy import *
from random import normalvariate
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

path = '/Users/konglingtong/PycharmProjects/kaggle/Titanic/data/data.csv'
featureNum = 7

def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))


def stocGradAscent(dataMatrix, classLabels, k, iter, alpha):
    m, n = shape(dataMatrix)
    w = zeros((n, 1))
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    for it in range(iter):
        for x in range(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            p = w_0 + dataMatrix[x] * w + interaction
            print("y: ", p)
            loss = sigmoid(classLabels[x] * p) - 1
            print("loss: ", loss)

            w_0 = w_0 - alpha * loss * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
                                    dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

    return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction

        pre = sigmoid(p)

        result.append(pre)

        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    data = pd.read_csv(path)
    X = np.asarray(data.drop(columns=['Survived']))
    y = np.asarray(data['Survived'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    date_startTrain = datetime.now()
    print("Start")
    w_0, w, v = stocGradAscent(mat(X_train), y_train, 20, 200, 0.15)
    print("Train Accuracy：%f" % (1 - getAccuracy(mat(X_train), y_train, w_0, w, v)))
    date_endTrain = datetime.now()
    print("Time：%s" % (date_endTrain - date_startTrain))
    print("Test")
    print("Test Accuracy：%f" % (1 - getAccuracy(mat(X_test), y_test, w_0, w, v)))
