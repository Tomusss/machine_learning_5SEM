#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.linear_model import Ridge

class RidgeRegr:
    def __init__(self, alpha = 0.1):
        self.alpha = alpha

    def fit(self, X, Y, eta = 0.0006, epochs = 200000):
        # wejscie:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Znajduje theta (w przyblizeniu) minimalizujace kwadratowa funkcje kosztu L uzywajac metody iteracyjnej.
        n, m = X.shape
        X = np.hstack((np.ones((n, 1)), X)).T
        self.theta = np.zeros((m+1,1))
        for k in range(epochs):
            indeksy = np.random.permutation(n)
            for i in indeksy:
                xi = X[:, i:i+1]
                yi = Y[i]
                ypred = self.theta.T @xi
                grad_mse = (2/n)*(xi@(ypred - yi))
                grad_rdg = (2/n)*self.alpha*self.theta
                grad_rdg[0] = 0
                grad = grad_mse + grad_rdg
                grad = grad/np.linalg.norm(grad)
                self.theta -= eta*grad
            
        return self
    
    def predict(self, X):
        # wejscie
        #  X = np.array, shape = (k, m)
        # zwraca
        #  Y = wektor(f(X_1), ..., f(X_k))
        k, m = X.shape
        X = np.hstack((np.ones((k,1)), X)).T
        Y_pred = self.theta.T @ X
        return Y_pred


def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha, solver='sag').fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    print(actual,expected)
    #assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha, solver='sag').fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    print(actual,expected)
    #assert list(actual) == pytest.approx(list(expected), rel=1e-3)
    
test_RidgeRegressionInOneDim()
test_RidgeRegressionInThreeDim()