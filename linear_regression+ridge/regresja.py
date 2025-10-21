#!/usr/bin/env python3

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

class LinearRegr:
    def fit(self, X, Y, eta = 0.012, epochs = 20000):
        # wejscie:
        #  X = np.array, shape = (m,n)
        #  Y = np.array, shape = (1,n)
        # Znajduje theta minimalizujace kwadratowa funkcje kosztu L uzywajac wzoru.
        # Uwaga: przed zastosowaniem wzoru do X nalezy dopisac kolumne zlozona z jedynek.
        m,n= X.shape
        X = np.hstack((np.ones((m, 1)), X)).T
        self.theta = np.zeros((n+1,1))
        for k in range(epochs):
        #Aktualizacja self.Beta zgodnie z metodą GD
            ypred = self.theta.T @ X
            grad = (2/m)*(X@(ypred - Y).T)
            self.theta -= eta*grad
        return self
    
    def predict(self, X):
        # wejscie
        #  X = np.array, shape = (m,k)
        # zwraca
        #  Y = wektor(f(X_1), ..., f(X_k))
        m,n= X.shape
        X = np.hstack((np.ones((m,1)), X)).T
        Y_pred = self.theta.T @ X
        return Y_pred


def test_RegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,2,10]).reshape((3,1))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print(expected, actual)
    #assert list(actual) == pytest.approx(list(expected))

def test_RegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print(expected, actual)
    #list(actual) == pytest.approx(list(expected))
    
test_RegressionInOneDim()
test_RegressionInThreeDim()