import numpy as np
import random

def partition(X, T=None, frac=0.8, by_column=0):
    if X is None or X.shape[0] == 0:
        print('Incorrect X input.')
        return
    if T is not None:
        if X.shape[0] != T.shape[0]:
            print('Incorrect input. No. of values and targets do not match.')
            return
    if by_column == 1:
        X = X.T
        if T is not None:
            T = T.T
    rand = np.array(sorted(random.sample(range(X.shape[0]), int(X.shape[0]*frac))))
    mask = np.ones(X.shape[0], dtype=bool)
    mask[rand] = False
    comp = np.arange(0, X.shape[0])[mask]
    if T is None:
        return X[rand], X[comp]
    else:
        return X[rand], X[comp], T[rand], T[comp]