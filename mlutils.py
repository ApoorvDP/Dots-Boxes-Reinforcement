import numpy as np, random

def move_to_onehot_24(move):
    onehot_move = [0]*24
    onehot_move[move] = 1
    return onehot_move

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
    mask, rand = np.ones(X.shape[0], dtype=bool), np.array(sorted(random.sample(range(X.shape[0]), int(X.shape[0]*frac))))
    mask[rand] = False
    comp = np.arange(0, X.shape[0])[mask]
    return X[rand], X[comp] if T is None else X[rand], X[comp], T[rand], T[comp]