from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

TOTAL_DATA_LENGTH = 1024
FORCED_DATA_LENGTH = 128
RANDOM_DATA_LENGTH = TOTAL_DATA_LENGTH - (FORCED_DATA_LENGTH * 2)

A = np.array([1, 1, 1, 1,
              1, 0, 0, 1,
              1, 0, 0, 1,
              1, 1, 1, 1])

B = np.array([1, 0, 0, 1,
              0, 1, 1, 0,
              0, 1, 1, 0,
              1, 0, 0, 1])

def random_pattern(length):
    return [[rnd.randint(0, 1) for i in range(length)]]

X = np.concatenate(([A]*FORCED_DATA_LENGTH, [B]*FORCED_DATA_LENGTH))
Y = np.array([], dtype=int)

for i in range(RANDOM_DATA_LENGTH):
    X = np.append(X, random_pattern(16), axis = 0)

np.random.shuffle(X)

for i in range(TOTAL_DATA_LENGTH):
    if np.array_equal(X[i], A):
        Y = np.append(Y, [1, 0])
    elif np.array_equal(X[i], B):
        Y = np.append(Y, [0, 1])
    else:
        Y = np.append(Y, [0, 0])

Y = np.reshape(Y, (TOTAL_DATA_LENGTH, 2))

regr = MLPRegressor(hidden_layer_sizes=(16, 8),
    activation = 'tanh',
    solver = 'adam',
    max_iter = 10000,
    verbose = False)

model = regr.fit(X, Y)

TEST = np.concatenate(([A], [B], random_pattern(16)))
print(TEST)
print(model.predict(TEST).round())