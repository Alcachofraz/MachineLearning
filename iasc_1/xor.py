from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

regr = MLPRegressor(hidden_layer_sizes=(4, 2),
                    activation='tanh',
                    solver='adam',
                    max_iter=10000,
                    n_iter_no_change=100,
                    verbose=False)

model = regr.fit(X, Y)

below = list(filter(lambda x: x < 0.1, model.loss_curve_))

if (len(below) > 0):
    print('Iteration at which loss got below 0.1: ' +
          str(model.loss_curve_.index(below[0]) + 1))
else:
    print('Loss never got below 0.1...')

plt.axhline(y=0.1, color='r', linestyle='-')
plt.plot(model.loss_curve_)
plt.show()
