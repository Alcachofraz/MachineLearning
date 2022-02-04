from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

learning_rates = [0.05, 0.25, 0.5, 1, 2]

for learning_rate in learning_rates:
    data = []

    for _ in range(10):
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        Y = [0, 1, 1, 0]

        regr = MLPRegressor(hidden_layer_sizes=(4, 2),
                            activation='relu',
                            solver='sgd',
                            max_iter=10000,
                            n_iter_no_change=100,
                            verbose=False,
                            momentum=0.5,
                            learning_rate_init=learning_rate,
                            shuffle=True)

        model = regr.fit(X, Y)

        below = list(filter(lambda x: x < 0.1, model.loss_curve_))

        if (len(below) > 0):
            print('Iteration at which loss got below 0.1: ' +
                  str(model.loss_curve_.index(below[0]) + 1))
            data.append(model.loss_curve_.index(below[0]) + 1)
        else:
            print('Loss never got below 0.1...')

    print('Mean: ' + str(np.mean(data)))

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.axhline(y=0.1, color='r', linestyle='-')
    plt.plot(model.loss_curve_)
    plt.show()
