from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

regr = MLPRegressor(hidden_layer_sizes = (4, 2),
                    activation = 'tanh',
                    solver = 'adam',
                    max_iter = 10000,
                    verbose = False)

model = regr.fit(X, Y)

below = list(filter(lambda x: x < 0.1, model.loss_curve_))

if (len(below) > 0):
    print('Iteration at which loss got below 0.1: ' + str(model.loss_curve_.index(below[0]) + 1))
else:
    print('Loss never got below 0.1...')

plt.axhline(y = 0.1, color = 'r', linestyle = '-')
plt.plot(model.loss_curve_)
plt.show()

"""
# Visualize results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
cax = plt.imshow(np.array(output), interpolation='nearest', vmin=0, vmax=1)
cbar = fig.colorbar(cax, ticks=[0, 1])
cbar.ax.tick_params(labelsize=15)
plt.set_cmap('gray')
plt.axis('off')

table = {'0, 0': (-2, -1),
         '0, 1': (-2, res+2),
         '1, 0': (res-2, -1),
         '1, 1': (res-2, res+2)}
for text, corner in table.items():
    ax.annotate(text, xy=corner, size=15, annotation_clip=False)

plt.show()
"""
