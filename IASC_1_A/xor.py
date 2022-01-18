from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

regr = MLPRegressor(hidden_layer_sizes=(2),
                    activation='tanh',
                    solver='lbfgs',
                    verbose=True)

model = regr.fit(X, Y)

print(model)

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
