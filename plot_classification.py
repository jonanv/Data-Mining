"""
=================================
Clasificación más cercana de los vecinos
=================================

Ejemplo de uso de la clasificación de vecinos más cercanos.
Trazará los límites de decisión para cada clase.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# importar algunos datos para jugar con
iris = datasets.load_iris()

# sólo tomamos las dos primeras características. Podríamos evitar este feo
# corte mediante el uso de un conjunto de datos de dos dimensiones
X = iris.data[:, :2]
y = iris.target

h = .02  # tamaño del paso en la malla

# Crear mapas de color
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
#for weights in ['uniform']:
    # creamos una instancia de Neighbors Classifier y ajustamos los datos.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Trazar el límite de decisión. Para ello, asignaremos un color a cada
    # punto en la malla [x_min, x_max] x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Coloca el resultado en un gráfico de color
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Trazar también los puntos de entrenamiento
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()