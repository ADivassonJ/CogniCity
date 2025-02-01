import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Definir las coordenadas de los puntos
points = np.array([[0, 0], [1, 4], [2, 1], [3, 3]])

# Calcular el diagrama de Voronoi
vor = Voronoi(points)

# Graficar el diagrama de Voronoi
voronoi_plot_2d(vor)
plt.plot(points[:,0], points[:,1], 'ro')  # Plotear los puntos
plt.xlim(vor.min_bound[0] - 1, vor.max_bound[0] + 1)
plt.ylim(vor.min_bound[1] - 1, vor.max_bound[1] + 1)
plt.show()