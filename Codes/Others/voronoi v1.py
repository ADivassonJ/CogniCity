import numpy as np
from scipy.spatial import Voronoi

# Definir las coordenadas de los puntos
coordenadas = np.array([[0, 0], [1, 4], [2, 1], [3, 3]])

# Calcular el diagrama de Voronoi
vor = Voronoi(coordenadas)

# Calcular las áreas de las regiones
areas = []
for region_index in range(len(coordenadas)):
    indices_puntos_region = vor.regions[vor.point_region[region_index]]
    if -1 not in indices_puntos_region:
        region = vor.vertices[indices_puntos_region]
        areas.append(abs(np.cross(region[:-1], region[1:]).sum()) / 2)
    else:
        areas.append(np.inf)  # Región infinita

print("Áreas de las regiones:", areas)

# Supongamos que tienes otras coordenadas
otras_coordenadas = np.array([[0.5, 0.5], [1.5, 2.5], [3.5, 3.5]])

# Función para encontrar la región a la que pertenece una coordenada
def encontrar_region(coordenada):
    indice_region = vor.point_region[np.argmin(np.linalg.norm(vor.points - coordenada, axis=1))]
    return indice_region

# Determinar a qué región corresponden las otras coordenadas
for i, coord in enumerate(otras_coordenadas):
    area = encontrar_region(coord)
    print(f"La coordenada {coord} se encuentra en la región {area}")

