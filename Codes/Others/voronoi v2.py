import numpy as np
from scipy.spatial import Voronoi
import pandas as pd

def add_bus_relation(df, coord_buses):
    # Definir las coordenadas de los puntos
    coord = [[0, 0], [1, 4], [2, 1], [3, 3]]
    coordenadas = np.array(coord)

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

    # Supongamos que tienes otras coordenadas
    otras_coordenadas = np.array(coord_buses)

    # Determinar a qué región corresponden las otras coordenadas
    for i, coord in enumerate(otras_coordenadas):
        area = vor.point_region[np.argmin(np.linalg.norm(vor.points - coord, axis=1))]
        print(f"La coordenada {coord} se encuentra en la región {area}")

if __name__ == "__main__":
    df = pd.DataFrame({'coord': [[0, 0], [1, 4], [2, 1], [3, 3]]})
    coord_buses = [[0.5, 0.5], [1.5, 2.5], [3.5, 3.5]]
    add_bus_relation(df, coord_buses)