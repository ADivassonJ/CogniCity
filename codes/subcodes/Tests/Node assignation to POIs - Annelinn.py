import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import voronoi_diagram

# --------------------------------------------------------
# 1. Leer datos
# --------------------------------------------------------
path = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Annelinn\population"

nodes = pd.read_excel(f"{path}\\electric_system_Annelinn.xlsx")     # columnas: lat, lon, i
edges = pd.read_excel(f"{path}\\electric_system_Annelinn.xlsx")          # columnas: i, j

# --------------------------------------------------------
# 2. Definir polígono de Kanaleneiland (lat, lon -> lon, lat)
# --------------------------------------------------------
boundary_latlon = [(58.37779995285961, 26.737546920776367),
                    (58.38207378048632,  26.74806118011475),
                    (58.38095016884387,  26.753661632537845),
                    (58.380230379627854, 26.761858463287357),
                    (58.37991546722738,  26.76752328872681),
                    (58.37947683455617,  26.77179336547852),
                    (58.379218151193385, 26.773359775543216),
                    (58.37876826256608,  26.77522659301758),
                    (58.377767239785584, 26.77874565124512),
                    (58.377013642098476, 26.77923917770386),
                    (58.375427660052644, 26.784152984619144),
                    (58.37414532457297,  26.7826509475708),
                    (58.371591763372905, 26.782929897308353),
                    (58.37125427449941,  26.783938407897953),
                    (58.369139270733456, 26.78344488143921),
                    (58.368970514972645, 26.782200336456302),
                    (58.36281344471082,  26.78001165390015),
                    (58.359865199696884, 26.77775859832764),
                    (58.358199670019,    26.7762565612793),
                    (58.35478959051058,  26.7717719078064),
                    (58.35464327610064,  26.770827770233158),
                    (58.35328311497021,  26.765420436859134),
                    (58.35779452929785,  26.76052808761597),
                    (58.359280022547246, 26.759669780731205),
                    (58.357108030241015, 26.754348278045658),
                    (58.35555491754449,  26.746966838836673),
                    (58.355836283607296, 26.7467737197876),
                    (58.356140156436815, 26.746795177459717),
                    (58.357198063664704, 26.7476749420166),
                    (58.35798584632884,  26.749970912933353),
                    (58.35900993751445,  26.751451492309574),
                    (58.360337835698445, 26.751902103424076),
                    (58.36097926015119,  26.75168752670288),
                    (58.36248712436589,  26.7497992515564),
                    (58.36366861475146,  26.748619079589847),
                    (58.364827562136334, 26.74872636795044),
                    (58.3659752201055,   26.75001382827759),
                    (58.36697657745643,  26.750378608703613),
                    (58.368247922822114, 26.749413013458252),
                    (58.36886670266924,  26.74803972244263),
                    (58.36998047905436,  26.744391918182377),
                    (58.37151045798987,  26.741452217102054),
                    (58.37556008204928,  26.738491058349613),
                    (58.377168553744575, 26.737632751464847),
                    (58.3777421867492,   26.73758983612061),
                     ]

# Convertir a (lon, lat)
boundary_lonlat = [(lon, lat) for (lat, lon) in boundary_latlon]
boundary_poly = Polygon(boundary_lonlat)

# --------------------------------------------------------
# 3. Crear Voronoi (recortado al polígono)
# --------------------------------------------------------
points_geom = [Point(row["lon"], row["lat"]) for _, row in nodes.iterrows()]
multi_points = MultiPoint(points_geom)

voronoi_multi = voronoi_diagram(multi_points, envelope=boundary_poly, edges=False)

buildings = pd.read_parquet(f"{path}\\pop_building.parquet")     # columnas: lat, lon, node, ...

# 3) KDTree con nodos
node_coords = np.column_stack([nodes["lon"].values, nodes["lat"].values])
tree = cKDTree(node_coords)

# 4) Coordenadas de edificios
building_coords = np.column_stack([buildings["lon"].values, buildings["lat"].values])

# 5) Buscar nodo más cercano
dist, idx = tree.query(building_coords, k=1)

# 6) Crear columna con el nodo asignado por distancias
nearest_nodes = nodes.iloc[idx]["i"].values

# 7) Determinar si cada edificio está dentro del boundary
inside = [
    boundary_poly.contains(Point(lon, lat))
    for lon, lat in building_coords
]

# 8) Asignar nodo si está dentro, 'unknown' si está fuera
buildings["node"] = [
    nearest_nodes[i] if inside[i] else "unknown"
    for i in range(len(buildings))
]

# 9) Guardar resultado
output_path = f"{path}\\pop_building_with_nodes.parquet"
buildings.to_parquet(output_path, index=False)

# Contar ocurrencias de cada nodo
counts = buildings["node"].value_counts()

print("Archivo guardado en:", output_path)