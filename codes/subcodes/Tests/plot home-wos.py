import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import osmnx as ox

# -----------------------------
# 0. Configuración y rutas
# -----------------------------
base_dir = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population"

citizen_parquet_path = os.path.join(base_dir, "pop_citizen.parquet")
building_parquet_path = os.path.join(base_dir, "pop_building.parquet")

# Nombre de las columnas de coordenadas en pop_building
LAT_COL = "lat"
LON_COL = "lon"

target_crs = "EPSG:28992"  # RD New (Países Bajos)

# -----------------------------
# 1. Leer los parquet
# -----------------------------
df_citizen = pd.read_parquet(citizen_parquet_path)
df_building = pd.read_parquet(building_parquet_path)

print(f"Citizens: {df_citizen.shape[0]} filas")
print(f"Buildings: {df_building.shape[0]} filas")

# -----------------------------
# 2. Normalizar claves WoS / osm_id como string
# -----------------------------
if "WoS" not in df_citizen.columns:
    raise KeyError("La columna 'WoS' no existe en pop_citizen.parquet")

if "osm_id" not in df_building.columns:
    raise KeyError("La columna 'osm_id' no existe en pop_building.parquet")

df_citizen["WoS_str"] = df_citizen["WoS"].astype(str)
df_building["osm_id_str"] = df_building["osm_id"].astype(str)

# Subconjunto de buildings con osm_id y coords, sin duplicados
cols_building = ["osm_id_str", LAT_COL, LON_COL]
for c in cols_building:
    if c not in df_building.columns:
        raise KeyError(f"La columna '{c}' no existe en pop_building.parquet")

df_building_sub = df_building[cols_building].drop_duplicates("osm_id_str")

# -----------------------------
# 3. Hacer el match WoS (citizens) -> osm_id (buildings)
# -----------------------------
df_citizen_wos = df_citizen[["WoS", "WoS_str"]].copy()

df_merge = df_citizen_wos.merge(
    df_building_sub,
    how="left",
    left_on="WoS_str",
    right_on="osm_id_str"
)

# Filas de citizen cuyo WoS NO tiene building asociado
mask_missing = df_merge[LAT_COL].isna() | df_merge[LON_COL].isna()
missing_count = mask_missing.sum()

# De esos missing, ¿cuántos tienen 'virtual' en el WoS?
missing_rows = df_merge[mask_missing].copy()
virtual_missing_count = missing_rows["WoS_str"].str.contains("virtual", case=False, na=False).sum()

print(f"Número de filas de pop_citizen cuyo WoS NO está en pop_building: {missing_count}")
print(f"De esos, número de WoS que contienen 'virtual': {virtual_missing_count}")

# Filas con match correcto
df_match = df_merge[~mask_missing].copy()

# -----------------------------
# 3b. Contar ciudadanos por WoS y asignar intensidad
# -----------------------------
# Número de ciudadanos por WoS
wos_counts = df_match.groupby("WoS_str").size().rename("n_citizens")

# Tomamos una fila por WoS para obtener una geometría por edificio,
# pero conservamos el conteo de ciudadanos
df_match_counts = (
    df_match.drop_duplicates("WoS_str")
            .merge(wos_counts, on="WoS_str", how="left")
)

# -----------------------------
# 4. GeoDataFrame de WoS (puntos) en WGS84 y reproyección
# -----------------------------
gdf_wos = gpd.GeoDataFrame(
    df_match_counts,
    geometry=[
        Point(xy) for xy in zip(df_match_counts[LON_COL], df_match_counts[LAT_COL])
    ],
    crs="EPSG:4326"  # lat/lon
)

gdf_wos_proj = gdf_wos.to_crs(target_crs)

# -----------------------------
# 5. Utrecht (R1433619) y Kanaleneiland
# -----------------------------
# Utrecht por OSM ID
gdf_utrecht = ox.geocode_to_gdf("R1433619", by_osmid=True)
if gdf_utrecht.crs is None:
    gdf_utrecht.set_crs(epsg=4326, inplace=True)
else:
    gdf_utrecht = gdf_utrecht.to_crs(epsg=4326)

# Polígono de Kanaleneiland (WGS84)
coords_kanaleneiland_latlon = [
    (52.07904398, 5.081736117),
    (52.07624318, 5.08308264),
    (52.06046958, 5.09756737),
    (52.06021839, 5.097758556),
    (52.06008988, 5.11164107),
    (52.06328398, 5.113065093),
    (52.06860149, 5.111588679),
    (52.07642504, 5.109425399),
    (52.07861645, 5.108711591),
    (52.08034774, 5.107271173),
    (52.08592257, 5.097037869),
    (52.08498639, 5.096460351),
    (52.08309467, 5.094751129),
    (52.0803543,  5.087985518),
    (52.07904398, 5.081736117),
]

coords_kanaleneiland = [(lon, lat) for (lat, lon) in coords_kanaleneiland_latlon]
kanaleneiland_geom = Polygon(coords_kanaleneiland)

gdf_kanaleneiland = gpd.GeoDataFrame(
    {"name": ["Kanaleneiland"]},
    geometry=[kanaleneiland_geom],
    crs="EPSG:4326"
)

# Reproyectar a CRS métrico
gdf_utrecht_proj = gdf_utrecht.to_crs(target_crs)
gdf_kanaleneiland_proj = gdf_kanaleneiland.to_crs(target_crs)

# -----------------------------
# 6. Dibujar mapa (sin marco, en grises, puntos más oscuros = más ciudadanos)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Perímetro de Utrecht (gris medio)
gdf_utrecht_proj.boundary.plot(
    ax=ax,
    linewidth=1,
    edgecolor="0.6"  # gris
)

# Perímetro de Kanaleneiland (gris oscuro, línea discontinua)
gdf_kanaleneiland_proj.boundary.plot(
    ax=ax,
    linewidth=2,
    linestyle="--",
    edgecolor="0.2"  # gris más oscuro
)

# Puntos de WoS: intensidad según número de ciudadanos (n_citizens)
counts = gdf_wos_proj["n_citizens"]

# Normalización para el mapa de color (min=blanco/gris claro, max=negro)
norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())

# Scatter manual para controlar color en escala de grises
sc = ax.scatter(
    gdf_wos_proj.geometry.x,
    gdf_wos_proj.geometry.y,
    s=10,
    c=counts,
    cmap="Greys",   # escala de blancos a negros
    norm=norm,
    alpha=1,
    linewidths=0
)

# Ajustar límites al perímetro de Utrecht
minx, miny, maxx, maxy = gdf_utrecht_proj.total_bounds
dx = (maxx - minx) * 0.05
dy = (maxy - miny) * 0.05

ax.set_xlim(minx - dx, maxx + dx)
ax.set_ylim(miny - dy, maxy + dy)

# 1) Sin marco/coordenadas
ax.set_axis_off()

ax.set_aspect("equal", adjustable="box")
ax.set_title("Utrecht (R1433619) + Kanaleneiland + intensidad WoS (nº de ciudadanos)")

plt.tight_layout()
plt.show()
