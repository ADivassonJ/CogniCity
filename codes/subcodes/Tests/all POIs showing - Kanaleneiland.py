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
building_parquet_path = os.path.join(base_dir, "pop_building.parquet")

# Nombre de las columnas de coordenadas en pop_building
LAT_COL = "lat"
LON_COL = "lon"

target_crs = "EPSG:28992"  # RD New (métrico)

# -----------------------------
# 1. Leer pop_building
# -----------------------------
df_building = pd.read_parquet(building_parquet_path)
print(f"Buildings: {df_building.shape[0]} filas")

# Validación de columnas
for c in [LAT_COL, LON_COL]:
    if c not in df_building.columns:
        raise KeyError(f"La columna '{c}' no existe en pop_building.parquet")

# Quitar filas sin coords
df_building = df_building.dropna(subset=[LAT_COL, LON_COL]).copy()

# (Opcional) deduplicación por osm_id con prioridad: Working > Intermediate > Salariat
if "osm_id" in df_building.columns and "archetype" in df_building.columns:
    df_building["archetype"] = df_building["archetype"].astype("string").str.strip()

    priority = ["Working", "Intermediate", "Salariat"]

    def dedup_group(g: pd.DataFrame) -> pd.DataFrame:
        # Elige la primera categoría disponible por prioridad
        for cat in priority:
            m = g["archetype"].eq(cat)
            if m.any():
                return g.loc[m].iloc[[0]]  # una sola fila
        # Si no hay ninguna de las 3, simplifica a una fila
        return g.iloc[[0]]

    df_building = (
        df_building
        .groupby("osm_id", group_keys=False, sort=False)
        .apply(dedup_group, include_groups=False)
        .reset_index(drop=True)
    )

elif "osm_id" in df_building.columns:
    # Fallback si no existe archetype
    df_building = df_building.drop_duplicates(subset=["osm_id"], keep="first")


# -----------------------------
# 2. GeoDataFrame de buildings (WGS84 -> CRS métrico)
# -----------------------------
gdf_buildings = gpd.GeoDataFrame(
    df_building,
    geometry=[Point(xy) for xy in zip(df_building[LON_COL], df_building[LAT_COL])],
    crs="EPSG:4326"
)

gdf_buildings_proj = gdf_buildings.to_crs(target_crs)

# -----------------------------
# 3. Aveiro (R1433619) y Kanaleneiland
# -----------------------------
gdf_Aveiro = ox.geocode_to_gdf("R1433619", by_osmid=True)
if gdf_Aveiro.crs is None:
    gdf_Aveiro.set_crs(epsg=4326, inplace=True)
else:
    gdf_Aveiro = gdf_Aveiro.to_crs(epsg=4326)

gdf = ox.geocode_to_gdf("Kanaleneiland, Utrecht, Netherlands")


# Ejemplo: quedarte con el primer polígono válido (si hubiera varios)
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].iloc[[0]]
gdf_Kanaleneiland_proj = gdf.to_crs(target_crs)

gdf_Aveiro_proj = gdf_Aveiro.to_crs(target_crs)

# -----------------------------
# 4. Plot (gris, sin ejes)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Perímetro Aveiro
gdf_Aveiro_proj.boundary.plot(
    ax=ax,
    linewidth=1,
    edgecolor="#0c343d"
)

# Perímetro Kanaleneiland
# Base blanca continua (halo)
gdf_Kanaleneiland_proj.boundary.plot(
    ax=ax,
    linewidth=3.0,
    linestyle="-",
    edgecolor="white",
    zorder=3
)

# Línea superior gris discontinua
gdf_Kanaleneiland_proj.boundary.plot(
    ax=ax,
    linewidth=2.0,
    linestyle="--",
    edgecolor="#0c343d",
    zorder=4
)

# -----------------------------
# Buildings como puntos: gris para todo menos archetype in {Salariat, Intermediate, Working}
# -----------------------------
# Asegura que existe la columna
if "archetype" not in gdf_buildings_proj.columns:
    raise KeyError("La columna 'archetype' no existe en pop_building.parquet")

# Normaliza a string (por si hay NaN o categorías)
arch = gdf_buildings_proj["archetype"].astype("string")

# Define categorías de interés
cats = ["Salariat", "Intermediate", "Working"]

# Colores: cambia estos 3 si quieres otros
color_map = {
    "Salariat": "#0c343d",       # azul
    "Intermediate": "#0c343d",   # naranja
    "Working": "#0c343d",        # verde
}

# Asigna color: por defecto gris
colors = arch.map(color_map).fillna("#a2c4c9")  # gris para el resto

ax.scatter(
    gdf_buildings_proj.geometry.x,
    gdf_buildings_proj.geometry.y,
    s=10,
    c=colors,
    alpha=0.75,
    linewidths=0,
    zorder=2
)

# Ajustar límites al perímetro de Aveiro
minx, miny, maxx, maxy = gdf_Aveiro_proj.total_bounds
dx = (maxx - minx) * 0.05
dy = (maxy - miny) * 0.05
ax.set_xlim(minx - dx, maxx + dx)
ax.set_ylim(miny - dy, maxy + dy)

ax.set_axis_off()
ax.set_aspect("equal", adjustable="box")

# -----------------------------
# Conteo simple de archetype en el dataset ORIGINAL
# -----------------------------
if "archetype" not in df_building.columns:
    raise KeyError("La columna 'archetype' no existe en pop_building.parquet")

arch = df_building["archetype"].astype("string").str.strip()

counts = arch.value_counts()

print("\nArchetype counts (dataset original):")
print(f"  Working:       {int(counts.get('Working', 0))}")
print(f"  Intermediate:  {int(counts.get('Intermediate', 0))}")
print(f"  Salariat:      {int(counts.get('Salariat', 0))}")
print(f"  Total rows:    {len(df_building)}\n")

plt.tight_layout()
plt.show()
