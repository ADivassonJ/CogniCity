import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import osmnx as ox

# -----------------------------
# 0. Configuración y rutas
# -----------------------------
base_dir = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Annelinn\population"
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
# 3. Tartu (R2153396) y Annelinn
# -----------------------------
gdf_Tartu = ox.geocode_to_gdf("R2153396", by_osmid=True)
if gdf_Tartu.crs is None:
    gdf_Tartu.set_crs(epsg=4326, inplace=True)
else:
    gdf_Tartu = gdf_Tartu.to_crs(epsg=4326)

coords_Annelinn_latlon = [(58.37779995285961, 26.737546920776367),
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


coords_Annelinn = [(lon, lat) for (lat, lon) in coords_Annelinn_latlon]
Annelinn_geom = Polygon(coords_Annelinn)

gdf_Annelinn = gpd.GeoDataFrame(
    {"name": ["Annelinn"]},
    geometry=[Annelinn_geom],
    crs="EPSG:4326"
)

gdf_Tartu_proj = gdf_Tartu.to_crs(target_crs)
gdf_Annelinn_proj = gdf_Annelinn.to_crs(target_crs)

# -----------------------------
# 4. Plot (gris, sin ejes)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Perímetro Tartu
gdf_Tartu_proj.boundary.plot(
    ax=ax,
    linewidth=1,
    edgecolor="0.25"
)

# Perímetro Annelinn
# Base blanca continua (halo)
gdf_Annelinn_proj.boundary.plot(
    ax=ax,
    linewidth=3.0,
    linestyle="-",
    edgecolor="white",
    zorder=3
)

# Línea superior gris discontinua
gdf_Annelinn_proj.boundary.plot(
    ax=ax,
    linewidth=2.0,
    linestyle="--",
    edgecolor="0.25",
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
    "Salariat": "#000000",       # azul
    "Intermediate": "#000000",   # naranja
    "Working": "#000000",        # verde
}

# Asigna color: por defecto gris
colors = arch.map(color_map).fillna("0.70")  # gris para el resto

ax.scatter(
    gdf_buildings_proj.geometry.x,
    gdf_buildings_proj.geometry.y,
    s=10,
    c=colors,
    alpha=0.75,
    linewidths=0,
    zorder=2
)

# Ajustar límites al perímetro de Tartu
minx, miny, maxx, maxy = gdf_Tartu_proj.total_bounds
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
