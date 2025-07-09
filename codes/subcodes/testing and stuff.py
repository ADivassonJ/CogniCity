import pandas as pd
import geopandas as gpd
from shapely import wkt
import numpy as np
import pydeck as pdk
from collections import defaultdict

# --- Parámetros de entrada ---
EXCEL_EVENTOS = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_level_2.xlsx"
EXCEL_EDIFICIOS = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_building.xlsx"

# Minuto que quieres visualizar (ej. 480 → 8:00)
minute = 480

# --- Cargar datos ---
# Eventos
df = pd.read_excel(EXCEL_EVENTOS, dtype={"osm_id": str})
# Edificios
pop_b = pd.read_excel(EXCEL_EDIFICIOS, dtype={"osm_id": str})

# --- Preprocesamiento eventos ---
df.replace([np.inf, float("inf")], 2500, inplace=True)
df["in"]  = df["in"].astype(int)
df["out"] = df["out"].astype(int)

# --- Preprocesamiento edificios ---
pop_b["geometry"] = pop_b["geometry"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(pop_b, geometry="geometry")

# --- Calcular ocupación en el minuto dado ---
occ = defaultdict(int)
for _, row in df.iterrows():
    if row["in"] <= minute < row["out"]:
        occ[row["osm_id"]] += 1

# Debug: imprimir primeras ocupaciones no-cero
print("Ocupaciones no nulas en el minuto", minute)
for osm, count in list(occ.items())[:10]:
    print(f"  {osm} → {count}")

# --- Asociar ocupación al GeoDataFrame ---
gdf["occupancy"] = gdf["osm_id"].map(occ).fillna(0).astype(int)

# --- Preparar coordenadas para pydeck ---
gdf["coordinates"] = gdf["geometry"].apply(lambda poly: poly.__geo_interface__["coordinates"])

# --- Escala de color ---
nonzero = gdf[gdf["occupancy"] > 0]["occupancy"]
min_occ = nonzero.min() if not nonzero.empty else 0
max_occ = nonzero.max() if not nonzero.empty else 1

def occupancy_to_color(val):
    if val == 0:
        return [255, 255, 255, 100]
    norm = (val - min_occ) / (max_occ - min_occ)
    r = int(255 * norm)
    g = int(255 * (1 - norm))
    return [r, g, 0, 180]

gdf["fill_color"] = gdf["occupancy"].apply(occupancy_to_color)

# --- Crear capa y mapa ---
layer = pdk.Layer(
    "PolygonLayer",
    gdf,
    get_polygon="coordinates",
    get_fill_color="fill_color",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=gdf.geometry.centroid.y.mean(),
    longitude=gdf.geometry.centroid.x.mean(),
    zoom=13
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Ocupación: {occupancy}"},
    map_style="mapbox://styles/mapbox/light-v9"
)

# --- Exportar HTML ---
output = f"mapa_ocupacion_{minute}min.html"
deck.to_html(output)
print(f"> Mapa generado: {output}")
