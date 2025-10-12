import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import requests

# --- RUTAS ---
path_buildings = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_building.parquet"
path_citizens  = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet"

# --- LEER LOS PARQUET ---
buildings = pd.read_parquet(path_buildings)
citizens  = pd.read_parquet(path_citizens)

# Normaliza nombres si vienen distintos
buildings = buildings.rename(columns={"latitude": "lat", "longitude": "lon"})

# --- AGREGAR PERSONAS POR WoS (peso original; no se usa para el mapa cronológico) ---
wos_weights = (
    citizens
    .dropna(subset=["WoS"])
    .groupby("WoS")
    .size()
    .rename("weight")
    .reset_index()
)

# --- CRUZAR con pop_building por osm_id ---
merged = wos_weights.merge(
    buildings,
    left_on="WoS",
    right_on="osm_id",
    how="inner"
)

# --- LIMPIEZA COORDENADAS ---
df = merged[["osm_id", "lat", "lon", "weight"]].copy()
df = df[(df["lat"].notna()) & (df["lon"].notna()) & (df["weight"] > 0)]

if df.empty:
    raise ValueError("No hay datos con coordenadas y pesos para el heatmap.")

# === PESO CRONOLÓGICO POR ORDEN DEL DF ===
# Si tu cronología depende de una columna temporal, ordénalo antes:
# df = df.sort_values('timestamp_col').reset_index(drop=True)
df = df.reset_index(drop=True)
chrono = np.arange(len(df), 0, -1, dtype=float)     # [N, N-1, ..., 1]
df["chrono_weight"] = pd.Series(chrono, index=df.index)

# --- OBTENER CONTORNO ADMINISTRATIVO DE UTRECHT (R1433619) ---
headers = {"User-Agent": "UtrechtBoundaryMapper/1.0 (contact: you@example.com)"}
resp = requests.get(
    "https://nominatim.openstreetmap.org/lookup",
    params={"osm_ids": "R1433619", "format": "json", "polygon_geojson": 1},
    headers=headers,
    timeout=30
)
resp.raise_for_status()
items = resp.json()
if not items or "geojson" not in items[0]:
    raise RuntimeError("No se pudo obtener el GeoJSON para R1433619.")

area_geojson = items[0]["geojson"]
bbox = items[0].get("boundingbox")

# --- CREAR MAPA BASE (CLARO) ---
center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(location=center, zoom_start=13, tiles=None)

# Mapa claro para buen contraste con el heatmap en negro
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr="© OpenStreetMap, © CARTO",
    name="CartoDB Positron (claro)",
    overlay=False,
    control=False
).add_to(m)

# --- HEATMAP CRONOLÓGICO EN BLANCO→NEGRO ---
# Gradiente monocromo: 0 = blanco, 1 = negro
grayscale_gradient = {
    0.0: "#ffffff",
    0.25: "#d9d9d9",
    0.5: "#b3b3b3",
    0.75: "#4d4d4d",
    1.0: "#000000"
}

heat_data = df[["lat", "lon", "chrono_weight"]].values.tolist()

HeatMap(
    heat_data,
    name="Heatmap cronológico (blanco→negro)",
    min_opacity=0.6,                 # asegura que lo bajo sea visible (blanco)
    radius=18,
    blur=18,
    max_zoom=18,
    max_val=float(df["chrono_weight"].max()),
    gradient=grayscale_gradient      # <-- clave para monocromo
).add_to(m)

# --- CONTORNO DE UTRECHT ---
folium.GeoJson(
    data=area_geojson,
    name="Utrecht administrativo (R1433619)",
    style_function=lambda feature: {"fillOpacity": 0, "color": "#000000", "weight": 2},
    tooltip="Utrecht administrativo"
).add_to(m)

# Ajustar vista
if bbox and len(bbox) == 4:
    south, north, west, east = map(float, bbox)
    m.fit_bounds([[south, west], [north, east]])
else:
    m.fit_bounds([[df["lat"].min(), df["lon"].min()],
                  [df["lat"].max(), df["lon"].max()]])

folium.LayerControl(collapsed=False).add_to(m)

# --- GUARDAR MAPA ---
output_path = "mapa_heatmap_cronologico_bw.html"
m.save(output_path)
print(f"Mapa guardado como '{output_path}'")
