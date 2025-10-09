import pandas as pd
import folium
import requests

# --- RUTAS ---
path_buildings = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_building.parquet"
path_citizens = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet"

# --- LEER LOS PARQUET ---
buildings = pd.read_parquet(path_buildings)
citizens = pd.read_parquet(path_citizens)

# Normaliza nombres si vienen distintos
buildings = buildings.rename(columns={"latitude": "lat", "longitude": "lon"})

# --- ELIMINAR DUPLICADOS DE 'Home' EN pop_citizen ---
# (asumo que 'Home' es una columna con el ID del edificio — ej. osm_id)
citizens_unique = citizens.drop_duplicates(subset=["Home"]).copy()

# --- CRUZAR con pop_building por osm_id ---
merged = citizens_unique.merge(
    buildings,
    left_on="Home",
    right_on="osm_id",
    how="inner"
)

# --- QUEDARSE SOLO CON COORDENADAS ---
df_home = merged[["osm_id", "lat", "lon"]].copy()
df_home = df_home[(df_home["lat"].notna()) & (df_home["lon"].notna())]

if df_home.empty:
    raise ValueError("No se encontraron coincidencias entre pop_citizen['Home'] y pop_building['osm_id'].")

# --- OBTENER CONTORNO ADMINISTRATIVO DE UTRECHT (R1433619) ---
headers = {"User-Agent": "UtrechtBoundaryMapper/1.0 (contact: you@example.com)"}
resp = requests.get(
    "https://nominatim.openstreetmap.org/lookup",
    params={
        "osm_ids": "R1433619",
        "format": "json",
        "polygon_geojson": 1
    },
    headers=headers,
    timeout=30
)
resp.raise_for_status()
items = resp.json()
if not items or "geojson" not in items[0]:
    raise RuntimeError("No se pudo obtener el GeoJSON para R1433619.")

area_geojson = items[0]["geojson"]
bbox = items[0].get("boundingbox")

# --- CREAR MAPA BASE (30% opacidad) ---
center = [df_home["lat"].mean(), df_home["lon"].mean()]
m = folium.Map(location=center, zoom_start=13, tiles=None)

tile = folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="© OpenStreetMap contributors",
    name="Mapa base (30%)",
    overlay=False,
    control=False
)
tile.add_to(m)

# Aplicar opacidad 30 %
m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-tile-pane { opacity: 0.3 !important; }
</style>
"""))

# --- AÑADIR LOS HOGARES (Home) ---
for _, row in df_home.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=3,
        fill=True,
        fill_opacity=0.8,
        color="#1f6feb",
        weight=1,
        popup=f"Home osm_id: {row['osm_id']}"
    ).add_to(m)

# --- CONTORNO DE UTRECHT ---
folium.GeoJson(
    data=area_geojson,
    name="Utrecht administrativo (R1433619)",
    style_function=lambda feature: {
        "fillOpacity": 0,
        "color": "#d00000",
        "weight": 2
    },
    tooltip="Utrecht administrativo"
).add_to(m)

# Ajustar vista
if bbox and len(bbox) == 4:
    south, north, west, east = map(float, bbox)
    m.fit_bounds([[south, west], [north, east]])
else:
    m.fit_bounds([[df_home["lat"].min(), df_home["lon"].min()],
                  [df_home["lat"].max(), df_home["lon"].max()]])

folium.LayerControl(collapsed=False).add_to(m)

# --- GUARDAR MAPA ---
m.save("mapa_home_citizen_Utrecht.html")
print("Mapa guardado como 'mapa_home_citizen_Utrecht.html'")
