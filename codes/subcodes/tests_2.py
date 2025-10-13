import pandas as pd
import numpy as np
import folium
import requests

# === Parámetros visuales ===
RADIUS_PX    = 6         # tamaño del punto en píxeles
MIN_OPACITY  = 0.0       # opacidad del último punto (0.0 = 0%)
COLOR_HEX    = "#000000" # negro

# --- RUTAS ---
path_buildings = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_building.parquet"
path_citizens  = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet"

# --- LEER PARQUET ---
buildings = pd.read_parquet(path_buildings)
citizens  = pd.read_parquet(path_citizens)

# --- FILTROS EN CITIZENS ---
# 1) independent_type == 1 (acepta numérico o string)
mask_ind = citizens["independent_type"] == 1

# 2) excluir WoS_subgroup == "Home" si existe
mask_home = citizens["WoS_subgroup"].astype(str).str.strip() != "Home"

# 3) filtrar y mantener weight
before = len(citizens)
citizens_f = citizens.loc[mask_ind & mask_home].dropna(subset=["WoS"]).copy()
after = len(citizens_f)
if citizens_f.empty:
    raise ValueError("Tras filtrar (independent_type==1 y excluir Home), no quedan citizens.")

# --- weight Y PESO (cronológico) ---
citizens_f = citizens_f.reset_index(drop=True)
citizens_f["weight"]  = citizens_f.index + 1         # 1..N

# (Opcional) si quieres 1 punto por WoS (evitar duplicados en mismo edificio):
# citizens_f = citizens_f.drop_duplicates(subset=["WoS"], keep="first").reset_index(drop=True)

# --- CRUCE SOLO PARA TRAER lat/lon (sin agregación) ---
# Reducimos buildings a columnas estrictamente necesarias para evitar conflictos
buildings_min = buildings[["osm_id", "lat", "lon"]].dropna(subset=["lat", "lon"])

# --- DICCIONARIO lat/lon DESDE BUILDINGS (sin merge, sin multiplicar filas) ---
# Si hay duplicados de osm_id en buildings, nos quedamos con el primero
buildings_min = (
    buildings[["osm_id", "lat", "lon"]]
    .dropna(subset=["lat", "lon"])
    .drop_duplicates(subset=["osm_id"], keep="first")
)

lat_map = dict(zip(buildings_min["osm_id"], buildings_min["lat"]))
lon_map = dict(zip(buildings_min["osm_id"], buildings_min["lon"]))

# Asignamos lat/lon a cada citizen por su WoS (que es el osm_id)
citizens_f["lat"] = citizens_f["WoS"].map(lat_map)
citizens_f["lon"] = citizens_f["WoS"].map(lon_map)

# --- OPACIDAD POR weight (1º = 100%, último = MIN_OPACITY) ---
N = len(citizens_f)
if N == 1:
    opacities = np.array([1.0], dtype=float)
else:
    frac = (citizens_f["weight"].values - 1) / (N - 1)     # 0 .. 1
    opacities = MIN_OPACITY + (1.0 - frac) * (1.0 - MIN_OPACITY)
citizens_f["fill_opacity"] = opacities

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

# --- MAPA BASE ---
center = [citizens_f["lat"].mean(), citizens_f["lon"].mean()]
m = folium.Map(location=center, zoom_start=13, tiles=None, prefer_canvas=True)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr="© OpenStreetMap, © CARTO",
    name="CartoDB Positron (claro)",
    overlay=False,
    control=False
).add_to(m)

# --- CAPA DE PUNTOS: negro, sin borde, opacidad por weight ---
layer_pts = folium.FeatureGroup(
    name=f"Puntos (negro, opacidad por weight; citizens {before}→{after}; puntos {N})",
    show=True
)

for _, r in citizens_f.iterrows():
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=RADIUS_PX,          # píxeles
        stroke=True,               # borde invisible con weight/opacity
        weight=0,
        opacity=0,
        color=COLOR_HEX,
        fill=True,
        fill_color=COLOR_HEX,      # siempre negro
        fill_opacity=float(r["fill_opacity"]),
        tooltip=f"#{int(r['weight'])}",
        popup=folium.Popup(
            f"weight: {int(r['weight'])}<br>"
            f"Opacidad: {r['fill_opacity']:.2f}<br>"
            f"osm_id (WoS): {r['WoS']}<br>"
            f"weight (idx+1): {int(r['weight'])}",
            max_width=260
        )
    ).add_to(layer_pts)

layer_pts.add_to(m)

# --- CONTORNO ---
folium.GeoJson(
    data=area_geojson,
    name="Utrecht administrativo (R1433619)",
    style_function=lambda feature: {"fillOpacity": 0, "color": "#000000", "weight": 2},
    tooltip="Utrecht administrativo"
).add_to(m)

# --- LEYENDA ---
legend_html = (
    '<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; '
    'background: rgba(255,255,255,0.9); padding: 10px 12px; border: 1px solid #ccc; '
    'border-radius: 6px; font-size: 12px;">'
    '<b>Opacidad por weight (1º→100%, último→'
    f'{int(MIN_OPACITY*100)}%)</b><br>'
    f'Citizens filtrados: {before} → {after}<br>'
    f'Puntos dibujados: {N}'
    '</div>'
)
m.get_root().html.add_child(folium.Element(legend_html))

# --- AJUSTAR VISTA ---
if bbox and len(bbox) == 4:
    south, north, west, east = map(float, bbox)
    m.fit_bounds([[south, west], [north, east]])
else:
    m.fit_bounds([[citizens_f["lat"].min(), citizens_f["lon"].min()],
                  [citizens_f["lat"].max(), citizens_f["lon"].max()]])

folium.LayerControl(collapsed=False).add_to(m)

# --- GUARDAR ---
output_path = "mapa_puntos_opacidad_negro_indep1_exclHome_order_weight.html"
m.save(output_path)
print(
    f"Mapa guardado como '{output_path}' — puntos: {N}, radio_px={RADIUS_PX}, "
    f"último_opacidad={MIN_OPACITY}, citizens filtrados: {before}->{after}"
)
