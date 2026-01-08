import pandas as pd
import numpy as np
import folium
import requests

# === Parámetros visuales ===
RADIUS_PX    = 6          # tamaño del punto en píxeles
MIN_OPACITY  = 0.0        # opacidad del último punto (0.0 = 0%)
COLOR_HEX    = "#000000"  # negro base

# === Interruptores ===
USE_OPACITY      = False   # True: opacidad por orden (negro); False: color negro->blanco por orden
SHOW_OSM_LABELS  = False   # True: añade etiquetas siempre visibles con el osm_id

# === ÁREAS LOCALES ===
AREAS = {
    "Kanaleneiland": [
        (52.07904398, 5.081736117),
        (52.07624318, 5.08308264 ),
        (52.06046958, 5.09756737 ),
        (52.06021839, 5.097758556),
        (52.06008988, 5.11164107 ),
        (52.06328398, 5.113065093),
        (52.06860149, 5.111588679),
        (52.07642504, 5.109425399),
        (52.07861645, 5.108711591),
        (52.08034774, 5.107271173),
        (52.08592257, 5.097037869),
        (52.08498639, 5.096460351),
        (52.08309467, 5.094751129),
        (52.0803543 , 5.087985518),
        (52.07904398, 5.081736117),
    ]
}

# --- RUTAS ---
path_buildings = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_building.parquet"
path_citizens  = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet"

# --- LEER PARQUET ---
buildings = pd.read_parquet(path_buildings)
citizens  = pd.read_parquet(path_citizens)

# --- FILTROS EN CITIZENS ---
mask_ind  = citizens["archetype"].isin(['c_arch_0', 'c_arch_1'])
mask_home = citizens["WoS_subgroup"].astype(str).str.strip() != "Home"
mask_unknown = citizens["WoS_subgroup"].astype(str).str.strip() != "unknown"
before = len(citizens)
citizens_f = citizens.loc[mask_ind & mask_home & mask_unknown].dropna(subset=["WoS"]).copy()
after = len(citizens_f)
if citizens_f.empty:
    raise ValueError("Tras filtrar (independent_type==1 y excluir Home), no quedan citizens.")

# --- ORDEN / PESO ---
citizens_f = citizens_f.reset_index(drop=True)
citizens_f["weight"] = citizens_f.index + 1  # 1..N

# --- TRAER LAT/LON ---
buildings_min = (
    buildings[["osm_id", "lat", "lon"]]
    .dropna(subset=["lat", "lon"])
    .drop_duplicates(subset=["osm_id"], keep="first")
)
lat_map = dict(zip(buildings_min["osm_id"], buildings_min["lat"]))
lon_map = dict(zip(buildings_min["osm_id"], buildings_min["lon"]))
citizens_f["lat"] = citizens_f["WoS"].map(lat_map)
citizens_f["lon"] = citizens_f["WoS"].map(lon_map)

# =========================
#   OPACIDAD / COLOR
# =========================
N = len(citizens_f)
if N == 1:
    frac = np.array([0.0])  # único punto como "primero"
else:
    frac = (citizens_f["weight"].values - 1) / (N - 1)

def gray_hex01(x: float) -> str:
    v = int(round(255 * np.clip(x, 0, 1)))
    return f"#{v:02X}{v:02X}{v:02X}"

if USE_OPACITY:
    if N == 1:
        opacities = np.array([1.0], dtype=float)
    else:
        opacities = MIN_OPACITY + (1.0 - frac) * (1.0 - MIN_OPACITY)
    citizens_f["fill_opacity"] = opacities
    citizens_f["fill_color"]   = COLOR_HEX
else:
    colors = [gray_hex01(x) for x in frac]  # negro -> blanco
    citizens_f["fill_opacity"] = 1.0
    citizens_f["fill_color"]   = colors

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

# --- MAPA ---
center = [citizens_f["lat"].mean(), citizens_f["lon"].mean()]
m = folium.Map(location=center, zoom_start=13, tiles=None, prefer_canvas=True)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr="© OpenStreetMap, © CARTO",
    name="CartoDB Positron (claro)",
    overlay=False,
    control=False
).add_to(m)

modo = "Opacidad por orden (negro)" if USE_OPACITY else "Color negro→blanco por orden"

# --- CAPA DE PUNTOS ---
layer_pts = folium.FeatureGroup(
    name=f"Puntos ({modo}; citizens {before}→{after}; puntos {N})",
    show=True
)

# (opcional) capa de etiquetas visibles
layer_labels = folium.FeatureGroup(
    name="Etiquetas osm_id",
    show=SHOW_OSM_LABELS
)

for _, r in citizens_f.iterrows():
    tooltip = folium.Tooltip(
        f"osm_id: {r['WoS']} • peso: {int(r['weight'])}",
        sticky=True,
        direction="top"
    )

    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=RADIUS_PX,
        stroke=True,
        weight=0,
        opacity=0,
        color=r["fill_color"],
        fill=True,
        fill_color=r["fill_color"],
        fill_opacity=float(r["fill_opacity"]),
        tooltip=tooltip,
        popup=folium.Popup(
            f"<b>osm_id:</b> {r['WoS']}<br>"
            f"<b>weight:</b> {int(r['weight'])}<br>"
            f"<b>Modo:</b> {modo}<br>"
            f"<b>Opacidad:</b> {float(r['fill_opacity']):.2f}<br>"
            f"<b>Color:</b> {r['fill_color']}",
            max_width=280
        )
    ).add_to(layer_pts)

    if SHOW_OSM_LABELS:
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=folium.DivIcon(
                html=(
                    '<div style="'
                    'font-size:10px; line-height:10px; '
                    'background: rgba(255,255,255,0.7); '
                    'border: 1px solid #aaa; border-radius: 3px; '
                    'padding: 2px 3px; white-space: nowrap;">'
                    f'osm_id: {r["WoS"]}'
                    '</div>'
                )
            )
        ).add_to(layer_labels)

layer_pts.add_to(m)
if SHOW_OSM_LABELS:
    layer_labels.add_to(m)

# --- CAPA DE ÁREAS: Kanaleneiland ---
layer_areas = folium.FeatureGroup(name="Áreas locales", show=True)
for area_name, coords in AREAS.items():
    # Cierra el anillo si no viene cerrado
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    folium.Polygon(
        locations=coords,             # (lat, lon)
        color="#000000",              # borde naranja
        weight=2,
        dash_array="2,2",             # borde discontinuo
        fill=False,
        tooltip=area_name
    ).add_to(layer_areas)
    
    folium.Polygon(
        locations=coords,             # (lat, lon)
        color="#FFFFFF",              # borde naranja
        weight=4,
        dash_array="2,2",             # borde discontinuo
        fill=False,
        tooltip=area_name
    ).add_to(layer_areas)
    
layer_areas.add_to(m)

# --- CONTORNO ADMINISTRATIVO ---
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
    f'<b>{modo}</b><br>'
    'Tooltip: muestra <i>osm_id</i> y <i>peso</i> al pasar el ratón.<br>'
    f'Etiquetas visibles: {"sí" if SHOW_OSM_LABELS else "no"}<br>'
    'Áreas: Kanaleneiland (borde naranja discontinuo)<br>'
    f'Citizens filtrados: {before} → {after}<br>'
    f'Puntos dibujados: {N}'
    '</div>'
)
m.get_root().html.add_child(folium.Element(legend_html))

# --- AJUSTAR VISTA (puntos + áreas + bbox de Utrecht si disponible) ---
fit_lats = list(citizens_f["lat"].dropna().values)
fit_lons = list(citizens_f["lon"].dropna().values)
for coords in AREAS.values():
    for lat, lon in coords:
        fit_lats.append(lat)
        fit_lons.append(lon)

if fit_lats and fit_lons:
    lat_min, lat_max = min(fit_lats), max(fit_lats)
    lon_min, lon_max = min(fit_lons), max(fit_lons)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
elif bbox and len(bbox) == 4:
    south, north, west, east = map(float, bbox)
    m.fit_bounds([[south, west], [north, east]])
else:
    m.fit_bounds([[citizens_f["lat"].min(), citizens_f["lon"].min()],
                  [citizens_f["lat"].max(), citizens_f["lon"].max()]])

folium.LayerControl(collapsed=False).add_to(m)

# --- GUARDAR ---
output_path = (
    "mapa_puntos_opacidad_negro_con_osm_y_areas.html" if USE_OPACITY
    else "mapa_puntos_color_bw_con_osm_y_areas.html"
)
m.save(output_path)
print(
    f"Mapa guardado como '{output_path}' — puntos: {N}, radio_px={RADIUS_PX}, "
    f"modo='{modo}', etiquetas_osm={'ON' if SHOW_OSM_LABELS else 'OFF'}, "
    f"áreas={list(AREAS.keys())}, citizens filtrados: {before}->{after}"
)
