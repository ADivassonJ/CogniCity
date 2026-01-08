import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import osmnx as ox

# -----------------------------
# 0. Configuración y rutas
# -----------------------------
base_dir = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Aradas\population"

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
# 5. Aveiro (R5325138) y Aradas
# -----------------------------
# Aveiro por OSM ID
gdf_Aveiro = ox.geocode_to_gdf("R5325138", by_osmid=True)
if gdf_Aveiro.crs is None:
    gdf_Aveiro.set_crs(epsg=4326, inplace=True)
else:
    gdf_Aveiro = gdf_Aveiro.to_crs(epsg=4326)

# Polígono de Aradas (WGS84)
coords_Aradas_latlon = [ (40.6260277,-8.6691095),
                    (40.6242125,-8.666836),
                    (40.6236329,-8.6659728),
                    (40.6234128,-8.6657849),
                    (40.6231059,-8.6656375),
                    (40.6225013,-8.6650637),
                    (40.6222507,-8.6646582),
                    (40.6220032,-8.663601),
                    (40.6216362,-8.6628016),
                    (40.6210708,-8.6622693),
                    (40.6202052,-8.6619314),
                    (40.61922,-8.6617375),
                    (40.6189017,-8.6617527),
                    (40.6186492,-8.6616989),
                    (40.6179653,-8.6614756),
                    (40.6175178,-8.661246),
                    (40.6171852,-8.6612232),
                    (40.6171653,-8.6610968),
                    (40.6173448,-8.6608471),
                    (40.6173608,-8.6607396),
                    (40.6172055,-8.6606055),
                    (40.6171618,-8.660501),
                    (40.6170468,-8.6603655),
                    (40.6140575,-8.6575236),
                    (40.613851,-8.6572765),
                    (40.6135878,-8.6568526),
                    (40.613393,-8.6566081),
                    (40.6130864,-8.6563332),
                    (40.6129204,-8.6560929),
                    (40.6127237,-8.6558836),
                    (40.6126708,-8.6557805),
                    (40.6126488,-8.6552235),
                    (40.6124499,-8.6548499),
                    (40.6123021,-8.6546783),
                    (40.6120758,-8.6545599),
                    (40.6119478,-8.6545386),
                    (40.6100666,-8.6545613),
                    (40.6096812,-8.6543448),
                    (40.6089189,-8.6535312),
                    (40.6082786,-8.652988),
                    (40.6079532,-8.6523902),
                    (40.6076241,-8.6519091),
                    (40.6072771,-8.6514912),
                    (40.60668,-8.6504698),
                    (40.6065747,-8.6504181),
                    (40.6063881,-8.6502435),
                    (40.6062332,-8.6500269),
                    (40.6054844,-8.6483856),
                    (40.6053114,-8.6480454),
                    (40.6052046,-8.6479189),
                    (40.6050178,-8.6476255),
                    (40.6045178,-8.6466887),
                    (40.6042572,-8.6463173),
                    (40.6040732,-8.6461176),
                    (40.6036906,-8.6458503),
                    (40.6035381,-8.6456551),
                    (40.6032476,-8.6454093),
                    (40.6030542,-8.6451902),
                    (40.6027228,-8.6447241),
                    (40.6025011,-8.6443065),
                    (40.6023171,-8.6440466),
                    (40.6022493,-8.6439981),
                    (40.6020254,-8.6439602),
                    (40.6013136,-8.6437384),
                    (40.6010098,-8.643593),
                    (40.6007614,-8.643566),
                    (40.6006448,-8.6435104),
                    (40.6003543,-8.6432293),
                    (40.5998495,-8.6423911),
                    (40.599316,-8.6417578),
                    (40.5991053,-8.6416322),
                    (40.5986892,-8.6416011),
                    (40.5984492,-8.6414653),
                    (40.5983617,-8.6413203),
                    (40.5983123,-8.6409809),
                    (40.5982499,-8.6408236),
                    (40.5980912,-8.640572),
                    (40.5979016,-8.6403789),
                    (40.5972284,-8.6400028),
                    (40.5969958,-8.6399705),
                    (40.5968975,-8.6399956),
                    (40.5964402,-8.6396011),
                    (40.5959139,-8.6389858),
                    (40.5954893,-8.6385972),
                    (40.5945613,-8.6379813),
                    (40.5937743,-8.6372131),
                    (40.5937678,-8.6366875),
                    (40.5937272,-8.6365929),
                    (40.5932003,-8.6359088),
                    (40.5929938,-8.6352565),
                    (40.5924366,-8.6339021),
                    (40.5923815,-8.6332999),
                    (40.5923066,-8.6329949),
                    (40.5921122,-8.6323842),
                    (40.5920556,-8.6320796),
                    (40.5918131,-8.6314779),
                    (40.591809,-8.6313249),
                    (40.5918757,-8.6309966),
                    (40.591891,-8.6305948),
                    (40.5917616,-8.6301748),
                    (40.5917761,-8.629417),
                    (40.5917162,-8.6291502),
                    (40.5916803,-8.6287324),
                    (40.5914751,-8.6282729),
                    (40.5914972,-8.6279974),
                    (40.5916615,-8.6276765),
                    (40.5916434,-8.6274363),
                    (40.5914849,-8.6269246),
                    (40.5883757,-8.6244006),
                    (40.5874733,-8.6235797),
                    (40.587295,-8.623363),
                    (40.587194,-8.6231671),
                    (40.5868211,-8.6223074),
                    (40.5866228,-8.622139),
                    (40.5864064,-8.6220873),
                    (40.5863086,-8.6217312),
                    (40.5862954,-8.6214654),
                    (40.586398,-8.6206219),
                    (40.5865712,-8.6199791),
                    (40.5867584,-8.6195026),
                    (40.5869593,-8.6191804),
                    (40.5870914,-8.6192253),
                    (40.5871382,-8.619014),
                    (40.5871432,-8.6186678),
                    (40.5871321,-8.618444),
                    (40.5871236,-8.618112),
                    (40.587257,-8.6180764),
                    (40.5895434,-8.6166483),
                    (40.5897483,-8.6166463),
                    (40.5920307,-8.6153937),
                    (40.5934133,-8.6147084),
                    (40.5939013,-8.6143535),
                    (40.5945506,-8.6140867),
                    (40.5960723,-8.6132704),
                    (40.5969792,-8.6135683),
                    (40.5977968,-8.6136263),
                    (40.6005637,-8.6142546),
                    (40.6006406,-8.6142345),
                    (40.6013907,-8.6145417),
                    (40.6019361,-8.6148515),
                    (40.6013476,-8.6157457),
                    (40.6011136,-8.6161966),
                    (40.6003201,-8.6181819),
                    (40.6052078,-8.6223089),
                    (40.6059547,-8.6230277),
                    (40.6062372,-8.6233187),
                    (40.6069917,-8.6242913),
                    (40.6089161,-8.6269292),
                    (40.6097193,-8.6277959),
                    (40.6101187,-8.6281644),
                    (40.6236293,-8.6395615),
                    (40.6238389,-8.6384898),
                    (40.6239732,-8.6383452),
                    (40.6241713,-8.6382761),
                    (40.6250741,-8.6388559),
                    (40.6249215,-8.6391849),
                    (40.6249961,-8.6394292),
                    (40.625771,-8.6411829),
                    (40.6261616,-8.6416184),
                    (40.6262771,-8.6417962),
                    (40.626503,-8.64201),
                    (40.6265785,-8.6422958),
                    (40.6266742,-8.6430671),
                    (40.6268177,-8.6434418),
                    (40.6269435,-8.6439459),
                    (40.6270621,-8.6446783),
                    (40.6274912,-8.6454454),
                    (40.6278877,-8.6464471),
                    (40.6278272,-8.6474518),
                    (40.6278976,-8.6478379),
                    (40.6279316,-8.6478432),
                    (40.6278874,-8.6485833),
                    (40.6278211,-8.648809),
                    (40.6270303,-8.6498279),
                    (40.6270053,-8.6500061),
                    (40.6269994,-8.6500623),
                    (40.6269936,-8.6501184),
                    (40.6269697,-8.650423),
                    (40.6268628,-8.6507406),
                    (40.6267409,-8.6508643),
                    (40.6265982,-8.6509256),
                    (40.6263341,-8.650965),
                    (40.6262965,-8.6511239),
                    (40.6262627,-8.6516432),
                    (40.626272,-8.6518649),
                    (40.6263288,-8.6521423),
                    (40.6265221,-8.6525869),
                    (40.6265496,-8.6526414),
                    (40.6267315,-8.6529555),
                    (40.6272151,-8.6545382),
                    (40.6272815,-8.6549749),
                    (40.6273554,-8.6561734),
                    (40.6273418,-8.6568172),
                    (40.6272684,-8.6578625),
                    (40.6271319,-8.658907),
                    (40.6268978,-8.6597033),
                    (40.6263523,-8.6621294),
                    (40.6264761,-8.6626895),
                    (40.6269026,-8.6639876),
                    (40.6270787,-8.664744),
                    (40.6271497,-8.6653192),
                    (40.6271319,-8.6657763),
                    (40.6270175,-8.6665162),
                    (40.6268942,-8.6669446),
                    (40.6267452,-8.6672412),
                    (40.6265183,-8.6674498),
                    (40.6260277,-8.6691095),
                ]

coords_Aradas = [(lon, lat) for (lat, lon) in coords_Aradas_latlon]
Aradas_geom = Polygon(coords_Aradas)

gdf_Aradas = gpd.GeoDataFrame(
    {"name": ["Aradas"]},
    geometry=[Aradas_geom],
    crs="EPSG:4326"
)

# Reproyectar a CRS métrico
gdf_Aveiro_proj = gdf_Aveiro.to_crs(target_crs)
gdf_Aradas_proj = gdf_Aradas.to_crs(target_crs)

# -----------------------------
# 6. Dibujar mapa (sin marco, en grises, puntos más oscuros = más ciudadanos)
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

# Perímetro de Aveiro (gris medio)
gdf_Aveiro_proj.boundary.plot(
    ax=ax,
    linewidth=1,
    edgecolor="0.6"  # gris
)

# Perímetro de Aradas (gris oscuro, línea discontinua)
gdf_Aradas_proj.boundary.plot(
    ax=ax,
    linewidth=2,
    linestyle="--",
    edgecolor="0.2"  # gris más oscuro
)

# Puntos de WoS: intensidad según número de ciudadanos (n_citizens)
counts = gdf_wos_proj["n_citizens"]

# Normalización para el mapa de color (min=blanco/gris claro, max=negro)
norm = plt.Normalize(vmin=counts.min()-10, vmax=counts.max())

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

# Ajustar límites al perímetro de Aveiro
minx, miny, maxx, maxy = gdf_Aveiro_proj.total_bounds
dx = (maxx - minx) * 0.05
dy = (maxy - miny) * 0.05

ax.set_xlim(minx - dx, maxx + dx)
ax.set_ylim(miny - dy, maxy + dy)

# 1) Sin marco/coordenadas
ax.set_axis_off()

ax.set_aspect("equal", adjustable="box")
ax.set_title("Aveiro (R5325138) + Aradas + intensidad WoS (nº de ciudadanos)")

plt.tight_layout()
plt.show()
