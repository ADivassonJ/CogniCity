import geopandas as gpd
import pandas as pd
import folium
from folium import Choropleth, LayerControl, GeoJson, GeoJsonTooltip

# 1. Cargar el archivo
path = "cbs_vk500_2024_v1.gpkg"
gdf = gpd.read_file(path, layer="cbs_vk500_2024")

# 2. Reemplazar valores nulos del CBS
gdf = gdf.replace([-99997, -99995], None)

# 3. Convertir columnas clave a numéricas (si no lo son)
gdf["aantal_inwoners"] = pd.to_numeric(gdf["aantal_inwoners"], errors="coerce")
gdf["omgevingsadressendichtheid"] = pd.to_numeric(gdf["omgevingsadressendichtheid"], errors="coerce")

# 4. Reproyectar a WGS84
gdf = gdf.to_crs(epsg=4326)

# 5. Crear mapa base
m = folium.Map(location=[52.1, 5.3], zoom_start=8, tiles="CartoDB positron")

# 6. Capa de coropletas
Choropleth(
    geo_data=gdf,
    data=gdf,
    columns=["crs28992res500m", "aantal_inwoners"],
    key_on="feature.properties.crs28992res500m",
    fill_color="YlGnBu",
    nan_fill_color="lightgray",
    fill_opacity=0.8,
    line_opacity=0.1,
    legend_name="Aantal inwoners (población total)",
).add_to(m)

# 7. Tooltips con información extra
GeoJson(
    gdf,
    name="CBS 500m grid",
    tooltip=GeoJsonTooltip(
        fields=["crs28992res500m", "aantal_inwoners", "omgevingsadressendichtheid"],
        aliases=["Cel:", "Población:", "Densidad direcciones:"],
        localize=True
    ),
    style_function=lambda x: {"fillOpacity": 0, "color": "transparent"}
).add_to(m)

# 8. Control de capas
LayerControl().add_to(m)

# 9. Guardar el mapa
m.save("mapa_cbs_vk500.html")
print("✅ Mapa guardado como 'mapa_cbs_vk500.html'. Ábrelo en tu navegador.")
