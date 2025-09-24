import geopandas as gpd
import osmnx as ox
from shapely.geometry import box
from shapely.ops import unary_union
import folium

def split_polygon_to_grid(polygon, size_m=1000, crs_in="EPSG:4326"):
    """
    Divide un polígono en celdas cuadradas de `size_m` x `size_m` (en metros).
    Devuelve un GeoDataFrame con las intersecciones (solo partes dentro del polígono).
    """
    # Asegura GeoSeries con CRS
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs_in).to_crs(epsg=3857)
    poly_m = gdf.geometry.iloc[0]

    minx, miny, maxx, maxy = poly_m.bounds
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x+size_m, y+size_m)
            inter = cell.intersection(poly_m)
            if not inter.is_empty:
                # Puede devolver multipolígonos; normalizamos a polígonos simples
                if inter.geom_type == "MultiPolygon":
                    cells.extend(list(inter.geoms))
                else:
                    cells.append(inter)
            y += size_m
        x += size_m

    if not cells:
        cells = [poly_m]

    grid = gpd.GeoDataFrame(geometry=cells, crs="EPSG:3857").to_crs(epsg=4326)
    # Área en km² por celda (aprox. por reproyección métrica previa)
    grid["area_km2"] = gpd.GeoDataFrame(geometry=cells, crs="EPSG:3857").area.values / 1e6
    return grid

# 1) Descarga del polígono
gdf_place = ox.geocode_to_gdf("Tartu")
polygon_ll = gdf_place.iloc[0].geometry  # WGS84

# 2) División en celdas de 1 km²
grid_gdf = split_polygon_to_grid(polygon_ll, size_m=1000)

print(f"Celdas generadas: {len(grid_gdf)} | Área total aprox: {grid_gdf['area_km2'].sum():.2f} km²")

# 3) Mapa interactivo con Folium
# Centro del mapa en el centroide del polígono original
centroid = gpd.GeoSeries([polygon_ll], crs="EPSG:4326").centroid.iloc[0]
m = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, tiles="CartoDB positron")

# Polígono original (borde grueso)
folium.GeoJson(
    polygon_ll,
    name="Área original",
    style_function=lambda x: {"fill": False, "color": "#1f77b4", "weight": 3}
).add_to(m)

# Celdas (relleno semitransparente)
folium.GeoJson(
    grid_gdf,
    name="Celdas 1 km²",
    style_function=lambda x: {
        "fillColor": "#ff7f0e",
        "color": "#ff7f0e",
        "weight": 1,
        "fillOpacity": 0.25
    },
    tooltip=folium.features.GeoJsonTooltip(fields=["area_km2"], aliases=["Área (km²)"], localize=True)
).add_to(m)

folium.LayerControl().add_to(m)

# 4) Mostrar en notebook (si procede) o guardar a HTML
m  # En Jupyter muestra el mapa
m.save("kanaleneiland_grid_1km.html")
print("Mapa guardado en 'kanaleneiland_grid_1km.html'")
