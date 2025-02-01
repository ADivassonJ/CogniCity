import folium
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from openpyxl import Workbook
# Configurar pandas para mostrar todas las columnas y filas
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def obtener_edificios_ciudad(ciudad):
    # Obtener los datos de todos los edificios dentro de la ciudad
    edificios = ox.features_from_place(ciudad, tags={'building': True})
    
    # Filtrar solo los polígonos y multipolígonos
    edificios_poligonos = edificios[edificios['geometry'].apply(lambda x: x.geom_type in ['Polygon', 'MultiPolygon'])]
    
    # Intercambiar latitud y longitud en las coordenadas de los edificios
    for i, edificio in edificios_poligonos.iterrows():
        if edificio.geometry.geom_type == 'Polygon':
            coords = [(coord[1], coord[0]) for coord in edificio.geometry.exterior.coords]
            edificios_poligonos.at[i, 'geometry'] = Polygon(coords)
        elif edificio.geometry.geom_type == 'MultiPolygon':
            new_polygons = []
            for polygon in edificio.geometry.geoms:
                coords = [(coord[1], coord[0]) for coord in polygon.exterior.coords]
                new_polygons.append(Polygon(coords))
            edificios_poligonos.at[i, 'geometry'] = MultiPolygon(new_polygons)
    
    # Obtener los ID de los edificios
    building_ID = edificios.index.get_level_values('osmid').tolist()
    buildings_type = edificios['building'].tolist()
    buildinds_amenity_type = edificios['amenity'].tolist()
    buildings_geometry = edificios['geometry'].tolist()
    
    return edificios_poligonos, building_ID, buildings_type, buildinds_amenity_type, buildings_geometry


def obtener_centroide(geom):
    # Calcular el centroide del polígono
    if geom.geom_type == 'Polygon':
        return list(geom.centroid.coords)[0]
    elif geom.geom_type == 'MultiPolygon':
        return list(geom.centroid.coords)[0]

def dataframe_to_rows(dataframe, index=True, header=True):
    # Obtener los nombres de las columnas
    columns = list(dataframe.columns)

    # Incluir los nombres de las columnas si header es True
    if header:
        yield tuple(columns if header is True else header)

    # Iterar sobre las filas del DataFrame
    for _, row in dataframe.iterrows():
        # Crear una tupla con los valores de la fila
        values = tuple(row)
        # Incluir el índice si index es True
        if index:
            values = (row.name,) + values
        # Devolver la tupla de valores
        yield values

def crear_archivo_excel(building_ID, buildings_type, buildinds_amenity_type, buildings_geometry):
    # Crear un DataFrame de pandas con los datos de los edificios
    df = pd.DataFrame({'building_ID': building_ID, 'buildings_type': buildings_type, 'buildinds_amenity_type': buildinds_amenity_type, 'buildings_geometry': buildings_geometry})
    
    # Guardar el DataFrame en un archivo Excel
    ruta_guardado_excel = 'C:/Users/asier.divasson/Desktop/V2G-QUETS - Code/datos_edificios.xlsx'
    df.to_excel(ruta_guardado_excel, index=False)
    print(f"Archivo Excel creado. Puedes encontrarlo en '{ruta_guardado_excel}'.")

def main():
    # Ciudad de interés
    ciudad = "Bilbao"
    
    # Obtener las geometrías de los edificios en la ciudad
    edificios, building_ID, buildings_type, buildinds_amenity_type, buildings_geometry = obtener_edificios_ciudad(ciudad)
    
    # Crear un archivo Excel con los datos de los edificios
    crear_archivo_excel(building_ID, buildings_type, buildinds_amenity_type, buildings_geometry)
    
    # Crear un mapa centrado en la ciudad
    centro_ciudad = (edificios['geometry'].iloc[0].centroid.x, edificios['geometry'].iloc[0].centroid.y)
    mapa = folium.Map(location=centro_ciudad, zoom_start=13)
    
    # Agregar marcadores para cada conjunto de coordenadas
    for i, edificio in edificios.iterrows():
        if edificio.geometry.geom_type == 'Polygon':
            coordenadas = list(edificio.geometry.exterior.coords)
            folium.Polygon(locations=coordenadas, color='blue', fill=True, fill_color='blue').add_to(mapa)
            # Obtener el centroide de los polígonos y agregar un marcador en esa ubicación
            centroide = obtener_centroide(edificio.geometry)
            #folium.Marker(location=centroide, popup=f'Edificio {i+1}', icon=folium.Icon(color='red')).add_to(mapa)
        elif edificio.geometry.geom_type == 'MultiPolygon':
            for polygon in edificio.geometry.geoms:
                coordenadas = list(polygon.exterior.coords)
                folium.Polygon(locations=coordenadas, color='blue', fill=True, fill_color='blue').add_to(mapa)
                centroide = obtener_centroide(polygon)
                #folium.Marker(location=centroide, popup=f'Edificio {i+1}', icon=folium.Icon(color='red')).add_to(mapa)
    
    # Guardar el mapa como un archivo HTML
    ruta_guardado_mapa = 'mapa_edificios' + ciudad + '.html'
    mapa.save(ruta_guardado_mapa)
    print(f"Mapa generado. Abre el archivo '{ruta_guardado_mapa}' en tu navegador para ver el resultado.")

if __name__ == "__main__":
    main()