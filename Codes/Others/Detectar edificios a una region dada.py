import folium
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon

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
    
    return edificios_poligonos

def obtener_centroide(geom):
    # Calcular el centroide del polígono
    if geom.geom_type == 'Polygon':
        return list(geom.centroid.coords)[0]
    elif geom.geom_type == 'MultiPolygon':
        return list(geom.centroid.coords)[0]

def main():
    # Ciudad de interés
    ciudad = "Bilbao"
    
    # Obtener las geometrías de los edificios en la ciudad
    edificios = obtener_edificios_ciudad(ciudad)
    
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
    ruta_guardado = r'C:\Users\asier.divasson\Desktop\V2G-QUETS - Code\mapa_edificios.html'
    mapa.save(ruta_guardado)
    
    print(f"Mapa generado. Abre el archivo '{ruta_guardado}' en tu navegador para ver el resultado.")

if __name__ == "__main__":
    main()
