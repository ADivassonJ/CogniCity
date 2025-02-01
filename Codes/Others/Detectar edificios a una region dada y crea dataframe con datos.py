import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely.wkt

def obtener_centroide_o_punto(geom):
    # Si la geometría es un polígono (Polygon o MultiPolygon)
    if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
        # Convertir la geometría en un objeto shapely si aún no lo es
        if not isinstance(geom, Point):
            centroid = geom.centroid
            return centroid.x, centroid.y  # Devolver las coordenadas del centroide
        else:
            # Devolver las coordenadas del punto
            return geom.x, geom.y
    else:
        return None  # Devolver None si la geometría no es reconocida

def obtener_edificios_ciudad(ciudad):
    # Obtener los datos de todos los edificios dentro de la ciudad
    edificios = ox.features_from_place(ciudad, tags={'building': True})
    
    # Obtener los ID de los edificios
    building_ID = edificios.index.get_level_values('osmid').tolist()
    
    # Crear una lista de valores y nombres de columnas
    values = {'osmid': building_ID, 'coordenadas': []}
    names = ['osmid', 'coordenadas']
    
    # Obtener coordenadas de los edificios
    for geom in edificios['geometry']:
        resultado = obtener_centroide_o_punto(geom)
        values['coordenadas'].append(resultado)
    
    # Definir variables de interés
    variables = ['building', 'amenity', 'geometry']
    
    # Obtener las variables de interés y agregarlas a la lista de valores y nombres de columnas
    for vari in variables:
        data_array = edificios[vari].tolist()
        names.append(vari)
        values[vari] = data_array
    
    # Crear el DataFrame con todas las variables
    buildings_data = pd.DataFrame(values, columns=names)

    return buildings_data

def crear_archivo_excel(data_frame): 
    # Guardar el DataFrame en un archivo Excel
    ruta_guardado_excel = 'C:/Users/asier.divasson/Desktop/V2G-QUETS - Code/datos_edificios.xlsx'
    data_frame.to_excel(ruta_guardado_excel, index=False)
    print(f"Archivo Excel creado. Puedes encontrarlo en '{ruta_guardado_excel}'.")

def main():
    # Ciudad de interés
    ciudad = "Bilbao"
    
    # Obtener las geometrías de los edificios en la ciudad
    buildings_df = obtener_edificios_ciudad(ciudad)
    
    # Crear un archivo Excel con los datos de los edificios
    crear_archivo_excel(buildings_df)
    
if __name__ == "__main__":
    main()
