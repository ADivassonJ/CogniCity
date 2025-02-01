import os
import random
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon

# Obtiene el centroide de una geometria dada (para obtener las coordenadas aprox de la ubicacion del servicio).
def obtener_centroide_o_punto(geom):
    # Si la geometría es un polígono (Polygon o MultiPolygon)
    if isinstance(geom, Polygon) or isinstance(geom, MultiPolygon):
        # Convertir la geometría en un objeto shapely si aún no lo es
        centroid = geom.centroid
        return centroid.y, centroid.x  # Devolver las coordenadas del centroide
    elif isinstance(geom, Point):
        return geom.y, geom.x  # Devolver las coordenadas del punto

# Obtiene las construcciones etiquetadas como building en el area dada
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

# Compara el ID de un edificio con el tipo de servicio que devería proveer
def type_service_ok(directory, df_area, osmid, type):
    # Look for the type of the amenity based on OSM data
    to_study = df_area.loc[df_area['osmid'] == osmid, 'amenity']
    if pd.isna(to_study.iloc[0]) or to_study.iloc[0] == 'yes':
        # Al detectarse la instancia de 'amenity' vacía (NaN), se pasa a observar el valor de 'building'
        to_study = df_area.loc[df_area['osmid'] == osmid, 'building']
        if pd.isna(to_study.iloc[0]):
            # Al detectarse la instancia de 'building' vacía (NaN), se considera imposible caracterizar el edificio
            print("NO DATA FROM BUILDING")
            return False
    
    # Convertir to_study en un string
    to_study_str = str(to_study.iloc[0])
    print(to_study_str)
    
    if to_study_str == 'yes': 
        random_value = random.choice(['Working', 'Commerce'])  # Solución temporal, a futuro es necesario especificar en porcentajes
        if random_value == type:
            return True
        else:
            return False    
    else:
        filepath = os.path.join(directory, 'building characterization.csv')
        df_chara = pd.read_csv(filepath)

        mask = df_chara.iloc[:, 0].str.contains(to_study_str, na=False)
        valores = df_chara.loc[mask, type]
        
        if not valores.empty:
            in_type = df_chara.loc[mask, type].iloc[0]
        else:
            in_type = None
            
        if in_type == 'x':
            return True
        else:
            return False


def main():
    # Definir variables
    ciudad = "Bilbao" # Área de interés
    directory = 'C:/Users/asier.divasson/Downloads' # Directorio
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']
    
    # Obtener las geometrías de los edificios en la ciudad
    print("Gathering data from selected area...")
    buildings_df = obtener_edificios_ciudad(ciudad)
    # Crear df de los datos por cada edificio y sus servicios
    services_buildings = {}
    for i in range(30):
    #for i in range(len(buildings_df)):
        ava_serv = []
        for type in type_services:
            is_ok = type_service_ok(directory, buildings_df, buildings_df['osmid'][i], type)
            if is_ok:
                ava_serv.append(type)
        print("-"*20)
        print(ava_serv)
        print("-"*20)
        services_buildings[buildings_df['osmid'][i]] = ava_serv
             
          
    
    
if __name__ == "__main__":
    main()
