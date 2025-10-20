import os
import sys
import geopandas as gpd
import random
import osmnx as ox
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor, as_completed  # o ThreadPoolExecutor
from functools import partial
import pandas as pd
from tqdm import tqdm

def update_warnings(message, path):
    """
    Escribe o agrega un mensaje al archivo 'warnings.txt' dentro de path['results'].
    
    Args:
        message (str): Mensaje a escribir en el archivo.
        path (dict): Diccionario con la clave 'results' que indica la ruta de la carpeta.
    """
    # Construir la ruta completa al archivo
    warnings_file = os.path.join(path['results'], 'warnings.txt')
    
    # Abrir el archivo en modo 'append' para agregar al final
    with open(warnings_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def ring_from_poi(lat, lon, x, y, crs="EPSG:4326"):
    poi = gpd.GeoSeries([Point(lon, lat)], crs=crs)
    poi_m = poi.to_crs(epsg=3857)
    outer = poi_m.buffer(x + y).iloc[0]
    inner = poi_m.buffer(max(x - y, 0)).iloc[0]
    ring = outer.difference(inner)
    ring_wgs84 = gpd.GeoSeries([ring], crs="EPSG:3857").to_crs(crs)
    return ring_wgs84.iloc[0]  # <- devuelve shapely.geometry.Polygon

def choose_id_in_ring(cand_df, ring_poly, include_border=True):
        """
        Devuelve un osm_id aleatorio de cand_df dentro del anillo/polígono.
        - cand_df: DataFrame con ['osm_id','lat','lon'] en EPSG:4326
        - ring_poly: shapely geometry, GeoSeries o GeoDataFrame
        - include_border: True -> cuenta puntos en el borde (intersects), False -> strictly within
        """
        if cand_df is None or len(cand_df) == 0:
            return None

        # 1) Candidatos como GeoDataFrame (WGS84)
        df = cand_df[['osm_id', 'lat', 'lon']].copy()
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['lon'], df['lat']),
            crs="EPSG:4326"
        )

        # 2) Extraer geometría del anillo y su CRS
        if isinstance(ring_poly, gpd.GeoDataFrame):
            ring_geom = ring_poly.geometry.unary_union
            ring_crs = ring_poly.crs
        elif isinstance(ring_poly, gpd.GeoSeries):
            ring_geom = ring_poly.unary_union
            ring_crs = ring_poly.crs
        else:
            ring_geom = ring_poly
            ring_crs = None

        if ring_geom is None:
            return None

        # 3) Alinear CRS si hace falta
        if ring_crs is not None and gdf.crs is not None and ring_crs != gdf.crs:
            ring_geom = gpd.GeoSeries([ring_geom], crs=ring_crs).to_crs(gdf.crs).iloc[0]

        # 4) Dos fases: sindex (intersects) -> filtro exacto (within/intersects)
        try:
            # usar intersects para prefiltrar SIEMPRE (más estable)
            idx_hits = list(gdf.sindex.query(ring_geom, predicate="intersects"))
            cand = gdf.iloc[idx_hits] if idx_hits else gdf  # si sindex vacío, cae al refinado global
        except Exception:
            cand = gdf  # sin sindex, refinado global

        # 5) Predicado exacto
        if include_border:
            mask = cand.geometry.intersects(ring_geom)  # cuenta borde
        else:
            mask = cand.geometry.within(ring_geom)      # estrictamente dentro

        subset = cand.loc[mask.values]
        if subset.empty:
            return None

        return subset['osm_id'].sample(1).iat[0]

def get_vehicle_stats(archetype, transport_archetypes, variables):
    results = {}   
    
    # Filtrar la fila correspondiente al arquetipo
    row = transport_archetypes[transport_archetypes['name'] == archetype]
    if row.empty:
        
        print(f"archetype: {archetype}")
        input(transport_archetypes)

        print(f"Strange mistake happend")
        return {}

    row = row.iloc[0]  # Extrae la primera (y única esperada) fila como Series

    for variable in variables:
        mu = float(row[f'{variable}_mu'])
        sigma = float(row[f'{variable}_sigma'])
        try:
            max_var = float(row[f'{variable}_max'])
        except Exception as e:
            max_var = float('inf')
        try:
            min_var = float(row[f'{variable}_min'])
        except Exception as e:
            min_var = float(0)
        
        var_result = np.random.normal(mu, sigma)
        var_result = max(min(var_result, max_var), min_var)
        results[variable] = var_result

    return results

def create_family_level_1_schedule(day, pop_building, family_df, activities, system_management, citizen_archetypes):
    """
      Summary: Crea la version inicial de los schedules de cada familia (level 1), 
    donde los agentes pueden realizar las actividades tal y como les apetezca, 
    independiente de si son o no capaces de hacerlo sin ayuda.

    Args:
        pop_building (DataFrame): Describe la poblacion de EDIFICIOS disponible, junto a sus caracteristicas.
        family_df (DataFrame): Describe las caracteristicas de cada agente CIUDADANO participe en una familia especifica.
        activities (list): Describe las actividades diarias a acometer por los agentes.

    Returns:
        todolist_family (DataFrame): Descripcion de level 1 del daily schedule de los agentes.
    """
    
    
    activities = ['WoS', 'Dutties']
    home_activities = ['Home_in', 'Home_out']
    
    wos_halftime = False

    if day in ['Fr', 'Sa', 'Su']:
        if day in ['Sa', 'Su']:
            if (system_management[f'{day}_morning'].iloc[0] == False) & (system_management[f'{day}_afternoon'].iloc[0] == False):
                activities = ['Dutties']
        if system_management[f'{day}_afternoon'].iloc[0] == False:
            wos_halftime = True
    
    # Inicializamos el df en el que meteremos los schedules
    todolist_family = []
    
    # Pasamos por cada agente que constitulle la familia 
    for _, row_f_df in family_df.iterrows():
        
        todolist_agent = []
        
        # Vamos actividad por actividad            
        for activity in activities:
            activity_amount = 1
            
            # Miramos si tenemos alguna cantidad de esta tarea a realizar (solo existe si la cantidad es != 1)
            if activity == 'Dutties':
                results = get_vehicle_stats(row_f_df['archetype'], citizen_archetypes, ['Dutties_amount', 'Dutties_time'])
                results = {k: int(round(v)) for k, v in results.items()}
                activity_amount = results['Dutties_amount']
                time2spend = results['Dutties_time']
            elif (activity == 'WoS') & (day in ['Sa', 'Su']) & (row_f_df['WoS_subgroup'] != 'Home'):
                WoS_key = row_f_df['WoS_subgroup'].split("_", 1)[0]             
                if row_f_df['class'] == 'Salariat' or WoS_key == 'office':
                    # Si es clase alta o trabaja en oficina, no trabaja los sabados y domingos
                    continue
            elif (activity == 'WoS') & (row_f_df['WoS_subgroup'] == 'Home'):
                # O no trabaja o lo hace desde casa, en cualquier caso, no computa
                continue
                
            # Hacemos un loop para realizar la suma de tareas la X cantidad de veces necesaria
            for _ in range(int(activity_amount)):
                if activity == 'Dutties':
                    # En caso de que el agente NO cuente con un edificio especifico para realizar la accion
                    # Elegimos, según el tipo de actividad que lista de edificios pueden ser validos
                    available_options = pop_building[pop_building['archetype'] == activity][['osm_id', 'lat', 'lon']] # ISSUE 33
                    try:
                        last_poi_data = pop_building[pop_building['osm_id'] == rew_row['osm_id']].iloc[0]
                    except Exception:
                        last_poi_data = pop_building[pop_building['osm_id'] == row_f_df['Home']].iloc[0]
                    ring = ring_from_poi(last_poi_data['lat'], last_poi_data['lon'], row_f_df['dist_poi_mu'], row_f_df['dist_poi_sigma'])
                    # Elegimos uno aleatorio del grupo de validos
                    osm_id = choose_id_in_ring(available_options, ring)
                else:
                    # En caso de que el agente cuente ya con un edificio especifico para realizar la accion acude a él
                    osm_id = row_f_df[activity.split('_')[0]]
                       
                if activity == 'WoS':
                    # Si el agente tiene una hora de accion especifica fixed True, si no False
                    fixed = True if row_f_df[f'WoS_fixed'] == 1 else False
                else:
                    # En caso de que el agente NO tenga una hora especifica de accceso y salida
                    fixed = False
                    
                # Si la actividad es el curro fixed sera WoS, si no Service 
                fixed_word = 'WoS' if ((activity == 'WoS') & fixed == False) else 'Service'
                
                # Los casos de work y home son distintos. En el documento a referenciar tienen etiquetas distintas a su nombre de actividad
                if activity == 'WoS':
                    activity_re = 'work'
                else:
                    activity_re = activity
                # Buscamos las horas de apertura y cierre del servicio/WoS
                
                if row_f_df['Home'] == row_f_df['WoS']:
                    opening = 0
                    closing = 24*60
                else:
                    opening = pop_building[(pop_building['osm_id'] == osm_id) & (pop_building['archetype'] == activity_re)][f'{fixed_word}_opening'].iloc[0]
                    closing = pop_building[(pop_building['osm_id'] == osm_id) & (pop_building['archetype'] == activity_re)][f'{fixed_word}_closing'].iloc[0]
                
                if activity == 'WoS':
                    # En caso de que el agente tenga un tiempo requerido de actividad
                    time2spend = int(row_f_df['WoS_time'])
                    if wos_halftime:
                        time2spend = time2spend/2
                
                node = pop_building[pop_building['osm_id']==osm_id]['node'].iloc[0]
                
                rew_row =[{
                    'agent': row_f_df['name'],
                    'archetype': row_f_df['archetype'],
                    'independent': row_f_df['independent_type'],
                    'todo': activity, 
                    'osm_id': osm_id, 
                    'node': node,
                    'opening': opening, 
                    'closing': closing, 
                    'fixed': fixed, 
                    'time2spend': time2spend, 
                    'family': row_f_df['family'],
                    'family_archetype': row_f_df['family_archetype'],
                    'trip': 1 if activity == 'WoS' else 2
                }]
                # La añadimos    
                todolist_agent.extend(rew_row)
            
        for h_activ in home_activities:
                
            home = row_f_df['Home']
            home_node = pop_building[pop_building['osm_id'] == row_f_df['Home']].iloc[0]['node']
                
            rew_row =[{
                'agent': row_f_df['name'],
                'archetype': row_f_df['archetype'],
                'independent': row_f_df['independent_type'],
                'todo': h_activ, 
                'osm_id': home, 
                'node': home_node,
                'opening': 0, 
                'closing': int(24*60), 
                'fixed': False, 
                'time2spend': 0, 
                'family': row_f_df['family'],
                'family_archetype': row_f_df['family_archetype'],
                'trip': 0 if h_activ == 'Home_out' else 3
            }]
            # La añadimos
            todolist_agent.extend(rew_row)
            
        if not any(d['todo'] in ("WoS", "Dutties") for d in todolist_agent):
            todolist_agent = [{
                'agent': row_f_df['name'],
                'archetype': row_f_df['archetype'],
                'independent': row_f_df['independent_type'],
                'todo': 'Home_out', 
                'osm_id': home, 
                'node': home_node,
                'opening': 0, 
                'closing': int(24*60), 
                'fixed': False, 
                'time2spend': 0, 
                'family': row_f_df['family'],
                'family_archetype': row_f_df['family_archetype'],
                'trip': 0
            }]
               
        # Añadimos a los resultados
        todolist_family.extend(todolist_agent)    
    
    # Devolvemos el df de salida
    return todolist_family

def sort_route(osm_ids, helper):
    # Esta funcion deberia devolver el df ordenado con los verdaderos siempor de out
    # recuerda que el helper siempre debe ser el primero

    dependants = osm_ids[osm_ids['osm_id'] != helper['osm_id'].iloc[0]].copy().reset_index(drop=True)  
    helper = osm_ids[osm_ids['osm_id'] == helper['osm_id'].iloc[0]].copy().reset_index(drop=True)
                       
    # Detectar si la columna 'in' o 'out' está presente
    target_col = 'in' if 'in' in dependants.columns else 'out'
    
    ascending = True
    if not dependants.empty:
        current_max = max(helper['conmutime'])
        
        for d_idx, d_row in dependants.iterrows():
            current_max = max([d_row['conmutime'], current_max])
            dependants.loc[d_idx, 'conmutime'] = current_max
            
        if target_col == 'in':
            # Aplicar la operación
            dependants.loc[:, target_col] = dependants[target_col].iloc[0] - dependants['conmutime'] * dependants.index
            if not dependants.empty:
                helper.at[helper.index[0], target_col] = (dependants[target_col].max() + helper['conmutime'].iloc[0])
            ascending = False
        else:
            # Aplicar la operación
            dependants.loc[:, target_col] = dependants[target_col].iloc[0] + dependants['conmutime'] * dependants.index
            if not dependants.empty:
                helper.at[helper.index[0], target_col] = (dependants[target_col].min() - helper['conmutime'].iloc[0])
            ascending = True
    
    combined_df = pd.concat([dependants, helper], ignore_index=True)
    combined_df = combined_df.sort_values(by=target_col, ascending=ascending).reset_index(drop=True)
    
    return combined_df       

# --- Helper a nivel de módulo: debe ser importable/pickleable ---
def _build_family_level1(family_tuple, day, pop_building, activities, citizen_archetypes, system_management):
    """
    family_tuple: (family_name, family_df)
    Devuelve: DataFrame con el level_1 de esa familia
    """
    
    family_name, family_df = family_tuple
    
    # Llama a tu función existente (debe ser importable a nivel de módulo)
    return create_family_level_1_schedule(day, pop_building, family_df, activities, system_management, citizen_archetypes)

def todolist_family_creation(
    study_area,
    df_citizens,
    pop_building,
    system_management,
    paths,
    day,
    citizen_archetypes,
    n_jobs=None,         # None -> usa número de CPUs disponibles
    use_threads=False,   # True si tu carga es I/O-bound
    chunksize=1          # >1 reduce overhead en muchísimas familias
):
    """
    Paralelizada por familia. Escribe a Excel una sola vez al final.
    """
    
    # Actividades una sola vez (evita recomputarlas por familia)
    activities = [a for a in system_management['activities'].tolist() if pd.notna(a)]

    # (Opcional) Únicos de building por si lo reactivas para level_2
    # pop_building_unique = pop_building.drop_duplicates(subset='osm_id')

    # Agrupar ciudadanos por familia
    families_iter = df_citizens.groupby('family')
    families = list(families_iter)  # materializamos para poder mostrar progreso
    total = len(families)

    # Elegir ejecutor
    Executor = ProcessPoolExecutor
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor
        Executor = ThreadPoolExecutor

    # Preparamos función parcial con parámetros constantes
    worker = partial(_build_family_level1, day=day, pop_building=pop_building,
                     activities=activities, citizen_archetypes=citizen_archetypes, system_management=system_management)
    # Lanzamos en paralelo
    results = []
    with Executor(max_workers=n_jobs) as ex:
        futures = {ex.submit(worker, fam): fam[0] for fam in families}
        # Progreso
        for fut in tqdm(as_completed(futures), total=total, desc=f"Families todo list creation ({day}): "):
            fam_name = futures[fut]
            try:
                df_level1 = fut.result()                
                results.extend(df_level1)
            except Exception as e:
                # No abortamos todo el run por una familia: registramos y seguimos
                print(f"[ERROR] familia '{fam_name}': {e}")
    
    '''# Iteración secuencial con barra de progreso
    results = []
    for fam in tqdm(families, total=total, desc=f"/secuential/ Families todo list creation ({day}): "):
        df_level1 = _build_family_level1(fam, day, pop_building, activities, citizen_archetypes, system_management)
        results.extend(df_level1)'''
    
    # Un solo concat al final (mucho más rápido)
    if results:
        level_1_schedule = pd.DataFrame(results)
    else:
        level_1_schedule = pd.DataFrame()

    # Escribir una sola vez (evita cientos de escrituras)
    level_1_schedule.to_excel(f"{paths['results']}/{study_area}_{day}_todolist.xlsx", index=False)

    return level_1_schedule

def load_filter_sort_reset(filepath):
    """
    Summary: 
       Load an Excel file, filter the rows where 'stat' is 'inactive' and return the DataFrame.
    Args:
       filepath (Path): path to the file that wants to be readed
    Returns:
       df: Readed dfs
    """
    df = pd.read_excel(filepath)
    df = df[df['state'] != 'inactive']
    return df

# Función principal
def main_td():
    # Input
    population = 450
    study_area = 'Kanaleneiland'
    
    ## Code initialization
    # Paths initialization
    paths = {}
    
    paths['main'] = Path(__file__).resolve().parent.parent.parent
    paths['system'] = paths['main'] / 'system'
    
    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    file_management = system_management[['file_1', 'file_2', 'pre']]

    # Paso 2: Bucle sobre filas del mini DF
    for index, row in file_management.iterrows():
        
        file_1 = paths[study_area] if row['file_1'] == 'study_area' else paths[row['file_1']]
        file_2 = study_area if row['file_2'] == 'study_area' else row['file_2']
        
        paths[file_2] = file_1 / file_2
        
        if not paths[file_2].exists():
            if row['pre'] == 'y':
                print(f"[Error] Critical file not detected:")
                print(f"{paths[file_2]}")
                print(f"Please solve the mentioned issue and reestart the model.")
                sys.exit()
            else:
                os.makedirs(paths[file_2], exist_ok=True)
    
    
    df_citizens = pd.read_parquet(f"{paths['population']}/pop_citizen.parquet")

    citizen_archetypes = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_citizen.xlsx")

    networks = ['drive', 'walk']
    networks_map = {}   
    for net_type in networks:           
        networks_map[net_type + "_map"] = ox.load_graphml(paths['maps'] / (net_type + '.graphml'))
    
    pop_building = pd.read_parquet(f"{paths['population']}/pop_building.parquet")
    
    ##############################################################################
    print(f'docs readed')
    
    days = {'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su'}

    found_schedule = set()
    found_vehicles = set()
    found_todolist = set()

    for file in paths['results'].glob('*.xlsx'):
        name = file.stem  # sin extensión
        parts = name.split('_')
        if len(parts) < 3:
            continue
        # asumiendo formato: study_area_day_kind
        study, day, kind = parts[-3], parts[-2], parts[-1].lower()
        if day not in days:
            continue
        if kind == 'schedule':
            found_schedule.add(day)
        elif kind == 'vehicles':
            found_vehicles.add(day)
        elif kind == 'todolist':
            found_todolist.add(day)

    # Faltantes por tipo
    missing_schedule = days - found_schedule
    missing_vehicles = days - found_vehicles

    days_missing_todolist = days - found_todolist
    # Faltantes en general (en al menos uno)
    days_missing_schedules = missing_schedule | missing_vehicles
    
    if days_missing_todolist:
        for day in days_missing_todolist:
            level_1_results = todolist_family_creation(study_area, df_citizens, pop_building, system_management, paths, day, citizen_archetypes)
    else:
        print(f"All days' todo lists already modeled.")

# Ejecución
if __name__ == '__main__':
    main_td()
    
    