import os
import sys
import geopandas as gpd
import random
from scipy.stats import lognorm
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

def ring_from_poi(row, lat, lon, mu_log, sigma_log, crs="EPSG:4326", return_point=False):
    """
    Crea un anillo (corona circular) alrededor de un punto (lat, lon) según una
    distribución LOG-normal (mu_log, sigma_log) y, opcionalmente, muestrea un
    punto dentro del anillo usando la lognormal truncada.

    Parámetros
    ----------
    row : pd.Series
        Datos del agente en cuestión. Debe contener 'dist_wos'.
    lat, lon : float
        Coordenadas del punto central (en grados si CRS=EPSG:4326).
    mu_log : float
        Media de la normal subyacente (log-espacio) de la log-normal de distancias.
    sigma_log : float
        Desviación estándar de la normal subyacente (log-espacio).
    crs : str
        CRS del punto de entrada (por defecto 'EPSG:4326').
    return_point : bool
        Si True, devuelve también un punto muestreado dentro del anillo
        respetando la log-normal truncada.

    Retorna
    -------
    shapely.geometry.Polygon
        Polígono del anillo en EPSG:4326.
    ó, si return_point=True:
    (Polygon, Point)
        El anillo y un punto muestreado dentro del mismo.
    """
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point
    from scipy.stats import norm, lognorm

    # ---------- 1. Helpers internos ----------

    def rango_sigma_lognormal(valor, mu_log, sigma_log):
        """
        Clasifica 'valor' (en metros) según cuántas sigma se aleja de la media
        en log-espacio.
        """
        if valor is None or valor <= 0:
            return None, None

        z = (np.log(valor) - mu_log) / sigma_log
        n = abs(z)

        if n <= 1:
            nivel = 1
        elif n <= 2:
            nivel = 2
        elif n <= 3:
            nivel = 3
        else:
            nivel = 4

        lado = "high" if z > 0 else "low"
        return nivel, lado

    def safe_ppf_log_normal(p, mu_log, sigma_log):
        """
        Cuantil seguro de la lognormal, evitando extremos 0 y 1.
        """
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return lognorm.ppf(p, s=sigma_log, scale=np.exp(mu_log))

    def sample_radius_lognorm_trunc(mu_log, sigma_log, low, high, rng=None):
        """
        Muestra UN radio de una lognormal truncada al intervalo [low, high].
        """
        if rng is None:
            rng = np.random.default_rng()

        F_low = lognorm.cdf(low,  s=sigma_log, scale=np.exp(mu_log))
        F_high = lognorm.cdf(high, s=sigma_log, scale=np.exp(mu_log))

        eps = 1e-9
        F_low  = np.clip(F_low,  eps, 1 - eps)
        F_high = np.clip(F_high, eps, 1 - eps)

        u = rng.uniform(F_low, F_high)
        r = lognorm.ppf(u, s=sigma_log, scale=np.exp(mu_log))
        return r

    def crear_anillo_sigma_y_radios(poi, mu_log, sigma_log, n, lado, crs_local):
        """
        Calcula el anillo (Polygon) y los radios low/high en metros.
        """
        poi_m = poi.to_crs(crs_local)

        # Percentiles en la normal estándar
        if lado == "high":
            p_low = norm.cdf(n - 1)
            p_high = norm.cdf(n)
        elif lado == "low":
            p_low = norm.cdf(-n)
            p_high = norm.cdf(-(n - 1))
        else:
            raise ValueError("lado debe ser 'high' o 'low'")

        # Radios en metros (lognormal)
        low = safe_ppf_log_normal(p_low,  mu_log, sigma_log)
        high = safe_ppf_log_normal(p_high, mu_log, sigma_log)

        if np.isnan(low) or np.isnan(high) or np.isclose(low, high):
            return None, None, None

        # Crear anillo geométrico en CRS local
        outer = poi_m.buffer(max(low, high)).iloc[0]
        inner = poi_m.buffer(min(low, high)).iloc[0]
        ring_local = outer.difference(inner)

        # Pasar anillo a WGS84
        ring_wgs84 = gpd.GeoSeries([ring_local], crs=crs_local).to_crs("EPSG:4326")
        return ring_wgs84.iloc[0], low, high

    # ---------- 2. Clasificar dist_wos en sigma-nivel ----------

    dist = row.get("dist_wos", None)
    n, lado = rango_sigma_lognormal(dist, mu_log, sigma_log)
    if n is None:
        return None if not return_point else (None, None)

    # ---------- 3. Crear punto base y CRS local ----------

    poi = gpd.GeoSeries([Point(lon, lat)], crs=crs)

    # Selección automática del CRS local en metros
    if 50 <= lat <= 54 and -1 <= lon <= 8:
        crs_local = "EPSG:28992"  # Países Bajos / entorno
    else:
        utm_zone = int((lon + 180) // 6) + 1
        hemisphere = "326" if lat >= 0 else "327"  # norte/sur
        crs_local = f"EPSG:{hemisphere}{utm_zone}"

    # ---------- 4. Crear anillo y obtener low/high ----------

    ring = None
    low = None
    high = None
    ring, low, high = crear_anillo_sigma_y_radios(poi, mu_log, sigma_log, n, lado, crs_local)

    if ring is None or low is None or high is None:
        return None if not return_point else (None, None)

    if not return_point:
        # Comportamiento antiguo: solo el polígono
        return ring

    # ---------- 5. Muestrear un punto dentro del anillo (opción 1) ----------

    # Punto central en CRS local
    poi_m = poi.to_crs(crs_local)
    x0 = poi_m.geometry.iloc[0].x
    y0 = poi_m.geometry.iloc[0].y

    # Radio desde la log-normal TRUNCADA a [low, high]
    r = sample_radius_lognorm_trunc(mu_log, sigma_log, low, high)

    # Ángulo uniforme
    theta = np.random.uniform(0, 2 * np.pi)

    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)

    pt_local = gpd.GeoSeries([Point(x, y)], crs=crs_local)
    pt_wgs84 = pt_local.to_crs("EPSG:4326").iloc[0]

    return ring, pt_wgs84

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

        chosen_id = None if subset.empty else subset['osm_id'].sample(1).iat[0]

        #visualizar_anillo_y_edificios(cand_df, ring_poly, chosen_id)

        return chosen_id

def get_vehicle_stats(archetype, transport_archetypes, variables):
    results = {}   
    
    # Filtrar la fila correspondiente al arquetipo
    row = transport_archetypes[transport_archetypes['name'] == archetype]
    if row.empty:
        print(f"Strange mistake happend")

        print(f"archetype: {archetype}")
        input(transport_archetypes)

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

def choose_closest_to_point_in_ring(cand_df, ring_poly, sample_point):
        """
        Elige el edificio más cercano a sample_point entre los candidatos de cand_df
        que estén dentro del anillo ring_poly.

        Supone que:
        - cand_df tiene columnas ['osm_id', 'lat', 'lon'] en EPSG:4326.
        - ring_poly está en EPSG:4326 (shapely Polygon o GeoSeries/GeoDataFrame).
        - sample_point es un shapely Point en EPSG:4326.

        Internamente:
        - Define un CRS proyectado local (UTM o 28992) a partir de la lat/lon del sample_point.
        - Reproyecta anillo, candidatos y punto a ese CRS.
        - Filtra candidatos que intersectan el anillo.
        - Devuelve el 'osm_id' del candidato más cercano al sample_point.
        """
        if cand_df is None or len(cand_df) == 0:
            return None

        # 1) Determinar CRS local proyectado a partir del sample_point
        lat = sample_point.y
        lon = sample_point.x

        if 50 <= lat <= 54 and -1 <= lon <= 8:
            # Aprox. NL / BE / oeste DE → EPSG:28992
            crs_local = "EPSG:28992"
        else:
            # UTM según longitud
            utm_zone = int((lon + 180) // 6) + 1
            hemisphere = "326" if lat >= 0 else "327"  # norte/sur
            crs_local = f"EPSG:{hemisphere}{utm_zone}"

        # 2) Construir GeoDataFrame de candidatos en WGS84 y reproyectar a CRS local
        gdf = gpd.GeoDataFrame(
            cand_df.copy(),
            geometry=gpd.points_from_xy(cand_df['lon'], cand_df['lat']),
            crs="EPSG:4326"
        ).to_crs(crs_local)

        # 3) Anillo como GeoSeries en WGS84 → CRS local
        if isinstance(ring_poly, (gpd.GeoSeries, gpd.GeoDataFrame)):
            ring_geom = ring_poly.unary_union
        else:
            ring_geom = ring_poly

        ring_gs = gpd.GeoSeries([ring_geom], crs="EPSG:4326").to_crs(crs_local)
        ring_local = ring_gs.iloc[0]

        # 4) Punto muestreado en WGS84 → CRS local
        sample_gs = gpd.GeoSeries([sample_point], crs="EPSG:4326").to_crs(crs_local)
        sample_local = sample_gs.iloc[0]

        # 5) Filtrar candidatos que caen dentro/intersectan el anillo
        try:
            idx_hits = list(gdf.sindex.query(ring_local, predicate="intersects"))
            cand = gdf.iloc[idx_hits] if idx_hits else gdf
        except Exception:
            cand = gdf

        mask_in_ring = cand.geometry.intersects(ring_local)
        gdf_in = cand.loc[mask_in_ring.values]

        if gdf_in.empty:
            return None

        # 6) Distancias correctas en metros en CRS proyectado local
        gdf_in = gdf_in.copy()
        gdf_in["dist_to_sample"] = gdf_in.geometry.distance(sample_local)

        # 7) Elegir el 'osm_id' más cercano
        chosen_id = gdf_in.sort_values("dist_to_sample").iloc[0]["osm_id"]
        return chosen_id

def create_family_level_1_schedule(day, pop_building, family_df, activities, system_management, citizen_archetypes, building_archetypes):
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
    def find_time(building_archetypes_df, activity_re):
        def round_to_30(x):
            return int(round(x / 30) * 30)

        # columnas *_mu presentes en la hoja de arquetipos
        list_building_variables = [
            col.rsplit('_', 1)[0]
            for col in building_archetypes_df.columns
            if col.endswith('_mu')
        ]

        # Debe devolver un dict con las variables pedidas
        list_building_values = get_vehicle_stats(activity_re, building_archetypes_df, list_building_variables)

        # Combinar ventanas: usa .get() para evitar KeyError
        # OJO: valida que realmente quieres sumarlas y no otra operación
        wo_open  = list_building_values.get('WoS_opening', 0) or 0
        sv_open  = list_building_values.get('Service_opening', 0) or 0
        wo_close = list_building_values.get('WoS_closing', 0) or 0
        sv_close = list_building_values.get('Service_closing', 0) or 0

        list_building_values['Service_opening'] = wo_open + sv_open
        list_building_values['Service_closing'] = wo_close + sv_close #ISSUE 54

        # Mezcla de vuelta en la fila (asumo que assign_data devuelve un dict/Serie listo para DataFrame)
        out_row = {k: round_to_30(v) for k, v in list_building_values.items()}

        # Ahora mismo da absolutamente igual, pues work tiene un tiempo de diferencia con service de 0, por lo que podriamos poner lo del else y ya
        # pero por si en un futuro le queremos meter nuevas funcionalidades.

        if activity_re == 'work':
            opening = out_row['WoS_opening']
            closing = out_row['WoS_closing']
        else:
            opening = out_row['Service_opening']
            closing = out_row['Service_closing']

        return opening, closing
    
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
                if row_f_df['s_class'] == 'Salariat' or WoS_key == 'office':
                    # Si es clase alta o trabaja en oficina, no trabaja los sabados y domingos
                    continue
            elif (activity == 'WoS') & (row_f_df['WoS_subgroup'] == 'Home'):
                # O no trabaja o lo hace desde casa, en cualquier caso, no computa
                continue
                
            # Hacemos un loop para realizar la suma de tareas la X cantidad de veces necesaria
            for idt in range(int(activity_amount)):
                if activity == 'Dutties':
                    # En caso de que el agente NO cuente con un edificio especifico para realizar la accion
                    # Elegimos, según el tipo de actividad que lista de edificios pueden ser validos
                    available_options = pop_building[pop_building['archetype'] == activity][['osm_id', 'lat', 'lon']] # ISSUE 33
                    try:
                        last_poi_data = pop_building[pop_building['osm_id'] == rew_row['osm_id']].iloc[0]
                    except Exception:
                        last_poi_data = pop_building[pop_building['osm_id'] == row_f_df['Home']].iloc[0]

                    ring, sample_point = ring_from_poi(
                        row_f_df,
                        last_poi_data['lat'],
                        last_poi_data['lon'],
                        row_f_df['dist_poi_mu']/1.293,
                        row_f_df['dist_poi_sigma']/1.293,
                        crs="EPSG:4326",
                        return_point=True
                    )

                    # Evitamos que el agente vuelva a su ubicacion anterior (puede concatenar en el mismo osm_id)
                    available_options = available_options[available_options['osm_id'] != last_poi_data['osm_id']].copy().reset_index(drop=True)

                    osm_id = choose_closest_to_point_in_ring(
                        available_options,
                        ring,
                        sample_point
                    )
                    
                    if osm_id == None:
                        osm_id = f"virtual_Dutties_{row_f_df['name']}_{idt}"
                              
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
                    activity_re = 'work' #Issue 62
                else:
                    activity_re = activity
                # Buscamos las horas de apertura y cierre del servicio/WoS
                if row_f_df['Home'] == row_f_df['WoS'] and activity_re == 'work':
                    opening = 0
                    closing = 24*60
                else:
                    if osm_id.startswith("virtual"):
                        opening, closing = find_time(building_archetypes, activity_re)

                    else:
                        opening = pop_building[(pop_building['osm_id'] == osm_id) & (pop_building['archetype'] == activity_re)][f'{fixed_word}_opening'].iloc[0]
                        closing = pop_building[(pop_building['osm_id'] == osm_id) & (pop_building['archetype'] == activity_re)][f'{fixed_word}_closing'].iloc[0]
                
                if activity == 'WoS':
                    # En caso de que el agente tenga un tiempo requerido de actividad
                    time2spend = int(row_f_df['WoS_time'])
                    if wos_halftime:
                        time2spend = time2spend/2
                
                try:
                    node = pop_building[pop_building['osm_id']==osm_id]['node'].iloc[0]
                except Exception:
                    node = 'unknown'
                
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
                    's_class': row_f_df['s_class'],
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
                's_class': row_f_df['s_class'],
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
                's_class': row_f_df['s_class'],
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
def _build_family_level1(family_tuple, day, pop_building, activities, citizen_archetypes, system_management, building_archetypes):
    """
    family_tuple: (family_name, family_df)
    Devuelve: DataFrame con el level_1 de esa familia
    """
    
    family_name, family_df = family_tuple
    
    # Llama a tu función existente (debe ser importable a nivel de módulo)
    return create_family_level_1_schedule(day, pop_building, family_df, activities, system_management, citizen_archetypes, building_archetypes)

def todolist_family_creation(
    study_area,
    df_citizens,
    pop_building,
    system_management,
    paths,
    day,
    citizen_archetypes,
    building_archetypes,
    n_jobs=None,         # None -> usa número de CPUs disponibles
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

    # Preparamos función parcial con parámetros constantes
    worker = partial(_build_family_level1, day=day, pop_building=pop_building,
                     activities=activities, citizen_archetypes=citizen_archetypes, system_management=system_management, building_archetypes=building_archetypes)
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
        df_level1 = _build_family_level1(fam, day, pop_building, activities, citizen_archetypes, system_management, building_archetypes)
        results.extend(df_level1)'''
    
    # Un solo concat al final (mucho más rápido)
    if results:
        level_1_schedule = pd.DataFrame(results).sort_values(by=["family", "agent", "trip"], ascending=[True, True, True])
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
    building_archetypes = pd.read_excel(f"{paths['archetypes']}/pop_archetypes_building.xlsx")

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
            level_1_results = todolist_family_creation(study_area, df_citizens, pop_building, system_management, paths, day, citizen_archetypes, building_archetypes)
    else:
        print(f"All days' todo lists already modeled.")

# Ejecución
if __name__ == '__main__':
    main_td()
    
    