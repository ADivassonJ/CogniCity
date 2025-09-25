# === Estándar de Python =======================================================
from __future__ import annotations

import itertools
import math
import folium
from shapely.geometry import box
import os
import random
import shutil
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

# === Terceros (instalados vía pip) ============================================


# --- DEBUG PLOT: home, ring, candidatos y elegido --------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# OSM y redes
import osmnx as ox
ox.settings.timeout = 500  # evita timeouts al bajar datos de OSM

# Numérico / datos / visualización
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree

# Geoespacial
import geopandas as gpd
import pyproj
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import MultiPolygon, Point, Polygon, LineString, LinearRing
from shapely.ops import clip_by_rect, transform, unary_union, voronoi_diagram

# Distancias geodésicas
from haversine import Unit, haversine

# === Configuración de warnings ================================================
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def add_ebus(paths, polygon, building_populations, study_area, buffer_m=500, proj_epsg=3857):
    """
    Asigna e-buses a edificios usando Voronoi recortados a un polígono buffered.
    
    - paths: dict con rutas de archivos, incluyendo 'population' para guardar resultados
    - polygon: shapely Polygon o MultiPolygon que define el área de estudio (EPSG:4326)
    - building_populations: DataFrame con edificios y columnas de lon/lat
    - study_area: parámetro para carga del sistema eléctrico
    - buffer_m: buffer en metros alrededor del polígono
    - proj_epsg: CRS métrico para cálculos en metros
    """
    
    # Cargar sistema eléctrico
    electric_system = e_sys_loading(paths, study_area)

    # Asignar edificios a los nodos usando los Voronoi
    building_populations_with_node = assign_buildings_to_nodes(building_populations, electric_system, polygon)
    
    # Guardar resultados
    building_populations_with_node.to_parquet(f"{paths['population']}/pop_building.parquet", index=False)
    
    return building_populations_with_node


def add_matches_to_stats_synpop(stats_synpop, df, name_column='name'):
    """
    Summary:
       Search for any '*' in the DataFrame 'df' and add the row and column namen into 'stats_synpop', so all statistical
      dependant situations are mapped on 'stats_synpop'.
    Args:
       stats_synpop (DataFrame): df with some archetypes' statistical values
       df (DataFrame): DataFrame to analyse
       name_column (str, optional): describes how the column on which names of the archetypes are saved. In case of 
      any difference with the standar value, add the new str here.
    Returns:
       stats_synpop (DataFrame): updated df with more archetypes' statistical values
    """
    
    
    for col in df.columns:
        # Saltar la columna 'name' ya que no es relevante para la búsqueda del '*'
        if col == name_column:
            continue
        
        # Iterar por cada celda de la columna
        for index, value in df[col].items():
            # Convertir el valor a string
            value_str = str(value)
            # Verificar si la celda contiene un '*'
            if '*' in value_str:
                # Obtener el valor de 'name' y la columna actual, manejando posibles valores NaN
                name_value = df.loc[index, name_column] if pd.notna(df.loc[index, name_column]) else "Unknown"
                # Añadir al DataFrame stats_synpop
                stats_synpop.loc[len(stats_synpop)] = [name_value, col, None, None, None, None]
    return stats_synpop


def Archetype_documentation_initialization(paths):
    """
    Summary:
       Loads all the dataframes related to archetype description. Also creates/loads (depending on the state of the files)
       the dataframes with all archetypes' statistical values.
    Args:
       paths (dict): Dictionary containing paths like 'archetypes'
    Returns:
       pop_archetypes (dict): Dictionary with pop archetypes DataFrames
       stats_synpop (DataFrame)
       stats_trans (DataFrame)
    """
    
    system_management = pd.read_excel(paths['system']/'system_management.xlsx')
    
    try:
        print(f'Loading archetypes data ...')
        
        archetypes_folder = Path(paths['archetypes'])
        file_names = system_management['archetypes'].dropna().tolist()
        
        files = [
            archetypes_folder / f"pop_archetypes_{name}.xlsx"
            for name in file_names
            if (archetypes_folder / f"pop_archetypes_{name}.xlsx").is_file()
        ]
        
        pop_archetypes = {
            f.stem.replace("pop_archetypes_", ""): load_filter_sort_reset(f)
            for f in files
        }
        
    except Exception as e:
        raise FileNotFoundError(f"[ERROR] Archetype files are not found in {paths['archetypes']}. Please fix and restart.") from e
    
    # Validar y cargar stats_synpop
    stats_configs = [
        ("stats_synpop.xlsx", create_stats_synpop, ["citizen", "family"]),
        ("stats_trans.xlsx", create_stats_trans, ["transport", "family"]),
        ("stats_class.xlsx", create_stats_class, ["transport", "family"]),
    ]

    stats = {}

    for filename, create_func, archetype_keys in stats_configs:
        inputs = [pop_archetypes.get(k) for k in archetype_keys]
        stats[Path(filename).stem] = load_or_create_stats(
            paths['archetypes'],
            filename,
            create_func,
            inputs
        )

    return pop_archetypes, stats

def assign_buildings_to_nodes(building_populations, electric_system, boundary_polygon):
    # GeoDataFrame edificios
    buildings_gdf = gpd.GeoDataFrame(
        building_populations.copy(),
        geometry=gpd.points_from_xy(building_populations['lon'], building_populations['lat']),
        crs="EPSG:4326"
    )

    # Voronoi para visualización
    nodes_gdf_proj, vor_gdf, boundary_proj = voronoi_from_nodes(electric_system, boundary_polygon)
    buildings_proj = buildings_gdf.to_crs(vor_gdf.crs)

    # Nodos como GDF
    nodes_gdf = nodes_gdf_proj[['node', 'geometry']].copy()

    # Para cada edificio, elegir nodo más cercano
    nearest_nodes = []
    for geom in buildings_proj.geometry:
        distances = nodes_gdf.distance(geom)
        nearest_idx = distances.idxmin()
        nearest_nodes.append(nodes_gdf.loc[nearest_idx, 'node'])

    buildings_proj['node'] = nearest_nodes
    buildings_proj = buildings_proj.to_crs("EPSG:4326")

    # Plot
    #plot_voronoi_with_buildings(nodes_gdf_proj, vor_gdf, boundary_proj, building_populations)

    return pd.DataFrame(buildings_proj.drop(columns='geometry'))


def assign_data(list_variables, list_values, row):
    """
    Asigna valores de list_values a una fila (Series o dict) según reglas:
      - *_type y *_amount → enteros redondeados
      - *_time → redondeado a múltiplos de 30
      - resto → tal cual
    """
    for variable in list_variables:
        val = list_values.get(variable, None)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue  # no asignamos nada si falta valor

        if variable.endswith(('_type', '_amount')):
            try:
                val = int(round(val))
            except Exception:
                pass  # por si val no es numérico
        elif variable.endswith('_time'):
            try:
                val = int(round(val / 30.0) * 30)
            except Exception:
                pass

        row[variable] = val

    return row


def buffer_value(electric_system):
    # Supongamos que tu df se llama electric_system y tiene 'lat' y 'long'
    # Encuentra los extremos
    left = electric_system.loc[electric_system['long'].idxmin()]
    right = electric_system.loc[electric_system['long'].idxmax()]
    top = electric_system.loc[electric_system['lat'].idxmax()]
    bottom = electric_system.loc[electric_system['lat'].idxmin()]

    # Lista de extremos
    extremos = [left, right, top, bottom]

    # Calcular todas las distancias entre pares
    max_distance = 0
    for i in range(len(extremos)):
        for j in range(i+1, len(extremos)):
            coord1 = (extremos[i]['lat'], extremos[i]['long'])
            coord2 = (extremos[j]['lat'], extremos[j]['long'])
            distance = haversine(coord1, coord2, unit=Unit.METERS)
            if distance > max_distance:
                max_distance = distance
    return int(max_distance)


def building_schedule_adding(osm_elements_df: pd.DataFrame,
                             building_archetypes_df: pd.DataFrame,
                             round_to: int = 30,
                             clamp_minutes: tuple[int,int] = (0, 24*60)):
    """
    Añade variables de 'building_archetypes_df' a cada building de 'osm_elements_df',
    redondeando tiempos a múltiplos de `round_to` minutos.
    """
    # columnas *_mu presentes en la hoja de arquetipos
    list_building_variables = [
        col.rsplit('_', 1)[0]
        for col in building_archetypes_df.columns
        if col.endswith('_mu')
    ]

    final_results = []

    for idx, row_oedf in osm_elements_df.iterrows():
        arche = row_oedf.get('archetype', None)
        if pd.isna(arche):
            # sin arquetipo → saltamos
            continue

        # Debe devolver un dict con las variables pedidas
        list_building_values = get_vehicle_stats(arche, building_archetypes_df, list_building_variables)

        if not list_building_values:
            # sin stats para ese arquetipo → loguea y sigue
            # print(f"[WARN] Archetype sin stats: {arche}")
            continue

        # Combinar ventanas: usa .get() para evitar KeyError
        # OJO: valida que realmente quieres sumarlas y no otra operación
        wo_open  = list_building_values.get('WoS_opening', 0) or 0
        sv_open  = list_building_values.get('Service_opening', 0) or 0
        wo_close = list_building_values.get('WoS_closing', 0) or 0
        sv_close = list_building_values.get('Service_closing', 0) or 0

        list_building_values['Service_opening'] = wo_open + sv_open
        list_building_values['Service_closing'] = wo_close + sv_close

        # Redondeo a múltiplos de `round_to` sólo para numéricos finitos
        for k, v in list(list_building_values.items()):
            if isinstance(v, (int, float)) and np.isfinite(v):
                rounded = int(round(v / float(round_to)) * round_to)
                # clamp opcional a [0, 1440]
                rounded = max(clamp_minutes[0], min(clamp_minutes[1], rounded))
                list_building_values[k] = rounded

        # Mezcla de vuelta en la fila (asumo que assign_data devuelve un dict/Serie listo para DataFrame)
        out_row = assign_data(list_building_variables, list_building_values, row_oedf)
        final_results.append(out_row)

    return pd.DataFrame(final_results).reset_index(drop=True)


def building_type(row, poss_ref):
    # Revisar cada categoría en orden de prioridad
    for category, values in poss_ref.items():
        actor = row[category] if category in row and pd.notna(row[category]) else False
        
        # Verificar si el valor de la categoría está en las prioridades definidas
        if actor:
            if isinstance(values, list):  # Si las prioridades son una lista
                if actor in values:
                    return f'{category}_{actor}'
            elif actor == values:  # Si la prioridad es un valor único
                return f'{category}_{actor}'

    # Si no se encuentra ninguna coincidencia
    return 'unknown'


def create_stats_synpop(archetypes_path, citizen_archetypes, family_archetypes): # Most probably, we should adapt this for getting all df needed, not just the specific two
    """
    Summary:
       It detects all the statistical dependence values and makes a table with two columns (item_1 and item_2) that
      relate the two variables in their archetype tables, and the columns mu, sigma, min and max, which describe the 
      characteristics of the normal curve to derive the statistical values.
    Args:
        archetypes_path (Path): Path to archetypes
        citizen_archetypes (DataFrame): df with all citizens' archetype data
        family_archetypes (DataFrame): df with all families' archetype data
    """
    
    stats_synpop = pd.DataFrame(columns=['item_1', 'item_2', 'mu', 'sigma', 'min', 'max'])
    stats_synpop = add_matches_to_stats_synpop(stats_synpop, citizen_archetypes)
    stats_synpop = add_matches_to_stats_synpop(stats_synpop, family_archetypes)
    stats_synpop.to_excel(archetypes_path/'stats_synpop.xlsx', index=False)
  
    
def create_stats_trans(archetypes_path, transport_archetypes, family_archetypes):
    # Obtener listas de 'name' de ambos DataFrames
    transport_names = transport_archetypes['name'].dropna().unique()
    family_names = family_archetypes['name'].dropna().unique()
    # Crear combinaciones entre todos los elementos (producto cartesiano)
    combinations = list(itertools.product(family_names, transport_names))
    # Crear DataFrame con las combinaciones
    stats_trans = pd.DataFrame(combinations, columns=['item_1', 'item_2'])
    stats_trans = stats_trans[~stats_trans['item_2'].isin(['walk', 'public'])] #estos no se requiere de asignacion a las familias
    # Agregar columnas vacías para los valores estadísticos
    stats_trans['mu'] = ''
    stats_trans['sigma'] = ''
    stats_trans['min'] = ''
    stats_trans['max'] = ''

    stats_trans.to_excel(archetypes_path/'stats_trans.xlsx', index=False)

    return stats_trans

def create_stats_class(archetypes_path, _, family_archetypes):
    
    # Obtener listas de 'name' de ambos DataFrames
    family_names = family_archetypes['name'].dropna().unique()
    # Crear DataFrame con las combinaciones
    stats_class = pd.DataFrame(family_names, columns=['name'])
    # Agregar columnas vacías para los valores estadísticos
    stats_class['Salariat'] = ''
    stats_class['Intermediate'] = ''
    stats_class['Working'] = ''

    stats_class.to_excel(archetypes_path/'stats_class.xlsx', index=False)

    return stats_class


def Citizen_inventory_creation(df, population):
    """
    Summary:
       Calculates the percentage of 'presence' and assigns to each row the population (rounded) proportional to that percentage.
    Args:
       df (DataFrame): df with archetype's data
       population (int): amount of citizens to evaluate
    Returns:
      - _ (DataFrame): df with info of citizens archetypes and each's population
      - total_presence (int): Citizens total presence from archetype data
    """

    # Check if any presence exists
    if df['presence'].sum() > 0:
        # Calculate presence percentage
        df['presence_percentage'] = df['presence'] / df['presence'].sum()
    else:
        # If no presence is detectet rise and error for the user
        print("        [Error] No available presences has been detected on citicens_archetype's file.")
        sys.exit()
    # Adds new row with population of each agent
    df['population'] = (df['presence_percentage'] * population).round().astype(int)

    return df[['name', 'population']].copy(), df['presence'].sum()

def Citizen_distribution_in_families(archetype_to_fill, df_distribution, total_presence, stats_synpop, pop_archetypes, ind_arch = 'f_arch_0'):
    """
    Summary:
       Creates dataframes for all families and citizens in the population, 
       describing their main characteristics.

    Args:
       archetype_to_fill (DataFrame): df with the archetypes data of the archetypes that can be applied
       df_distribution (DataFrame): df with info of citizens archetypes and each's population
       total_presence (int): Citizens total presence from archetype data
       stats_synpop (DataFrame): df with all archetypes' statistical values
       pop_archetypes (dict): Dictionary of archetype DataFrames
       ind_arch (str, optional): Archetype name for individual homes 
         (families with just one citizen). Defaults to 'f_arch_0'.
         If single-person households use a different archetype, specify it here.

    Returns:
       df_distribution (DataFrame): Remaining citizens that could not be assigned to a family
       df_citizens (DataFrame): All created citizens with their characteristics
       df_families (DataFrame): All created families with their members
    """
    # Initialize result DataFrames
    df_families = pd.DataFrame(columns=['name', 'archetype', 'description', 'members'])
    df_citizens = pd.DataFrame(columns=['name', 'archetype', 'description'])

    # Temporary citizen buffer: used during family creation.  
    # Sometimes a family starts being created but cannot be completed due to constraints.  
    # In those cases, this buffer is cleared without merging into the real df_citizens.
    df_part_citizens = pd.DataFrame(columns=df_citizens.columns)

    # Stats DataFrame to monitor expected vs. created families
    df_stats_families = pd.DataFrame({
        'archetype': archetype_to_fill['name'],
        'presence': archetype_to_fill['presence'],
        'percentage': archetype_to_fill['presence'] / archetype_to_fill['presence'].sum()*100,
        'stat_presence': 0,
        'stat_percentage': 0,
        'error': 0
    })
    
    while True:
        # Flag to skip invalid family scenarios
        flag = False

        # Check if there are any archetypes still possible
        archetype_to_fill = is_it_any_archetype(archetype_to_fill, df_distribution, ind_arch)
        if archetype_to_fill.empty:
            break

        # Keep stats only for still-available archetypes
        df_stats_families = df_stats_families[df_stats_families['archetype'].isin(archetype_to_fill['name'])]

        # Count how many families of each archetype exist so far
        archetype_counts = df_families['archetype'].value_counts()

        # Update stats with actual distribution
        df_stats_families = df_stats_families.copy()
        df_stats_families.loc[:, 'stat_presence'] = df_stats_families['archetype'].map(archetype_counts).fillna(0).astype(int)
        df_stats_families.loc[:, 'stat_percentage'] = (df_stats_families['stat_presence'] / df_stats_families['stat_presence'].sum()) * 100
        df_stats_families.loc[:, 'error'] = df_stats_families.apply(
            lambda row: (row['stat_percentage'] - row['percentage']) / row['percentage'] if row['percentage'] != 0 else 0,
            axis=1
        )

        # First iteration → start with the archetype with the largest presence
        if df_stats_families['stat_presence'].sum() == 0:
            arch_to_fill = df_stats_families.loc[df_stats_families['presence'].idxmax(), 'archetype']
        # Later iterations → continue with the archetype with the largest statistical error
        else:
            arch_to_fill = df_stats_families.loc[df_stats_families['error'].idxmin(), 'archetype']

        # Merge distribution and archetype info for the chosen family
        merged_df = process_arch_to_fill(archetype_to_fill, arch_to_fill, df_distribution)
        
        # --- Case 1: Individual families ---
        if arch_to_fill == ind_arch:
            # Keep only candidates marked with '*' and join with statistical values
            merged_result = (
                merged_df[merged_df['participants'].str.contains(r'\*', na=False)]
                .merge(
                    stats_synpop[stats_synpop['item_1'] == ind_arch],
                    left_on='name',
                    right_on='item_2',
                    how='inner'
                ).dropna(subset=['population'])
                .query("population != 0")
            )

            # If no candidates → remove this archetype and skip
            if merged_result.empty:
                archetype_to_fill = archetype_to_fill[archetype_to_fill['name'] != arch_to_fill].reset_index(drop=True)
                continue

            # Compute probabilities from 'mu' values
            merged_result.loc[:, 'probability'] = merged_result['mu'] / merged_result['mu'].sum()
            # Randomly select one citizen archetype
            random_choice = np.random.choice(merged_result['name'], p=merged_result['probability'])
            # Update merged_df: chosen archetype gets 1 participant, others 0
            merged_df.loc[:, 'participants'] = np.where(merged_df['name'] == random_choice, 1, 0)
            # Create the new citizen
            citizen_description = pop_archetypes['citizen'].loc[pop_archetypes['citizen']['name'] == random_choice, 'description'].values[0]
            new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': random_choice, 'description': citizen_description}
            df_part_citizens.loc[len(df_part_citizens)] = new_row

        # --- Case 2: Collective families ---
        else:
            for idx in merged_df.index:
                row = merged_df.loc[idx]

                # Skip if no archetype available
                if pd.notna(row['participants']):
                    value = row['participants']
                    # If '*', use statistical values
                    stats_value = get_stats_value(value, stats_synpop, arch_to_fill, row['name'])                 
                    merged_df.at[idx, 'participants'] = stats_value

                    # Case: family fits within available population
                    if merged_df.at[idx, 'participants'] <= row['population']:
                        for _ in range(int(merged_df.at[idx, 'participants'])):
                            citizen_description = pop_archetypes['citizen'].loc[pop_archetypes['citizen']['name'] == row['name'], 'description'].values[0]
                            new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': row['name'], 'description': citizen_description}
                            df_part_citizens.loc[len(df_part_citizens)] = new_row

                    # Case: family does NOT fit population constraints
                    else:                                      
                        fila = archetype_to_fill.loc[archetype_to_fill["name"] == arch_to_fill]
                        if not (fila.isin(['*']).any().any()):
                            # No '*' means this family will never fit → remove it
                            archetype_to_fill = archetype_to_fill[archetype_to_fill["name"] != arch_to_fill].reset_index(drop=True)
                            df_part_citizens = df_part_citizens.drop(df_part_citizens.index)
                            flag = True
                            break
                        else:
                            # With '*', check if minimum statistical values allow a valid family
                            cols_with_star = [col for col in fila.columns if fila.iloc[0][col] == '*']
                            name_value = fila.iloc[0]['name']
                            filtered_df = stats_synpop[(stats_synpop['item_1'] == name_value) & (stats_synpop['item_2'].isin(cols_with_star))][['item_2', 'min']]

                            for idy in filtered_df.index:
                                if filtered_df.at[idy, 'min'] <= row['population']:
                                    # Replace '*' with minimum feasible value
                                    merged_df.loc[merged_df['name'] == filtered_df.at[idy, 'item_2'], 'participants'] = filtered_df.at[idy, 'min']
                                    for _ in range(int(merged_df.at[idx, 'participants'])):
                                        citizen_description = pop_archetypes['citizen'].loc[pop_archetypes['citizen']['name'] == row['name'], 'description'].values[0]
                                        new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': row['name'], 'description': citizen_description}
                                        df_part_citizens.loc[len(df_part_citizens)] = new_row
                                else:
                                    # Even min values don’t fit → remove archetype
                                    archetype_to_fill = archetype_to_fill[archetype_to_fill["name"] != arch_to_fill].reset_index(drop=True)
                                    df_part_citizens = df_part_citizens.drop(df_part_citizens.index)
                                    flag = True
                                    break

        # Skip further processing if the family creation failed
        if flag:
            continue

        # Update population distribution
        mask = df_distribution['population'].notna() & merged_df['participants'].notna()
        df_distribution.loc[mask, 'population'] = df_distribution.loc[mask, 'population'] - merged_df.loc[mask, 'participants']

        # Add created citizens
        df_citizens = pd.concat([df_citizens, df_part_citizens], ignore_index=True)

        # Add new family
        family_description = pop_archetypes['family'].loc[pop_archetypes['family']['name'] == arch_to_fill, 'description'].values[0]
        new_row_2 = {'name': f'family_{len(df_families)}', 'archetype': arch_to_fill, 'description': family_description, 'members': df_part_citizens['name'].tolist()}
        df_families.loc[len(df_families)] = new_row_2

        # Reset buffer
        df_part_citizens = df_part_citizens.drop(df_part_citizens.index)

        # Stop if no citizens remain
        if df_distribution['population'].sum() == 0:
            break

    return df_distribution, df_citizens, df_families


def computate_stats(row):
    mu = float(row['mu'])
    sigma = float(row['sigma'])
    min_val = int(row['min'])
    max_val = int(row['max'])

    stats_value = round(np.random.normal(mu, sigma))
    stats_value = max(min(stats_value, max_val), min_val)

    return stats_value

def split_polygon_to_grid(
    polygon,
    size_m=1000,
    crs_in="EPSG:4326",
    min_area_ratio=0.01,  # filtra fragmentos <1% del área de la celda
):
    """
    Divide `polygon` en celdas de `size_m` x `size_m` (metros), devolviendo sólo la
    parte interior al polígono. Devuelve un GeoDataFrame en EPSG:4326 con:
      - geometry: geometría recortada por el polígono
      - area_km2: área de cada celda (km², calculada en CRS métrico)
      - row, col: índices de la rejilla
      - cell_id: identificador entero único (row-major)
    
    Parámetros
    ----------
    polygon : shapely Polygon/MultiPolygon (en `crs_in`)
    size_m : float
        Tamaño del lado de la celda (m).
    crs_in : str
        CRS de entrada del polígono.
    min_area_ratio : float
        Umbral para descartar intersecciones minúsculas (relativo al área de la celda).
    """

    # 0) Normaliza a GeoDataFrame y proyecta a CRS métrico (UTM local)
    gdf_in = gpd.GeoDataFrame(geometry=[polygon], crs=crs_in)
    gdf_m = ox.projection.project_gdf(gdf_in)   # elige UTM apropiado
    poly_m = gdf_m.geometry.iloc[0]
    crs_m = gdf_m.crs

    # 1) Bounds y alineación a la rejilla (múltiplos de size_m)
    minx, miny, maxx, maxy = poly_m.bounds
    start_x = np.floor(minx / size_m) * size_m
    start_y = np.floor(miny / size_m) * size_m
    end_x   = np.ceil(maxx / size_m) * size_m
    end_y   = np.ceil(maxy / size_m) * size_m

    xs = np.arange(start_x, end_x, size_m)
    ys = np.arange(start_y, end_y, size_m)

    # 2) Genera celdas e interseca
    cells = []
    rows, cols = [], []
    cell_area = (size_m * size_m)  # m²
    area_cut = cell_area * float(min_area_ratio)

    # recorre en orden row-major: y (filas) externo, x (columnas) interno
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            cell = box(x, y, x + size_m, y + size_m)
            inter = cell.intersection(poly_m)
            if inter.is_empty:
                continue
            # Puede ser MultiPolygon: añadimos cada parte por separado
            if inter.geom_type == "MultiPolygon":
                for geom in inter.geoms:
                    if geom.area >= area_cut:
                        cells.append(geom)
                        rows.append(r)
                        cols.append(c)
            else:
                if inter.area >= area_cut:
                    cells.append(inter)
                    rows.append(r)
                    cols.append(c)

    # 3) Si no hay celdas, devolvemos el polígono completo como una sola
    if not cells:
        cells = [poly_m]
        rows = [0]
        cols = [0]

    # 4) Construimos GDF en métrico y añadimos métricas/índices
    grid_m = gpd.GeoDataFrame({"row": rows, "col": cols, "geometry": cells}, crs=crs_m)
    grid_m["cell_id"] = (grid_m["row"] * len(xs) + grid_m["col"]).astype(int)
    grid_m["area_km2"] = grid_m.area.values / 1e6  # m² → km²

    # 5) Devolvemos en WGS84 para el resto del pipeline
    grid = grid_m.to_crs(4326).reset_index(drop=True)
    # Orden agradable de columnas
    grid = grid[["cell_id", "row", "col", "area_km2", "geometry"]]

    return grid


def save_split_poligon_map(polygon, grid_gdf):
    
    # 3) Mapa interactivo con Folium
    # Centro del mapa en el centroide del polígono original
    centroid = gpd.GeoSeries([polygon], crs="EPSG:4326").centroid.iloc[0]
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=13, tiles="CartoDB positron")

    # Polígono original (borde grueso)
    folium.GeoJson(
        polygon,
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
    m.save("kanaleneiland_grid_1km.html")
    print("Mapa guardado en 'kanaleneiland_grid_1km.html'")


import os, sys, json
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon

def _project_to_metric(geom_wgs84):
    """Proyecta una geometría WGS84 a un CRS métrico (UTM elegido por OSMnx)."""
    gdf = gpd.GeoDataFrame(geometry=[geom_wgs84], crs=4326)
    gdf_m = ox.projection.project_gdf(gdf)  # UTM apropiado
    return gdf_m.geometry.iloc[0], gdf_m.crs

def _write_empty_geojson(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": []}, f)

def download_pois(study_area, paths, building_archetypes_df, special_areas_coords, city_district):
    print(f'    [WARNING] Data is missing, it needs to be downloaded.')

    study_area_path = paths[study_area]

    # Documento crítico
    try:
        print(f"        Reading 'Class matrix for ESeC 2008 rev.xlsx' ...")
        Services_Group_relationship = pd.read_excel(f'{study_area_path}/Class matrix for ESeC 2008 rev.xlsx')
    except Exception:
        print(f"    [ERROR] File 'Class matrix for ESeC 2008 rev.xlsx' is not found in the data folder ({study_area_path}).")
        sys.exit(1)

    print(f'        Processing data (it might take a while)...')

    services_groups = services_groups_creation(Services_Group_relationship, building_archetypes_df['name'].unique())

    # Ciudad a descargar (completa)
    city = city_district.get(study_area)

    # Polígono de la ciudad
    gdf_city = ox.geocode_to_gdf(city).to_crs(4326)
    polygon_wgs84 = gdf_city.iloc[0].geometry

    # 1) Grid 1 km²: trabajar en métrico y volver a WGS84
    poly_metric, metric_crs = _project_to_metric(polygon_wgs84)
    grid_gdf_metric = split_polygon_to_grid(polygon_wgs84, size_m=1000, crs_in="EPSG:4326")
    grid_gdf = gpd.GeoDataFrame(grid_gdf_metric, geometry='geometry', crs=metric_crs).to_crs(4326).reset_index(drop=True)
    print(f"            '{city}' was divided into {len(grid_gdf)} cells for management.")

    # 2) Carpeta cache
    cache_path = os.path.join(paths['maps'], "cache")
    os.makedirs(cache_path, exist_ok=True)

    # 3) Celdas ya descargadas
    geojson_files = [f for f in os.listdir(cache_path) if f.endswith(".geojson") and f[:-8].isdigit()]
    done_ids = {int(f[:-8]) for f in geojson_files}
    missing_ids = [i for i in range(len(grid_gdf)) if i not in done_ids]

    # 4) Procesar celdas faltantes
    for grid_id in missing_ids:
        grid_geom = grid_gdf.at[grid_id, "geometry"]
        all_osm_data = []

        for group_name, group_ref in services_groups.items():
            
            input(services_groups)
            
            
            try:
                df_group = get_osm_elements(grid_geom, group_ref)  # espera geom en WGS84
                if df_group is not None and hasattr(df_group, "empty") and not df_group.empty:
                    if 'archetype' not in df_group.columns:
                        df_group['archetype'] = group_name.replace('_list', '')
                    # normalizamos CRS por si acaso
                    if getattr(df_group, "crs", None) is not None and df_group.crs != "EPSG:4326":
                        df_group = df_group.to_crs(4326)
                    all_osm_data.append(df_group)
            except Exception as e:
                print(f"                [ERROR] {group_name} in cell {grid_id}: {e}")

        out_fp = f"{cache_path}/{grid_id}.geojson"
        if all_osm_data:
            osm_elements_df = gpd.GeoDataFrame(pd.concat(all_osm_data, ignore_index=True), crs=4326)
            # columnas mínimas esperadas (relleno si faltan)
            for col in ['archetype','building_type','osm_id','lat','lon']:
                if col not in osm_elements_df.columns and col != 'geometry':
                    osm_elements_df[col] = pd.NA
            try:
                osm_elements_df.to_file(out_fp, driver="GeoJSON")
            except ValueError:
                # si por lo que sea queda vacío, escribimos placeholder
                _write_empty_geojson(out_fp)
        else:
            _write_empty_geojson(out_fp)

        total = sum(len(df) for df in all_osm_data) if all_osm_data else 0
        print(f"                Cell {grid_id}: {total} elements found")

    # 5) Concatenar todo (asumimos archivos OK)
    geojson_paths = [os.path.join(cache_path, f) for f in os.listdir(cache_path) if f.endswith(".geojson")]
    gdfs = [gpd.read_file(fp).to_crs(4326) for fp in geojson_paths]
    if len(gdfs) == 0:
        osm_elements_df = gpd.GeoDataFrame(columns=['archetype','building_type','osm_id','geometry','lat','lon'], crs=4326)
    else:
        osm_elements_df = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=4326)

    # 6) Añadir schedule / relación con arquetipos
    SG_relationship = building_schedule_adding(osm_elements_df, building_archetypes_df)

    # 7) Polígono del distrito (tus coords vienen como (lat, lon) → convertimos a (lon, lat))
    coords = special_areas_coords[study_area]
    coords_corrected = [(lon, lat) for (lat, lon) in coords]
    distric_polygon = Polygon(coords_corrected)

    # 8) Añadir buses eléctricos
    buildings_populations = add_ebus(paths, distric_polygon, SG_relationship, study_area)

    return buildings_populations

def e_sys_loading(paths, study_area):
    try:
        electric_system = pd.read_excel(paths['maps'] / 'electric_system.xlsx')
    except Exception as e:
        try:
            electric_system = pd.read_excel(paths['desktop'] / f'electric_system_{study_area}.xlsx')
            electric_system.to_parquet(f"{paths['maps']}/electric_system.parquet", index=False)
        except Exception as e:
            print(f"    [WARNING] The file relating to the electrical network for the “{study_area}” case has not been found.\n       Please locate the .xlsx file on your computer desktop with the following name:")
            print(f"            electric_system_{study_area}.xlsx")
            code = 'NOT_DONE'
            while code != 'DONE':
                code = input(f"        Once you have completed the requested action, enter “DONE”.\n")
                if code != 'DONE':
                    print(f'        Incorrect continuation code.')
                else:
                    try:
                        electric_system = pd.read_excel(paths['desktop'] / f'electric_system_{study_area}.xlsx')
                        electric_system.to_parquet(f"{paths['maps']}/electric_system.parquet", index=False)
                    except Exception as e:
                        print(f'        [ERROR] Archivo no encontrado en la ubiccion solicitada.')
                        print(f"        ({paths['desktop'] / f'electric_system_{study_area}.xlsx'})")
                        code = 'NOT_DONE'
    return electric_system


def find_group(name, df_families, row_out):
    for idx, row in df_families.iterrows():
        if name in row['members']:
            return row[row_out]
    return None


def Geodata_initialization(study_area, paths, pop_archetypes):
    agent_populations = {}
    # Obtener redes activas desde transport_archetypes
    networks = get_active_networks(pop_archetypes['transport'])
    
    # Diccionario con coordenadas de los territorios especiales
    special_areas_coords = {
        "Aradas": [(1, 1), (1, 1)],
        "Kanaleneiland": [(52.07892763457244, 5.081179665783377), 
                          (52.071082860598274, 5.087677559318499), 
                          (52.060700337662205, 5.097493321101714), 
                          (52.0589253058436, 5.111134343014198),
                          (52.06371772987415, 5.113155235149382),
                          (52.06713423672216, 5.112072614362676),
                          (52.07698296226893, 5.109504220222101),
                          (52.07814350260757, 5.108891797422314),
                          (52.079586294469394, 5.107820057522688),
                          (52.081311310482626, 5.106084859589962),
                          (52.0818131208049, 5.105013119690336),
                          (52.08520019308004, 5.09822543371076),
                          (52.08291081138339, 5.094959178778566),
                          (52.08102903986475, 5.090570148713432),
                        ],
        "Annelinn": [(1, 1), (1, 1)],
    }
    
    city_district = {
        "Aradas": "Aveiro",
        "Kanaleneiland": "Utrecht",
        "Annelinn": "Tartu",
    }

    # Paso 1: Cargar o descargar POIs
    agent_populations['building'] = load_or_download_pois(study_area, paths, pop_archetypes['building'], special_areas_coords, city_district)

    # Paso 2: Cargar o descargar redes
    networks_map = load_or_download_networks(study_area, paths['maps'], networks, city_district)

    return agent_populations, networks_map


def get_active_networks(transport_archetypes_df):
    active_maps = transport_archetypes_df.loc[
        transport_archetypes_df['state'] == 'active', 'map'
    ].dropna().unique().tolist()
    
    # Asegurar que 'walk' esté incluido
    if 'walk' not in active_maps:
        active_maps.append('walk')
    
    return active_maps


def get_stats_value(value, stats_synpop: pd.DataFrame,
                    family_arch: str, citizen_arch: str,
                    fallback_min_if_missing: bool = True) -> int:
    """
    Resuelve '*' principalmente con 'mu' (redondeado >=0).
    Si no hay mu, intenta 'min' si existe, y si no, 1 (o 0 si no hay fallback).
    """
    if not (isinstance(value, str) and value.strip() == '*'):
        # ya es número
        try:
            return max(int(value), 0)
        except Exception:
            return 0

    subset = stats_synpop[(stats_synpop['item_1'] == family_arch) & (stats_synpop['item_2'] == citizen_arch)]
    if subset.empty:
        return 1 if fallback_min_if_missing else 0

    mu = subset['mu'].dropna()
    if not mu.empty:
        return max(int(round(mu.iloc[0])), 0)

    mn = subset['min'].dropna()
    if not mn.empty:
        return max(int(mn.iloc[0]), 0)

    return 1 if fallback_min_if_missing else 0


def get_wos_action(archetype, pop_archetypes, stats_synpop):
    value = pop_archetypes['citizen'].loc[
        pop_archetypes['citizen']['name'] == archetype, 'WoS_action'
    ].values[0]
    return get_stats_value(value, stats_synpop, archetype, 'WoS_action')


import pandas as pd
import geopandas as gpd
import osmnx as ox

def get_osm_elements(polygon, poss_ref):
    """
    Descarga y filtra elementos de OSM dentro de un polígono dado para UN servicio (poss_ref).
    Calcula centroides en un CRS proyectado y retorna columnas estandarizadas.

    Parámetros:
    - polygon (shapely Polygon/MultiPolygon): área de búsqueda.
    - poss_ref (dict): reglas de filtrado {key: [values] | value} (p.ej. {'amenity':['school','hospital']}).

    Retorna:
    - GeoDataFrame con columnas: ['building_type','osm_id','geometry','lat','lon'] (CRS EPSG:4326).
    """

    # 1) Preparar etiquetas (una sola descarga por servicio)
    if not poss_ref or not isinstance(poss_ref, dict):
        return gpd.GeoDataFrame(columns=['building_type','osm_id','geometry','lat','lon'], crs="EPSG:4326")

    tags = {key: True for key in poss_ref.keys()}

    # 2) Descargar datos
    try:
        gdf = ox.geometries_from_polygon(polygon, tags).reset_index()
    except Exception:
        # Si Overpass falla, devolvemos vacío con esquema correcto
        return gpd.GeoDataFrame(columns=['building_type','osm_id','geometry','lat','lon'], crs="EPSG:4326")

    if gdf.empty:
        return gpd.GeoDataFrame(columns=['building_type','osm_id','geometry','lat','lon'], crs="EPSG:4326")

    # 3) Filtrar según poss_ref (OR entre keys/values)
    mask = pd.Series(False, index=gdf.index)
    for key, values in poss_ref.items():
        if key not in gdf.columns:
            continue
        if isinstance(values, list):
            mask |= gdf[key].isin(values)
        else:
            mask |= (gdf[key] == values)

    filtered = gdf.loc[mask].copy()
    if filtered.empty:
        return gpd.GeoDataFrame(columns=['building_type','osm_id','geometry','lat','lon'], crs="EPSG:4326")

    # 4) Centroides correctos (proyectado)
    gdf_geo = gpd.GeoDataFrame(filtered, geometry='geometry', crs="EPSG:4326")
    try:
        target_crs = gpd.GeoSeries([polygon], crs="EPSG:4326").estimate_utm_crs()
    except Exception:
        target_crs = gdf_geo.estimate_utm_crs()

    gdf_proj = gdf_geo.to_crs(target_crs)
    centroids_proj = gdf_proj.geometry.centroid
    centroids_geo = gpd.GeoSeries(centroids_proj, crs=target_crs).to_crs(epsg=4326)

    gdf_geo['lat'] = centroids_geo.y
    gdf_geo['lon'] = centroids_geo.x

    # 5) Campos derivados (usa tus funciones existentes)
    #    Nota: 'building_type' depende de poss_ref del servicio actual
    gdf_geo['building_type'] = gdf_geo.apply(lambda row: building_type(row, poss_ref), axis=1)
    gdf_geo['osm_id'] = gdf_geo.apply(osmid_reform, axis=1)

    return gdf_geo[['building_type','osm_id','geometry','lat','lon']]


def get_vehicle_stats(archetype, transport_archetypes, variables):
    results = {}   
    
    # Filtrar la fila correspondiente al arquetipo
    row = transport_archetypes[transport_archetypes['name'] == archetype]
    if row.empty:
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


def is_it_any_archetype(archetype_to_fill, df_distribution, ind_arch):
    """
    Summary:
       Analyse the df archetype_to_fill df to evaluate if the archetypes shown in this df can really be 
      created or if, due to a lack of any archetype of citizen, it is no longer possible (e.g. if family 
      archetype 3 needs one citizen type 1, one type 3 and 3 type 4 and we don't have enough of any type 
      left, we already know that this family archetype can never be generated again).
    Args:
       archetype_to_fill (DataFrame): df with the archetypes data of the archetypes that can be applied
       df_distribution (DataFrame): All citizens that are to be added to a family
       ind_arch (str): Archetype name on which individuals (families with just one citizen) exists.
    Returns:
       archetype_to_fill (DataFrame): updated df with the archetypes data of the archetypes that can be applied
    """
    
    # Determine names with any 0 or NaN (excluding the first column)
    condition = df_distribution.iloc[:, 1:].eq(0) | df_distribution.iloc[:, 1:].isna()
    problematic_names = set(df_distribution.loc[condition.any(axis=1), 'name'])
    # If there is no problem, we return the original DataFrame.
    if not problematic_names:
        return archetype_to_fill
    # Columns containing archetype data in archetype_to_fill (from fifth column onwards)
    archetype_cols = archetype_to_fill.columns[4:]
    # Create a Boolean mask:
    # For each row, evaluate whether there is at least one column in which, being non-null and other than 0,
    # the column name is in problematic_names.
    mask = archetype_to_fill.apply(
        lambda row: any(col in problematic_names 
                        for col, val in row[archetype_cols].items() 
                        if pd.notna(val) and val != 0),
        axis=1
    )
    # Exclude from deletion the row whose ‘name’ matches ind_arch
    mask &= (archetype_to_fill['name'] != ind_arch)
    
    # Filtering and resetting the index
    return archetype_to_fill[~mask].reset_index(drop=True)


def load_or_create_stats(archetypes_path, filename, creation_func, creation_args):
    filepath = archetypes_path / filename
    try:
        print(f'Loading statistical data from {filename} ...')
        df_stats = pd.read_excel(filepath)
        if df_stats.isnull().sum().sum() != 0:
            print(f"    [ERROR] {filename} is incomplete.\n    Please fill μ, σ, max, min values on this file on the following path and rerun:\n{filepath}")
            sys.exit()
    except Exception:
        # Si falla, crear archivo vacío y lanzar error para que el usuario lo rellene
        creation_func(archetypes_path, *creation_args)
        print(f"    [ERROR] {filename} is incomplete.\n    Please fill μ, σ, max, min values on this file on the following path and rerun:\n{filepath}")
        sys.exit()
    return df_stats
  

def load_or_download_pois(study_area, paths, pop_archetypes_building, special_areas_coords, city_district):
    print(f'Loading POIs data ...')
    
    pop_path = paths['population']
    # Creamos una lista con los servicios que deveriamos tener en el documento
    building_types = (pop_archetypes_building['name']
        .dropna()
        .astype(str).str.strip()
        .unique().tolist()
    )
    try:
        # Intentamos leer el doc
        pop_building = pd.read_parquet(f'{pop_path}/pop_building.parquet')
    except Exception:
        # Ya que no se ha podido leer, creamos el df
        pop_building = download_pois(study_area, paths, pop_archetypes_building, special_areas_coords, city_district)
    # Devolvemos el df completo
    return pop_building
    


def load_or_download_networks(study_area, study_area_path, networks, city_district):
    networks_map = {}
    missing_networks = []
    city = city_district.get(study_area)
    
    print(f'Loading maps ...')

    for net_type in networks:
        try:
            graph = ox.load_graphml(study_area_path / f"{net_type}.graphml")
            networks_map[f"{net_type}_map"] = graph
        except Exception:
            print(f'    [WARNING] {net_type} map is missing.')
            missing_networks.append(net_type)

    if missing_networks:
        print(f'    Downloading missing maps: {missing_networks}')
        for net_type in missing_networks:
            try:
                print(f'        Downloading {net_type} network from {study_area} ...')

                # Transformar el polígono a metros para aplicar buffer
                project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
                gdf = ox.geocode_to_gdf('Kanaleneiland')
                polygon = gdf.iloc[0].geometry

                # Guardar polígono original
                polygon_original = polygon

                # Aplicar buffer a todos
                project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
                polygon_m = transform(project, polygon_original)
                if net_type == 'walk':
                    buff = 300
                else:
                    buff = 1000
                    
                polygon_buffered_m = polygon_m.buffer(buff)

                # Volver a lat/lon
                project_back = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
                polygon_buffered = transform(project_back, polygon_buffered_m)

                # Obtener el grafo de la red usando el buffer
                graph = ox.graph_from_polygon(polygon_buffered, network_type=net_type)
                ox.save_graphml(graph, study_area_path / f"{net_type}.graphml")
                networks_map[f"{net_type}_map"] = graph

                # ---------- PLOTEAR POLÍGONO Y RED SUPERPUESTOS ----------
                # ---------- CREAR GEODATAFRAMES ----------
                gdf_original = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon_original])
                gdf_buffered = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon_buffered])
                # ---------- REPROYECTAR A CRS MÉTRICO ----------
                gdf_original_proj = gdf_original.to_crs(epsg=3857)
                gdf_buffered_proj = gdf_buffered.to_crs(epsg=3857)
                graph_proj = ox.project_graph(graph, to_crs="EPSG:3857")  # Reproyecta la red
                # ---------- CREAR FIGURA ----------
                fig, ax = plt.subplots(figsize=(10, 10))
                # Dibujar buffer
                gdf_buffered_proj.plot(ax=ax, facecolor='lightblue', edgecolor='blue', alpha=0.3, label='Buffer')
                # Dibujar polígono original
                gdf_original_proj.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='Área original')
                # Dibujar grafo de la red en negro sin nodos
                ox.plot_graph(graph_proj, ax=ax, node_size=0, edge_color='black', show=False, close=False)
                # Ajustar la visualización
                ax.set_aspect('equal')  # Mantener proporción real
                ax.set_xlim(gdf_buffered_proj.total_bounds[[0, 2]])  # xmin, xmax del buffer
                ax.set_ylim(gdf_buffered_proj.total_bounds[[1, 3]])  # ymin, ymax del buffer
                # Título y etiquetas
                ax.set_title(f"{study_area} ({net_type}) with buffer of {buff} meters.")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                plt.show()

            except Exception as e:
                print(f'        [ERROR] Failed to download {net_type} network: {e}')
                raise e

    return networks_map

   
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


def osmid_reform(row):
    osmid = row.get('osmid')
    element_type = row.get('element_type')
    
    if pd.isna(osmid) or pd.isna(element_type):
        return None  # Si faltan datos, devolver None
    
    # Determinar el prefijo basado en el tipo del elemento
    if element_type.lower() == 'node':
        return f'N{osmid}'
    elif element_type.lower() == 'relation':
        return f'R{osmid}'
    elif element_type.lower() == 'way':
        return f'W{osmid}'

    # Si no es un tipo válido, devolver None
    return None


def process_arch_to_fill(archetype_df, arch_name, df_distribution):
    """
    Para un 'arch_name' seleccionado, filtra la fila correspondiente en el DataFrame
    'archetype_df', conserva solo las columnas que contengan la palabra 'archetype',
    transpone el resultado y lo une con 'df_distribution' para poder compararlo.
    
    Retorna el DataFrame mergeado.
    """    
    row = archetype_df[archetype_df['name'] == arch_name]
    # Seleccionar columnas que contengan "arch" (sin importar mayúsculas/minúsculas)
    columns_to_keep = [col for col in row.columns if 'arch' in col.lower()] ########################## CUIDADO CON ESTO ASIER DEL FUTURO
    row_filtered = row[columns_to_keep]
    # Transponer y reformatear el DataFrame
    transposed = row_filtered.T.reset_index()
    transposed.columns = ['name', 'participants']  
    # Unir con df_distribution para comparar los valores
    merged_df = pd.merge(df_distribution, transposed, on='name', how='left')
    return merged_df


def plot_voronoi_with_buildings(nodes_gdf_proj, vor_gdf, boundary_proj, buildings_df, lon_col='lon', lat_col='lat'):
    """
    Dibuja los nodos, el polígono límite, las celdas de Voronoi y los edificios.
    
    nodes_gdf_proj : GeoDataFrame de nodos (ya proyectado)
    vor_gdf : GeoDataFrame de polígonos Voronoi
    boundary_proj : Polygon o MultiPolygon límite (ya proyectado)
    buildings_df : DataFrame con edificios, debe tener columnas de lon/lat
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Polígono límite
    if isinstance(boundary_proj, (Polygon, MultiPolygon)):
        boundary_gs = gpd.GeoSeries([boundary_proj], crs=nodes_gdf_proj.crs)
    else:
        boundary_gs = boundary_proj
    boundary_gs.boundary.plot(ax=ax, color="black", linewidth=2, label="Boundary")
    
    # Voronoi
    vor_gdf.plot(ax=ax, alpha=0.3, edgecolor="blue", facecolor="lightblue", label="Voronoi")
    
    # Nodos
    nodes_gdf_proj.plot(ax=ax, color="red", markersize=50, label="Nodes")
    
    # Edificios: convertir a GeoDataFrame en mismo CRS
    buildings_gdf = gpd.GeoDataFrame(
        buildings_df,
        geometry=gpd.points_from_xy(buildings_df[lon_col], buildings_df[lat_col]),
        crs="EPSG:4326"
    ).to_crs(nodes_gdf_proj.crs)
    buildings_gdf.plot(ax=ax, color="green", markersize=20, alpha=0.6, label="Buildings")
    
    plt.legend()
    plt.title("Voronoi Diagram con Nodos y Edificios")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()
   

def random_name_surname(ref_dict):
    if not ref_dict:
        return None
    
    name = random.choice(list(ref_dict.keys()))
    if not ref_dict[name]:
        return None

    surname = random.choice(ref_dict[name])
    return f"{name}_{surname}"


def services_groups_creation(df, to_keep):

    # Diccionarios de salida, uno por grupo
    group_dicts = {}
    
    # Para cada grupo, filtramos filas con 'x' y construimos el diccionario correspondiente
    for group in to_keep:
        group_df = df[df[group] == 'x'][['name', 'surname']].dropna()
        group_ref = {}
        
        for _, row in group_df.iterrows():
            name = row['name']
            surname = row['surname']
            
            if name not in group_ref:
                group_ref[name] = []
            
            if surname not in group_ref[name]:
                group_ref[name].append(surname)
        
        group_dicts[group + "_list"] = group_ref
    
    return group_dicts

        
def Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats, paths, study_area):
    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    stats_synpop = stats['stats_synpop']
    stats_trans = stats['stats_trans']
    stats_class =stats['stats_class']
    SG_relationship = agent_populations['building']
    
    try:
        print(f"Loading synthetic population data ...") 
        
        for type_population in system_management['archetypes'].dropna():
            agent_populations[type_population] = pd.read_parquet(f"{paths['population']}/pop_{type_population}.parquet")
            
    except Exception as e:     
        print(f'    [WARNING] Data is missing.') 
        print(f'        Creating synthetic population (it might take a while) ...')

        ## Synthetic population generation
        # Section added just in case in the future we want to optimize the error of the synthetic population
        archetype_to_analyze = pop_archetypes['citizen']
        archetype_to_fill = pop_archetypes['family']
        
        print("            Citizen inventory creation ...")
        agent_populations['distribution'], total_presence = Citizen_inventory_creation(archetype_to_analyze, population)
        
        print("            Distributing citizens among families ...")
        agent_populations['distribution'], agent_populations['citizen'], agent_populations['family'] = Citizen_distribution_in_families(archetype_to_fill, agent_populations['distribution'], total_presence, stats_synpop, pop_archetypes)
        
        print("            Social class assingment ...")
        agent_populations['family'] = social_class_assingment(agent_populations['family'], stats_class)
        
        print("            Utilities assignment ...")
        agent_populations['family'], agent_populations['citizen'], agent_populations['transport'] = Utilities_assignment(agent_populations['citizen'], agent_populations['family'], pop_archetypes, paths, SG_relationship, stats_synpop, stats_trans)

        print(f"        Saving data ...")

        for type_population in system_management['archetypes'].dropna():
            agent_populations[type_population].to_parquet(
                paths['population'] / f"pop_{type_population}.parquet",
                index=False
            )
            
    return agent_populations

import numpy as np
import pandas as pd

def pick_class_from_stats(archetype, stats_class, class_cols=None):
    """
    Devuelve 'Salariat'/'Intermediate'/'Working' para un archetype dado
    usando los pesos de stats_class. Tolera strings, NaN, y que no sumen 1.
    """
    row = stats_class.loc[stats_class['name'] == archetype]
    if row.empty:
        return np.nan  # o el fallback que prefieras

    s = row.iloc[0]

    # Quedarse solo con las columnas de clase
    if class_cols is None:
        # Inferir: quitar 'name' y cualquier no-numérico tras to_numeric
        s = s.drop(labels=['name'], errors='ignore')
    else:
        s = s.reindex(class_cols)  # asegura orden/consistencia

    # Convertir a numérico, forzar NaN a 0 y evitar negativos
    s = pd.to_numeric(s, errors='coerce').fillna(0).clip(lower=0)

    # Si todos los pesos son 0 -> muestreo uniforme entre las clases presentes
    total = s.sum()
    if total <= 0:
        return np.random.choice(s.index.tolist())

    probs = (s / total).values.astype(float)
    return np.random.choice(s.index.to_list(), p=probs)


def social_class_assingment(family_populations, stats_class):
    
    for idx, family in family_populations.iterrows():
        choice = pick_class_from_stats(
            family['archetype'],
            stats_class,
            class_cols=['Salariat', 'Intermediate', 'Working']  # opcional pero recomendado
        )
        # por ejemplo, guardarlo:
        family_populations.at[idx, 'class'] = choice
    
    return family_populations

def ring_from_poi(lat, lon, x, y, crs="EPSG:4326"):
    poi = gpd.GeoSeries([Point(lon, lat)], crs=crs)
    poi_m = poi.to_crs(epsg=3857)
    outer = poi_m.buffer(x + y).iloc[0]
    inner = poi_m.buffer(max(x - y, 0)).iloc[0]
    ring = outer.difference(inner)
    ring_wgs84 = gpd.GeoSeries([ring], crs="EPSG:3857").to_crs(crs)
    return ring_wgs84.iloc[0]  # <- devuelve shapely.geometry.Polygon


def _as_geometry_and_crs(geom_like):
    """Devuelve (geom_shapely, crs or None) desde geometry / GeoSeries / GeoDataFrame."""
    if geom_like is None:
        return None, None
    if isinstance(geom_like, gpd.GeoDataFrame):
        return geom_like.geometry.unary_union, getattr(geom_like, "crs", None)
    if isinstance(geom_like, gpd.GeoSeries):
        return geom_like.unary_union, getattr(geom_like, "crs", None)
    # shapely geometry
    return geom_like, None

def _best_utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    hemi = 326 if lat >= 0 else 327  # Norte/Sur
    return f"EPSG:{hemi}{zone:02d}"

def plot_wos_debug(
    home_lat, home_lon,
    ring_poly,
    work_df=None, study_df=None,
    chosen_id=None,
    ring_crs="EPSG:4326",
    title=None,
    show=True,
    ax=None,
    draw_mid_circle=True
):
    """
    Visualiza home, anillo, candidatos (work/study) dentro del anillo y el elegido.
    Dibuja además un círculo concéntrico en el centro del anillo, con radio medio
    entre el círculo exterior y el interior.
    """
    ring_geom, ring_geom_crs = _as_geometry_and_crs(ring_poly)
    if ring_geom is None:
        return

    # reproyectar anillo a WGS84 si hace falta
    if ring_geom_crs is not None and ring_geom_crs != "EPSG:4326":
        ring_geom = gpd.GeoSeries([ring_geom], crs=ring_geom_crs).to_crs("EPSG:4326").iloc[0]

    # GeoDataFrame home
    gdf_home = gpd.GeoDataFrame(
        {"name": ["home"]},
        geometry=[Point(home_lon, home_lat)],
        crs="EPSG:4326"
    )

    def _prep_candidates(df):
        if df is None or len(df) == 0:
            return None, None
        gdf = gpd.GeoDataFrame(
            df[['osm_id','lat','lon']].copy(),
            geometry=gpd.points_from_xy(df['lon'], df['lat']),
            crs="EPSG:4326"
        )
        mask = gdf.geometry.within(ring_geom)
        return gdf, gdf.loc[mask.values]

    gdf_work, gdf_work_in = _prep_candidates(work_df)
    gdf_study, gdf_study_in = _prep_candidates(study_df)

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_ax = True

    # Anillo exterior
    gpd.GeoSeries([ring_geom], crs="EPSG:4326").boundary.plot(
        ax=ax, color="black", linewidth=1.5, alpha=0.8, label="Ring"
    )

    # === NUEVO: círculo concéntrico de radio medio ===
    if draw_mid_circle:
        metric_crs = _best_utm_epsg(home_lon, home_lat)
        ring_proj = gpd.GeoSeries([ring_geom], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
        center_proj = gpd.GeoSeries([Point(home_lon, home_lat)], crs="EPSG:4326").to_crs(metric_crs).iloc[0]

        # radio exterior
        r_outer = center_proj.distance(LineString(ring_proj.exterior.coords))
        # radio interior (si hay varios huecos, coger el más cercano)
        if getattr(ring_proj, "interiors", None):
            r_inner = min(center_proj.distance(LineString(r.coords)) for r in ring_proj.interiors)
        else:
            r_inner = 0  # si no hay hueco, se asume círculo sólido

        # radio medio
        r_mid = 0.5 * (r_outer + r_inner)

        # construir círculo medio
        mid_circle_proj = center_proj.buffer(r_mid, resolution=128)
        mid_circle_geo = gpd.GeoSeries([mid_circle_proj], crs=metric_crs).to_crs("EPSG:4326").iloc[0]

        # dibujar perímetro del círculo medio
        gpd.GeoSeries([mid_circle_geo.boundary], crs="EPSG:4326").plot(
            ax=ax, linewidth=1.2, linestyle="--", color="black",
            label=f"Mid circle (~{2*r_mid/1000:.2f} km Ø)"
        )

    # Todos los candidatos (gris claro)
    if gdf_work is not None and len(gdf_work) > 0:
        gdf_work.plot(ax=ax, markersize=8, color="0.8", alpha=0.4, label="Work (all)")
    if gdf_study is not None and len(gdf_study) > 0:
        gdf_study.plot(ax=ax, markersize=8, color="0.8", alpha=0.4, label="Study (all)")

    # Dentro del anillo (gris más oscuro)
    if gdf_work_in is not None and len(gdf_work_in) > 0:
        gdf_work_in.plot(ax=ax, markersize=20, color="0.4", label="Work in ring")
    if gdf_study_in is not None and len(gdf_study_in) > 0:
        gdf_study_in.plot(ax=ax, markersize=20, color="0.4", label="Study in ring")

    # Home (cuadrado negro)
    gdf_home.plot(ax=ax, markersize=60, marker="s", color="black", label="Home")

    # Elegido (estrella negra grande)
    if chosen_id is not None:
        for gdf_cand in (gdf_work, gdf_study):
            if gdf_cand is not None and not gdf_cand.empty:
                hit = gdf_cand[gdf_cand['osm_id'] == chosen_id]
                if not hit.empty:
                    hit.plot(ax=ax, markersize=150, marker="*", color="black", label="Chosen WoS")
                    break

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")

    if created_ax and show:
        plt.show()


def Utilities_assignment(
    df_citizens: pd.DataFrame,
    df_families: pd.DataFrame,
    pop_archetypes: dict,
    paths,
    SG_relationship: pd.DataFrame,
    stats_synpop: pd.DataFrame,
    stats_trans: pd.DataFrame,
    ring_crs: str = "EPSG:4326",
    expand_factor: float = 2.0,
    max_iters: int = 4):
    
    # --- helpers ---
    def pick_building_type(osm_id):
        bt = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, 'building_type']
        return bt.iat[0] if not bt.empty else np.nan

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

    # --- 1) Variables de arquetipo de transporte (una vez) ---
    variables = [c.rsplit('_', 1)[0] for c in pop_archetypes['transport'].columns if c.endswith('_mu')]

    # --- 2) Vehículos privados por familia ---
    df_priv_vehicle = pd.DataFrame(columns=['name', 'archetype', 'family', 'ubication'] + variables)

    Home_ids = SG_relationship.loc[SG_relationship['archetype'] == 'Home', 'osm_id'].tolist()
    if not Home_ids:
        raise ValueError("No hay edificios Home en SG_relationship.")

    shuffled_Home_ids = random.sample(Home_ids, len(Home_ids))
    counter = 0

    for idx_df_f, row_df_f in df_families.iterrows():
        if counter >= len(shuffled_Home_ids):
            shuffled_Home_ids = random.sample(Home_ids, len(Home_ids))
            counter = 0

        Home_id = shuffled_Home_ids[counter]
        counter += 1

        df_families.at[idx_df_f, 'Home'] = Home_id
        df_families.at[idx_df_f, 'Home_type'] = pick_building_type(Home_id)

        # vehículos según stats_trans para este arquetipo de familia
        filtered_st_trans = stats_trans[stats_trans['item_1'] == row_df_f['archetype']]
        for _, row_fs in filtered_st_trans.iterrows():
            stats_value = computate_stats(row_fs)  # asumimos que existe
            for _ in range(int(stats_value)):
                stats_variables = get_vehicle_stats(row_fs['item_2'], pop_archetypes['transport'], variables)
                new_vehicle_row = {
                    'name': f'priv_vehicle_{len(df_priv_vehicle)}',
                    'archetype': row_fs['item_2'],
                    'family': row_df_f['name'],
                    'ubication': Home_id,  # Idealmente plaza de aparcamiento
                    **stats_variables}
                df_priv_vehicle.loc[len(df_priv_vehicle)] = new_vehicle_row

    # --- 3) Atributos de familia y Home por ciudadano ---
    df_citizens['family'] = df_citizens['name'].apply(lambda n: find_group(n, df_families, 'name'))
    df_citizens['family_archetype'] = df_citizens['name'].apply(lambda n: find_group(n, df_families, 'archetype'))
    df_citizens['Home'] = df_citizens['name'].apply(lambda n: find_group(n, df_families, 'Home'))
    df_citizens['class'] = df_citizens['name'].apply(lambda n: find_group(n, df_families, 'class'))

    # --- 4) Work/Study pools (usar LISTAS de columnas, no sets) ---
    work_df = SG_relationship.loc[SG_relationship['archetype'].isin(['Salariat', 'Intermediate', 'Working']), ['archetype', 'osm_id', 'lat', 'lon']].copy()
    # Añadir columna vacía llamada 'pop'
    work_df['pop'] = 0
    study_df = SG_relationship.loc[SG_relationship['archetype'] == 'study', ['osm_id', 'lat', 'lon']].copy()

    # --- 5) Variables de arquetipo de ciudadano (una vez) ---
    citizen_vars = [c.rsplit('_', 1)[0] for c in pop_archetypes['citizen'].columns if c.endswith('_mu')]

    # --- 6) Asignación por ciudadano ---
    for idx, row in df_citizens.iterrows():
        
        class_work_df = work_df[(work_df['archetype'] == row['class']) & (work_df['pop'] < 15)]
        
        # 6.1) escribir variables de arquetipo de ciudadano
        arche = row['archetype']
        citizen_vals = get_vehicle_stats(arche, pop_archetypes['citizen'], citizen_vars)
        row_updated = assign_data(citizen_vars, citizen_vals, row.copy())
        df_citizens.loc[idx, row_updated.index] = row_updated

        # 6.2) home georef
        home_id = df_citizens.at[idx, 'Home']
        home_row = SG_relationship.loc[SG_relationship['osm_id'] == home_id, ['osm_id', 'lat', 'lon']]
        if home_row.empty or home_row[['lat', 'lon']].isna().any(axis=None):
            continue
        home_lat = float(home_row.iloc[0]['lat'])
        home_lon = float(home_row.iloc[0]['lon'])

        # 6.3) anillo con expansión
        
        DEBUG_PLOTS = True # <----- Modificar esto para que podamos plotear los donuts
        
        arch_citizen = pop_archetypes['citizen']
        data_filtered = arch_citizen[arch_citizen['name'] == row_updated['archetype']].iloc[0]

        rx, ry = float(data_filtered['dist_wos_mu']), float(data_filtered['dist_wos_sigma'])
        WoS_id = None
        
        # Asifgnamos los valores de poi_mu y poi_sigma
        df_citizens['dist_poi_mu'], df_citizens['dist_poi_sigma'] = float(data_filtered['dist_poi_mu']), float(data_filtered['dist_poi_sigma'])

        # Si el ciudadano es "fijo" y ya hay WoS en la familia, reutiliza y evita anillos
        if df_citizens.at[idx, 'WoS_fixed'] == 1:
            fam = df_citizens.at[idx, 'family']
            fam_fixed = df_citizens[
                (df_citizens['family'] == fam) &
                (df_citizens['WoS_fixed'] == 1) &
                (df_citizens['WoS'].notna())
            ]
            if not fam_fixed.empty:
                WoS_id = fam_fixed['WoS'].iloc[0]

        for it in range(max_iters):
            if WoS_id is not None:
                break  # ya resuelto por reutilización

            ring = ring_from_poi(home_lat, home_lon, rx, ry, crs=ring_crs)

            if DEBUG_PLOTS:
                plot_wos_debug(
                    home_lat, home_lon,
                    ring_poly=ring,
                    work_df=class_work_df if df_citizens.at[idx, 'WoS_fixed'] != 1 else None,
                    study_df=study_df if df_citizens.at[idx, 'WoS_fixed'] == 1 else None,
                    chosen_id=None,
                    ring_crs=ring_crs,
                    #title=f"Citizen {idx} (dist_mu={rx:.2f}, dist_sigma={ry:.2f})"
                )

            if df_citizens.at[idx, 'WoS_fixed'] != 1:
                # elige uno aleatorio entre los work dentro del anillo
                WoS_id = choose_id_in_ring(class_work_df, ring)
            else:
                # si no había reutilización, busca entre study dentro del anillo
                WoS_id = choose_id_in_ring(study_df, ring)

            if WoS_id is not None:
                break  # encontrado → salir
            ry *= expand_factor
            
            if ry < 0:
                ry = 0

        # Si no se encontró dentro de anillos, hacer fallback aleatorio # ISSUE 46
        if WoS_id is None:
            if df_citizens.at[idx, 'WoS_fixed'] != 1:
                WoS_id = work_df['osm_id'].sample(1).iat[0] if not work_df.empty else None
            else:
                if not study_df.empty:
                    WoS_id = study_df['osm_id'].sample(1).iat[0]
                elif not work_df.empty:
                    WoS_id = work_df['osm_id'].sample(1).iat[0]

        if WoS_id is None:
            continue  # no hay candidatos
        
        df_citizens.at[idx, 'WoS'] = WoS_id
        df_citizens.at[idx, 'WoS_subgroup'] = pick_building_type(WoS_id)

        # Añadimos los agenets al osm_id para asegurar que no usamos espacios overbooked
        if df_citizens.at[idx, 'WoS_fixed'] != 1:
            idx = work_df.index[work_df['osm_id'] == WoS_id]
            for index in idx:
                work_df.at[index, 'pop'] += 1
    
    return df_families, df_citizens, df_priv_vehicle


def voronoi_from_nodes(electric_system: pd.DataFrame, boundary_polygon: Polygon):
    """
    Genera un diagrama de Voronoi recortado por un polígono límite.

    Parámetros:
    -----------
    electric_system : pd.DataFrame
        DataFrame con nodos. La primera columna (sin nombre) contiene los nombres.
        Debe tener columnas 'long' y 'lat'.
    boundary_polygon : shapely.geometry.Polygon
        Polígono límite.

    Retorna:
    --------
    nodes_gdf_proj : gpd.GeoDataFrame
        GeoDataFrame de nodos proyectados.
    vor_gdf : gpd.GeoDataFrame
        Polígonos de Voronoi recortados al límite.
    boundary_proj : shapely.geometry.Polygon
        Polígono límite proyectado y unificado.
    """
    
    # Renombrar la primera columna a 'node'
    electric_system = electric_system.rename(columns={electric_system.columns[0]: "node"})

    # Renombrar la primera columna como 'node'
    df = electric_system.rename(columns={electric_system.columns[0]: "node"}).copy()

    # Crear GeoDataFrame de nodos
    nodes_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["long"], df["lat"]),
        crs="EPSG:4326"
    )

    # Proyectar a un CRS métrico (UTM automático)
    utm_crs = nodes_gdf.estimate_utm_crs()
    nodes_gdf_proj = nodes_gdf.to_crs(utm_crs)
    boundary_proj = gpd.GeoSeries([boundary_polygon], crs="EPSG:4326").to_crs(utm_crs).unary_union

    # Generar Voronoi
    multipoint = nodes_gdf_proj.unary_union
    voronoi = voronoi_diagram(multipoint, envelope=boundary_proj)

    # Convertir polígonos Voronoi a GeoDataFrame
    vor_polys = []
    node_ids = []
    for idx, point in enumerate(nodes_gdf_proj.geometry):
        poly = [cell for cell in voronoi.geoms if cell.contains(point)]
        if poly:
            clipped = poly[0].intersection(boundary_proj)
            vor_polys.append(clipped)
            node_ids.append(nodes_gdf_proj.iloc[idx]["node"])

    vor_gdf = gpd.GeoDataFrame({"node": node_ids, "geometry": vor_polys}, crs=utm_crs)
    
    return nodes_gdf_proj, vor_gdf, boundary_proj


def main():
    # Input
    population = 450
    study_area = 'Kanaleneiland'
    
    ## Code initialization
    # Paths initialization
    paths = {}
    
    paths['main'] = Path(__file__).resolve().parent.parent.parent
    paths['system'] = paths['main'] / 'system'
    paths['desktop'] = Path.home() / "Desktop"
    
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
            elif row['pre'] == 'p':
                user_is_stupid = True
                while user_is_stupid:    
                    response = input(f"Data for the case study '{study_area}' was not found.\nDo you want to copy data from standar scenario or do you want to create your own? [Y (copy)/N (create)]\n")
                    if response == 'Y':
                        user_is_stupid = False
                        shutil.copytree(paths['base_scenario'], paths[file_2])
                    elif response == 'N':
                        user_is_stupid = False
                        os.makedirs(paths[file_2], exist_ok=True)
                    else:
                        print(f"Your response was not valid, please respond Y (yes) or N (no).")
            else:
                os.makedirs(paths[file_2], exist_ok=True)
    
    print('#'*20, ' System initialization ','#'*20)
    # Archetype documentation initialization
    pop_archetypes, stats = Archetype_documentation_initialization(paths)
    # Geodata initialization
    agent_populations, networks_map = Geodata_initialization(study_area, paths, pop_archetypes)
    # Synthetic population initialization
    agent_populations = Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats, paths, study_area)
    print('#'*20, ' Initialization finalized ','#'*20)
    
if __name__ == '__main__':
    main()
