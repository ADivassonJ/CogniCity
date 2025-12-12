from __future__ import annotations

# estándar
import itertools
import os
import random
import shutil
import sys
from pathlib import Path
from scipy.stats import norm

# terceros
import folium
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.stats import lognorm
import pyproj
from itertools import cycle
from tqdm import tqdm
from haversine import Unit, haversine
from scipy.spatial import Voronoi, cKDTree
from shapely.errors import ShapelyDeprecationWarning
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import box, MultiPolygon, Point, Polygon
from shapely.ops import clip_by_rect, transform, unary_union, voronoi_diagram


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


def assign_data(list_variables, list_values, row_old):
    """
    Asigna valores de list_values a una fila (Series o dict) según reglas:
      - *_type y *_amount → enteros redondeados
      - *_time → redondeado a múltiplos de 30
      - resto → tal cual
    """

    row = row_old.copy() 
    
    for variable in list_variables:
        val = list_values.get(variable, None)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue  # no asignamos nada si falta valor

        if variable.endswith(('_type', '_amount', '_fixed')):
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

        # Combinar ventanas: usa .get() para evitar KeyError
        # OJO: valida que realmente quieres sumarlas y no otra operación
        wo_open  = list_building_values.get('WoS_opening', 0) or 0
        sv_open  = list_building_values.get('Service_opening', 0) or 0
        wo_close = list_building_values.get('WoS_closing', 0) or 0
        sv_close = list_building_values.get('Service_closing', 0) or 0

        list_building_values['Service_opening'] = wo_open + sv_open
        list_building_values['Service_closing'] = wo_close + sv_close #ISSUE 54

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
        'stat_percentage': 0.0,
        'error': 0.0
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
        df_stats_families['stat_percentage'] = (
            df_stats_families['stat_presence'].astype(float)
            / df_stats_families['stat_presence'].sum()
        ) * 100
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
        # Update population distribution
        mask = df_distribution['population'].notna() & merged_df['participants'].notna()

        # Aseguramos que ambos sean numéricos y compatibles
        df_distribution['population'] = pd.to_numeric(df_distribution['population'], errors='coerce')
        merged_df['participants'] = pd.to_numeric(merged_df['participants'], errors='coerce')

        # Realizamos la resta como float (para evitar errores de tipo)
        df_distribution.loc[mask, 'population'] = (
            df_distribution.loc[mask, 'population'].astype(float)
            - merged_df.loc[mask, 'participants'].astype(float)
        )

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

def download_pois(study_area, paths, building_archetypes_df, special_areas_coords, city_district, building_types):
    # User Interface
    print(f'    [WARNING] Data is missing, it needs to be downloaded.')
    
    # Sacamos los datos de paths
    study_area_path = paths[study_area]
    cache_path = os.path.join(paths['maps'], "cache")
    
    # Nos aseguramos de que contamos con documentos criticos
    try:
        print(f"        Reading 'Class matrix for ESeC 2008 rev.xlsx' ...")
        Services_Group_relationship = pd.read_excel(f'{study_area_path}/Class matrix for ESeC 2008 rev.xlsx')
    except Exception:
        print(f"    [ERROR] File 'Class matrix for ESeC 2008 rev.xlsx' is not found in the data folder ({study_area_path}).")
        sys.exit()
        
    # User Interface
    print(f'        Processing data (it might take a while)...')
    # En base al distrito de analisis, sacamos la ciudad a descargar
    city = city_district.get(study_area)
    
    gdf = ox.geocode_to_gdf(city)
    #gdf = ox.geocode_to_gdf('Kanaleneiland')

    polygon = gdf.iloc[0].geometry
    # Crear si no existe
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path, exist_ok=True)
    # Listar archivos .geojson
    geojson_files = [f.replace(".geojson", "") for f in os.listdir(cache_path) if f.endswith(".geojson")]
    missing_files = [b for b in building_types if b not in geojson_files]
    not_missing_files = [b for b in building_types if b in geojson_files]
    
    # User Interface
    for not_miss in not_missing_files:
        print(f"            {not_miss}: retrieved from cache.")
    
    # Creamos el diccionario con los datos de etiquetas a considerar
    services_groups = services_groups_creation(Services_Group_relationship, missing_files)

    for group_name, group_ref in services_groups.items():
        
        if not group_name in ['work', 'Entertainment', 'Salariat', 'Intermediate', 'Working', 'Home']:
            try:
                # Descargamos datos
                df_group = get_osm_elements(polygon, group_ref)
                df_group['archetype'] = group_name
                # Guardamos resultados
                df_group.to_file(f"{cache_path}/{group_name}.geojson", driver="GeoJSON")
                # UserInterface
                print(f"            {group_name}: {len(df_group)} elements found")
            except Exception as e:
                print(f"            [ERROR] Failed to get data for {group_name}: {e}")
        else:
            
            if not os.path.isdir(f"{cache_path}/{group_name}"):
                    os.makedirs(f"{cache_path}/{group_name}", exist_ok=True)
                
            # Listar archivos .geojson
            geojson_files_2 = [f.replace(".geojson", "") for f in os.listdir(f"{cache_path}/{group_name}") if f.endswith(".geojson")]
            missing_files_2 = [b for b in group_ref if b not in geojson_files_2]
            not_missing_files_2 = [b for b in group_ref if b in geojson_files_2]
            
            # User Interface
            for not_miss in not_missing_files_2:
                print(f"            {group_name}_{not_miss}: retrieved from cache.")
            
            for key in missing_files_2:
                
                smaller = {key: group_ref[key]}
                
                try:
                    # Descargamos datos
                    df_group = get_osm_elements(polygon, smaller)
                    df_group['archetype'] = group_name
                    # Guardamos resultados
                    df_group.to_file(f"{cache_path}/{group_name}/{key}.geojson", driver="GeoJSON")
                    # UserInterface
                    print(f"            {group_name}_{key}: {len(df_group)} elements found")
                except Exception as e:
                    print(f"            [ERROR] Failed to get data for {group_name}_{key}: {e}")
            # Concatenar todos los DataFrames
            # Leer todos los .geojson del path
            geojson_files_2 = [os.path.join(f"{cache_path}/{group_name}", f) for f in os.listdir(f"{cache_path}/{group_name}") if f.endswith(".geojson")]
            
            # Cargar y unir en un solo GeoDataFrame
            gdfs_2 = [gpd.read_file(f) for f in geojson_files_2]
            osm_elements_df_2 = gpd.GeoDataFrame(pd.concat(gdfs_2, ignore_index=True), crs=gdfs_2[0].crs)
            osm_elements_df_2.to_file(f"{cache_path}/{group_name}.geojson", driver="GeoJSON")
            
    # Concatenar todos los DataFrames
    # Leer todos los .geojson del path
    geojson_files = [os.path.join(cache_path, f) for f in os.listdir(cache_path) if f.endswith(".geojson")]

    # Cargar y unir en un solo GeoDataFrame
    gdfs = [gpd.read_file(f) for f in geojson_files]
    osm_elements_df = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    
    SG_relationship = building_schedule_adding(osm_elements_df, building_archetypes_df)
    
    coords = special_areas_coords[study_area]
    # Intercambiar lat y lon a lon, lat
    coords_corrected = [(lon, lat) for lat, lon in coords]
    distric_polygon = Polygon(coords_corrected)
    
    # Añadir datos de buses eléctricos aquí
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

def Geodata_initialization(study_area, paths, pop_archetypes, special_areas_coords, city_district):
    agent_populations = {}
    # Obtener redes activas desde transport_archetypes
    networks = get_active_networks(pop_archetypes['transport'])

    # Paso 1: Cargar o descargar POIs
    agent_populations['building'] = load_or_download_pois(study_area, paths, pop_archetypes['building'], special_areas_coords, city_district)
    pop_building = agent_populations['building']
    
    paths['new_POIs'] = os.path.join(paths['maps'], 'new_POIs')
    
    ## Guardamos los charging points (si hay) en paths['maps']
    # Crear si no existe
    if not os.path.isdir(paths['new_POIs']):
        os.makedirs(paths['new_POIs'], exist_ok=True)
        
    if not os.path.isfile(f"{paths['new_POIs']}/charging_station.xlsx"):
        charging_station = pop_building[pop_building['archetype'] == 'charging_station'].reset_index(drop=True)
        charging_station.to_excel(f"{paths['new_POIs']}/charging_station.xlsx", index=False)
        print(f"    [WARNING] No document relating to “charging_station” has been detected in the “new_POIs” folder.")
        print(f"    Unless the user modifies this file, the system will only consider the current quantity and")
        print(f"    distribution of “charging_station”, with a total of {len(charging_station)} for the current case.")
    else:
        pop_building_simp = pop_building[pop_building['archetype'] != 'charging_station']
        charging_station = pd.read_excel(f"{paths['new_POIs']}/charging_station.xlsx")
        agent_populations['building'] = pd.concat([pop_building_simp, charging_station], ignore_index=True).reset_index(drop=True)
        print(f"    [NOTE] Document relating to “charging_station” detected [Data copied].")
    
    if not os.path.isfile(f"{paths['new_POIs']}/share_mob_hubs.xlsx"):
        print(f"    [WARNING] No document relating to “share_mob_hubs” has been detected in the “new_POIs” folder.")
        print(f"    Unless the user inserts this file, the system will not consider this service.")
    
    # Guardar resultados
    agent_populations['building'].to_parquet(f"{paths['population']}/pop_building.parquet", index=False)
    
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
        gdf = ox.features_from_polygon(polygon, tags).reset_index(drop=True)
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
        
        if variable in ('dist_wos', 'dist_poi'):
            var_result = np.random.lognormal(mean=mu, sigma=sigma)
        else:
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
            print(f"    [ERROR] {filename} is incomplete.\n    Please fill μ,  sigma, max, min values on this file on the following path and rerun:\n{filepath}")
            sys.exit()
    except Exception:
        # Si falla, crear archivo vacío y lanzar error para que el usuario lo rellene
        creation_func(archetypes_path, *creation_args)
        print(f"    [ERROR] {filename} is incomplete.\n    Please fill μ,  sigma, max, min values on this file on the following path and rerun:\n{filepath}")
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
        pop_building = pd.read_parquet(os.path.join(pop_path, 'pop_building.parquet'))
    except Exception:
        # Ya que no se ha podido leer, creamos el df
        pop_building = download_pois(study_area, paths, pop_archetypes_building, special_areas_coords, city_district, building_types)
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
        
        group_dicts[group] = group_ref
    
    return group_dicts

        
def Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats, paths, study_area, special_areas_coords, city_district):
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
        agent_populations['family'], agent_populations['citizen'], agent_populations['transport'] = Utilities_assignment(agent_populations['citizen'], agent_populations['family'], pop_archetypes, paths, SG_relationship, stats_synpop, stats_trans, study_area, special_areas_coords)

        print(f"        Saving data ...")

        for type_population in system_management['archetypes'].dropna():
            agent_populations[type_population].to_parquet(
                paths['population'] / f"pop_{type_population}.parquet",
                index=False
            )
            
    return agent_populations

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
        family_populations.at[idx, 's_class'] = choice
    
    return family_populations

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

def only_inside_district(Home_ids, geometry_latlon, pop_building):
    # Convertir a (lon, lat) para shapely
    poly_lonlat = [(lon, lat) for (lat, lon) in geometry_latlon]
    poly = Polygon(poly_lonlat)  # se cierra automáticamente

    # --- Filtrar pop_building a solo los Home_ids y limpiar coords ---
    buildings_home = pop_building[pop_building['osm_id'].isin(Home_ids)].copy()
    buildings_home = buildings_home.rename(columns={"latitude": "lat", "longitude": "lon"})  # por si acaso
    buildings_home = buildings_home[pd.notna(buildings_home['lat']) & pd.notna(buildings_home['lon'])]
    buildings_home[['lat','lon']] = buildings_home[['lat','lon']].astype(float)

    # --- Puntos dentro del polígono (estrictamente dentro) ---
    # Si quieres incluir los del borde, cambia 'contains' por 'covers'
    mask_inside = buildings_home.apply(
        lambda r: poly.contains(Point(r['lon'], r['lat'])),
        axis=1
    )
    homes_inside = buildings_home[mask_inside].copy()
    
    return homes_inside['osm_id'].tolist()

def calculate_centroid(coords):
    """
    Calcula el centroide (latitud, longitud) de un polígono definido por coordenadas.
    
    Args:
        coordenadas (list of tuple): Lista de tuplas (lat, lon)
    
    Returns:
        tuple: (latitud_centroide, longitud_centroide)
    """
    # Shapely usa el orden (x, y) → (longitud, latitud)
    poligono = Polygon([(lon, lat) for lat, lon in coords])
    centroide = poligono.centroid
    return (centroide.y, centroide.x)

def Utilities_assignment(
    df_citizens: pd.DataFrame,
    df_families: pd.DataFrame,
    pop_archetypes: dict,
    paths,
    SG_relationship: pd.DataFrame,
    stats_synpop: pd.DataFrame,
    stats_trans: pd.DataFrame,
    study_area, 
    special_areas_coords,
    ring_crs: str = "EPSG:4326",
    max_iters: int = 4,
    disk: bool = False,
    max_ocupancy: int = 14):
    
    # --- helpers ---
    def pick_building_type(osm_id):
        if "_" in osm_id:
            return 'outside'
        bt = SG_relationship.loc[SG_relationship['osm_id'] == osm_id, 'building_type']
        return bt.iat[0] if not bt.empty else np.nan
    
    def choose_id(pop_buildings):
        sorted = pop_buildings.sort_values(by='distr_dist').reset_index(drop=True)
        
        return sorted['osm_id'].iloc[0] if not sorted.empty else None

    def visualizar_anillo_y_edificios(cand_df, ring_poly, chosen_id=None):
        """
        Visualiza un anillo (ring_poly) y los edificios (cand_df) en escala de grises.
        - Gris claro: edificios fuera
        - Gris oscuro: edificios dentro
        - Blanco con borde negro: edificio elegido
        - Gris medio translúcido: anillo
        """

        # Crear GeoDataFrame
        gdf = gpd.GeoDataFrame(
            cand_df.copy(),
            geometry=gpd.points_from_xy(cand_df['lon'], cand_df['lat']),
            crs="EPSG:4326"
        )

        # Asegurar geometría del anillo
        if isinstance(ring_poly, (gpd.GeoSeries, gpd.GeoDataFrame)):
            ring_geom = ring_poly.unary_union
        else:
            ring_geom = ring_poly

        ring_gs = gpd.GeoSeries([ring_geom], crs="EPSG:4326")

        # Pasar a CRS proyectado (metros)
        gdf_proj = gdf.to_crs("EPSG:3857")
        ring_proj = ring_gs.to_crs("EPSG:3857")

        # Determinar qué puntos están dentro
        mask_inside = gdf_proj.geometry.within(ring_proj.iloc[0])

        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 8))

        # --- ANILLO ---
        if not ring_proj.is_empty.all():
            ring_proj.plot(ax=ax, color="lightgrey", edgecolor="dimgray", alpha=0.3)

        # --- EDIFICIOS FUERA ---
        gdf_out = gdf_proj[~mask_inside]
        if not gdf_out.empty:
            gdf_out.plot(ax=ax, color="gainsboro", markersize=25, label="Fuera")

        # --- EDIFICIOS DENTRO ---
        gdf_in = gdf_proj[mask_inside]
        if not gdf_in.empty:
            gdf_in.plot(ax=ax, color="dimgray", markersize=35, label="Dentro")

        # --- EDIFICIO ELEGIDO ---
        if chosen_id is not None and chosen_id in gdf_proj['osm_id'].values:
            elegido = gdf_proj.loc[gdf_proj['osm_id'] == chosen_id]
            elegido.plot(ax=ax, facecolor="white", edgecolor="black",
                        markersize=80, marker="o", zorder=3, label="Elegido")

        # --- LEYENDA MANUAL ---
        legend_elements = [
            mpatches.Patch(facecolor='lightgrey', edgecolor='dimgray', alpha=0.3, label='Ring'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='dimgray', markersize=8, label='Inside'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gainsboro', markersize=8, label='Outside'),
            Line2D([0], [0], marker='o', color='black', markerfacecolor='white', markersize=10, label='Choosen ID')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Estilo general
        ax.set_aspect('equal')
        ax.set_facecolor("white")
        plt.tight_layout()
        plt.show()

    def generate_private_vehicles(df_families, pop_archetypes, stats_trans,
                              SG_relationship, special_areas_coords, study_area,
                              computate_stats, get_vehicle_stats, pick_building_type,
                              only_inside_district):
        """Genera los vehículos privados por familia según arquetipos de transporte."""
        
        # --- 1) Variables de transporte ---
        transport_vars = [
            col.rsplit('_', 1)[0] 
            for col in pop_archetypes['transport'].columns 
            if col.endswith('_mu')
        ]

        # --- 2) Filtrar hogares dentro del distrito ---
        home_ids = SG_relationship.loc[SG_relationship['archetype'] == 'Home', 'osm_id'].tolist()
        homes_inside = only_inside_district(home_ids, special_areas_coords[study_area], SG_relationship)

        if not homes_inside:
            raise ValueError("No se encontraron edificios tipo 'Home' dentro del área de estudio.")

        # --- 3) Preparar iterador infinito de hogares ---
        homes_cycle = cycle(random.sample(homes_inside, len(homes_inside)))

        # --- 4) Crear lista de vehículos ---
        vehicles = []

        for _, family in tqdm(df_families.iterrows(), total=len(df_families), desc="                Vehicles generation"):
            home_id = next(homes_cycle)

            # Actualizamos df_families en memoria (sin escribir cada iteración)
            family_name = family['name']
            family_archetype = family['archetype']
            df_families.at[_, 'Home'] = home_id
            df_families.at[_, 'Home_type'] = pick_building_type(home_id)

            # --- 5) Vehículos según el arquetipo de la familia ---
            trans_subset = stats_trans[stats_trans['item_1'] == family_archetype]
            for _, row in trans_subset.iterrows():
                n_vehicles = int(computate_stats(row))
                for _ in range(n_vehicles):
                    vehicle_vars = get_vehicle_stats(row['item_2'], pop_archetypes['transport'], transport_vars)
                    vehicles.append({
                        'name': f"priv_vehicle_{len(vehicles)}",
                        'archetype': row['item_2'],
                        'family': family_name,
                        'ubication': home_id,
                        **vehicle_vars
                    })
            
            ## Public transport
            vehicle_vars = get_vehicle_stats('UB_diesel', pop_archetypes['transport'], transport_vars)
            # Creamos la nueva fila
            vehicles.append({
                    'name': f"Public_transport",
                    'archetype': 'UB_diesel',
                    'family': family_name,
                    'ubication': home_id,
                    **vehicle_vars
                })


            # --- 6) Construir DataFrame final ---
            df_priv_vehicle = pd.DataFrame(vehicles, columns=['name', 'archetype', 'family', 'ubication'] + transport_vars)

        return df_priv_vehicle, df_families

    def assign_family_attributes(df_citizens, df_families):
        """Asigna atributos de familia y hogar a cada ciudadano de forma vectorizada y eficiente."""
        
        # Crear un mapeo de miembro -> datos de familia
        member_to_family = {}
        for _, row in tqdm(df_families.iterrows(), total=len(df_families), desc="                Assigning family attributes"):
            for member in row['members']:
                member_to_family[member] = {
                    'family': row['name'],
                    'family_archetype': row['archetype'],
                    'Home': row['Home'],
                    's_class': row['s_class']
                }
        
        # Convertir el mapeo en un DataFrame temporal
        df_map = pd.DataFrame.from_dict(member_to_family, orient='index')
        df_map.index.name = 'name'
        df_map.reset_index(inplace=True)
        
        # Hacer merge directo en base al nombre
        df_result = df_citizens.merge(df_map, on='name', how='left')
        
        return df_result
    
    def dist_real_calculation(home_df, row, df, WoS_id):

        if WoS_id == None:
            input(f"Strange error ocurred (WTF)")

        if WoS_id.startswith("virtual"):
            return row['dist_wos']

        Home_id = row['Home']

        home_lat, home_lon = home_df.loc[home_df['osm_id'] == Home_id, ['lat', 'lon']].iloc[0]
        try:
            wos_lat, wos_lon = df.loc[df['osm_id'] == WoS_id, ['lat', 'lon']].iloc[0]
        except Exception:
            # Esto se da si el WoS es el propio hogar
            return 0

        return haversine((home_lat, home_lon), (wos_lat, wos_lon), unit=Unit.METERS) * 1.293

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


    def assign_utilities(df_citizens, df_families, df_priv_vehicle,
                        work_df, study_df, home_df, SG_relationship, pop_archetypes,
                        citizen_vars, ring_crs, disk, max_iters, max_ocupancy):

        # Preindex SG_relationship para acceso directo
        SG_relationship_unique = SG_relationship.drop_duplicates(subset=['osm_id'], keep='first')
        sg_map = SG_relationship_unique.set_index('osm_id')[['lat', 'lon']].to_dict('index')

        # Preindex arquetipos ciudadanos
        arch_stats = pop_archetypes['citizen'].set_index('name')

        # Preindex work_df por nombre de arquetipo
        work_df_idx = {arch: grp for arch, grp in work_df.groupby('archetype')}

        # Index para actualizar población más rápido
        work_df = work_df.set_index('osm_id')

        # Pasamos df a arrays temporales para evitar accesos repetidos
        citizens_arr = df_citizens.copy()

        # Bucle más rápido con itertuples
        for row in tqdm(citizens_arr.itertuples(index=True), total=citizens_arr.shape[0],
                        desc="                Utilities assignation: "):

            idx = row.Index
            arch = row.archetype
            s_class = row.s_class

            # Tomar subconjunto de work/study ya prefiltrado por arquetipo
            class_work_df = work_df_idx.get(s_class, None)
            class_work_df = class_work_df[class_work_df['pop'] < max_ocupancy]

            # Obtener estadísticas del vehículo y asignar valores
            citizen_vals = get_vehicle_stats(arch, pop_archetypes['citizen'], citizen_vars)
            row_updated = assign_data(citizen_vars, citizen_vals, citizens_arr.loc[idx])
            citizens_arr.loc[idx, citizen_vars] = row_updated[citizen_vars]

            # Localizar vivienda
            home_id = citizens_arr.at[idx, 'Home']
            home_info = sg_map.get(home_id, None)
            if home_info is None or pd.isna(home_info['lat']) or pd.isna(home_info['lon']):
                continue
   
            data_filtered = arch_stats.loc[row_updated['archetype']]

            # Guardar parámetros por fila
            citizens_arr.at[idx, 'dist_poi_mu'] = float(data_filtered['dist_poi_mu'])
            citizens_arr.at[idx, 'dist_poi_sigma'] = float(data_filtered['dist_poi_sigma'])

            # Probabilidad de quedarse en casa
            homestay_cond = random.random() < citizens_arr.at[idx, 'homestay']
            WoS_id = None

            # Caso fijo por familia
            if citizens_arr.at[idx, 'WoS_fixed'] == 1:
                fam = citizens_arr.at[idx, 'family']
                fam_fixed = citizens_arr[
                    (citizens_arr['family'] == fam) &
                    (citizens_arr['WoS_fixed'] == 1) &
                    (citizens_arr['WoS'].notna())
                ]
                if not fam_fixed.empty:
                    WoS_id = fam_fixed['WoS'].iloc[0]
            
            if homestay_cond:
                WoS_id = home_id

            # Selección WoS si no fijo
            if WoS_id is None:
                if disk:
                    WoS_id = choose_id(class_work_df if citizens_arr.at[idx, 'WoS_fixed'] != 1 else study_df)
                else:
                    home_lat, home_lon = float(home_info['lat']), float(home_info['lon'])

                    # OJO: aquí mu y sigma deben ser los parámetros en log-espacio si ring_from_poi
                    # los interpreta como mu_log y sigma_log.
                    mu_log    = float(data_filtered['dist_wos_mu'])
                    sigma_log = float(data_filtered['dist_wos_sigma'])

                    print(f' archetype: {row.archetype}')
                    input(f"mu_log: {mu_log}, sigma_log: {sigma_log}")

                    ring, sample_point = ring_from_poi(
                        row_updated,
                        home_lat,
                        home_lon,
                        mu_log/1.293,
                        sigma_log/1.293,
                        crs=ring_crs,
                        return_point=True  # <<< importante
                    )

                    cand_df = class_work_df if citizens_arr.at[idx, 'WoS_fixed'] != 1 else study_df

                    WoS_id = choose_closest_to_point_in_ring(
                        cand_df,
                        ring,
                        sample_point
                    )


            # Si no se encuentra lugar válido
            if WoS_id is None:
                WoS_id = f"virtual_WoS_{row.name}"
                virtual_WoS = True
            else:
                virtual_WoS = False

            citizens_arr.at[idx, 'WoS'] = WoS_id

            # Distancia real
            citizens_arr.at[idx, 'dist_wos_real'] = dist_real_calculation(
                home_df, citizens_arr.loc[idx],
                class_work_df if citizens_arr.at[idx, 'WoS_fixed'] != 1 else study_df,
                WoS_id
            )

            # Subgrupo WoS
            if WoS_id == home_id:
                citizens_arr.at[idx, 'WoS_subgroup'] = 'Home'
            elif virtual_WoS:
                citizens_arr.at[idx, 'WoS_subgroup'] = 'unknown'
            else:
                citizens_arr.at[idx, 'WoS_subgroup'] = pick_building_type(WoS_id)

            # Actualizar población en work_df si no fijo y no virtual
            if citizens_arr.at[idx, 'WoS_fixed'] != 1 and WoS_id in work_df.index and not virtual_WoS:
                work_df.at[WoS_id, 'pop'] += 1

        return df_families, citizens_arr, df_priv_vehicle

    
    df_priv_vehicle, df_families = generate_private_vehicles(df_families, pop_archetypes, stats_trans,
                                    SG_relationship, special_areas_coords, study_area,
                                    computate_stats, get_vehicle_stats, pick_building_type,
                                    only_inside_district)

    df_citizens = assign_family_attributes(df_citizens, df_families)
   
    centroid = calculate_centroid(special_areas_coords[study_area])
    
    work_df = SG_relationship.loc[SG_relationship['archetype'].isin(['Salariat', 'Intermediate', 'Working']), ['archetype', 'osm_id', 'lat', 'lon']].copy()
    
    # Añadir columna vacía llamada 'pop'
    work_df['pop'] = 0
    study_df = SG_relationship.loc[SG_relationship['archetype'] == 'study', ['osm_id', 'lat', 'lon']].copy()
    home_df = SG_relationship.loc[SG_relationship['archetype'] == 'Home', ['osm_id', 'lat', 'lon']].copy()

    if disk:
        for idx_wd, row_wd in work_df.iterrows():
            work_df.at[idx_wd,'distr_dist'] = haversine(centroid, (row_wd['lat'], row_wd['lon']), unit=Unit.METERS) * 1.293
        
        for idx_wd, row_wd in study_df.iterrows():
            study_df.at[idx_wd,'distr_dist'] = haversine(centroid, (row_wd['lat'], row_wd['lon']), unit=Unit.METERS) * 1.293

    # --- 5) Variables de arquetipo de ciudadano (una vez) ---
    citizen_vars = [c.rsplit('_', 1)[0] for c in pop_archetypes['citizen'].columns if c.endswith('_mu')]

    df_families, df_citizens, df_priv_vehicle = assign_utilities(df_citizens, df_families, df_priv_vehicle,
                                                                work_df, study_df, home_df, SG_relationship, pop_archetypes,
                                                                citizen_vars, ring_crs, disk, max_iters, max_ocupancy)
    
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

def paths_initialization(study_area):
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
                    response = input(f"Data for the case study '{study_area}' was not found.\nDo you want to copy data from standar scenario or do you want to create your own? [Y/N]\n")
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
    return paths, system_management

def Documents_initialisation(population, study_area):
    print('#'*20, ' System initialization ','#'*20)
    
    # Diccionario con coordenadas de los territorios especiales
    special_areas_coords = {
        "Aradas": [ (40.6260277,-8.6691095),
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
                ],
        "Kanaleneiland": [  (52.07904398,	5.081736117),
                            (52.07624318,	5.08308264),
                            (52.06046958,	5.09756737),
                            (52.06021839,	5.097758556),
                            (52.06008988,	5.11164107),
                            (52.06328398,	5.113065093),
                            (52.06860149,	5.111588679),
                            (52.07642504,	5.109425399),
                            (52.07861645,	5.108711591),
                            (52.08034774,	5.107271173),
                            (52.08592257,	5.097037869),
                            (52.08498639,	5.096460351),
                            (52.08309467,	5.094751129),
                            (52.0803543, 	5.087985518),
                            (52.07904398,	5.081736117),
                            ],
        "Annelinn": [(58.37779995285961, 26.737546920776367),
                    (58.38207378048632,  26.74806118011475),
                    (58.38095016884387,  26.753661632537845),
                    (58.380230379627854, 26.761858463287357),
                    (58.37991546722738,  26.76752328872681),
                    (58.37947683455617,  26.77179336547852),
                    (58.379218151193385, 26.773359775543216),
                    (58.37876826256608,  26.77522659301758),
                    (58.377767239785584, 26.77874565124512),
                    (58.377013642098476, 26.77923917770386),
                    (58.375427660052644, 26.784152984619144),
                    (58.37414532457297,  26.7826509475708),
                    (58.371591763372905, 26.782929897308353),
                    (58.37125427449941,  26.783938407897953),
                    (58.369139270733456, 26.78344488143921),
                    (58.368970514972645, 26.782200336456302),
                    (58.36281344471082,  26.78001165390015),
                    (58.359865199696884, 26.77775859832764),
                    (58.358199670019,    26.7762565612793),
                    (58.35478959051058,  26.7717719078064),
                    (58.35464327610064,  26.770827770233158),
                    (58.35328311497021,  26.765420436859134),
                    (58.35779452929785,  26.76052808761597),
                    (58.359280022547246, 26.759669780731205),
                    (58.357108030241015, 26.754348278045658),
                    (58.35555491754449,  26.746966838836673),
                    (58.355836283607296, 26.7467737197876),
                    (58.356140156436815, 26.746795177459717),
                    (58.357198063664704, 26.7476749420166),
                    (58.35798584632884,  26.749970912933353),
                    (58.35900993751445,  26.751451492309574),
                    (58.360337835698445, 26.751902103424076),
                    (58.36097926015119,  26.75168752670288),
                    (58.36248712436589,  26.7497992515564),
                    (58.36366861475146,  26.748619079589847),
                    (58.364827562136334, 26.74872636795044),
                    (58.3659752201055,   26.75001382827759),
                    (58.36697657745643,  26.750378608703613),
                    (58.368247922822114, 26.749413013458252),
                    (58.36886670266924,  26.74803972244263),
                    (58.36998047905436,  26.744391918182377),
                    (58.37151045798987,  26.741452217102054),
                    (58.37556008204928,  26.738491058349613),
                    (58.377168553744575, 26.737632751464847),
                    (58.3777421867492,   26.73758983612061),
                     ],
    }
    
    city_district = {
        "Aradas": "Aveiro",
        "Kanaleneiland": "Utrecht",
        "Annelinn": "Tartu",
    }
    
    # Paths initialization
    paths, system_management = paths_initialization(study_area)
    
    # Archetype documentation initialization
    pop_archetypes, stats = Archetype_documentation_initialization(paths)
    
    # Geodata initialization
    agent_populations, networks_map = Geodata_initialization(study_area, paths, pop_archetypes, special_areas_coords, city_district)
    
    # Synthetic population initialization
    agent_populations = Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats, paths, study_area, special_areas_coords, city_district)
    
    print('#'*20, ' Initialization finalized ','#'*20)
    
    # Return generated or loaded data
    return paths, system_management, pop_archetypes, agent_populations, networks_map
    
if __name__ == '__main__':
    
    # Input
    population = 10
    study_area = 'Kanaleneiland'
    
    Documents_initialisation(population, study_area)
