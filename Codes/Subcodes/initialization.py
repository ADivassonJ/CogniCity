import os
import sys
import shutil
import random
import osmnx as ox
ox.settings.timeout = 500
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

############# Due to py 3.7. some things are 'rusty'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
#############


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
    try:
        print(f'Loading archetypes data ...')
        archetypes_folder = Path(paths['archetypes'])
        files = [f for f in archetypes_folder.iterdir() if f.is_file() and f.name.startswith('pop_archetypes_') and f.suffix == '.xlsx']

        pop_archetypes = {
            f.stem.replace('pop_archetypes_', ''): load_filter_sort_reset(f)
            for f in files
        }
    except Exception as e:
        raise FileNotFoundError(f"[ERROR] Archetype files are not found in {paths['archetypes']}. Please fix and restart.") from e
    # Validar y cargar stats_synpop
    stats_synpop = load_or_create_stats(
        paths['archetypes'],
        'stats_synpop.xlsx',
        create_stats_synpop,
        [pop_archetypes.get('citizens'), pop_archetypes.get('families')]
    )

    # Validar y cargar stats_trans
    stats_trans = load_or_create_stats(
        paths['archetypes'],
        'stats_trans.xlsx',
        create_stats_trans,
        [pop_archetypes.get('transport'), pop_archetypes.get('families')]
    )

    return pop_archetypes, stats_synpop, stats_trans

def load_or_create_stats(archetypes_path, filename, creation_func, creation_args):
    filepath = archetypes_path / filename
    try:
        print(f'Loading statistical data from {filename} ...')
        df_stats = pd.read_excel(filepath)
        if df_stats.isnull().sum().sum() != 0:
            raise ValueError(f"{filename} has missing values.")
    except Exception:
        # Si falla, crear archivo vacío y lanzar error para que el usuario lo rellene
        creation_func(archetypes_path, *creation_args)
        raise ValueError(f"[ERROR] {filepath} is missing or incomplete. Please fill μ, σ, max, min values and rerun.")

    return df_stats
        
def Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats_synpop, paths, SG_relationship, study_area, stats_trans):
    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    
    try:
        print(f"Loading synthetic population data ...") 
        
        for type_population in system_management['archetypes'].dropna():
            agent_populations[type_population] = pd.read_excel(f"{paths['population']}/pop_{type_population}.xlsx")
            
    except Exception as e:     
        print(f'    [WARNING] Data is missing.') 
        print(f'        Creating synthetic population (it might take a while) ...')

        ## Synthetic population generation
        # Section added just in case in the future we want to optimize the error of the synthetic population
        archetype_to_analyze = pop_archetypes['citizen']
        archetype_to_fill = pop_archetypes['family']
        
        # Citizen_inventory_creation
        agent_populations['distribution'], total_presence = Citizen_inventory_creation(archetype_to_analyze, population)
        # Citizen_distribution_in_families
        agent_populations['distribution'], agent_populations['citizen'], agent_populations['family'] = Citizen_distribution_in_families(archetype_to_fill, agent_populations['distribution'], total_presence, stats_synpop, pop_archetypes)
        # Utilities_assignment
        agent_populations['family'], agent_populations['citizen'], agent_populations['transport'] = Utilities_assignment(agent_populations['citizen'], agent_populations['family'], pop_archetypes, paths, SG_relationship, stats_synpop, stats_trans)

        print(f"        Saving data ...")

        for type_population in system_management['archetypes'].dropna():
            agent_populations[type_population].to_excel(f"{paths['population']}/pop_{type_population}.xlsx", index=False)
            
    return agent_populations

def Geodata_initialization(study_area, paths, pop_archetypes):
    agent_populations = {}
    # Obtener redes activas desde transport_archetypes
    networks = get_active_networks(pop_archetypes['transport'])

    # Paso 1: Cargar o descargar POIs
    agent_populations['building'] = load_or_download_pois(study_area, paths[study_area], paths['population'], pop_archetypes['building'])

    # Paso 2: Cargar o descargar redes
    networks_map = load_or_download_networks(study_area, paths['maps'], networks)

    # TO_DO: Añadir datos de buses eléctricos aquí

    return agent_populations, networks_map

def get_active_networks(transport_archetypes_df):
    active_maps = transport_archetypes_df.loc[
        transport_archetypes_df['state'] == 'active', 'map'
    ].dropna().unique().tolist()
    
    # Asegurar que 'walk' esté incluido
    if 'walk' not in active_maps:
        active_maps.append('walk')
    
    return active_maps

def load_or_download_pois(study_area, study_area_path, pop_path, building_archetypes_df):
    try:
        print(f'Loading POIs data ...')
        return pd.read_excel(f'{pop_path}/pop_building.xlsx')
    except Exception:
        print(f'    [WARNING] Data is missing, it needs to be downloaded.') 
        return download_pois(study_area, study_area_path, pop_path, building_archetypes_df)

def download_pois(study_area, study_area_path, pop_path, building_archetypes_df):
    try:
        print(f'        Downloading services data from {study_area} ...')
        Services_Group_relationship = pd.read_excel(f'{study_area_path}/Services-Group relationship.xlsx')
    except Exception:
        print(f"    [ERROR] File 'Services-Group relationship.xlsx' is not found in the data folder ({study_area_path}).")
        raise FileNotFoundError("Required file missing.")

    print(f'        Processing data (it might take a while)...')
    
    # Sacamos las columnas de arquetipos de servicios actualmente activos
    to_keep = building_archetypes_df['name'].unique()

    services_groups = services_groups_creation(Services_Group_relationship, to_keep)
    
    all_osm_data = []

    # Paralelization
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(get_osm_elements, study_area, group_ref): group_name
            for group_name, group_ref in services_groups.items()
        }
        for future in as_completed(futures):
            group_name = futures[future]
            try:
                df_group = future.result()
                df_group['archetype'] = group_name.replace('_list', '')
                all_osm_data.append(df_group)
            except Exception as e:
                print(f"    [ERROR] Failed to get data for {group_name}: {e}")
    osm_elements_df = pd.concat(all_osm_data, ignore_index=True)
    
    SG_relationship = building_schedule_adding(osm_elements_df, building_archetypes_df)
    
    SG_relationship.to_excel(f'{pop_path}/pop_building.xlsx', index=False)

    return SG_relationship

def building_schedule_adding(osm_elements_df, building_archetypes_df):
    '''
    Esta funcion esta para sumar las caracteristicas a la poblacion de los building
    '''
    list_building_variables = [col.rsplit('_', 1)[0] for col in building_archetypes_df.columns if col.endswith('_mu')]
    
    for idx, row_oedf in osm_elements_df.iterrows():
        list_building_values = get_vehicle_stats(row_oedf['archetype'], building_archetypes_df, list_building_variables)
        
        if list_building_values == {}:
            input(f"row_oedf['archetype']: {row_oedf['archetype']}")
            continue
        
        list_building_values['Service_opening'] = list_building_values['WoS_opening'] + list_building_values['Service_opening']
        list_building_values['Service_closing'] = list_building_values['WoS_closing'] + list_building_values['Service_closing']
        
        osm_elements_df = assign_data(list_building_values, list_building_values, osm_elements_df, idx)

    return osm_elements_df

def load_or_download_networks(study_area, study_area_path, networks):
    networks_map = {}
    missing_networks = []
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
                graph = ox.graph_from_place(study_area, network_type=net_type)
                ox.save_graphml(graph, study_area_path / f"{net_type}.graphml")
                networks_map[f"{net_type}_map"] = graph
            except Exception as e:
                print(f'        [ERROR] Failed to download {net_type} network: {e}')
                raise e

    return networks_map

def Citizen_distribution_in_families(archetype_to_fill, df_distribution, total_presence, stats_synpop, pop_archetypes, ind_arch = 'f_arch_0'):
    """
    Summary:
       Creates df for all families and citizens population, where their characteristics are described
    Args:
       archetype_to_fill (DataFrame): df with the archetypes data of the archetypes that can be applied
       df_distribution (DataFrame): df with info of citizens archetypes and each's population
       total_presence (int): Citizens total presence from archetype data
       stats_synpop (DataFrame): df with all archetypes' statistical values
       ind_arch (str, optional): Archetype name on which individuals (families with just one citizen) 
      exists. If individial Homes are an archetype different from 'f_arch_0', especify here under variable 'ind_arch'.
    Returns:
       df_distribution (DataFrame): All citizens that were not able to add to a family
       df_citizens (DataFrame): df with all citizens of the system and theis main characteristics
       df_families (DataFrame): df with all families of the system and theis main characteristics
    """
    # Creates some dfs for post-prior use
    df_families = pd.DataFrame(columns=['name', 'archetype', 'description', 'members'])
    df_citizens = pd.DataFrame(columns=['name', 'archetype', 'description'])
    # df for citizens while they are on the creation loop (sometimes, it creates some agents on a family, before
    # acknoledgin it is not posible to create that famili because one specific characteristic. In those cases
    # this df is cleaned in the same loop, disenabeling the merge with the real df_citizens)
    df_part_citizens = pd.DataFrame(columns=df_citizens.columns)
    # df for statistical correction while creating the families
    df_stats_families = pd.DataFrame({
        'archetype': archetype_to_fill['name'],
        'presence': archetype_to_fill['presence'],
        'percentage': archetype_to_fill['presence'] / archetype_to_fill['presence'].sum()*100,
        'stat_presence': 0,
        'stat_percentage': 0,
        'error': 0
    })
    
    while 1==1:
        
        # flag for jumping scenarios where we dont want any data to be saved
        flag = False
        # Evaluation of the families that can be created
        archetype_to_fill = is_it_any_archetype(archetype_to_fill, df_distribution, ind_arch)
        # If no families can be created, leaves loop
        if archetype_to_fill.empty:
            break
        # df_stats_families adapts to archetype_to_fill's available archetypes
        df_stats_families = df_stats_families[df_stats_families['archetype'].isin(archetype_to_fill['name'])]
        # The actual presence of each archetype in df_families is evaluated and saved in archetype_counts
        archetype_counts = df_families['archetype'].value_counts()
        # df_stats_families is updated
        df_stats_families = df_stats_families.copy()
        df_stats_families.loc[:, 'stat_presence'] = df_stats_families['archetype'].map(archetype_counts).fillna(0).astype(int)
        df_stats_families.loc[:, 'stat_percentage'] = (df_stats_families['stat_presence'] / df_stats_families['stat_presence'].sum()) * 100
        df_stats_families.loc[:, 'error'] = df_stats_families.apply(
            lambda row: (row['stat_percentage'] - row['percentage']) / row['percentage'] if row['percentage'] != 0 else 0,
            axis=1
        )
        # If is the very first time it begins with the archetype with most presence in their archetype data df
        if df_stats_families['stat_presence'].sum() == 0:
            arch_to_fill = df_stats_families.loc[df_stats_families['presence'].idxmax(), 'archetype']
        # If is NOT the first time, it continues with the archetype currently worstly represented in the sample
        else:
            arch_to_fill = df_stats_families.loc[df_stats_families['error'].idxmin(), 'archetype']
        # We obtain a df with the actual number of citizens to be distributed in households and the number of 
        # citizens of the chosen household that, the archetype of the household chosen in this iteration, 
        # consumes from the available sample.
        merged_df = process_arch_to_fill(archetype_to_fill, arch_to_fill, df_distribution)
        
        # For the case of the individual family, the usage of stats_synpop is a bit different than in the rest
        if arch_to_fill == ind_arch:
            # merged_result is created to work with this new paradigma
            merged_result = (
                merged_df[merged_df['participants'].str.contains(r'\*', na=False)]
                .merge(
                    stats_synpop[stats_synpop['item_1'] == ind_arch],
                    left_on='name',
                    right_on='item_2',
                    how='inner'
                ).dropna(subset=['population'])  # Elimina NaN
                .query("population != 0")  # Elimina ceros
            )
            # If merge_results is empty, no variable of this individual family can be created, so the option is deleted
            # from archetype_to_fill, so nexts steps of the loop will no consider this family archetype.
            if merged_result.empty:
                archetype_to_fill = archetype_to_fill[archetype_to_fill['name'] != arch_to_fill].reset_index(drop=True)
                continue
            # Basically, in individual families mu value is equal to presence values in other dfs so it creates de 
            # probability values of each type of citizen to be the one composing this family
            merged_result.loc[:, 'probability'] = merged_result['mu'] / merged_result['mu'].sum()
            # Then, and using the previously calculated values, an stochastic value is selected
            random_choice = np.random.choice(merged_result['name'], p=merged_result['probability'])
            # The non selected archetypes are added as 0 while the selected one as 1
            merged_df.loc[:, 'participants'] = np.where(merged_df['name'] == random_choice, 1, 0)
            # The data for the new citizen is created   
            citizen_description = pop_archetypes['citizen'].loc[pop_archetypes['citizen']['name'] == random_choice, 'description'].values[0]
            new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': random_choice, 'description': citizen_description}
            df_part_citizens.loc[len(df_part_citizens)] = new_row
            
        # For the case of the colective families, the usage of stats_synpop is conventional
        else:
            # merged_df is analyzed row by row
            for idx in merged_df.index:
                row = merged_df.loc[idx]
                # If that row's value for participans is NaN (we dont have any archetypes of that type to distribute) we jump to next loop's act
                if pd.notna(row['participants']):
                    ## Statistic value application
                    value = row['participants']
                    # If value is '*' then 'stats_synpop' is consulted to get statistical values
                    stats_value = get_stats_value(value, stats_synpop, arch_to_fill, row['name'])                 
                    merged_df.at[idx, 'participants'] = stats_value
                    ## Fitness evaluation    
                    # If the presented famly suits on the actual population availability ...                    
                    if merged_df.at[idx, 'participants'] <= row['population']:
                        # Citizens are created
                        for _ in range(int(merged_df.at[idx, 'participants'])):
                            citizen_description = pop_archetypes['citizen'].loc[pop_archetypes['citizen']['name'] == row['name'], 'description'].values[0]
                            new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': row['name'], 'description': citizen_description}
                            df_part_citizens.loc[len(df_part_citizens)] = new_row
                    # If the presented famly DOES NOT suit on the actual population availability ...     
                    else:                                      
                        # Evaluate issue
                        fila = archetype_to_fill.loc[archetype_to_fill["name"] == arch_to_fill]
                        # Evaluate if there is any '*'
                        if not (fila.isin(['*']).any().any()):
                            # If it's not, then no scenario exist on which that family may fit into the current population, so
                            # this family option is deleted from the archetype_to_fill
                            archetype_to_fill = archetype_to_fill[archetype_to_fill["name"] != arch_to_fill].reset_index(drop=True)
                            # Also, the family, did not fit at the end, so 'df_part_citizens' is also deleted
                            df_part_citizens = df_part_citizens.drop(df_part_citizens.index)
                            # This flags assures that no calculation will be done after 'break' and the loop will go on
                            flag = True
                            break
                        else:
                            # If the issue comes from a '*' then we valuate if it may exist any statistical scenario where that family fits
                            # All columns with '*' are saved
                            cols_with_star = [col for col in fila.columns if fila.iloc[0][col] == '*']
                            # Issue family archetype name is saved
                            name_value = fila.iloc[0]['name']
                            # stats_synpop gets filtered so we only work this the needed data
                            filtered_df = stats_synpop[(stats_synpop['item_1'] == name_value) & (stats_synpop['item_2'].isin(cols_with_star))][['item_2', 'min']]
                            # All values with '*' of this family are evaluated
                            for idy in filtered_df.index:
                                # If the issue comes from a '*' which assigned value is higher than 'min' for that value, AND that min suits into the population
                                # the min value is assigned instead of the original one.
                                if filtered_df.at[idy, 'min'] <= row['population']:
                                    merged_df.loc[merged_df['name'] == filtered_df.at[idy, 'item_2'], 'participants'] = filtered_df.at[idy, 'min']
                                    for _ in range(int(merged_df.at[idx, 'participants'])):
                                        # Citizens are created
                                        citizen_description = pop_archetypes['citizen'].loc[pop_archetypes['citizen']['name'] == row['name'], 'description'].values[0]
                                        new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': row['name'], 'description': citizen_description}
                                        df_part_citizens.loc[len(df_part_citizens)] = new_row
                                # If the issue comes from a '*' which assigned value is higher than 'min' for that value, BUT that min DOES NOT suit into the 
                                # population, then there is no way that family can make it, so it is deleted from archetype_to_fill.
                                else:
                                    archetype_to_fill = archetype_to_fill[archetype_to_fill["name"] != arch_to_fill].reset_index(drop=True)
                                    # Also, the family, did not fit at the end, so 'df_part_citizens' is also deleted
                                    df_part_citizens = df_part_citizens.drop(df_part_citizens.index)
                                    # This flags assures that no calculation will be done after 'break' and the loop will go on
                                    flag = True
                                    break
        # This flags assures that no calculation will be done after previous missed tries and the loop will go on
        if flag:
            flag = False
            continue
        # Update remaining population for that archetype
        mask = df_distribution['population'].notna() & merged_df['participants'].notna()
        df_distribution.loc[mask, 'population'] = df_distribution.loc[mask, 'population'] - merged_df.loc[mask, 'participants']
        # Add new citizens to df_citizens
        df_citizens = pd.concat([df_citizens, df_part_citizens], ignore_index=True)
        # Create new family
        family_description = pop_archetypes['family'].loc[pop_archetypes['family']['name'] == arch_to_fill, 'description'].values[0]
        new_row_2 = {'name': f'family_{len(df_families)}', 'archetype': arch_to_fill, 'description': family_description, 'members': df_part_citizens['name'].tolist()}
        df_families.loc[len(df_families)] = new_row_2
        # df_part_citizens is done with its job, so it get reinitiated
        df_part_citizens = df_part_citizens.drop(df_part_citizens.index)
        # If no citizens are left to be distributed, breaks the loop
        if df_distribution['population'].sum() == 0:
            break

    return df_distribution, df_citizens, df_families  

def get_stats_value(value, stats_doc, item_1, item_2):
    if str(value).strip() == '*':
        row = stats_doc[
            (stats_doc['item_1'] == item_1) & 
            (stats_doc['item_2'] == item_2)
        ]
        if row.empty:
            raise ValueError(f"No stats found for ({item_1}, {item_2}) in the correspondent doc. n\ Most probably is because the archetypes have been updated but the stats doc has not been updated.") # Si pasa esto es porque se an actualizado los arquetipos pero no se a actualizado el doc de stats
        
        stats_value = computate_stats(row)

        return stats_value

    return int(value)

def computate_stats(row):
    mu = float(row['mu'])
    sigma = float(row['sigma'])
    min_val = int(row['min'])
    max_val = int(row['max'])

    stats_value = round(np.random.normal(mu, sigma))
    stats_value = max(min(stats_value, max_val), min_val)

    return stats_value

def get_wos_action(archetype, pop_archetypes, stats_synpop):
    value = pop_archetypes['citizen'].loc[
        pop_archetypes['citizen']['name'] == archetype, 'WoS_action'
    ].values[0]
    return get_stats_value(value, stats_synpop, archetype, 'WoS_action')

def assign_data(list_variables, list_values, df_pop, idx):
    for variable in list_variables:
        value = list_values[variable]
        if variable.endswith('_type') or variable.endswith('_amount'):
            value = int(round(value)) # ISSUE XXCC
        df_pop.at[idx, variable] = value
    return df_pop

def Utilities_assignment(df_citizens, df_families, pop_archetypes, paths, SG_relationship, stats_synpop, stats_trans):
    variables = [col.rsplit('_', 1)[0] for col in pop_archetypes['transport'].columns if col.endswith('_mu')]
    # Suponiendo que variables ya está definida en algún lado
    df_priv_vehicle = pd.DataFrame(columns=['name', 'archetype', 'family', 'ubication'] + variables)

    # Filtramos los valores posibles de 'osm_id' donde 'archetype' es 'Home'
    Home_ids = SG_relationship[SG_relationship['archetype'] == 'Home']['osm_id'].tolist()

    # Inicializamos una copia barajada de Home_ids
    shuffled_Home_ids = random.sample(Home_ids, len(Home_ids))
    counter = 0  # contador para avanzar en la lista barajada

    # Asignamos un valor de shuffled_Home_ids a cada fila en df_families
    for idx_df_f, row_df_f in df_families.iterrows():
        if counter >= len(shuffled_Home_ids):
            shuffled_Home_ids = random.sample(Home_ids, len(Home_ids))
            counter = 0

        Home_id = shuffled_Home_ids[counter]
        counter += 1
        
        df_families.at[idx_df_f, 'Home'] = Home_id
        df_families.at[idx_df_f, 'Home_type'] = SG_relationship.loc[
            SG_relationship['osm_id'] == Home_id, 'building_type'
        ].values[0]

        filtered_st_trans = stats_trans[stats_trans['item_1'] == row_df_f['archetype']]
        
        for _, row_fs in filtered_st_trans.iterrows():
            stats_value = computate_stats(row_fs)
            for new_vehicle in range(stats_value):
                stats_variables = get_vehicle_stats(row_fs['item_2'], pop_archetypes['transport'], variables)
                
                new_vehicle_row = {
                    'name': f'priv_vehicle_{len(df_priv_vehicle)}',
                    'archetype': row_fs['item_2'],
                    'family': row_df_f['name'],
                    'ubication': Home_id} #CUIDADO CON ESTO; ES UNA SIMPLIFICACION; DEBERIA SER UN PARKING SPOT
                new_vehicle_row.update(stats_variables)
                df_priv_vehicle.loc[len(df_priv_vehicle)] = new_vehicle_row        
    
    # Asignar familia a cada ciudadano
    df_citizens['family'] = df_citizens['name'].apply(lambda name: find_group(name, df_families, 'name'))
    
    # Asignar hogar a cada ciudadano
    df_citizens['Home'] = df_citizens['name'].apply(lambda name: find_group(name, df_families, 'Home'))
    
    work_ids = SG_relationship[SG_relationship['archetype'] == 'work']['osm_id'].tolist()  
    study_ids = SG_relationship[SG_relationship['archetype'] == 'study']['osm_id'].tolist()  
    
    for idx in range(len(df_citizens)):
        
        list_citizen_variables = [col.rsplit('_', 1)[0] for col in pop_archetypes['citizen'].columns if col.endswith('_mu')]
        list_citizen_values = get_vehicle_stats(df_citizens['archetype'][idx], pop_archetypes['citizen'], list_citizen_variables)
        
        df_citizens = assign_data(list_citizen_variables, list_citizen_values, df_citizens, idx)

        if df_citizens['WoS_action_type'][idx] == 1:
            WoS_id = random.choice(work_ids)
        else:
            WoS_id = random.choice(study_ids)
        df_citizens.at[idx, 'WoS'] = WoS_id
        df_citizens.at[idx, 'WoS_subgroup'] = SG_relationship.loc[
            SG_relationship['osm_id'] == WoS_id, 'building_type'
        ].values[0]  # Esto obtiene el nombre correspondiente

    return df_families, df_citizens, df_priv_vehicle

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

def find_group(name, df_families, row_out):
    for idx, row in df_families.iterrows():
        if name in row['members']:
            return row[row_out]
    return None

def random_name_surname(ref_dict):
    if not ref_dict:
        return None
    
    name = random.choice(list(ref_dict.keys()))
    if not ref_dict[name]:
        return None

    surname = random.choice(ref_dict[name])
    return f"{name}_{surname}"

def services_groups_creation(df, to_keep):
    # Primero quitamos las filas marcadas como 'not considered'
    df = df[df['not considered'] != 'x'].reset_index(drop=True)

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

def get_osm_elements(area_name, poss_ref):
    """
    Obtiene y filtra los elementos de OSM según un conjunto fijo de etiquetas y prioridades,
    y devuelve un DataFrame con la estructura personalizada.

    Parámetros:
    - area_name (str): Nombre del área para buscar datos.
    - poss_ref (dict): Reglas de prioridad para las etiquetas.

    Retorna:
    - DataFrame: Datos filtrados con las claves `osm_id`, `geometry`, `lat`, `lon`.
    """
    # Conjunto fijo de etiquetas a consultar
    tags = {key: True for key in poss_ref.keys()}
    
    # Descargar datos de la zona especificada
    gdf = ox.geometries_from_place(area_name, tags)
    
    # Convertir a DataFrame
    data = gdf.reset_index()

    # Crear la estructura personalizada
    filtered_data = []

    for _, row in data.iterrows():
        for key, values in poss_ref.items():
            if key in row and pd.notna(row[key]):
                if isinstance(values, list):  # Si hay una lista de valores aceptados
                    if row[key] in values:
                        break
                elif row[key] == values:  # Si el valor aceptado es único
                    break
        else:
            # Si ninguna clave cumple, continuar con la siguiente fila
            continue
        building_type_name = building_type(row, poss_ref)
        osmid_reformed = osmid_reform(row)
        
        # Añadir los datos con la estructura personalizada
        filtered_data.append({
            'building_type': building_type_name,
            'osm_id': osmid_reformed,
            'geometry': row['geometry'],
            'lat': row['geometry'].centroid.y if row['geometry'] and not row['geometry'].is_empty else None,
            'lon': row['geometry'].centroid.x if row['geometry'] and not row['geometry'].is_empty else None,
        })

    # Convertir a DataFrame final
    return pd.DataFrame(filtered_data)

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

def obtener_geometrias(city_name, pos_ref):
    data = get_osm_elements(city_name, pos_ref)
    return data

def obtener_dataframe_direcciones(city, pos_ref): 
    """Obtiene un DataFrame con las coordenadas de cada dirección y asigna un 'Territorio' (distrito)."""
    print('Procesando datos...')

    refug_data = obtener_geometrias(city, pos_ref)
    df_refug_data = pd.DataFrame(refug_data)
    gdf_refug_data = gpd.GeoDataFrame(df_refug_data, geometry='geometry')
    gdf_refug_data.set_crs("EPSG:4326", inplace=True)
    
    return gdf_refug_data

def main():
    # Input
    population = 450
    study_area = 'Abando'
    
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
    
    print('#'*20, ' System initialization ','#'*20)
    # Archetype documentation initialization
    pop_archetypes, stats_synpop, stats_trans = Archetype_documentation_initialization(paths)
    # Geodata initialization
    agent_populations, networks_map = Geodata_initialization(study_area, paths, pop_archetypes)
    # Synthetic population initialization
    agent_populations = Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats_synpop, paths, agent_populations['building'], study_area, stats_trans)
    print('#'*20, ' Initialization finalized ','#'*20)
    
if __name__ == '__main__':
    main()