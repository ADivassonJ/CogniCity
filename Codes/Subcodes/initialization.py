import os
import sys
import random
import osmnx as ox
ox.settings.timeout = 500
import numpy as np
import pandas as pd
import geopandas as gpd

def Archetype_documentation_initialization(main_path, archetypes_path):
    """
    Summary:
       Loads all the df related to archetype description. Also creates/loads (depending on the state of the files) the df with
      all archetypes' statistical values.
    Args:
       main_path (Path): Main path where the code and all data is saved
       archetypes_path (Path): Path to archetypes
    Returns:
       citizen_archetypes (DataFrame): df with all citizens' archetype data
       family_archetypes (DataFrame): df with all families' archetype data
       s_archetypes (DataFrame): ### WORK IN PROGRESS ###
       stats_synpop (DataFrame): df with all archetypes' statistical values
    """
    # Read archetypes data from files
    try:
        print(f'Loading archetypes data ...')
        citizen_archetypes = load_filter_sort_reset(archetypes_path / 'citizen_archetypes.xlsx')
        family_archetypes = load_filter_sort_reset(archetypes_path / 'family_archetypes.xlsx')
        s_archetypes = load_filter_sort_reset(archetypes_path / 's_archetypes.xlsx')
    except Exception as e:
        print(f"    [ERROR] File regarding archetypes are not found in the intended folder ({archetypes_path}).")
        print(f"    Please fix the problem and restart the program.")
        sys.exit()
    ## Evaluate stats_synpop (file where statistics values of some characteristis are defined)
    try:
        # If the file exists, verify that all needed data is there
        print(f'Loading statistical data ...')
        stats_synpop = pd.read_excel(archetypes_path / 'stats_synpop.xlsx')
        if stats_synpop.isnull().sum().sum() != 0:
            # If any data is missing, ask the user to fill it
            print(f'    [ERROR] {archetypes_path}/stats_synpop has one or more values empty.')
            print(f'    Please include all μ, σ, max and min for each detected scenario and run the code again.')
            sys.exit()
        # If everyhting is OK, go on
        return citizen_archetypes, family_archetypes, s_archetypes, stats_synpop
    except Exception:
        # If the file DOES NOT exist, create the file and ask the user to fill the missing data
        create_stats_synpop(archetypes_path, citizen_archetypes, family_archetypes)
        print(f'    [ERROR] {archetypes_path}/stats_synpop has no information.')
        print(f'    Please include all μ, σ, max and min for each detected scenario and run the code again.')
        sys.exit()

def Synthetic_population_initialization(citizen_archetypes, family_archetypes, population, stats_synpop, data_path, services_groups, study_area):
    study_area_path = data_path / study_area
    try:
        print(f"Loading synthetic population data ...") 
        df_distribution = pd.read_excel(f'{study_area_path}/df_distribution.xlsx')
        df_families = pd.read_excel(f'{study_area_path}/df_families.xlsx')
        df_citizens = pd.read_excel(f'{study_area_path}/df_citizens.xlsx')     
    except Exception as e:     
        print(f'    [WARNING] Data is missing.') 
        print(f'        Creating synthetic population (it might take a while) ...')
        ## Synthetic population generation
        # Section added just in case in the future we want to optimize the error of the synthetic population
        archetype_to_analyze = citizen_archetypes
        archetype_to_fill = family_archetypes
        
        # Citizen_inventory_creation
        df_distribution, total_presence = Citizen_inventory_creation(archetype_to_analyze, population)
        # Citizen_distribution_in_families
        df_distribution, df_citizens, df_families = Citizen_distribution_in_families(archetype_to_fill, df_distribution, total_presence, stats_synpop, citizen_archetypes, family_archetypes)
        # Utilities_assignment
        df_families, df_citizens = Utilities_assignment(df_citizens, df_families, citizen_archetypes, family_archetypes, data_path, services_groups, stats_synpop)

        print(f"        Saving data ...")
        df_distribution.to_excel(f'{study_area_path}/df_distribution.xlsx', index=False)
        df_families.to_excel(f'{study_area_path}/df_families.xlsx', index=False)
        df_citizens.to_excel(f'{study_area_path}/df_citizens.xlsx', index=False)

    return df_citizens, df_families

def Geodata_initialization(study_area, data_path):
    study_area_path = data_path / study_area
    os.makedirs(study_area_path, exist_ok=True)
    networks = ['all', 'bike', 'drive', 'drive_service', 'walk']
    try:
        print(f'Loading POIs data ...')
        osm_elements_df = pd.read_excel(f'{study_area_path}/SG_relationship.xlsx')
    except Exception as e:
        print(f'    [WARNING] Data is missing, it needs to be downloaded.') 
        try:
            print(f'        Downloading services data from {study_area} ...')
            print(f'        Saving data ...')
            SG_relationship = pd.read_excel(f'{data_path}/Services-Group relationship.xlsx')
        except Exception as e:
            print(f"    [ERROR] File 'Services-Group relationship.xlsx' is not found in the data folder ({data_path}).")
            print(f"    Please fix the problem and restart the program.")
            sys.exit()
        print(f'        Processing data ...')
        services_groups = services_groups_creation(SG_relationship)
        # Lista para acumular los resultados
        all_osm_data = []
        ##### Esto hay que paralelizarlooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
        for group_name, group_ref in services_groups.items():
            # group_ref es el diccionario que se pasa como poss_ref
            df_group = get_osm_elements(study_area, group_ref)
            # Añadir columna indicando a qué grupo pertenece
            df_group['service_group'] = group_name.replace('_list', '')
            all_osm_data.append(df_group)
        # Unir todos los DataFrames en uno solo
        osm_elements_df = pd.concat(all_osm_data, ignore_index=True)
        osm_elements_df.to_excel(f'{study_area_path}/SG_relationship.xlsx', index=False)
    
    networks_map = {}   
    try:
        print(f'Loading maps ...')
        for net_type in networks:           
            networks_map[net_type + "_map"] = ox.load_graphml(study_area_path / (net_type + '.graphml'))
    except Exception as e:
        print(f'    [WARNING] Data is missing, it needs to be downloaded.') 
        print(f'    Since the maps may have changed in part, all maps will be downloaded: {networks}')
        for net_type in networks: 
            print(f'        Downloading data for {net_type} network from {study_area} ...')
            try:
                graph = ox.graph_from_place(study_area, network_type=net_type)
                ox.save_graphml(graph, study_area_path / f"{net_type}.graphml")
                networks_map[f"{net_type}_map"] = graph
            except Exception as e:
                print(f'        [ERROR] Failed to download {net_type} network.')
                print(f'        {e}')
                sys.exit()
    
    ####CUIDADOO!!! HAY QUE AÑADIR LOS DATOS DE LOS BUSES ELECTRICOS A LOS QUE ASIGNAMOS LOS DISTINTOS POIs

    return osm_elements_df, networks_map



def Citizen_distribution_in_families(archetype_to_fill, df_distribution, total_presence, stats_synpop, citizen_archetypes, family_archetypes, ind_arch = 'f_arch_0'):
    """
    Summary:
       Creates df for all families and citizens population, where their characteristics are described
    Args:
       archetype_to_fill (DataFrame): df with the archetypes data of the archetypes that can be applied
       df_distribution (DataFrame): df with info of citizens archetypes and each's population
       total_presence (int): Citizens total presence from archetype data
       stats_synpop (DataFrame): df with all archetypes' statistical values
       ind_arch (str, optional): Archetype name on which individuals (families with just one citizen) 
      exists. If individial homes are an archetype different from 'f_arch_0', especify here under variable 'ind_arch'.
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
            citizen_description = citizen_archetypes.loc[citizen_archetypes['name'] == random_choice, 'description'].values[0]
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
                            citizen_description = citizen_archetypes.loc[citizen_archetypes['name'] == row['name'], 'description'].values[0]
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
                                        citizen_description = citizen_archetypes.loc[citizen_archetypes['name'] == row['name'], 'description'].values[0]
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
        family_description = family_archetypes.loc[family_archetypes['name'] == arch_to_fill, 'description'].values[0]
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
            raise ValueError(f"No stats found for ({item_1}, {item_2}) in stats_doc")
        mu = row.iloc[0]['mu']
        sigma = row.iloc[0]['sigma']
        min_val = row.iloc[0]['min']
        max_val = row.iloc[0]['max']

        stats_value = round(np.random.normal(mu, sigma))
        stats_value = max(min(stats_value, int(max_val)), int(min_val))

        return stats_value

    return int(value)



def Utilities_assignment(df_citizens, df_families, citizen_archetypes, family_archetypes, data_path, services_groups, stats_synpop):
    # Filtramos los valores posibles de 'osm_id' donde 'service_group' es 'home'
    home_ids = services_groups[services_groups['service_group'] == 'home']['osm_id'].tolist()

    # Inicializamos una copia barajada de home_ids
    shuffled_home_ids = random.sample(home_ids, len(home_ids))
    counter = 0  # contador para avanzar en la lista barajada

    # Asignamos un valor de shuffled_home_ids a cada fila en df_families
    for idx in range(len(df_families)):
        # Si hemos usado todos los home_ids, barajamos de nuevo
        if counter >= len(shuffled_home_ids):
            shuffled_home_ids = random.sample(home_ids, len(home_ids))
            counter = 0

        home_id = shuffled_home_ids[counter]
        df_families.at[idx, 'home'] = home_id
        df_families.at[idx, 'home_type'] = services_groups.loc[
            services_groups['osm_id'] == home_id, 'building_type'
        ].values[0]
        
        counter += 1  # avanzamos al siguiente
    
    # Asignar familia a cada ciudadano
    df_citizens['family'] = df_citizens['name'].apply(lambda name: find_group(name, df_families, 'name'))
    # Asignar hogar a cada ciudadano
    df_citizens['home'] = df_citizens['name'].apply(lambda name: find_group(name, df_families, 'home'))
    
    # Filtramos los valores posibles de 'name' donde 'group' es 'home'
    work_ids = services_groups[services_groups['service_group'] == 'work']['osm_id'].tolist()
    for idx in range(len(df_citizens)):
        work_id = random.choice(work_ids)
        df_citizens.at[idx, 'WoS'] = work_id
        df_citizens.at[idx, 'WoS_subgroup'] = services_groups.loc[
            services_groups['osm_id'] == work_id, 'building_type'
        ].values[0]  # Esto obtiene el nombre correspondiente
        value = citizen_archetypes.loc[citizen_archetypes['name'] == df_citizens.at[idx, 'archetype'], 'WoS_type'].values[0]
        stats_value = get_stats_value(value, stats_synpop, df_citizens.at[idx, 'archetype'], 'WoS_type')
        df_citizens.at[idx, 'WoS_type'] = stats_value

    return df_families, df_citizens

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

def services_groups_creation(df):
    # Primero quitamos las filas marcadas como 'not considered'
    df = df[df['not considered'] != 'x'].reset_index(drop=True)

    # Lista de columnas que definen los grupos
    groups = [
        "home", "study", "work", "entertainment", "healthcare",
        "public_transportation", "private_transportation", "charging_station"
    ]

    # Diccionarios de salida, uno por grupo
    group_dicts = {}

    # Para cada grupo, filtramos filas con 'x' y construimos el diccionario correspondiente
    for group in groups:
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

    