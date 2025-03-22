import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
pd.set_option('mode.chained_assignment', 'raise')  # Convierte el warning en error

### Functions in use
def add_matches_to_cond_archetypes(cond_archetypes, df, name_column='name'):
    """
    Summary:
       Search for any '*' in the DataFrame 'df' and add the row and column namen into 'cond_archetypes', so all statistical
      dependant situations are mapped on 'cond_archetypes'.
    Args:
       cond_archetypes (DataFrame): df with some archetypes' statistical values
       df (DataFrame): DataFrame to analyse
       name_column (str, optional): describes how the column on which names of the archetypes are saved. In case of 
      any difference with the standar value, add the new str here.
    Returns:
       cond_archetypes (DataFrame): updated df with more archetypes' statistical values
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
                # Añadir al DataFrame cond_archetypes
                cond_archetypes.loc[len(cond_archetypes)] = [name_value, col, None, None, None, None]
    return cond_archetypes

def citizen_archetypes_distribution(df, population):
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
        print("[Error] No available presences has been detected on citicens_archetype's file.")
        sys.exit()
    # Adds new row with population of each agent
    df['population'] = (df['presence_percentage'] * population).round().astype(int)

    return df[['name', 'population']].copy(), df['presence'].sum()

def create_cond_archetypes(archetypes_path, citizen_archetypes, family_archetypes): # Most probably, we should adapt this for getting all df needed, not just the specific two
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
    
    cond_archetypes = pd.DataFrame(columns=['item_1', 'item_2', 'mu', 'sigma', 'min', 'max'])
    cond_archetypes = add_matches_to_cond_archetypes(cond_archetypes, citizen_archetypes)
    cond_archetypes = add_matches_to_cond_archetypes(cond_archetypes, family_archetypes)
    cond_archetypes.to_excel(archetypes_path/'cond_archetypes.xlsx', index=False)

def families_creation(archetype_to_fill, df_distribution, total_presence, cond_archetypes, citizen_archetypes, family_archetypes, ind_arch = 'f_arch_0'):
    """
    Summary:
       Creates df for all families and citizens population, where their characteristics are described
    Args:
       archetype_to_fill (DataFrame): df with the archetypes data of the archetypes that can be applied
       df_distribution (DataFrame): df with info of citizens archetypes and each's population
       total_presence (int): Citizens total presence from archetype data
       cond_archetypes (DataFrame): df with all archetypes' statistical values
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
    
    print('Creating synthetic population... (it might take a while)')
    
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
        
        # For the case of the individual family, the usage of cond_archetypes is a bit different than in the rest
        if arch_to_fill == ind_arch:
            # merged_result is created to work with this new paradigma
            merged_result = (
                merged_df[merged_df['participants'].str.contains(r'\*', na=False)]
                .merge(
                    cond_archetypes[cond_archetypes['item_1'] == ind_arch],
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
            new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': random_choice, 'description': 'Cool guy'}
            df_part_citizens.loc[len(df_part_citizens)] = new_row
            
        # For the case of the colective families, the usage of cond_archetypes is conventional
        else:
            # merged_df is analyzed row by row
            for idx in merged_df.index:
                row = merged_df.loc[idx]
                # If that row's value for participans is NaN (we dont have any archetypes of that type to distribute) we jump to next loop's act
                if pd.notna(row['participants']):
                    ## Statistic value application
                    valor = row['participants']
                    # If value is '*' then 'cond_archetypes' is consulted to get statistical values
                    if isinstance(valor, str) and valor.strip() == '*':
                        row_2 = cond_archetypes[(cond_archetypes['item_1'] == arch_to_fill) & (cond_archetypes['item_2'] == row['name'])]
                        # Assing an statistical value (stochastically)
                        stat_value = round(np.random.normal(row_2['mu'], row_2['sigma'])[0])
                        # If assigned value is out of min and max assign min or max
                        if stat_value < int(row_2['min']):
                            stat_value = int(row_2['min'])
                        if stat_value > int(row_2['max']):
                            stat_value = int(row_2['max'])
                        # Sustitutes '*' by the value on 'stat_value'
                        merged_df.at[idx, 'participants'] = stat_value
                    else:
                        # If it is a number (just in case) in gets swaped to int if its possible
                        try:
                            merged_df.at[idx, 'participants'] = int(valor)
                        except ValueError:
                            pass
                    ## Fitness evaluation    
                    # If the presented famly suits on the actual population availability ...                    
                    if merged_df.at[idx, 'participants'] <= row['population']:
                        # Citizens are created
                        for _ in range(int(merged_df.at[idx, 'participants'])):
                            new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': row['name'], 'description': 'Cool guy'}
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
                            # cond_archetypes gets filtered so we only work this the needed data
                            filtered_df = cond_archetypes[(cond_archetypes['item_1'] == name_value) & (cond_archetypes['item_2'].isin(cols_with_star))][['item_2', 'min']]
                            # All values with '*' of this family are evaluated
                            for idy in filtered_df.index:
                                # If the issue comes from a '*' which assigned value is higher than 'min' for that value, AND that min suits into the population
                                # the min value is assigned instead of the original one.
                                if filtered_df.at[idy, 'min'] <= row['population']:
                                    merged_df.loc[merged_df['name'] == filtered_df.at[idy, 'item_2'], 'participants'] = filtered_df.at[idy, 'min']
                                    for _ in range(int(merged_df.at[idx, 'participants'])):
                                        # Citizens are created
                                        new_row = {'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}', 'archetype': row['name'], 'description': 'Cool guy'}
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
        new_row_2 = {'name': f'family_{len(df_families)}', 'archetype': arch_to_fill, 'description': 'Cool family', 'members': df_part_citizens['name'].tolist()}
        df_families.loc[len(df_families)] = new_row_2
        # df_part_citizens is done with its job, so it get reinitiated
        df_part_citizens = df_part_citizens.drop(df_part_citizens.index)
        # If no citizens are left to be distributed, breaks the loop
        if df_distribution['population'].sum() == 0:
            break
        
    print("    [DONE]")
    
    # Suponiendo que df_citizens y df_families ya están definidos
    df_final_stats_citizens = df_citizens['archetype'].value_counts().reset_index()
    df_final_stats_citizens.columns = ['archetype', 'count']

    # Para df_families
    df_final_stats_families = df_families['archetype'].value_counts().reset_index()
    df_final_stats_families.columns = ['archetype', 'count']

    # Usar directamente los valores de citizen_archetypes y family_archetypes
    df_final_stats_citizen_archetypes = citizen_archetypes[['name', 'presence']].copy()
    df_final_stats_citizen_archetypes.columns = ['name', 'count']

    df_final_stats_family_archetypes = family_archetypes[['name', 'presence']].copy()
    df_final_stats_family_archetypes.columns = ['name', 'count']

    # Combinar df_final_stats_families con df_final_stats_family_archetypes
    merged_families = df_final_stats_families.merge(df_final_stats_family_archetypes, left_on='archetype', right_on='name', how='outer', suffixes=('_families', '_family_archetypes')).drop(columns=['name'])
    merged_families.fillna(0, inplace=True)
    merged_families['rate_families'] = merged_families['count_families'] / merged_families['count_families'].sum()
    merged_families['rate_family_archetypes'] = merged_families['count_family_archetypes'] / merged_families['count_family_archetypes'].sum()
    merged_families['rate_difference'] = abs((merged_families['rate_families'] - merged_families['rate_family_archetypes'])/ merged_families['rate_family_archetypes']*100)

    # Combinar df_final_stats_citizens con df_final_stats_citizen_archetypes
    merged_citizens = df_final_stats_citizens.merge(df_final_stats_citizen_archetypes, left_on='archetype', right_on='name', how='outer', suffixes=('_citizens', '_citizen_archetypes')).drop(columns=['name'])
    merged_citizens.fillna(0, inplace=True)
    merged_citizens['rate_citizens'] = merged_citizens['count_citizens'] / merged_citizens['count_citizens'].sum()
    merged_citizens['rate_citizen_archetypes'] = merged_citizens['count_citizen_archetypes'] / merged_citizens['count_citizen_archetypes'].sum()
    merged_citizens['rate_difference'] = abs((merged_citizens['rate_citizens'] - merged_citizens['rate_citizen_archetypes'])/merged_citizens['rate_citizen_archetypes']*100)

    # Mostrar el promedio de rate_difference
    print("Abs error on citizens:", round(merged_citizens['rate_difference'].mean(), 4), "%")
    print("Abs error on families:", round(merged_families['rate_difference'].mean(), 4), "%")

    return df_distribution, df_citizens, df_families  

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

def load_archetype_data(main_path, archetypes_path):
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
       cond_archetypes (DataFrame): df with all archetypes' statistical values
    """
    
    # Read archetypes data from files
    citizen_archetypes = load_filter_sort_reset(archetypes_path / 'citizen_archetypes.xlsx')
    family_archetypes = load_filter_sort_reset(archetypes_path / 'family_archetypes.xlsx')
    s_archetypes = load_filter_sort_reset(archetypes_path / 's_archetypes.xlsx')
    
    ## Evaluate cond_archetypes (file where statistics values of some characteristis are defined)
    try:
        # If the file exists, verify that all needed data is there
        cond_archetypes = pd.read_excel(archetypes_path / 'cond_archetypes.xlsx')
        if cond_archetypes.isnull().sum().sum() != 0:
            # If any data is missing, ask the user to fill it
            print(f'{archetypes_path}/cond_archetypes has one or more values empty,')
            print('please include all μ, σ, max and min for each detected scenario and run the code again.')
            sys.exit()
        # If everyhting is OK, go on
        return citizen_archetypes, family_archetypes, s_archetypes, cond_archetypes
    except Exception:
        # If the file DOES NOT exist, create the file and ask the user to fill the missing data
        create_cond_archetypes(archetypes_path, citizen_archetypes, family_archetypes)
        print(f'{archetypes_path}/cond_archetypes has no information,')
        print('please include all μ, σ, max and min for each detected scenario and run the code again.')
        sys.exit()

def load_filter_sort_reset(filepath):
    """
    Summary: 
       Load an Excel file, filter the rows where 'stat' is 'inactive' and return the DataFrame.
    Args:
       filepath (Path): path to the file that wants to be readed
    Returns:
       df: Readed dfs
    """

    try:
        df = pd.read_excel(filepath)
        df = df[df['state'] != 'inactive']
        return df
    except Exception as e:
        print(f"{filepath.name} not found or error loading: {e}")
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

### Functions not in use
def random_arch(df):
    """
    Selecciona aleatoriamente un 'name' del DataFrame basado en la 
    distribución de probabilidades calculada a partir de la columna 'presence'.
    """
    if df is not None and not df.empty:
        presence_probabilities = df['presence'] / df['presence'].sum()
        random_name = np.random.choice(df['name'], p=presence_probabilities)
        return random_name
    else:
        print("Error en random_arch: DataFrame vacío o None.")
        return None

### Main
def main():
    # Input
    population = 45000
    study_area = 'Kanaleneiland'
    
    
    ## Code initialization
    # Paths initialization
    main_path = Path(__file__).resolve().parent.parent
    subcodes_path = main_path / 'Subcodes'
    archetypes_path = main_path / 'Archetypes'
    data_path = main_path / 'Data'
    results_path = main_path / 'Results'
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    # Data loading
    citizen_archetypes, family_archetypes, s_archetypes, cond_archetypes = load_archetype_data(main_path, archetypes_path)
    
    
    ## Synthetic population generation
    # Section added just in case in the future we want to optimize the error of the synthetic population
    archetype_to_analyze = citizen_archetypes
    archetype_to_fill = family_archetypes
    # Populations characterization
    df_distribution, total_presence = citizen_archetypes_distribution(archetype_to_analyze, population)
    # Creation of families and citizens df
    df_distribution, df_citizens, df_families = families_creation(archetype_to_fill, df_distribution, total_presence, cond_archetypes, citizen_archetypes, family_archetypes)
    
    
    
    
    print(f"Distribución final guradada en {results_path}")
    df_distribution.to_excel(f'{results_path}/df_distribution.xlsx', index=False)
    df_families.to_excel(f'{results_path}/df_families.xlsx', index=False)
    df_citizens.to_excel(f'{results_path}/df_citizens.xlsx', index=False)

if __name__ == '__main__':
    main()

