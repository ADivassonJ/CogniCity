def responsability_matrix_creation(todolist_family, SG_relationship_unique):   
    # Creamos la matriz de los resultados
    responsability_matrix = pd.DataFrame()
    # DataFrame con todo_type > 0 (dependientes con trips que requieren asistencia)
    dependents = todolist_family[todolist_family["todo_type"] > 0].add_suffix('_d')
    # DataFrame con todo_type == 0 (independientes)
    helpers = todolist_family[todolist_family["todo_type"] == 0].add_suffix('_h')
    # Eliminamos los independientes pero no capaces de ayudar (aquellos que en WoS son dependientes)
    agents_with_wos = todolist_family[(todolist_family['todo'] == 'WoS') & (todolist_family['todo_type'] != 0)]['agent'].unique()
    helpers = helpers[~helpers['agent_h'].isin(agents_with_wos)].reset_index(drop=True)
    # Producto cartesiano (todas las combinaciones posibles)
    df_combinado = helpers.merge(dependents, how='cross')
    # Calculamos todas las convinaciones
    for idx_df_conb, row_df_conb in df_combinado.iterrows():
        # Si la entrada es 0, sera la actividad de Home_out, por lo que lo ignoramos, no se plantean actividades previas a esta
        if row_df_conb['in_h'] == 0:
            continue
        # Sacamos las latitudes y longitudes de las posiciones de helper y dependant
        lat_h, lon_h = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_h'], ['lat', 'lon']].values[0]
        lat_d, lon_d = SG_relationship_unique.loc[SG_relationship_unique['osm_id'] == row_df_conb['osm_id_d'], ['lat', 'lon']].values[0]
        ## Sacamos los valores de las distancias
        # Distancia geografica
        geo_dist = haversine(lat_h, lon_h, lat_d, lon_d)
        # Distancia temporal
        time_dist = row_df_conb['in_d'] - row_df_conb['in_h'] # si esto da negativo, el agente helper tiene más tiempo para gestionar al dependant
        # Distancia social
        soc_dist = 1 # Aqui habrá que poner algo estadistico o algo
        # Calculamos la puntuación
        score = geo_dist + abs(time_dist)/100 + soc_dist # quizas cada uno entre el maximo?
        
        try: 
            h_schedule = todolist_family[(todolist_family['agent'] == row_df_conb['agent_h']) & (todolist_family['out'] <= row_df_conb['in_h'])] 
            
            print('h_schedule')
            print(h_schedule)
            print('todolist_family')
            print(todolist_family)
                         
            h_pre_step = h_schedule[h_schedule['out'] == max(h_schedule['out'])]
        except Exception:
            input('sa petao')
        
        d_schedule = todolist_family[(todolist_family['agent'] == row_df_conb['agent_d']) & (todolist_family['out'] <= row_df_conb['in_d'])]              
        d_pre_step = d_schedule[d_schedule['out'] == max(d_schedule['out'])]
        
        new_row = {
            'helper': row_df_conb['agent_h'],
            'dependent': row_df_conb['agent_d'],
            'osm_id_h0': h_pre_step['osm_id'].iloc[0],
            'osm_id_h1': row_df_conb['osm_id_h'],
            'osm_id_d0': d_pre_step['osm_id'].iloc[0],
            'osm_id_d1': row_df_conb['osm_id_d'],
            'geo_dist': geo_dist,
            'time_dist': time_dist,
            'soc_dist': soc_dist,
            'score': score,
            'out_h': h_pre_step['out'].iloc[0],
            'in_h': row_df_conb['in_h'],
            'out_d': d_pre_step['out'].iloc[0],
            'in_d': row_df_conb['in_d']
        }

        responsability_matrix = pd.concat([responsability_matrix, pd.DataFrame([new_row])], ignore_index=True)

    if not responsability_matrix.empty:
        responsability_matrix = responsability_matrix.loc[
            responsability_matrix.groupby(['dependent', 'osm_id_d1'])['score'].idxmin()
        ].reset_index(drop=True)
        # Agrupar por 'out_h' y sumar los valores de 'score'
        grouped = responsability_matrix.groupby('out_h')['score'].sum()
        # Encontrar el grupo con menor suma
        min_group = grouped.idxmin()
        # Filtrar el DataFrame original para quedarte solo con ese grupo
        responsability_matrix = responsability_matrix[responsability_matrix['out_h'] == min_group]
    else:
        print(f'family XX has no responsables.')
    
    ## Nos deshacemos de los casos de un helper ayuda al mismo dependant para multiples tareas (evitamos futuros conflictos)
    # Agrupamos por las columnas 'helper' y 'dependent' y obtenemos el índice del menor 'score'
    idx_min_scores = responsability_matrix.groupby(['helper', 'dependent'])['score'].idxmin()
    # Seleccionamos solo esas filas
    responsability_matrix = responsability_matrix.loc[idx_min_scores].reset_index(drop=True)
    
    return responsability_matrix
