def agent_collection(prev_matrix2cover, matrix2cover, helper):
    # Inicializacion del df de los resultados
    new_new_list = pd.DataFrame()
    ## Creación de ruta de recogida
    # DataFrame con datos de outs
    out_osm_ids = pd.DataFrame(columns=['osm_id', 'out', 'conmu_time'])
    # Agrupamos para crear ruta de recogida
    osm_id_groups = prev_matrix2cover.groupby('osm_id')
    # Pasamos por todos los grupos de la salida
    for name_group, oi_group in osm_id_groups:
        # Buscamos el valor maximo de out en el grupo que tenga time2spend != 0 (quién condiciona)
        filtered = oi_group[oi_group['time2spend']!=0]
        # Asignamos tiempo de conmutación del grupo
        group_conmu_time = oi_group['conmu_time'].max()
        
        ### NO ME ACLARO CON ESTA SECCION IGUAL ES FALLO ###
        
        # Asignamos tiempo de salida del grupo
        if filtered.empty:
            filtered = matrix2cover[matrix2cover['fixed'] == True]
            if filtered.empty:
                group_out_time = oi_group['out'].min()
            else:
                group_out_time = filtered['in'].max() - group_conmu_time*len(filtered)
        else:
            group_out_time = filtered['out'].max()
            
        ###################
            
        # Añadir nueva fila de datos
        rew_row ={ 
            'osm_id': name_group,
            'out': group_out_time,
            'conmu_time': group_conmu_time
        }   
        # Suma a dataframe
        out_osm_ids = pd.concat([out_osm_ids, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='out', ascending=False).reset_index(drop=True)
    
    # Crear la ruta ordenada
    sorted_route = sort_route(out_osm_ids, helper)
    
    ## Crear el nuevo schedule (parte de recogida de agentes)
    # Iteramos todos los osm_id de salida
    for _, name_group in sorted_route.iterrows():
        # Sacamos el grupo relativo al trip actual
        group = osm_id_groups.get_group(name_group['osm_id'])
        # Sacamos los valores a asignar para este grupo
        group_out_time = name_group['out']
        group_conmu_time = name_group['conmu_time']
        # Miramos los agenets que ya estan en movimiento
        previous_agents = new_new_list['agent'].unique()
        # Iniciamos con los agentes en movimiento
        for p_agent in previous_agents:
            # Nueva fila
            rew_row ={
                'agent': p_agent,
                'todo': 'Collect', 
                'osm_id': name_group['osm_id'], 
                'todo_type': 0, 
                'opening': 0, 
                'closing': float('inf'), 
                'fixed': False, 
                'time2spend': 0, 
                'in': group_out_time, 
                'out': group_out_time,
                'conmu_time': group_conmu_time
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
        # Despues agentes que se mueven por primera vez
        for _, agent in group.iterrows():
            # En caso de que el agente tenga que estar un tiempo especifico, de espera o se haya cerrado el servicio en el que estan, tendrá que esperar (sin poder hacer nada más)
            if (group_out_time > agent['closing']) or (group_out_time > agent['out'] and agent['time2spend'] != 0):
                ## Calculamos el tiempo de espera
                waiting_time = group_out_time - min([agent['out'], agent['closing']])
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'todo': f'Waiting collection', 
                    'osm_id': agent['osm_id'],  # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related 
                    'fixed': agent['fixed'], 
                    'time2spend': waiting_time, 
                    'in': agent['out'], 
                    'out': group_out_time,
                    'conmu_time': group_conmu_time
                }   
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)

            # Actualización del caso original del agente
            rew_row ={
                'agent': agent['agent'],
                'todo': f"{agent['todo']}", 
                'osm_id': agent['osm_id'], 
                'todo_type': agent['todo_type'], 
                'opening': agent['opening'], 
                'closing': agent['closing'], 
                'fixed': agent['fixed'], 
                'time2spend': agent['time2spend'], 
                'in': agent['in'], 
                'out': group_out_time,
                'conmu_time': group_conmu_time
            }   
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
    return new_new_list

def agent_delivery(new_new_list, new_list, matrix2cover, helper):
    ## Creación de ruta de recogida
    # DataFrame con datos de ins
    in_osm_ids = pd.DataFrame(columns=['osm_id', 'in', 'conmu_time'])
    # Agrupamos para crear ruta de entrega
    osm_id_groups = matrix2cover.groupby('osm_id')
    # Pasamos por todos los grupos de la salida
    for name_group, oi_group in osm_id_groups:
        # Buscamos el valor minimo de in en el grupo que tenga fixed == True (quién condiciona)
        filtered = oi_group[oi_group['fixed'] == True]
        # Asignamos tiempo de conmutación del grupo
        group_conmu_time = oi_group['conmu_time'].max()
        # Asignamos tiempo de llegada del grupo
        if filtered.empty: # No tiene más condiciones, porque si es fixed tendra un time2spend seguro, no hace falta comprobar
            group_in_time = oi_group['in'].max()
        else:
            group_in_time = filtered['in'].min()
        # Añadir nueva fila de datos
        rew_row ={ 
            'osm_id': name_group,
            'in': group_in_time,
            'conmu_time': group_conmu_time
        }   
        # Suma a dataframe
        in_osm_ids = pd.concat([in_osm_ids, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=False).reset_index(drop=True)
    
    # Crear la ruta ordenada
    sorted_route = sort_route(in_osm_ids, helper)
    
    ## Crear el nuevo schedule (parte de recogida de agentes)
    # Iteramos todos los osm_id de salida
    for _, name_group in sorted_route.iterrows():
        # Sacamos el grupo relativo al trip actual
        group = osm_id_groups.get_group(name_group['osm_id'])
        # Sacamos los valores a asignar para este grupo
        group_in_time = name_group['in']
        group_conmu_time = name_group['conmu_time']
        # Miramos los agentes que ya estan en movimiento (si estan presentes en new_new_list, son de otro ciclo, porque new_new_list empieza limpio)
        previous_agents = new_new_list['agent'].unique()
        # Iniciamos con los agentes en movimiento
        for p_agent in previous_agents:
            # Nueva fila
            rew_row ={
                'agent': p_agent,
                'todo': 'Delivery', 
                'osm_id': name_group['osm_id'], 
                'todo_type': 0, 
                'opening': 0, 
                'closing': float('inf'), 
                'fixed': False, 
                'time2spend': 0, 
                'in': group_in_time, 
                'out': group_in_time,
                'conmu_time': group_conmu_time
            }
            # Suma a dataframe
            new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
        # Despues agentes que se mueven por primera vez
        for _, agent in group.iterrows():
            # En caso de que el agente tenga que estar un tiempo especifico de espera en el servicio en el que estan (sin poder hacer nada más)
            if (group_in_time < agent['opening']) or (group_in_time < agent['in'] and agent['fixed'] == True):
                # Calculamos el tiempo de espera
                waiting_time = max([agent['in'], agent['opening']]) - group_in_time # Tecnicamente [agent['in'], agent['opening']] deberian ser iguales si es fix, pero bue
                # Nueva fila
                rew_row ={
                    'agent': agent['agent'],
                    'todo': f'Waiting opening', 
                    'osm_id': agent['osm_id'],  # Issue 17
                    'todo_type': 0, 
                    'opening': 0,               # Es una accion not-place-related, pero sí time-related
                    'closing': float('inf'),    # Es una accion not-place-related, pero sí time-related
                    'fixed': agent['fixed'], 
                    'time2spend': waiting_time, 
                    'in': group_in_time, 
                    'out': agent['in'],
                    'conmu_time': group_conmu_time
                }   
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                
                ################ diferencias a partir de aqui
                
                new_out = (agent['in'] + agent['time2spend']) if agent['time2spend'] != 0 else min([agent['closing'], agent['out']])
                
                if new_out <= agent['closing']:
                    rew_row ={
                        'agent': agent['agent'],
                        'todo': f"{agent['todo']}", 
                        'osm_id': agent['osm_id'], 
                        'todo_type': 0, 
                        'opening': agent['opening'], 
                        'closing': agent['closing'], 
                        'fixed': agent['fixed'], 
                        'time2spend': agent['time2spend'], 
                        'in': agent['in'], 
                        'out': new_out,
                        'conmu_time': group_conmu_time
                    }   
                    # Suma a dataframe
                    new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                else:
                    print(f"Due to 'Accompany' {agent['agent']} was not able to fullfill '{agent['todo']}' at {agent['in']}.")
            else:
                # Actualización del caso original del agente
                new_out = (group_in_time + agent['time2spend']) if agent['time2spend'] != 0 else agent['closing']
                print(f"para {agent['agent']} se calcula group_in_time ({group_in_time}):")
                print(agent)
                
                if agent['todo'] in ['Collect', 'Delivery']:
                    new_out = group_in_time
                elif new_out > agent['closing'] and agent['fixed'] == False:
                    non_active_time = new_out - agent['closing'] 
                    new_out = agent['closing']
                    print(f"Due to 'Accompany' {agent['agent']} lost {non_active_time} minutes of'{agent['todo']}'.")
                elif new_out > agent['closing'] and agent['fixed'] == True:
                    print(f"Due to 'Accompany' {agent['agent']} was not able to fullfill '{agent['todo']}' at {agent['in']}.")
                    continue
                    
                rew_row ={
                    'agent': agent['agent'],
                    'todo': agent['todo'], 
                    'osm_id': agent['osm_id'], 
                    'todo_type': 0, 
                    'opening': agent['opening'], 
                    'closing': agent['closing'], 
                    'fixed': agent['fixed'], 
                    'time2spend': agent['time2spend'], 
                    'in': group_in_time, 
                    'out': new_out,
                    'conmu_time': group_conmu_time
                }   
                print('name_group')
                print(name_group)
                print('rew_row')
                input(rew_row)
                # Suma a dataframe
                new_new_list = pd.concat([new_new_list, pd.DataFrame([rew_row])], ignore_index=True).sort_values(by='in', ascending=True)
                
    return new_new_list
