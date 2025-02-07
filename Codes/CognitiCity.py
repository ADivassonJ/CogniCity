import os
from pathlib import Path
from Subcodes.initialization import initialization
from Subcodes.simulate_day import simulate_day
from Subcodes.file_playgroud import save_dataframes


if __name__ == "__main__":
    ##################  SIMULATION EVALUATION STUFF  ################## 
    area = "Kanaleneiland"
    population_citizens = 200
    steps_captured = 24
    days_captured = 1
    EV_percentage = 0.05
    ###################################################################

    # Definición de variables principales
    main_path = Path(__file__).resolve().parent.parent
    data_path = main_path / f'Data/{area}'
    results_path = main_path / f'Results/{area}'
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    type_services = ['Living', 'Working', 'Commerce', 'Healthcare', 'Education', 'Entertainment']

    # Inicialización
    df_buildings, df_actions, df_citizens, df_vehicles, df_moving_agents = initialization(area, population_citizens, EV_percentage, main_path, type_services)
    
    # Bucle de simulación
    for day in range(days_captured):
        print('#' * 22, 'day:', day + 1, '#' * 22)                           ####### COSORRO no se puc actualizar un df en un for
        df_actions, df_moving_agents = simulate_day(df_actions, df_citizens, df_buildings, df_moving_agents, day, steps_captured, main_path)
       
    # Guardar los DataFrames
    df_names = ['df_actions', 'df_citizens', 'df_vehicles', 'df_moving_agents']
    df_group = [df_actions, df_citizens, df_vehicles, df_moving_agents]
    save_dataframes(results_path, df_group, df_names)
    
