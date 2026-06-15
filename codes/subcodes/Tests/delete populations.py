import os
import glob
import sys
import pandas as pd
from pathlib import Path

study_areas = ["Annelinn", "Aradas", "Kanaleneiland"]

for study_area in study_areas:
    # -----------------------------
    # 0. Limpieza de archivos pop_* (excepto pop_building)
    paths = {}
        
    paths['main'] = Path(__file__).resolve().parent.parent.parent.parent
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
    

# -----------------------------
data_dir = paths['data']
# Si tu base_dir ya apunta dentro de un escenario concreto, ajusta data_dir a la ruta raíz que contiene s0..s4

deleted_files = []

for s in range(5):  # s0 a s4
    s_folder = os.path.join(data_dir, f"s{s}")
    if not os.path.isdir(s_folder):
        continue

    # Recorre cada carpeta de escenario dentro de sN (Annelinn, Aradas, Kanaleneiland, etc.)
    for scenario_name in os.listdir(s_folder):
        scenario_path = os.path.join(s_folder, scenario_name)
        if not os.path.isdir(scenario_path):
            continue

        population_path = os.path.join(scenario_path, "population")
        if not os.path.isdir(population_path):
            continue

        # Busca todos los archivos que empiecen por "pop_"
        for f in glob.glob(os.path.join(population_path, "pop_*")):
            filename = os.path.basename(f)

            # Excepción: no borrar pop_building (con cualquier extensión)
            if filename.startswith("pop_building"):
                continue

            if os.path.isfile(f):
                try:
                    os.remove(f)
                    deleted_files.append(f)
                except OSError as e:
                    print(f"No se pudo eliminar {f}: {e}")

print(f"Limpieza completada: {len(deleted_files)} archivo(s) eliminado(s)")
for f in deleted_files:
    print(f"  - {f}")