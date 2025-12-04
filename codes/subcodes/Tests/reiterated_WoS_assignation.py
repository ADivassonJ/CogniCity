import subprocess
import pandas as pd
from pathlib import Path
import time
import os

import sys
import shutil

# ============================
# CONFIGURACIÓN
# ============================
N_REPETICIONES = 100  # ← número de veces que quieres repetir el proceso
study_area = "Kanaleneiland"

def paths_initialization(study_area):
    # Paths initialization
    paths = {}
    
    paths['main'] = Path(__file__).resolve().parent.parent.parent.parent
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

# Paths initialization
paths, system_management = paths_initialization(study_area)

# Scripts
path_init = paths['codes'] / "CognitiCity.py"
path_test = paths['subcodes'] / "Tests" / "WoS_assignation_evaluation.py"

# Archivos de datos y resultados
path_summary_csv = paths['results'] / "resumen_dist_wos_lognormal.csv"
path_results_xlsx = paths['results'] / "test_results.xlsx"

# Archivos a eliminar tras cada iteración
files_to_delete = [
    paths['results'] / "Kanaleneiland_schedule_citizen.xlsx",
    paths['results'] / "Kanaleneiland_schedule_vehicle.xlsx",
    paths['results'] / "Kanaleneiland_todolist.xlsx",
    paths['results'] / "dist_wos_por_citizen.csv",
    paths['results'] / "resumen_dist_wos_lognormal.csv"
]

path_citizens = paths['population'] / "pop_citizen.parquet"


def resumen_spended_to_csv(df, path_summary_csv):
    """
    A partir de un df con columnas:
      - 'agent'
      - 'archetype'
      - 'out'
      - 'in'
      
    Hace:
      1) Crea columna 'spended' = out - in
      2) Agrupa por 'agent' para obtener spended_tot = 1440 - sum(spended)
      3) Calcula:
           - media y std GLOBAL de spended_tot
           - media y std de spended_tot POR archetype
      4) Construye un df de resultados:
           - fila GLOBAL
           - filas por archetype
      5) Añade ese df al CSV path_summary_csv (append).
    """

    df = df.copy()

    # 1) Nueva columna
    df["spended"] = df["out"] - df["in"]

    # 2) Agrupar por agent (pero con spended_tot = 1440 - sum)
    df_agentes = (
        df.groupby("agent")
          .agg(
              archetype=("archetype", "first"),
              suma_spended=("spended", "sum")
          )
          .reset_index()
    )

    # Aplicar la transformación deseada
    df_agentes["spended_tot"] = 1440 - df_agentes["suma_spended"]

    # Ya no necesitamos la columna suma_spended
    df_agentes.drop(columns=["suma_spended"], inplace=True)

    # 3) Estadísticas globales
    media_global = df_agentes["spended_tot"].mean()
    std_global = df_agentes["spended_tot"].std()

    # Estadísticas por archetype
    stats_archetype = (
        df_agentes.groupby("archetype")["spended_tot"]
                  .agg(["mean", "std"])
                  .reset_index()
    )

    # 4) Crear df de salida GLOBAL + archetypes
    fila_global = pd.DataFrame({
        "archetype": ["GLOBAL"],
        "mean_spended_tot": [media_global],
        "std_spended_tot": [std_global]
    })

    filas_archetype = stats_archetype.rename(
        columns={
            "mean": "mean_spended_tot",
            "std": "std_spended_tot"
        }
    )

    df_resultados = pd.concat([fila_global, filas_archetype], ignore_index=True)

    # 5) Escribir / añadir al CSV
    if os.path.exists(path_summary_csv):
        df_resultados.to_csv(path_summary_csv, mode="a", header=False, index=False)
    else:
        df_resultados.to_csv(path_summary_csv, index=False)

    return df_agentes, (media_global, std_global), stats_archetype, df_resultados

# ============================
# FUNCIÓN PRINCIPAL
# ============================
def ejecutar_pipeline(n_reps=N_REPETICIONES):
    """
    Ejecuta n_reps veces la secuencia:
      1. Ejecuta Documents_initialisation.py
      2. Ejecuta WoS assignation evaluation - Kanaleneiland.py
      3. Lee resumen_dist_wos_lognormal.csv
      4. Añade los resultados a test_results.xlsx
      5. Borra los archivos Parquet intermedios
    """

    for i in range(1, n_reps + 1):
        print(f"\n=== EJECUCIÓN {i}/{n_reps} ===")

        # 1️⃣ Ejecutar Documents_initialisation.py
        print("→ Ejecutando Documents_initialisation.py ...")
        subprocess.run(["python", str(path_init)], check=True)

        df = pd.read_excel(paths['results'] / "Kanaleneiland_schedule_citizen.xlsx")

        # 2️⃣ Ejecutar WoS assignation evaluation - Kanaleneiland.py
        print("→ Ejecutando WoS assignation evaluation - Kanaleneiland.py ...")
        resumen_spended_to_csv(df, path_summary_csv)

        # 3️⃣ Leer resumen CSV
        time.sleep(2)  # espera breve
        if not path_summary_csv.exists():
            print(f"⚠️  No se encontró {path_summary_csv}, se omite esta iteración.")
            continue

        resumen = pd.read_csv(path_summary_csv)
        resumen["iteration"] = i
        resumen["timestamp"] = pd.Timestamp.now()

        # 4️⃣ Añadir al Excel acumulado
        if path_results_xlsx.exists():
            existing = pd.read_excel(path_results_xlsx)
            combined = pd.concat([existing, resumen], ignore_index=True)
        else:
            combined = resumen

        combined.to_excel(path_results_xlsx, index=False)
        print(f"✅ Resultados añadidos a {path_results_xlsx.name}")

        # 5️⃣ Borrar archivos intermedios
        print("🗑️  Borrando archivos Parquet intermedios...")
        for f in files_to_delete:
            try:
                if f.exists():
                    os.remove(f)
                    print(f"   - Eliminado: {f.name}")
                else:
                    print(f"   - No existe (omitido): {f.name}")
            except Exception as e:
                print(f"   ⚠️  No se pudo borrar {f.name}: {e}")

        # Pausa entre iteraciones
        time.sleep(1)

    print("\n🎯 Proceso completado con éxito.")
    print(f"Resultados acumulados en: {path_results_xlsx.resolve()}")

# ============================
# EJECUCIÓN DIRECTA
# ============================
if __name__ == "__main__":
    ejecutar_pipeline(N_REPETICIONES)
