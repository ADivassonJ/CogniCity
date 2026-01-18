import subprocess
import pandas as pd
from pathlib import Path
import time
import os
import sys
import shutil
import json
from datetime import datetime

# ============================
# CONFIGURACIÓN
# ============================
N_REPETICIONES = 100  # número de iteraciones
study_area = "Kanaleneiland"

import re

def get_replicates_root(paths: dict, study_area: str) -> Path:
    return paths['results'] / "replicates" / study_area

def get_existing_iterations(paths: dict, study_area: str) -> list[int]:
    """
    Busca carpetas tipo iter_0001 en results/replicates/<study_area>/
    Devuelve lista de ints (iterations) ordenada ascendente.
    """
    root = get_replicates_root(paths, study_area)
    if not root.exists():
        return []

    iters = []
    pattern = re.compile(r"^iter_(\d{4})$")
    for p in root.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                iters.append(int(m.group(1)))
    return sorted(iters)

def get_next_iteration(paths: dict, study_area: str) -> int:
    existing = get_existing_iterations(paths, study_area)
    return (existing[-1] + 1) if existing else 1


def paths_initialization(study_area):
    # Paths initialization
    paths = {}

    paths['main'] = Path(__file__).resolve().parent.parent.parent.parent
    paths['system'] = paths['main'] / 'system'
    paths['desktop'] = Path.home() / "Desktop"

    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')

    file_management = system_management[['file_1', 'file_2', 'pre']]
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
                    response = input(
                        f"Data for the case study '{study_area}' was not found.\n"
                        f"Do you want to copy data from standar scenario or do you want to create your own? [Y/N]\n"
                    )
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

# Archivos a borrar tras cada iteración (se archivarán antes)
files_to_delete = [
    paths['results'] / "Kanaleneiland_schedule_citizen.xlsx",
    paths['results'] / "Kanaleneiland_schedule_vehicle.xlsx",
    paths['results'] / "Kanaleneiland_todolist.xlsx",
    paths['results'] / "dist_wos_por_citizen.csv",
    paths['results'] / "resumen_dist_wos_lognormal.csv"
]

path_citizens = paths['population'] / "pop_citizen.parquet"


# ============================
# UTILIDADES ARCHIVADO
# ============================
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def archive_iteration_outputs(iteration: int, study_area: str, paths: dict, files_to_archive: list, resumen_df: pd.DataFrame):
    """
    Crea:
      results/replicates/<study_area>/iter_XXXX/{micro,agg,meta}
    Copia microdatos y outputs antes de borrarlos y guarda agregados por iteración.
    """
    root = ensure_dir(paths['results'] / "replicates" / study_area)
    iter_dir = ensure_dir(root / f"iter_{iteration:04d}")
    micro_dir = ensure_dir(iter_dir / "micro")
    agg_dir = ensure_dir(iter_dir / "agg")
    meta_dir = ensure_dir(iter_dir / "meta")

    archived = []
    for f in files_to_archive:
        try:
            if f.exists():
                dst = micro_dir / f.name
                shutil.copy2(f, dst)
                archived.append({"src": str(f), "dst": str(dst)})
            else:
                archived.append({"src": str(f), "dst": None, "note": "missing"})
        except Exception as e:
            archived.append({"src": str(f), "dst": None, "error": str(e)})

    agg_path = agg_dir / f"agg_archetype_iter_{iteration:04d}.csv"
    resumen_df.to_csv(agg_path, index=False)

    meta = {
        "study_area": study_area,
        "iteration": iteration,
        "created_at": datetime.now().isoformat(),
        "archived_files": archived,
        "agg_file": str(agg_path),
    }
    meta_path = meta_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    return iter_dir


# ============================
# RESUMEN POR ITERACIÓN
# ============================
def resumen_spended_to_csv(df, path_summary_csv):
    """
    A partir de df con columnas: agent, archetype, out, in
    - spended = out - in
    - por agent: spended_tot = 1440 - sum(spended)
    - stats: GLOBAL + por archetype (mean, std)
    - append al CSV global
    Devuelve:
      df_agentes, (media_global, std_global), stats_archetype, df_resultados
    """
    df = df.copy()

    df["spended"] = df["out"] - df["in"]

    df_agentes = (
        df.groupby("agent")
          .agg(
              archetype=("archetype", "first"),
              suma_spended=("spended", "sum")
          )
          .reset_index()
    )

    df_agentes["spended_tot"] = 1440 - df_agentes["suma_spended"]
    df_agentes.drop(columns=["suma_spended"], inplace=True)

    media_global = df_agentes["spended_tot"].mean()
    std_global = df_agentes["spended_tot"].std()

    stats_archetype = (
        df_agentes.groupby("archetype")["spended_tot"]
                  .agg(["mean", "std"])
                  .reset_index()
    )

    fila_global = pd.DataFrame({
        "archetype": ["GLOBAL"],
        "mean_spended_tot": [media_global],
        "std_spended_tot": [std_global]
    })

    filas_archetype = stats_archetype.rename(
        columns={"mean": "mean_spended_tot", "std": "std_spended_tot"}
    )

    df_resultados = pd.concat([fila_global, filas_archetype], ignore_index=True)

    # append CSV global
    if os.path.exists(path_summary_csv):
        df_resultados.to_csv(path_summary_csv, mode="a", header=False, index=False)
    else:
        df_resultados.to_csv(path_summary_csv, index=False)

    return df_agentes, (media_global, std_global), stats_archetype, df_resultados


# ============================
# FUNCIÓN PRINCIPAL
# ============================
def ejecutar_pipeline(n_reps=N_REPETICIONES, resume=True):
    """
    Si resume=True:
      - detecta iteraciones ya archivadas en results/replicates/<study_area>/iter_XXXX
      - continúa desde la siguiente
      - ejecuta hasta completar n_reps iteraciones TOTALES (no adicionales)

    Si resume=False:
      - empieza desde 1 y ejecuta n_reps iteraciones (modo "nuevo")
    """

    # 0) Determinar rango de iteraciones a ejecutar
    if resume:
        existing = get_existing_iterations(paths, study_area)
        start_i = (existing[-1] + 1) if existing else 1

        # Interpretación: n_reps es TOTAL deseado
        # Si ya hay 37 iteraciones, y n_reps=100 => ejecuta 38..100
        end_i = n_reps

        if start_i > end_i:
            print(f"✅ Ya existen {len(existing)} iteraciones archivadas (hasta {existing[-1]:04d}).")
            print(f"No hay nada que ejecutar porque n_reps={n_reps}.")
            return

        print(f"🔁 Modo reanudar activado.")
        print(f"   - Iteraciones existentes: {len(existing)} ({existing[0]:04d}..{existing[-1]:04d})" if existing else "   - No hay iteraciones existentes.")
        print(f"   - Ejecutando desde iter_{start_i:04d} hasta iter_{end_i:04d} (total objetivo={n_reps}).")

    else:
        start_i, end_i = 1, n_reps
        print(f"🆕 Modo nuevo (sin reanudar). Ejecutando 1..{end_i:04d}")

        # Opcional (si quieres limpiar outputs previos):
        # - Borrar/renombrar path_summary_csv y path_results_xlsx para no mezclar corridas
        # - O moverlos a un backup con timestamp

    # 1) Loop principal
    for i in range(start_i, end_i + 1):
        print(f"\n=== EJECUCIÓN {i}/{end_i} ===")

        # 1️⃣ Ejecutar CognitiCity.py
        print("→ Ejecutando CognitiCity.py ...")
        subprocess.run(["python", str(path_init)], check=True)

        # 2️⃣ Leer microdatos
        citizen_xlsx = paths['results'] / f"{study_area}_schedule_citizen.xlsx"
        if not citizen_xlsx.exists():
            print(f"⚠️  No se encontró {citizen_xlsx}, se omite esta iteración.")
            continue

        df = pd.read_excel(citizen_xlsx)

        # 3️⃣ Calcular resumen y append al CSV global
        print("→ Calculando resumen por archetype ...")
        df_agentes, (media_global, std_global), stats_archetype, df_resultados = resumen_spended_to_csv(df, path_summary_csv)

        # 4️⃣ Archivar outputs antes de borrarlos
        print("📦 Archivando outputs de la iteración (micro + agregados) ...")
        iter_dir = archive_iteration_outputs(
            iteration=i,
            study_area=study_area,
            paths=paths,
            files_to_archive=files_to_delete,
            resumen_df=df_resultados
        )
        print(f"   - Archivado en: {iter_dir}")

        # 5️⃣ Actualizar Excel acumulado evitando duplicados por iteration
        time.sleep(0.5)
        if not path_summary_csv.exists():
            print(f"⚠️  No se encontró {path_summary_csv}, se omite esta iteración.")
            continue

        resumen = pd.read_csv(path_summary_csv)
        resumen["iteration"] = i
        resumen["timestamp"] = pd.Timestamp.now()

        if path_results_xlsx.exists():
            existing_xlsx = pd.read_excel(path_results_xlsx)
            combined = pd.concat([existing_xlsx, resumen], ignore_index=True)

            # Evitar duplicados por iteración (si re-ejecutas una misma i)
            # Mantiene el último registro de esa iteración.
            if "iteration" in combined.columns:
                combined = combined.sort_values("timestamp").drop_duplicates(subset=["iteration", "archetype"], keep="last")
        else:
            combined = resumen

        combined.to_excel(path_results_xlsx, index=False)
        print(f"✅ Resultados añadidos a {path_results_xlsx.name}")

        # 6️⃣ Borrar outputs del directorio results (ya archivados)
        print("🗑️  Borrando outputs (ya archivados) ...")
        for f in files_to_delete:
            try:
                if f.exists():
                    os.remove(f)
                    print(f"   - Eliminado: {f.name}")
                else:
                    print(f"   - No existe (omitido): {f.name}")
            except Exception as e:
                print(f"   ⚠️  No se pudo borrar {f.name}: {e}")

        time.sleep(0.5)

    print("\n🎯 Proceso completado con éxito.")
    print(f"Resultados acumulados en: {path_results_xlsx.resolve()}")
    print(f"Réplicas archivadas en: {(paths['results'] / 'replicates' / study_area).resolve()}")


# ============================
# EJECUCIÓN DIRECTA
# ============================
if __name__ == "__main__":
    ejecutar_pipeline(N_REPETICIONES, resume=True)
