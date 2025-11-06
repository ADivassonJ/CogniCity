import subprocess
import pandas as pd
from pathlib import Path
import time
import os

# ============================
# CONFIGURACIÓN
# ============================
N_REPETICIONES = 5  # ← número de veces que quieres repetir el proceso

# Scripts
path_init = Path(r"C:\Users\asier\Documents\GitHub\CogniCity\codes\subcodes\Documents_initialisation.py")
path_test = Path(r"C:\Users\asier\Documents\GitHub\CogniCity\codes\subcodes\tests_3.py")

# Archivos de datos y resultados
path_summary_csv = Path(r"C:\Users\asier\resumen_dist_wos_lognormal.csv")
path_results_xlsx = Path(r"C:\Users\asier\Documents\GitHub\CogniCity\results\test_results.xlsx")

# Archivos a eliminar tras cada iteración
files_to_delete = [
    Path(r"C:\Users\asier\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet"),
    Path(r"C:\Users\asier\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_distribution.parquet"),
    Path(r"C:\Users\asier\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_family.parquet"),
    Path(r"C:\Users\asier\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_transport.parquet"),
]

# ============================
# FUNCIÓN PRINCIPAL
# ============================
def ejecutar_pipeline(n_reps=N_REPETICIONES):
    """
    Ejecuta n_reps veces la secuencia:
      1. Ejecuta Documents_initialisation.py
      2. Ejecuta tests_3.py
      3. Lee resumen_dist_wos_lognormal.csv
      4. Añade los resultados a test_results.xlsx
      5. Borra los archivos Parquet intermedios
    """

    for i in range(1, n_reps + 1):
        print(f"\n=== EJECUCIÓN {i}/{n_reps} ===")

        # 1️⃣ Ejecutar Documents_initialisation.py
        print("→ Ejecutando Documents_initialisation.py ...")
        subprocess.run(["python", str(path_init)], check=True)

        # 2️⃣ Ejecutar tests_3.py
        print("→ Ejecutando tests_3.py ...")
        subprocess.run(["python", str(path_test)], check=True)

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
    ejecutar_pipeline(100)
