import os
import glob
import pandas as pd
import numpy as np

BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

results = []

for s in S_FOLDERS:
    for scen in SCENARIOS:
        folder = os.path.join(BASE_PATH, s, scen)

        # Busca cualquier XLSX que contenga schedule_vehicle
        pattern = os.path.join(folder, f"*{scen}_schedule_vehicle*.xlsx")
        files = glob.glob(pattern)

        if not files:
            print(f"[WARN] No files found for {s}/{scen}")
            continue

        for fp in files:
            try:
                df = pd.read_excel(fp)
            except Exception as e:
                print(f"[ERROR] Could not read {fp}: {e}")
                continue

            required_cols = {"agent", "archetype", "todo"}
            if not required_cols.issubset(df.columns):
                print(f"[SKIP] Missing required columns in {fp}")
                continue

            d = df[["agent", "archetype", "todo"]].copy()

            # Limpieza básica
            d = d.dropna(subset=["agent", "archetype"])

            # Interpretar vacío como NaN o string vacío/espacios
            d["todo_empty"] = d["todo"].isna() | (d["todo"].astype(str).str.strip() == "")

            # Eliminar duplicados por agente
            d = d.drop_duplicates(subset=["agent"], keep="first")

            # TOTAL por archivo
            total_agents = d["agent"].nunique()
            total_empty = int(d["todo_empty"].sum())
            total_pct_empty = (total_empty / total_agents * 100) if total_agents else np.nan

            results.append({
                "S": s,
                "Scenario": scen,
                "File": os.path.basename(fp),
                "Archetype": "TOTAL",
                "Vehicles": total_agents,
                "TodoEmpty": total_empty,
                "PctTodoEmpty": total_pct_empty
            })

            # DESGLOSE por archetype
            for arch, g in d.groupby("archetype"):
                n = g["agent"].nunique()
                ne = int(g["todo_empty"].sum())
                pct = (ne / n * 100) if n else np.nan

                results.append({
                    "S": s,
                    "Scenario": scen,
                    "File": os.path.basename(fp),
                    "Archetype": str(arch),
                    "Vehicles": n,
                    "TodoEmpty": ne,
                    "PctTodoEmpty": pct
                })

# Tabla final
out = pd.DataFrame(results)

if out.empty:
    raise ValueError("No se encontró ningún fichero válido. Revisa nombres/rutas.")

out = out.sort_values(["S", "Scenario", "File", "Archetype"]).reset_index(drop=True)

# Guardado
out.to_excel("summary_vehicles_todo_by_S_scenario.xlsx", index=False)
out.to_csv("summary_vehicles_todo_by_S_scenario.csv", index=False)

print("OK -> Generados:")
print(" - summary_vehicles_todo_by_S_scenario.xlsx")
print(" - summary_vehicles_todo_by_S_scenario.csv")

print("\nPreview:")
print(out)
