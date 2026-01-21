import os
import glob
import pandas as pd
import numpy as np

BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

results = []
coverage = []

for s in S_FOLDERS:
    for scen in SCENARIOS:
        folder = os.path.join(BASE_PATH, s, scen)
        pattern = os.path.join(folder, f"*{scen}_schedule_vehicle*.xlsx")
        files = glob.glob(pattern)

        coverage.append({
            "S": s,
            "Scenario": scen,
            "FilesFound": len(files),
            "FolderExists": os.path.isdir(folder),
        })

        # Si no hay ficheros: NO metemos filas de resultados
        if not files:
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
            d = d.dropna(subset=["agent", "archetype"])

            d["todo_empty"] = d["todo"].isna() | (d["todo"].astype(str).str.strip() == "")
            d = d.drop_duplicates(subset=["agent"], keep="first")

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

out = pd.DataFrame(results).sort_values(["S", "Scenario", "File", "Archetype"]).reset_index(drop=True)
cov = pd.DataFrame(coverage).sort_values(["S", "Scenario"]).reset_index(drop=True)

# Guardado
out.to_excel("summary_vehicles_todo_by_S_scenario.xlsx", index=False)
out.to_csv("summary_vehicles_todo_by_S_scenario.csv", index=False)

cov.to_excel("coverage_files_found.xlsx", index=False)
cov.to_csv("coverage_files_found.csv", index=False)

print("OK -> Generados:")
print(" - summary_vehicles_todo_by_S_scenario.xlsx / .csv (solo donde hay datos)")
print(" - coverage_files_found.xlsx / .csv (cobertura de ficheros por S/Scenario)")
