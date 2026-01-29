import os
import glob
import pandas as pd
import numpy as np

BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

results = []
coverage = []
modal_dist = []  # <-- NUEVO

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

        if not files:
            continue

        for fp in files:
            try:
                df = pd.read_excel(fp)
            except Exception as e:
                print(f"[ERROR] Could not read {fp}: {e}")
                continue

            # =========================
            # PARTE ORIGINAL (todo vacío)
            # =========================
            required_cols = {"agent", "archetype", "todo"}
            if required_cols.issubset(df.columns):
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
            else:
                print(f"[SKIP] Missing required columns (agent/archetype/todo) in {fp}")

            # ==========================================
            # NUEVO: DISTRIBUCIÓN MODAL POR USUARIO ÚNICO
            # ==========================================
            required_modal_cols = {"user", "archetype"}
            if not required_modal_cols.issubset(df.columns):
                print(f"[SKIP] Missing required columns (user/archetype) in {fp}")
                continue

            m = df[["user", "archetype"]].copy()

            # limpiar user: eliminar NaN y strings vacíos/espacios
            m["user"] = m["user"].astype(str)
            m["user_clean"] = m["user"].str.strip()
            m = m[(m["user_clean"].notna()) & (m["user_clean"] != "")]

            # eliminar NaN de archetype también (si aplica)
            m = m.dropna(subset=["archetype"])

            # usuarios únicos (primer registro)
            m = m.drop_duplicates(subset=["user_clean"], keep="first")

            total_users = m["user_clean"].nunique()

            # fila TOTAL (opcional, útil para checks)
            modal_dist.append({
                "S": s,
                "Scenario": scen,
                "File": os.path.basename(fp),
                "Archetype": "TOTAL",
                "Users": total_users,
                "PctUsers": 100.0 if total_users else np.nan
            })

            # distribución por archetype
            counts = (
                m.groupby("archetype")["user_clean"]
                 .nunique()
                 .sort_values(ascending=False)
            )

            for arch, n_users in counts.items():
                pct = (n_users / total_users * 100) if total_users else np.nan
                modal_dist.append({
                    "S": s,
                    "Scenario": scen,
                    "File": os.path.basename(fp),
                    "Archetype": str(arch),
                    "Users": int(n_users),
                    "PctUsers": pct
                })

            # ==========================================
            # NUEVO: DISTRIBUCIÓN MODAL POR USUARIO ÚNICO
            # (EXCLUYE independent != 1)
            # ==========================================
            required_modal_cols = {"user", "archetype", "independent"}  # <-- añadimos independent
            if not required_modal_cols.issubset(df.columns):
                print(f"[SKIP] Missing required columns (user/archetype/independent) in {fp}")
                continue

            # Filtrar SOLO independent == 1
            df_modal = df[df["independent"] == 1].copy()

            m = df_modal[["user", "archetype"]].copy()

            # limpiar user: eliminar NaN y strings vacíos/espacios
            m["user"] = m["user"].astype(str)
            m["user_clean"] = m["user"].str.strip()
            m = m[(m["user_clean"].notna()) & (m["user_clean"] != "")]

            # eliminar NaN de archetype también (si aplica)
            m = m.dropna(subset=["archetype"])

            # usuarios únicos (primer registro)
            m = m.drop_duplicates(subset=["user_clean"], keep="first")

            total_users = m["user_clean"].nunique()

            # fila TOTAL
            modal_dist.append({
                "S": s,
                "Scenario": scen,
                "File": os.path.basename(fp),
                "Archetype": "TOTAL",
                "Users": total_users,
                "PctUsers": 100.0 if total_users else np.nan
            })

            # distribución por archetype
            counts = (
                m.groupby("archetype")["user_clean"]
                 .nunique()
                 .sort_values(ascending=False)
            )

            for arch, n_users in counts.items():
                pct = (n_users / total_users * 100) if total_users else np.nan
                modal_dist.append({
                    "S": s,
                    "Scenario": scen,
                    "File": os.path.basename(fp),
                    "Archetype": str(arch),
                    "Users": int(n_users),
                    "PctUsers": pct
                })


# DataFrames finales
out = pd.DataFrame(results).sort_values(["S", "Scenario", "File", "Archetype"]).reset_index(drop=True)
cov = pd.DataFrame(coverage).sort_values(["S", "Scenario"]).reset_index(drop=True)

modal_out = (
    pd.DataFrame(modal_dist)
      .sort_values(["S", "Scenario", "File", "Archetype"])
      .reset_index(drop=True)
)

# Guardado
out.to_excel(os.path.join(BASE_PATH, "summary_vehicles_todo_by_S_scenario.xlsx"), index=False)
out.to_csv(os.path.join(BASE_PATH, "summary_vehicles_todo_by_S_scenario.csv"), index=False)

cov.to_excel(os.path.join(BASE_PATH, "coverage_files_found.xlsx"), index=False)
cov.to_csv(os.path.join(BASE_PATH, "coverage_files_found.csv"), index=False)

# NUEVO guardado distribución modal
modal_out.to_excel(os.path.join(BASE_PATH, "modal_distribution_by_S_scenario.xlsx"), index=False)
modal_out.to_csv(os.path.join(BASE_PATH, "modal_distribution_by_S_scenario.csv"), index=False)

print("OK -> Generados:")
print(" - summary_vehicles_todo_by_S_scenario.xlsx / .csv (solo donde hay datos)")
print(" - coverage_files_found.xlsx / .csv (cobertura de ficheros por S/Scenario)")
print(" - modal_distribution_by_S_scenario.xlsx / .csv (distribución modal por user único)")
