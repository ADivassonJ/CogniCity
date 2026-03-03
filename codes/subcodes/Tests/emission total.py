import os
import pandas as pd

BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

SHEET_NAME = "vehicle_daily_last"

summary_results = []

for s in S_FOLDERS:
    for scen in SCENARIOS:

        # Construcción correcta del nombre del Excel
        excel_file = os.path.join(
            BASE_PATH,
            s,
            scen,
            f"{scen}_daily_total_stats_inferred_24.xlsx"
        )

        if not os.path.exists(excel_file):
            print(f"[SKIP] Missing Excel -> {excel_file}")
            continue

        print(f"[OK] Processing -> {s} / {scen}")

        # -----------------------------
        # Leer hoja concreta
        # -----------------------------
        df = pd.read_excel(excel_file, sheet_name=SHEET_NAME)

        # -----------------------------
        # Limpieza columnas numéricas
        # -----------------------------
        for col in ["mjkm", "emissions"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

        # -----------------------------
        # Totales generales
        # -----------------------------
        total_mjkm = df["mjkm"].sum()
        total_emissions = df["emissions"].sum()

        # -----------------------------
        # Solo eléctricos
        # -----------------------------
        electric_mjkm = df.loc[
            df["archetype"] == "PC_electric", "mjkm"
        ].sum()

        # -----------------------------
        # Guardar resultados
        # -----------------------------
        summary_results.append({
            "scenario_folder": s,
            "case_study": scen,
            "total_mjkm": total_mjkm,
            "total_emissions": total_emissions,
            "electric_mjkm": electric_mjkm
        })

# DataFrame final
summary_df = pd.DataFrame(summary_results)

print(summary_df)