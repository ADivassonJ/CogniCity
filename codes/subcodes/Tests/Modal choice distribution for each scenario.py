import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# TAMAÑO PAPER (mm → pulgadas)
# -----------------------------
fig_width = 368 / 25.4
fig_height = 78 / 25.4

# -----------------------------
# RUTAS
# -----------------------------
BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"
DATA_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

# -----------------------------
# CONTENEDORES
# -----------------------------
modal_clean = []      # matriz condicional
modal_total = []      # distribución global ← NUEVO

# -----------------------------
# LOOP PRINCIPAL
# -----------------------------
for s in S_FOLDERS:
    for scen in SCENARIOS:

        excel_file = os.path.join(BASE_PATH, s, scen, f"{scen}_schedule_vehicle.xlsx")
        parquet_file = os.path.join(DATA_PATH, s, scen, "population", "pop_citizen.parquet")

        if not os.path.exists(excel_file) or not os.path.exists(parquet_file):
            continue

        df = pd.read_excel(excel_file)
        pop = pd.read_parquet(parquet_file)

        required_cols = {"user", "archetype", "in"}
        if not required_cols.issubset(df.columns):
            continue

        # -----------------------------
        # SIMPLIFICACIÓN POR USUARIO
        # -----------------------------
        df_simplified = (
            df.loc[df.groupby("user")["in"].idxmin()]
            .reset_index(drop=True)
        )

        # =====================================================
        # ✅ 1️⃣ MATRIZ CONDICIONAL (como antes)
        # =====================================================
        pop_subset = pop[["name", "archetype"]].rename(
            columns={"name": "user", "archetype": "citizen_archetype"}
        )

        df_conditional = df_simplified.merge(pop_subset, on="user", how="left")

        modal_matrix = pd.crosstab(
            df_conditional["citizen_archetype"],
            df_conditional["archetype"],
            normalize="index"
        ) * 100

        modal_matrix["Scenario"] = scen
        modal_matrix["S"] = s

        modal_clean.append(modal_matrix.reset_index())

        # =====================================================
        # ✅ 2️⃣ DISTRIBUCIÓN GLOBAL ← NUEVO
        # =====================================================
        modal_share = (
            df_simplified["archetype"]
            .value_counts(normalize=True) * 100
        )

        modal_share["Scenario"] = scen
        modal_share["S"] = s

        modal_total.append(modal_share)

# -----------------------------
# CONSOLIDACIÓN
# -----------------------------
modal_df = pd.concat(modal_clean, ignore_index=True)
modal_total_df = pd.DataFrame(modal_total).fillna(0)

# -----------------------------
# COLORES PAPER
# -----------------------------
colors = {
    "walk": "#EAD8A6",
    "UB_diesel": "#D4A017",
    "PC_petrol": "#9C7A00",
    "PC_electric": "#4F3B00",
}

scenario_order = ["Annelinn", "Aradas", "Kanaleneiland"]

# ============================================================
# ✅ FIGURA → RESUMEN GLOBAL POR ESCENARIO
# Barras = S
# ============================================================
for scen in scenario_order:

    scen_df = modal_total_df[modal_total_df["Scenario"] == scen]

    pivot = (
        scen_df
        .set_index("S")
        .drop(columns=["Scenario"])
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bottom = np.zeros(len(pivot))

    for mode in pivot.columns:
        values = pivot[mode].values

        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=mode,
            color=colors.get(mode, None)
        )

        bottom += values

    ax.set_ylabel("Modal share [%]")
    ax.set_xlabel("Simulation run (S)")
    ax.set_ylim(0, 100)

    ax.set_title(f"{scen} — Global modal distribution")

    ax.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
