import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# TAMAÑO FIGURA
# -----------------------------
fig_width = 368 / 25.4
fig_height = 78 / 25.4

# -----------------------------
# RUTAS
# -----------------------------
BASE_PATH_1 = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (wp3 true)"
BASE_PATH_2 = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (wp3 false)"
DATA_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

# -----------------------------
# RENOMBRES
# -----------------------------
rename_scenarios = {
    "s0": "Baseline Scenario",
    "s1": "EU Trinity",
    "s2": "NECP Essentials",
    "s3": "REPowerEU",
    "s4": "Go RES"
}

rename_archetypes = {
    "c_arch_0": "Adult male",
    "c_arch_1": "Adult female",
    "c_arch_2": "Children",
    "c_arch_3": "Elder",
    "c_arch_4": "Youth"
}

MOBILITY_LABELS = {
    "UB_diesel": "Public Transportation",
    "PC_petrol": "Combustion car",
    "PC_electric": "Electric car",
    "walk": "Walk",
    "CS_electric": "Car sharing"
}

scenario_order = ["Annelinn", "Aradas", "Kanaleneiland"]

# -----------------------------
# CONTENEDORES
# -----------------------------
modal_clean = []
modal_total = []
transition_results = []
cs_flows = []

# -----------------------------
# FUNCION AUXILIAR
# -----------------------------
def simplify(df):
    return df.loc[df.groupby("user")["in"].idxmin()].reset_index(drop=True)

# -----------------------------
# LOOP PRINCIPAL
# -----------------------------
for s in S_FOLDERS:
    for scen in SCENARIOS:

        # NEW (5 modos)
        new_excel = os.path.join(BASE_PATH_1, s, f"{scen}_schedule_vehicle.xlsx")
        # OLD (4 modos)
        old_excel = os.path.join(BASE_PATH_2, s, f"{scen}_schedule_vehicle.xlsx")

        parquet_file = os.path.join(DATA_PATH, s, scen, "population", "pop_citizen.parquet")

        if not (os.path.exists(new_excel) and os.path.exists(old_excel) and os.path.exists(parquet_file)):
            continue

        print(f"[OK] {s} / {scen}")

        df_new = pd.read_excel(new_excel)
        df_old = pd.read_excel(old_excel)
        pop = pd.read_parquet(parquet_file)

        # -----------------------------
        # SIMPLIFICACIÓN
        # -----------------------------
        df_new_s = simplify(df_new)
        df_old_s = simplify(df_old)

        # -----------------------------
        # MERGE ARCHETYPE
        # -----------------------------
        pop_subset = pop[["name", "archetype"]].rename(
            columns={"name": "user", "archetype": "citizen_archetype"}
        )

        df_new_s = df_new_s.merge(pop_subset, on="user", how="left")

        # -----------------------------
        # MATRIZ MODAL (como antes)
        # -----------------------------
        modal_matrix = pd.crosstab(
            df_new_s["citizen_archetype"],
            df_new_s["archetype"],
            normalize="index"
        ) * 100

        modal_matrix["Scenario"] = scen
        modal_matrix["S"] = s
        modal_clean.append(modal_matrix.reset_index())

        # -----------------------------
        # DISTRIBUCIÓN GLOBAL
        # -----------------------------
        modal_share = df_new_s["archetype"].value_counts(normalize=True) * 100
        modal_share["Scenario"] = scen
        modal_share["S"] = s
        modal_total.append(modal_share)

        # =====================================================
        # 🔥 TRANSICIÓN 4 → 5
        # =====================================================
        transition_df = df_old_s[["user", "archetype"]].rename(
            columns={"archetype": "mode_old"}
        ).merge(
            df_new_s[["user", "archetype"]].rename(
                columns={"archetype": "mode_new"}
            ),
            on="user",
            how="inner"
        )

        # MATRIZ TRANSICIÓN
        transition_matrix = pd.crosstab(
            transition_df["mode_old"],
            transition_df["mode_new"],
            normalize="index"
        ) * 100

        transition_matrix["Scenario"] = scen
        transition_matrix["S"] = s
        transition_results.append(transition_matrix.reset_index())

        # FLUJO HACIA CAR SHARING
        cs_flow = transition_df[transition_df["mode_new"] == "CS_electric"]
        cs_dist = cs_flow["mode_old"].value_counts(normalize=True) * 100
        cs_dist["Scenario"] = scen
        cs_dist["S"] = s
        cs_flows.append(cs_dist)

# -----------------------------
# CONSOLIDACIÓN
# -----------------------------
modal_df = pd.concat(modal_clean, ignore_index=True)
modal_total_df = pd.concat(modal_total, axis=1).T.fillna(0)
transition_all = pd.concat(transition_results, ignore_index=True)
cs_flow_df = pd.concat(cs_flows, axis=1).T.fillna(0)

# -----------------------------
# ESTÉTICA GLOBAL
# -----------------------------
plt.rcParams.update({
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================
# 🔥 FIGURA 2 — HEATMAP TRANSICIÓN
# ============================================================
for scen in scenario_order:
    scen_df = transition_all[
        (transition_all["Scenario"] == scen) &
        (transition_all["S"] == "s0")
    ]

    pivot = (
        scen_df
        .groupby("mode_old")
        .mean(numeric_only=True)
    )

    pivot = pivot.drop(columns=["S"], errors="ignore")

    # -----------------------------
    # 🔥 RENOMBRAR FILAS Y COLUMNAS
    # -----------------------------
    pivot.index = pivot.index.map(lambda x: MOBILITY_LABELS.get(x, x))
    pivot.columns = [MOBILITY_LABELS.get(col, col) for col in pivot.columns]

    plt.figure(figsize=(6,4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")

    plt.title(f"Transition 4 → 5 modes ({scen})")
    plt.xlabel("New mode")
    plt.ylabel("Old mode")

    plt.tight_layout()
    plt.show()

