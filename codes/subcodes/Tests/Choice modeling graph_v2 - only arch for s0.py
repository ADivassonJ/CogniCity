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

S_FOLDERS = [f"s{i}" for i in range(1)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]
SCENARIOS = ["Annelinn"]

# -----------------------------
# DICCIONARIOS DE RENOMBRADO
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

# -----------------------------
# CONTENEDOR GLOBAL
# -----------------------------
modal_clean = []
modal_total = []      # distribución global

# -----------------------------
# LOOP PRINCIPAL
# -----------------------------
for s in S_FOLDERS:
    for scen in SCENARIOS:
        excel_file = os.path.join(BASE_PATH, s, f"{scen}_schedule_vehicle.xlsx")
        parquet_file = os.path.join(DATA_PATH, s, scen, "population", "pop_citizen.parquet")

        if not os.path.exists(excel_file):
            print(f"[SKIP] Missing Excel -> {excel_file}")
            continue

        if not os.path.exists(parquet_file):
            print(f"[SKIP] Missing Parquet -> {parquet_file}")
            continue

        print(f"[OK] Processing -> {s} / {scen}")

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

        # -----------------------------
        # MERGE citizen archetype
        # -----------------------------
        pop_subset = pop[["name", "archetype"]].rename(
            columns={"name": "user", "archetype": "citizen_archetype"}
        )

        df_simplified = df_simplified.merge(pop_subset, on="user", how="left")

        # -----------------------------
        # MATRIZ MODAL
        # -----------------------------
        modal_matrix = pd.crosstab(
            df_simplified["citizen_archetype"],
            df_simplified["archetype"],
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
# COLORES FIJOS PAPER-STYLE
# -----------------------------
colors_aradas = {
    
}

colors_annelinn = {
    
}

colors_kanaleneiland = {
    
}

MOBILITY_LABELS = {
    "UB_diesel": "Public Transportation",
    "PC_petrol": "Combustion car",
    "PC_electric": "Electric car",
    "walk": "Walk",
    "CS_electric": "Carsharing"
}

scenario_order = ["Annelinn", "Aradas", "Kanaleneiland"]

# -----------------------------
# CONFIGURACIÓN DE ESTÉTICA
# -----------------------------
# Puedes ajustar esto globalmente para que todo el documento sea igual
plt.rcParams.update({
    'axes.grid': True,
    'axes.axisbelow': True, # Para que el grid quede detrás de las barras
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================
# ✅ FIGURA 2 → RESUMEN GLOBAL (CORREGIDA)
# ============================================================

# 1. Mapeamos los nombres de S0, S1... a los nombres del Paper
modal_total_df["S"] = modal_total_df["S"].map(rename_scenarios).fillna(modal_total_df["S"])

# 2. DEFINIR EL ORDEN (Esto es lo que faltaba)
s_order = ["Baseline Scenario", "EU Trinity", "NECP Essentials", "REPowerEU", "Go RES"]

for scen in scenario_order:
    colors = colors_annelinn if scen == "Annelinn" else colors_aradas if scen == "Aradas" else colors_kanaleneiland
    
    scen_df = modal_total_df[modal_total_df["Scenario"] == scen]

    pivot = (
        scen_df
        .set_index("S")
        .drop(columns=["Scenario"])
        .reindex(s_order)   # <--- Ahora ya encontrará 's_order'
        .sort_index(axis=1) # Orden alfabético de transportes
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bottom = np.zeros(len(pivot))

    for mode in pivot.columns:
        ax.bar(
            pivot.index, 
            pivot[mode], 
            bottom=bottom, 
            label=MOBILITY_LABELS.get(mode, mode), 
            color=colors.get(mode),
            zorder=3
        )
        bottom += pivot[mode].values

    # Estética estilo tu función
    ax.set_title(f"Modal Share por {scen}")  # ← aquí añades el título
    ax.set_ylabel("Modal share [%]")
    ax.set_xlabel("Simulation Scenarios")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()