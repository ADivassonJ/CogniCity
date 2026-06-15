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
BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (0)"
DATA_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data"

S_FOLDERS = [f"s{i}" for i in range(5)]
S_FOLDERS = [f"s{i}" for i in range(1)]

SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

DAY_ORDER = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

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

MOBILITY_LABELS = {
    "UB_diesel": "Public Transportation",
    "PC_petrol": "Combustion car",
    "PC_electric": "Electric car",
    "walk": "Walk"
}

colors_annelinn = {
    "walk": "#d5a6bd",
    "UB_diesel": "#a64d79",
    "PC_petrol": "#741b47",
    "PC_electric": "#49102e",
}
colors_aradas = {
    "walk": "#ffe599",
    "UB_diesel": "#f1c232",
    "PC_petrol": "#bf9000",
    "PC_electric": "#574100",
}
colors_kanaleneiland = {
    "walk": "#a2c4c9",
    "UB_diesel": "#45818e",
    "PC_petrol": "#134f5c",
    "PC_electric": "#072329",
}

plt.rcParams.update({
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# -----------------------------
# CONTENEDOR GLOBAL
# -----------------------------
modal_clean = []

# -----------------------------
# LOOP PRINCIPAL
# -----------------------------
for s in S_FOLDERS:
    for scen in SCENARIOS:
        excel_file = os.path.join(BASE_PATH, s, f"{scen}_schedule_vehicle.xlsx")
        parquet_file = os.path.join(DATA_PATH, s, scen, "population", "pop_citizen.parquet")

        if not os.path.exists(excel_file) or not os.path.exists(parquet_file):
            print(f"[SKIP] Missing files -> {s} / {scen}")
            continue

        print(f"[OK] Processing -> {s} / {scen}")

        df = pd.read_excel(excel_file)
        pop = pd.read_parquet(parquet_file)

        required_cols = {"user", "archetype", "in", "day"}
        if not required_cols.issubset(df.columns):
            print(f"[SKIP] Missing columns in {excel_file}")
            continue

        # -----------------------------
        # SIMPLIFICACIÓN POR USUARIO + DÍA
        # → primer viaje de cada usuario por día
        # -----------------------------
        df_simplified = (
            df.loc[df.groupby(["user", "day"])["in"].idxmin()]
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
        # MATRIZ MODAL por día y citizen_archetype
        # -----------------------------
        modal_matrix = (
            df_simplified
            .groupby(["citizen_archetype", "day", "archetype"])
            .size()
            .reset_index(name="count")
        )

        # Normalizar por citizen_archetype + day
        modal_matrix["pct"] = (
            modal_matrix["count"]
            / modal_matrix.groupby(["citizen_archetype", "day"])["count"].transform("sum")
            * 100
        )

        modal_matrix["Scenario"] = scen
        modal_matrix["S"] = s
        modal_clean.append(modal_matrix)

# -----------------------------
# CONSOLIDACIÓN
# -----------------------------
modal_df = pd.concat(modal_clean, ignore_index=True)

# -----------------------------
# PLOT POR ESCENARIO
# -----------------------------
for scen in SCENARIOS:
    colors = (colors_annelinn if scen == "Annelinn"
              else colors_aradas if scen == "Aradas"
              else colors_kanaleneiland)

    scen_df = modal_df[modal_df["Scenario"] == scen].copy()

    # Renombrar citizen_archetype
    scen_df["citizen_archetype"] = (
        scen_df["citizen_archetype"].map(rename_archetypes).fillna(scen_df["citizen_archetype"])
    )

    # =====================================================
    # GLOBAL: agregar todos los arquetipos juntos por día
    # =====================================================
    global_df = (
        scen_df.groupby(["day", "archetype"])["count"]
        .sum()
        .reset_index()
    )
    
    global_df["pct"] = (
        global_df["count"]
        / global_df.groupby("day")["count"].transform("sum")
        * 100
    )
    global_df["citizen_archetype"] = "GLOBAL"

    scen_df = pd.concat([scen_df, global_df], ignore_index=True)

    # =====================================================
    # ORDEN DE ARQUETIPOS: primero individuales, luego GLOBAL
    # =====================================================
    archetype_order = (
        [v for v in rename_archetypes.values() if v in scen_df["citizen_archetype"].unique()]
        + ["GLOBAL"]
    )

    n_archetypes = len(archetype_order)
    fig, axes = plt.subplots(
        1, n_archetypes,
        figsize=(fig_width, fig_height),
        sharey=True
    )

    # Días disponibles en orden
    days_available = [d for d in DAY_ORDER if d in scen_df["day"].unique()]
    # Etiquetas cortas para el eje X
    day_labels = days_available  

    modes = [m for m in ["walk", "UB_diesel", "PC_petrol", "PC_electric"]
         if m in modal_df["archetype"].unique()]

    handles, labels = [], []

    for ax, arch in zip(axes, archetype_order):
        arch_df = scen_df[scen_df["citizen_archetype"] == arch].copy()

        # Pivot: filas = días, columnas = modos
        pivot = (
            arch_df.groupby(["day", "archetype"])["pct"]
            .mean()
            .unstack(fill_value=0)
            .reindex(days_available, fill_value=0)
        )

        # Asegurar que todas las columnas de modo existen
        for m in modes:
            if m not in pivot.columns:
                pivot[m] = 0
        pivot = pivot[modes]

        bottom = np.zeros(len(pivot))

        for mode in modes:
            bars = ax.bar(
                day_labels,
                pivot[mode].values,
                bottom=bottom,
                label=MOBILITY_LABELS.get(mode, mode),
                color=colors.get(mode),
                zorder=3,
                width=0.6
            )
            bottom += pivot[mode].values

        ax.set_title(arch, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Day")
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        if ax == axes[0]:
            ax.set_ylabel("Modal share [%]")

        # Recoger leyenda del primer subplot
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    fig.suptitle(f"Modal share by day and archetype — {scen}", fontsize=10, fontweight="bold", y=1.01)
    fig.legend(handles, labels, title="Mode", bbox_to_anchor=(1.01, 0.9),
               loc="upper left", frameon=True, fontsize=8)

    plt.tight_layout()
    plt.show()