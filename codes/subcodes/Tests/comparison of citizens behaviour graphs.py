import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# ==========================================
# CONFIGURACIÓN
# ==========================================

arquetipos = ["Adult male (c_arch_0)", "Adult female (c_arch_1)", "Children (c_arch_2)", "Elder (c_arch_3)", "Youth (c_arch_4)"]
escenarios = ["Baseline Scenario", "EU Trinity", "NECP Essentials", "REPowerEU", "Go RES"]
zonas = ["Annelinn", "Aradas", "Kanaleneiland"]

n_escenarios = len(escenarios)
x = np.arange(len(arquetipos))
width = 0.15

path = "C:/Users/asier.divasson/Downloads/responses/"

fig_width = 368 / 25.4
fig_height = 78 / 25.4

# ==========================================
# COLORES
# ==========================================

colores_annelin       = ["#000000", "#741b47", "#a64d79", "#d5a6bd", "#ead1dc"]
colores_aradas        = ["#000000", "#bf9000", "#f1c232", "#ffe599", "#fff2cc"]
colores_kanaleneiland = ["#000000", "#134f5c", "#45818e", "#a2c4c9", "#d0e0e3"]
colores_zonas         = ["#741b47", "#bf9000", "#134f5c"]

zone_colors = {
    "Annelinn":       colores_annelin,
    "Aradas":         colores_aradas,
    "Kanaleneiland":  colores_kanaleneiland,
}

# ==========================================
# RUTAS — WP3 TRUE (nuevo) y WP3 FALSE (baseline/original)
# ==========================================

BASE_PATH_TRUE  = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (wp3 true)"
BASE_PATH_FALSE = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (wp3 false)"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

# ==========================================
# CARGA DE DATOS
# ==========================================

def load_metrics(base_path):
    citizen_metrics = {}
    vehicle_metrics = {}

    for s in S_FOLDERS:
        for scen in SCENARIOS:
            excel_file = os.path.join(
                base_path,
                s,
                f"{scen}_daily_total_stats_inferred_24.xlsx"
            )

            if not os.path.exists(excel_file):
                print(f"⚠ Not found: {excel_file}")
                continue

            # ---- CITIZENS ----
            df_cit = pd.read_excel(excel_file, sheet_name="cit_by_archetype", decimal=",")
            df_cit = df_cit[df_cit["archetype"].str.startswith("c_arch")]
            df_cit = df_cit.sort_values("archetype")

            citizen_metrics[(s, scen)] = {
                "walk":   df_cit["walk_time__mean"].values,
                "travel": df_cit["travel_time__mean"].values,
                "cost":   df_cit["cost__mean"].values,
                "co2":    df_cit["emissions__mean"].values,
            }

            # ---- VEHICLES ----
            df_veh = pd.read_excel(excel_file, sheet_name="veh_by_archetype", decimal=",")
            df_veh = df_veh[df_veh["archetype"].str.startswith("PC")]
            df_veh = df_veh.sort_values("archetype")

            total_mjkm    = df_veh["mjkm__sum"].sum()
            total_co2     = df_veh["emissions__sum"].sum()
            electric_mjkm = df_veh.loc[df_veh["archetype"] == "PC_electric", "mjkm__sum"].sum()

            vehicle_metrics[(s, scen)] = {
                "mjkm_sum":         total_mjkm,
                "co2_sum":          total_co2,
                "electric_mjkm_sum": electric_mjkm,
            }

    return citizen_metrics, vehicle_metrics


print("Loading wp3 false (baseline)...")
cit_false, veh_false = load_metrics(BASE_PATH_FALSE)

print("Loading wp3 true...")
cit_true, veh_true = load_metrics(BASE_PATH_TRUE)

print("✔ Data loaded")

# ==========================================
# CÁLCULO DE VARIACIÓN PORCENTUAL
# pct_change = (true - false) / |false| * 100
# ==========================================

def pct_change(new_val, base_val):
    """Element-wise % change. Returns 0 where base is 0 to avoid division errors."""
    base = np.where(base_val == 0, np.nan, base_val)
    return (new_val - base_val) / np.abs(base) * 100


# ==========================================
# FUNCIONES DE GRÁFICO
# ==========================================

def crear_grafico_pct(valores_pct, colores, nombre_pdf, description):
    """Barras agrupadas con variación porcentual en el eje Y."""
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i in range(n_escenarios):
        ax.bar(x + i * width, valores_pct[:, i], width,
               label=escenarios[i], color=colores[i])

    # Línea de referencia en 0%
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    ax.set_xlabel("Archetypes")
    ax.set_ylabel(f"Change in {description} [%]")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(arquetipos)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig.savefig(f"{path}{nombre_pdf}", format='pdf', bbox_inches='tight')
    plt.show()


def crear_grafico_lineas_zonas_pct(valores_pct, colores, nombre_pdf, description, barras_error_pct):
    """Líneas por zona con variación porcentual en el eje Y."""
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_esc = np.arange(len(escenarios))
    marcadores = ['o', 's', 'D']

    for i in range(valores_pct.shape[1]):
        ax.errorbar(
            x_esc,
            valores_pct[:, i],
            yerr=barras_error_pct[i],
            label=zonas[i],
            color=colores[i],
            marker=marcadores[i],
            linewidth=1.5,
            capsize=5,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

    ax.set_xlabel("Scenarios")
    ax.set_ylabel(f"Change in {description} [%]")
    ax.set_xticks(x_esc)
    ax.set_xticklabels(escenarios, rotation=0, ha='center')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig.savefig(f"{path}{nombre_pdf}", format='pdf', bbox_inches='tight')
    plt.show()


# ==========================================
# 1️⃣  GRÁFICOS POR ZONA — variación % por arquetipo
# ==========================================

metric_map = {
    "walk":   "Walk time [min]",
    "travel": "Travel time [min]",
    "cost":   "Cost [€]",
    "co2":    "CO2 emission [kg/day]",
}

for zone in SCENARIOS:
    for metric_key, ylabel in metric_map.items():

        # Matriz (arquetipos × escenarios) para cada versión
        base = np.array([cit_false[(s, zone)][metric_key] for s in S_FOLDERS]).T  # shape (5 arch, 5 scen)
        new  = np.array([cit_true [(s, zone)][metric_key] for s in S_FOLDERS]).T

        valores_pct = pct_change(new, base)

        nombre_pdf = f"{zone.lower()}_{metric_key}_pct.pdf"
        crear_grafico_pct(valores_pct, zone_colors[zone], nombre_pdf, ylabel)
        print(f"✔ Generated {nombre_pdf}")


# ==========================================
# 2️⃣  GRÁFICOS COMPARATIVOS POR ZONA — variación % vehículos
# ==========================================

comparative_metrics = {
    "mjkm_sum":          "Total energy consumption [MWh/day]",
    "electric_mjkm_sum": "EV consumption [MWh/day]",
    "co2_sum":           "CO2 emission [tons/day]",
}

for metric_key, ylabel in comparative_metrics.items():

    base = np.array([
        [veh_false[(s, "Annelinn")][metric_key],
         veh_false[(s, "Aradas")][metric_key],
         veh_false[(s, "Kanaleneiland")][metric_key]]
        for s in S_FOLDERS
    ])  # shape (5 scen, 3 zonas)

    new = np.array([
        [veh_true[(s, "Annelinn")][metric_key],
         veh_true[(s, "Aradas")][metric_key],
         veh_true[(s, "Kanaleneiland")][metric_key]]
        for s in S_FOLDERS
    ])

    valores_pct = pct_change(new, base)

    # Barras de error propagadas al porcentaje (mismos valores relativos que antes)
    barras_error_pct = [0.75, 0.86, 0.66]  # en puntos porcentuales

    nombre_pdf = f"comparison_{metric_key}_pct.pdf"
    crear_grafico_lineas_zonas_pct(valores_pct, colores_zonas, nombre_pdf, ylabel, barras_error_pct)
    print(f"✔ Generated {nombre_pdf}")