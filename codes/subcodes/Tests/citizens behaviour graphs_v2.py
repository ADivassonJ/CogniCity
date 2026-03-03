import numpy as np
import matplotlib.pyplot as plt

# Datos comunes
arquetipos = ["Adult male (c_arch_0)", "Adult female (c_arch_1)", "Children (c_arch_2)", "Elder (c_arch_3)", "Youth (c_arch_4)"]
escenarios = ["Baseline Scenario", "EU Trinity", "NECP Essentials", "REPowerEU", "Go RES"]
zonas = ["Annelinn", "Aradas", "Kanaleneiland"]

n_escenarios = len(escenarios)
x = np.arange(len(arquetipos))
width = 0.15
path = "C:/Users/asier.divasson/Downloads/responses/"

# Convertir tamaño a pulgadas (1 pulgada = 25.4 mm)
fig_width = 368 / 25.4
fig_height = 78 / 25.4

def crear_grafico(valores, colores, nombre_pdf, description, errores_pct):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Calcular error absoluto a partir del porcentaje
    errores_abs = valores * errores_pct[:, np.newaxis]
    
    # Dibujar barras
    for i in range(n_escenarios):
        ax.bar(x + i*width, valores[:, i], width, label=escenarios[i], 
               yerr=errores_abs[:, i], capsize=5, color=colores[i])
    
    # Etiquetas y formato
    ax.set_xlabel("Archetypes")
    ax.set_ylabel(description)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(arquetipos)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(f"{path}{nombre_pdf}", format='pdf', bbox_inches='tight')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def crear_grafico_lineas_zonas(valores, colores, nombre_pdf, description, barras_error, logaritmic=False):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    x = np.arange(len(escenarios))
    
    # Definimos las barras de error para cada zona (en valores absolutos)
    # Si los valores están en porcentaje, conviértelo a la misma escala que 'valores'
    marcadores = ['o', 's', 'D']  # círculo, cuadrado, diamante
    
    for i in range(valores.shape[1]):
        ax.errorbar(
            x,
            valores[:, i],
            yerr=barras_error[i],
            label=zonas[i],
            color=colores[i],
            marker=marcadores[i],
            linewidth=1.5,
            capsize=5  # para añadir "tapas" a las barras de error
        )
    
    ax.set_xlabel("Scenarios")
    ax.set_ylabel(description)
    ax.set_xticks(x)
    ax.set_xticklabels(escenarios, rotation=0, ha='center')
    
    if logaritmic:
        ax.set_yscale('log')

    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(f"{path}{nombre_pdf}", format='pdf', bbox_inches='tight')
    plt.show()



colores_annelin =       ["#000000", "#741b47", "#a64d79", "#d5a6bd", "#ead1dc"]  
colores_aradas =        ["#000000", "#bf9000", "#f1c232", "#ffe599", "#fff2cc"]  
colores_kanaleneiland = ["#000000", "#134f5c", "#45818e", "#a2c4c9", "#d0e0e3"]   



colores_zonas = ["#741b47", "#bf9000", "#134f5c"]


import os
import pandas as pd

BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

citizen_metrics = {}
vehicle_metrics = {}

for s in S_FOLDERS:
    for scen in SCENARIOS:

        excel_file = os.path.join(
            BASE_PATH,
            s,
            scen,
            f"{scen}_daily_total_stats_inferred_24.xlsx"
        )

        if not os.path.exists(excel_file):
            continue

        # -------- CITIZENS --------
        df_cit = pd.read_excel(excel_file, sheet_name="cit_by_archetype", decimal=",")

        df_cit = df_cit[df_cit["archetype"].str.startswith("c_arch")]
        df_cit = df_cit.sort_values("archetype")

        citizen_metrics[(s, scen)] = {
            "walk": df_cit["walk_time__mean"].values,
            "travel": df_cit["travel_time__mean"].values,
            "cost": df_cit["cost__mean"].values,
            "co2": df_cit["emissions__mean"].values,
        }

        # -------- VEHICLES --------
        df_veh = pd.read_excel(excel_file, sheet_name="veh_by_archetype", decimal=",")

        df_veh = df_veh[df_veh["archetype"].str.startswith("PC")]
        df_veh = df_veh.sort_values("archetype")

        df_veh = pd.read_excel(excel_file, sheet_name="veh_by_archetype", decimal=",")

        df_veh = df_veh[df_veh["archetype"].str.startswith("PC")]
        df_veh = df_veh.sort_values("archetype")

        total_mjkm = df_veh["mjkm__sum"].sum()
        total_co2 = df_veh["emissions__sum"].sum()

        electric_mjkm = df_veh.loc[
            df_veh["archetype"] == "PC_electric",
            "mjkm__sum"
        ].sum()

        vehicle_metrics[(s, scen)] = {
            "mjkm_sum": total_mjkm,
            "co2_sum": total_co2,
            "electric_mjkm_sum": electric_mjkm
        }

print("✔ Data loaded automatically")

# ==========================================
# AUTO-GENERADOR DE TODOS LOS GRÁFICOS
# ==========================================

metric_map = {
    "walk": ("walk_time__mean", "Walk time [min]"),
    "travel": ("travel_time__mean", "Travel time [min]"),
    "cost": ("cost__mean", "Cost [€]"),
    "co2": ("emissions__mean", "CO2 emission [kg/day]"),
}

zone_colors = {
    "Annelinn": colores_annelin,
    "Aradas": colores_aradas,
    "Kanaleneiland": colores_kanaleneiland,
}

# -----------------------------
# 1️⃣ GRÁFICOS POR ZONA (barras)
# -----------------------------

for zone in SCENARIOS:

    for metric_key, (metric_col, ylabel) in metric_map.items():

        valores = np.array([
            citizen_metrics[(s, zone)][metric_key]
            for s in S_FOLDERS
        ]).T

        errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100

        nombre_pdf = f"{zone.lower()}_{metric_key}.pdf"

        crear_grafico(
            valores,
            zone_colors[zone],
            nombre_pdf,
            ylabel,
            errores_pct
        )

        print(f"✔ Generated {nombre_pdf}")


# -----------------------------
# 2️⃣ GRÁFICOS COMPARATIVOS POR ZONA (líneas)
# -----------------------------

comparative_metrics = {
    "mjkm_sum": ("Total energy consumption [MWh/day]", False),
    "electric_mjkm_sum": ("Electric vehicle consumption [MWh/day]", False),
    "co2_sum": ("CO2 emission [tons/day]", False),
}

for metric_key, (ylabel, log_scale) in comparative_metrics.items():

    valores = np.array([
        [
            vehicle_metrics[(s, "Annelinn")][metric_key],
            vehicle_metrics[(s, "Aradas")][metric_key],
            vehicle_metrics[(s, "Kanaleneiland")][metric_key],
        ]
        for s in S_FOLDERS
    ])

    barras_error = [0.0075, 0.0086, 0.0066]  # puedes parametrizar si quieres

    nombre_pdf = f"comparison_{metric_key}.pdf"

    crear_grafico_lineas_zonas(
        valores,
        colores_zonas,
        nombre_pdf,
        ylabel,
        barras_error,
        logaritmic=log_scale
    )

    print(f"✔ Generated {nombre_pdf}")