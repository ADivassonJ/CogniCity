import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Datos comunes
arquetipos = ["Adult male (c_arch_0)", "Adult female (c_arch_1)", "Children (c_arch_2)", "Elder (c_arch_3)", "Youth (c_arch_4)"]
escenarios = ["Baseline Scenario", "EU Trinity", "NECP Essentials", "REPowerEU", "Go RES"]
zonas = ["Annelinn", "Aradas", "Kanaleneiland"]

# Escenarios futuros para el eje X
escenarios_futuros = escenarios[1:] 

path = "C:/Users/asier.divasson/Downloads/responses/"
BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

S_FOLDERS = [f"s{i}" for i in range(5)]
SCENARIOS = ["Annelinn", "Aradas", "Kanaleneiland"]

# Convertir tamaño a pulgadas (1 pulgada = 25.4 mm)
fig_width = 368 / 25.4
fig_height = 78 / 25.4

colores_zonas = ["#741b47", "#bf9000", "#134f5c"]

def crear_grafico_lineas_reduccion(valores, colores, nombre_pdf, description, barras_error, logaritmic=False):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    x = np.arange(len(escenarios_futuros))
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
            capsize=5  
        )
    
    ax.set_xlabel("Future Scenarios")
    ax.set_ylabel(description)
    ax.set_xticks(x)
    ax.set_xticklabels(escenarios_futuros, rotation=0, ha='center')
    
    if logaritmic:
        ax.set_yscale('log')

    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(f"{path}{nombre_pdf}", format='pdf', bbox_inches='tight')
    plt.show()


# ==========================================
# CARGA DE DATOS (Solo Vehículos - CO2)
# ==========================================
vehicle_metrics = {}

for s in S_FOLDERS:
    for scen in SCENARIOS:
        excel_file = os.path.join(BASE_PATH, s, f"{scen}_daily_total_stats_inferred_24.xlsx")

        if not os.path.exists(excel_file):
            continue

        df_veh = pd.read_excel(excel_file, sheet_name="veh_by_archetype", decimal=",")
        df_veh = df_veh[df_veh["archetype"].str.startswith("PC")]
        
        total_co2 = df_veh["emissions__sum"].sum()

        vehicle_metrics[(s, scen)] = {
            "co2_sum": total_co2
        }

print("✔ Data loaded automatically (Only CO2 sum)")


# ==========================================
# CÁLCULO DE REDUCCIÓN PORCENTUAL
# ==========================================

# 1. Matriz original de valores absolutos (5 escenarios x 3 zonas)
valores_absolutos = np.array([
    [
        vehicle_metrics[(s, "Annelinn")]["co2_sum"],
        vehicle_metrics[(s, "Aradas")]["co2_sum"],
        vehicle_metrics[(s, "Kanaleneiland")]["co2_sum"],
    ]
    for s in S_FOLDERS
])

# 2. Separar el Baseline (s0) y los escenarios futuros (s1 a s4)
baseline = valores_absolutos[0, :]       
valores_futuros = valores_absolutos[1:, :] 

# 3. Cálculo de la reducción en PORCENTAJE (%)
valores_reduccion_pct = ((baseline - valores_futuros) / baseline) * 100


# ==========================================
# 🔥 NUEVO: MOSTRAR DATOS EN TEXTO (PRINT)
# ==========================================
print("\n" + "="*60)
print("   REPORTE DE REDUCCIÓN DE CO2 (%) RESPECTO AL BASELINE")
print("="*60)

for j, zona in enumerate(zonas):
    print(f"\n📍 ZONA: {zona}")
    print("-" * 30)
    for i, scen_futuro in enumerate(escenarios_futuros):
        porcentaje = valores_reduccion_pct[i, j]
        print(f"  ↳ {scen_futuro}: {porcentaje:.2f}% de reducción")

print("\n" + "="*60)


# ==========================================
# CONFIGURACIÓN Y GENERACIÓN DE GRÁFICO
# ==========================================
errores_absolutos_base = np.array([0.0075, 0.0086, 0.0066])
barras_error_pct = (errores_absolutos_base / baseline) * 100

nombre_pdf = "comparison_co2_reduction_percentage.pdf"
ylabel = "CO2 emission reduction [%]"

crear_grafico_lineas_reduccion(
    valores_reduccion_pct,
    colores_zonas,
    nombre_pdf,
    ylabel,
    barras_error_pct,
    logaritmic=False
)

print(f"✔ Generated {nombre_pdf}")