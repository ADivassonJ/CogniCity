import numpy as np
import matplotlib.pyplot as plt

# Datos comunes
arquetipos = ["c_arch_0", "c_arch_1", "c_arch_2", "c_arch_3", "c_arch_4"]
escenarios = ["Base case-scenario", "EU Trinity", "NECP Essentials", "REPowerEU", "Go RES"]

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100  

n_escenarios = len(escenarios)
x = np.arange(len(arquetipos))
width = 0.15
path = "C:/Users/asier.divasson/Downloads/responses/"

# Convertir tamaño a pulgadas (1 pulgada = 25.4 mm)
fig_width = 368 / 25.4
fig_height = 78 / 25.4

def crear_grafico(valores, colores, nombre_pdf):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Calcular error absoluto a partir del porcentaje
    errores_abs = valores * errores_pct[:, np.newaxis]
    
    # Dibujar barras
    for i in range(n_escenarios):
        ax.bar(x + i*width, valores[:, i], width, label=escenarios[i], 
               yerr=errores_abs[:, i], capsize=5, color=colores[i])
    
    # Etiquetas y formato
    ax.set_xlabel("Archetypes")
    ax.set_ylabel("Walk time [min]")
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(arquetipos)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(f"{path}{nombre_pdf}", format='pdf', bbox_inches='tight')
    plt.show()

colores_annelin =       ["#000000", "#741b47", "#a64d79", "#d5a6bd", "#ead1dc"]  
colores_aradas =        ["#000000", "#bf9000", "#f1c232", "#ffe599", "#fff2cc"]  
colores_kanaleneiland = ["#000000", "#134f5c", "#45818e", "#a2c4c9", "#d0e0e3"]   

# ----- ann_walk -----
valores = np.array([
    [0, 0.00000, 0.00000, 0.00000, 0.00000],
    [0.045104, 0.08949, 0.32661, 1.02491, 6.68060],
    [37.04237, 38.11252, 37.59464, 43.58866, 48.49672],
    [18.12158, 17.75333, 28.89758, 35.39679, 40.97440],
    [0.014625, 0.00000, 0.02217, 0.23955, 1.48025]
])

crear_grafico(valores, colores_annelin, "ann_walk.pdf")

# ----- ann_mot_tra -----
valores = np.array([
    [77.96014, 78.06, 76.07, 72.71, 68.04],
    [75.51887, 75.86, 73.67, 70.96, 66.60],
    [34.29097, 33.93, 33.48, 31.38, 29.32],
    [39.3689, 39.46, 38.12, 37.36, 35.54],
    [73.10321, 73.41, 71.46, 69.89, 63.09]
])

crear_grafico(valores, colores_annelin, "ann_mot_tra.pdf")

# ----- ann_costs -----
valores = np.array([
    [16.90821, 18.85, 12.38, 9.71, 0.50],
    [13.39092, 16.51, 10.58, 8.31, 0.74],
    [9.185368, 9.14, 6.50, 5.02, 0.00],
    [8.964833, 9.80, 6.54, 5.08, 0.21],
    [16.81137, 18.22, 12.79, 10.00, 0.17]
])

crear_grafico(valores, colores_annelin, "ann_costs.pdf")

# ----- ara_walk -----
valores = np.array([
    [29.21234328, 28.65, 44.75, 51.36, 58.74],
    [37.10882631, 38.05, 58.00, 68.41, 77.35],
    [22.23559607, 21.59, 21.26, 21.18, 20.62],
    [21.53454413, 20.75, 35.12, 42.51, 48.80],
    [24.32974087, 23.60, 35.78, 42.67, 49.82]
])

crear_grafico(valores, colores_aradas, "ara_walk.pdf")


# ----- ara_mot_tra -----
valores = np.array([
    [10.50748818, 10.13, 10.00, 8.92, 8.16],
    [13.14897464, 13.39, 12.56, 11.60, 10.51],
    [20.29476374, 20.47, 19.58, 18.52, 17.63],
    [7.980279056, 7.95, 7.41, 7.05, 6.54],
    [11.05150289, 11.22, 9.96, 10.21, 8.75]
])

crear_grafico(valores, colores_aradas, "ara_mot_tra.pdf")

# ----- ara_costs -----
valores = np.array([
    [0.719665383, 0.72, 0.79, 0.59, 0.22],
    [0.877984406, 0.97, 0.90, 0.77, 0.25],
    [5.505995704, 5.48, 3.81, 2.91, 0.00],
    [0.596547572, 0.55, 0.59, 0.47, 0.16],
    [1.141327725, 1.24, 1.01, 0.89, 0.19]
])

crear_grafico(valores, colores_aradas, "ara_cost.pdf")


# ----- kan_walk -----
valores = np.array([
    [0.007936, 0.03, 0.00, 0.01, 2.91],
    [15.24873, 14.23, 18.12, 21.65, 62.59],
    [50.68499, 52.18, 52.15, 49.58, 64.99],
    [51.04167, 48.74, 52.09, 55.55, 46.74],
    [2.014273, 1.42, 1.94, 1.69, 10.14]
])

crear_grafico(valores, colores_kanaleneiland, "kan_walk.pdf")

# ----- kan_mot_tra -----
valores = np.array([
    [77.96014, 78.06, 76.07, 72.71, 68.04],
    [75.51887, 75.86, 73.67, 70.96, 66.60],
    [34.29097, 33.93, 33.48, 31.38, 29.32],
    [39.3689, 39.46, 38.12, 37.36, 35.54],
    [73.10321, 73.41, 71.46, 69.89, 63.09]
])

crear_grafico(valores, colores_kanaleneiland, "kan_mot_tra.pdf")


# ----- kan_cost -----
valores = np.array([
    [16.90821, 18.85, 12.38, 9.71, 0.50],
    [13.39092, 16.51, 10.58, 8.31, 0.74],
    [9.185368, 9.14, 6.50, 5.02, 0.00],
    [8.964833, 9.80, 6.54, 5.08, 0.21],
    [16.81137, 18.22, 12.79, 10.00, 0.17]
])
crear_grafico(valores, colores_kanaleneiland, "kan_cost.pdf")
