import numpy as np
import matplotlib.pyplot as plt

# Datos comunes
arquetipos = ["c_arch_0", "c_arch_1", "c_arch_2", "c_arch_3", "c_arch_4"]
escenarios = ["Base case-scenario", "EU Trinity", "NECP Essentials", "REPowerEU", "Go RES"]
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

# ----- Electricity consumption -----
valores = np.array([
    [2.470,  .554,  4.187],
    [2.011,  .654,  3.080],
    [12.035, 4.492, 21.182],
    [12.469, 4.850, 18.604],
    [9.901,  4.185, 14.355]
])

barras_error = [0.0074716, 0.0085729, 0.0066398]  # 0.75%, 0.86%, 0.66%
crear_grafico_lineas_zonas(valores, colores_zonas, "Comparison of electricity consumption.pdf", "Consumption [MWh/day]", barras_error)

# ----- Comparison of energy consumption -----
valores = np.array([
    [81.842, 2187.576, 6722.996],
    [80.989, 2162.827, 6776.578],
    [51.419, 1034.812, 3250.047],
    [25.523, 512.043, 1803.410],
    [9.500, 231.487, 620.112]
])

barras_error = [0.0355799, 0.0375963, 0.0324568]  # 0.75%, 0.86%, 0.66%
crear_grafico_lineas_zonas(valores, colores_zonas, "Comparison of energy consumption.pdf", "Consumption [MWh/day]", barras_error, True)


# ----- CO2 emission -----
valores = np.array([
    [19.573716, 7.874719, 24.198599],
    [19.486697, 7.785523, 24.392602],
    [8.711785, 3.720833, 11.678988],
    [4.196039, 1.838503, 6.473673],
    [2.872845, .829166, 2.218048]
])

barras_error = [0.0355799, 0.0375963, 0.0324568]  # 0.75%, 0.86%, 0.66%
crear_grafico_lineas_zonas(valores, colores_zonas, "CO2 emission.pdf", "CO2 emission [tons/day]", barras_error)

# ----- ann_walk -----
valores = np.array([
    [0, 0.00000, 0.00000, 0.00000, 0.00000],
    [0.045104, 0.08949, 0.32661, 1.02491, 6.68060],
    [37.04237, 38.11252, 37.59464, 43.58866, 48.49672],
    [18.12158, 17.75333, 28.89758, 35.39679, 40.97440],
    [0.014625, 0.00000, 0.02217, 0.23955, 1.48025]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100  

crear_grafico(valores, colores_annelin, "ann_walk.pdf", "Walk time [min]", errores_pct)

# ----- ann_mot_tra -----
valores = np.array([
    [77.96014, 78.06, 76.07, 72.71, 68.04],
    [75.51887, 75.86, 73.67, 70.96, 66.60],
    [34.29097, 33.93, 33.48, 31.38, 29.32],
    [39.3689, 39.46, 38.12, 37.36, 35.54],
    [73.10321, 73.41, 71.46, 69.89, 63.09]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100  

crear_grafico(valores, colores_annelin, "ann_mot_tra.pdf", "travel time [min]",errores_pct)

# ----- ann_costs -----
valores = np.array([
    [16.90821, 18.85, 12.38, 9.71, 0.50],
    [13.39092, 16.51, 10.58, 8.31, 0.74],
    [9.185368, 9.14, 6.50, 5.02, 0.00],
    [8.964833, 9.80, 6.54, 5.08, 0.21],
    [16.81137, 18.22, 12.79, 10.00, 0.17]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.3591/14.6398, 0.3462/9.3833, 0.3195/8.0798, 0.4210/6.6147, 0.2966/14.7502])

crear_grafico(valores, colores_annelin, "ann_costs.pdf", "cost [€]", errores_pct)

# ----- ann_CO2 -----
valores = np.array([
    [2.702056, 2.716919, 1.508877, .921628, .55018],
    [3.560383, 3.564637, 1.790105, 1.025214, .5673612],
    [.5545351, .5497063, .3825624, .2963819, .2210887],
    [1.251956, 1.264421, .610722, .430941, .278881],
    [2.090307, 2.015528, 1.106727, .8244485, .5012588]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([32.7956/628.7314, 30.9171/627.9426, 6.1525/150.4788, 33.9092/285.6389, 28.8523/550.7885])

crear_grafico(valores, colores_annelin, "ann_co2.pdf", "CO2 emission [kg/day]", errores_pct)

# ----- ara_walk -----
valores = np.array([
    [29.21234328, 28.65, 44.75, 51.36, 58.74],
    [37.10882631, 38.05, 58.00, 68.41, 77.35],
    [22.23559607, 21.59, 21.26, 21.18, 20.62],
    [21.53454413, 20.75, 35.12, 42.51, 48.80],
    [24.32974087, 23.60, 35.78, 42.67, 49.82]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100  

crear_grafico(valores, colores_aradas, "ara_walk.pdf", "Walk time [min]", errores_pct)

# ----- ara_mot_tra -----
valores = np.array([
    [10.50748818, 10.13, 10.00, 8.92, 8.16],
    [13.14897464, 13.39, 12.56, 11.60, 10.51],
    [20.29476374, 20.47, 19.58, 18.52, 17.63],
    [7.980279056, 7.95, 7.41, 7.05, 6.54],
    [11.05150289, 11.22, 9.96, 10.21, 8.75]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100  

crear_grafico(valores, colores_aradas, "ara_mot_tra.pdf", "travel time [min]", errores_pct)

# ----- ara_costs -----
valores = np.array([
    [0.719665383, 0.72, 0.79, 0.59, 0.22],
    [0.877984406, 0.97, 0.90, 0.77, 0.25],
    [5.505995704, 5.48, 3.81, 2.91, 0.00],
    [0.596547572, 0.55, 0.59, 0.47, 0.16],
    [1.141327725, 1.24, 1.01, 0.89, 0.19]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.3591/14.6398, 0.3462/9.3833, 0.3195/8.0798, 0.4210/6.6147, 0.2966/14.7502])

crear_grafico(valores, colores_aradas, "ara_costs.pdf", "cost [€]", errores_pct)

# ----- ara_CO2 -----
valores = np.array([
    [0.893645077, 0.8590180637, 0.4434137054, 0.2073958369, 0.09469305483],
    [1.148427751, 1.149562875, 0.5472967791, 0.2741389721, 0.1315315162],
    [0.3260568771, 0.3278727702, 0.22641328, 0.1797884082, 0.1271973555],
    [0.700836151, 0.6955659075, 0.3247988226, 0.1681129778, 0.07645254694],
    [0.7983543778, 0.812821233, 0.3425290476, 0.209480296, 0.08780061338]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([32.7956/628.7314, 30.9171/627.9426, 6.1525/150.4788, 33.9092/285.6389, 28.8523/550.7885])

crear_grafico(valores, colores_aradas, "ara_co2.pdf", "CO2 emission [kg/day]", errores_pct)

# ----- kan_walk -----
valores = np.array([
    [0.007936, 0.03, 0.00, 0.01, 2.91],
    [15.24873, 14.23, 18.12, 21.65, 62.59],
    [50.68499, 52.18, 52.15, 49.58, 64.99],
    [51.04167, 48.74, 52.09, 55.55, 46.74],
    [2.014273, 1.42, 1.94, 1.69, 10.14]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100  

crear_grafico(valores, colores_kanaleneiland, "kan_walk.pdf", "Walk time [min]", errores_pct)

# ----- kan_mot_tra -----
valores = np.array([
    [77.96014, 78.06, 76.07, 72.71, 68.04],
    [75.51887, 75.86, 73.67, 70.96, 66.60],
    [34.29097, 33.93, 33.48, 31.38, 29.32],
    [39.3689, 39.46, 38.12, 37.36, 35.54],
    [73.10321, 73.41, 71.46, 69.89, 63.09]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.31, 0.48, 1.21, 2.09, 0.48]) / 100  

crear_grafico(valores, colores_kanaleneiland, "kan_mot_tra.pdf", "travel time [min]", errores_pct)

# ----- kan_costs -----
valores = np.array([
    [16.90821, 18.85, 12.38, 9.71, 0.50],
    [13.39092, 16.51, 10.58, 8.31, 0.74],
    [9.185368, 9.14, 6.50, 5.02, 0.00],
    [8.964833, 9.80, 6.54, 5.08, 0.21],
    [16.81137, 18.22, 12.79, 10.00, 0.17]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([0.3591/14.6398, 0.3462/9.3833, 0.3195/8.0798, 0.4210/6.6147, 0.2966/14.7502])

crear_grafico(valores, colores_kanaleneiland, "kan_costs.pdf", "cost [€]", errores_pct)

# ----- kan_CO2 -----
valores = np.array([
    [2.592864, 2.597628, 1.472764, 0.9879613, 0.564957773],
    [3.526475, 3.555813, 1.931369, 1.155861, 0.5898487978],
    [0.490608, 0.478368, 0.3645678, 0.2822337, 0.1848069646],
    [1.072849, 1.090605, 0.6373819, 0.4398749, 0.2794700617],
    [2.088417, 2.057353, 1.19371, 0.910399, 0.4908111899]
])

# Error porcentual por arquetipo (convertido a decimal)
errores_pct = np.array([32.7956/628.7314, 30.9171/627.9426, 6.1525/150.4788, 33.9092/285.6389, 28.8523/550.7885])

crear_grafico(valores, colores_kanaleneiland, "kan_co2.pdf", "CO2 emission [kg/day]", errores_pct)




