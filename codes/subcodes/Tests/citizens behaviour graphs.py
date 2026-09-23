import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_PATH = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"

S_FOLDERS = ["s01", "s02", "s03", "s04", "s05"]

DISTRICTS = ["Kanaleneiland", "Oog in Al"]

ARQUETIPOS = [
    "Adult male",
    "Adult female",
    "Children",
    "Elder",
    "Youth"
]

path = "C:/Users/asier.divasson/Downloads/responses/"

# Tamaño de figura en pulgadas
fig_width = 368 / 25.4
fig_height = 120 / 25.4


# ============================================================
# COLORES
# ============================================================

colores_distritos = {
    "Kanaleneiland": "#134f5c",
    "Oog in Al": "#741b47"
}


# ============================================================
# CARGA DE DATOS
# ============================================================

citizen_metrics = {}

vehicle_metrics = {}


for s in S_FOLDERS:

    for district in DISTRICTS:

        excel_file = os.path.join(
            BASE_PATH,
            s,
            f"{district}_daily_total_stats_inferred_24.xlsx"
        )

        if not os.path.exists(excel_file):
            print(f"⚠ No encontrado: {excel_file}")
            continue

        # ----------------------------------------------------
        # CITIZENS
        # ----------------------------------------------------

        df_cit = pd.read_excel(
            excel_file,
            sheet_name="cit_by_archetype",
            decimal=","
        )

        df_cit = df_cit[
            df_cit["archetype"].str.startswith("c_arch")
        ].copy()

        df_cit = df_cit.sort_values("archetype")

        citizen_metrics[(s, district)] = {
            "walk": df_cit["walk_time__mean"].values,
            "travel": df_cit["travel_time__mean"].values,
            "cost": df_cit["cost__mean"].values,
            "co2": df_cit["emissions__mean"].values,
        }

        # ----------------------------------------------------
        # VEHICLES
        # ----------------------------------------------------

        df_veh = pd.read_excel(
            excel_file,
            sheet_name="veh_by_archetype",
            decimal=","
        )

        df_veh = df_veh[
            df_veh["archetype"].str.startswith("PC")
        ].copy()

        total_mjkm = df_veh["mjkm__sum"].sum()
        total_co2 = df_veh["emissions__sum"].sum()

        electric_mjkm = df_veh.loc[
            df_veh["archetype"] == "PC_electric",
            "mjkm__sum"
        ].sum()

        vehicle_metrics[(s, district)] = {
            "mjkm_sum": total_mjkm,
            "co2_sum": total_co2,
            "electric_mjkm_sum": electric_mjkm
        }


print("✔ Data loaded automatically")


# ============================================================
# FUNCIÓN PARA CREAR VIOLIN PLOTS
# ============================================================

def crear_violin_arquetipos(
    valores_distritos,
    nombre_pdf,
    ylabel,
    logaritmic=False
):
    """
    valores_distritos:
        {
            "Kanaleneiland": [
                [valores de s01 para archetype 1],
                [valores de s02 para archetype 1],
                ...
            ],
            "Oog in Al": [...]
        }

    Cada distrito contiene una distribución por arquetipo.
    """

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    n_arquetipos = len(ARQUETIPOS)
    n_distritos = len(DISTRICTS)

    # Separación entre los dos distritos dentro de cada arquetipo
    separation = 0.35

    posiciones = []

    for i, archetype in enumerate(ARQUETIPOS):

        centro = i + 1

        # Posiciones de los dos violines
        pos_kan = centro - separation
        pos_oog = centro + separation

        posiciones.append(
            [pos_kan, pos_oog]
        )

    # --------------------------------------------------------
    # Dibujar cada distrito
    # --------------------------------------------------------

    for district_index, district in enumerate(DISTRICTS):

        data = valores_distritos[district]

        district_positions = [
            posiciones[i][district_index]
            for i in range(n_arquetipos)
        ]

        violin = ax.violinplot(
            data,
            positions=district_positions,
            widths=0.55,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )

        # Colorear cuerpos
        for body in violin["bodies"]:
            body.set_facecolor(
                colores_distritos[district]
            )
            body.set_edgecolor(
                colores_distritos[district]
            )
            body.set_alpha(0.65)

        # Líneas del violin
        for key in [
            "cmeans",
            "cmedians",
            "cbars",
            "cmins",
            "cmaxes"
        ]:
            if key in violin:
                violin[key].set_color(
                    colores_distritos[district]
                )
                violin[key].set_linewidth(1)

    # --------------------------------------------------------
    # Ejes
    # --------------------------------------------------------

    ax.set_xlabel("Archetypes")
    ax.set_ylabel(ylabel)

    ax.set_xticks(
        np.arange(1, n_arquetipos + 1)
    )

    ax.set_xticklabels(ARQUETIPOS)

    if logaritmic:
        ax.set_yscale("log")

    # --------------------------------------------------------
    # Leyenda
    # --------------------------------------------------------

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=colores_distritos[district],
            edgecolor=colores_distritos[district],
            alpha=0.65,
            label=district
        )
        for district in DISTRICTS
    ]

    ax.legend(
        handles=legend_elements,
        loc="best"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.7
    )

    plt.tight_layout()

    fig.savefig(
        os.path.join(path, nombre_pdf),
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# MAPA DE MÉTRICAS
# ============================================================

metric_map = {
    "walk": (
        "walk_time__mean",
        "Walk time [min]"
    ),

    "travel": (
        "travel_time__mean",
        "Travel time [min]"
    ),

    "cost": (
        "cost__mean",
        "Cost [€]"
    ),

    "co2": (
        "emissions__mean",
        "CO2 emission [g/day]"
    )
}


# ============================================================
# CREAR DISTRIBUCIONES
# ============================================================

for metric_key, (metric_col, ylabel) in metric_map.items():

    valores_distritos = {}

    for district in DISTRICTS:

        # ----------------------------------------------------
        # Aquí juntamos s01, s02, s03, s04 y s05
        # ----------------------------------------------------

        datos_arquetipos = []

        for archetype_index in range(len(ARQUETIPOS)):

            valores_arquetipo = []

            for s in S_FOLDERS:

                key = (s, district)

                if key not in citizen_metrics:
                    continue

                valor = citizen_metrics[key][metric_key][
                    archetype_index
                ]

                valores_arquetipo.append(valor)

            datos_arquetipos.append(
                valores_arquetipo
            )

        valores_distritos[district] = datos_arquetipos

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------

    nombre_pdf = f"comparison_{metric_key}_violin.pdf"

    crear_violin_arquetipos(
        valores_distritos,
        nombre_pdf,
        ylabel
    )

    print(f"✔ Generated {nombre_pdf}")

# ============================================================
# VIOLIN PLOT: WALK TIME + TRAVEL TIME
# ============================================================

valores_distritos_total_time = {}

for district in DISTRICTS:

    datos_arquetipos = []

    for archetype_index in range(len(ARQUETIPOS)):

        valores_arquetipo = []

        for s in S_FOLDERS:

            key = (s, district)

            if key not in citizen_metrics:
                continue

            walk = citizen_metrics[key]["walk"][archetype_index]
            travel = citizen_metrics[key]["travel"][archetype_index]

            # Suma de walk + travel
            total_time = walk + travel

            valores_arquetipo.append(total_time)

        datos_arquetipos.append(valores_arquetipo)

    valores_distritos_total_time[district] = datos_arquetipos


# ============================================================
# CREAR GRÁFICO
# ============================================================

crear_violin_arquetipos(
    valores_distritos_total_time,
    "comparison_total_time_violin.pdf",
    "Total travel time [min]"
)

print("✔ Generated comparison_total_time_violin.pdf")