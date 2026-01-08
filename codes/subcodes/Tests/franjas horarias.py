import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Cargar datos
# --------------------------------------------------------

file_path = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_schedule_citizen.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

df["in"] = pd.to_numeric(df["in"], errors="coerce")
df["out"] = pd.to_numeric(df["out"], errors="coerce")

# --------------------------------------------------------
# 2. Clasificación de actividad
# --------------------------------------------------------

def map_activity(todo):
    todo = str(todo)
    if "Home" in todo:
        return "Home"
    elif "WoS" in todo:
        return "WoS"
    elif "Dutties" in todo:
        return "Dutties"
    else:
        return "Other"

df["activity"] = df["todo"].apply(map_activity)
df = df[df["activity"].isin(["Home", "WoS", "Dutties"])]

# --------------------------------------------------------
# 3. Probabilidades por hora y arquetipo
# --------------------------------------------------------

archetypes = sorted(df["archetype"].unique())

for arch in archetypes:
    df_arch = df[df["archetype"] == arch].copy()
    n_agents = df_arch["agent"].nunique()
    if n_agents == 0:
        continue

    prob_list = []

    for h in range(24):
        start = h * 60
        end = (h + 1) * 60

        overlap = np.minimum(df_arch["out"], end) - np.maximum(df_arch["in"], start)
        overlap = overlap.clip(lower=0)
        df_arch["overlap"] = overlap

        hourly = df_arch.groupby("activity")["overlap"].sum()

        total_possible = n_agents * 60.0
        probs = hourly.reindex(["Home", "WoS", "Dutties"], fill_value=0) / total_possible
        prob_list.append(probs)

    prob_df = pd.DataFrame(prob_list, index=range(24))
    prob_df *= 100.0  # convertir a %

    # --------------------------------------------------------
    # 4. Graficar en escala de grises
    # --------------------------------------------------------

    # Tres tonos de gris bien diferenciados
    colors = ["#111111", "#777777", "#CCCCCC"]  # Home, WoS, Dutties

    fig, ax = plt.subplots()

    ax.stackplot(
        prob_df.index,
        prob_df["Home"],
        prob_df["WoS"],
        prob_df["Dutties"],
        labels=["Home", "WoS", "Dutties"],
        colors=colors
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 24, 1))

    # Leyenda sin borde de color para mantener estilo gris
    legend = ax.legend(loc="upper right", frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    plt.tight_layout()
    plt.show()
