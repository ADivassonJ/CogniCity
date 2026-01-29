import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Cargar datos
# --------------------------------------------------------

file_path = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_schedule_vehicle.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

df["in"] = pd.to_numeric(df["in"], errors="coerce")
df["out"] = pd.to_numeric(df["out"], errors="coerce")

# --------------------------------------------------------
# 2. Clasificación de actividad (Vehicles)
# --------------------------------------------------------

def map_activity_vehicle(todo):
    # Normalizar: NaN -> "", strip espacios
    if pd.isna(todo):
        todo = ""
    todo = str(todo).strip()

    # Regla: vacío -> Home (always)
    if todo == "":
        return "Home_always"

    # Home_in -> Home(before leaving)
    if "Home_in" in todo:
        return "Home_before_leaving"

    # Home_out -> Home(after arriving)
    if "Home_out" in todo:
        return "Home_after_arriving"

    # WoS y Dutties -> Outside
    if ("WoS" in todo) or ("Dutties" in todo):
        return "Outside"

    # Si aparece algo distinto, lo descartamos luego
    return "Other"

df["activity"] = df["todo"].apply(map_activity_vehicle)

# Nos quedamos solo con las 3 clases principales (y la opción "always" si quieres incluirla)
# Si "Home_always" equivale conceptualmente a Home, puedes integrarlo como Home_always o sumarlo a "Home_before/after".
keep_acts = ["Home_before_leaving", "Home_after_arriving", "Outside", "Home_always"]
df = df[df["activity"].isin(keep_acts)].copy()

# Si prefieres que "Home_always" cuente como Home (always) separado, lo dejamos como está.
# Si prefieres sumarlo a "Home_after_arriving" o a un "Home" genérico, dímelo y lo ajusto.

# --------------------------------------------------------
# 3. Probabilidades por hora y arquetipo
# --------------------------------------------------------

archetypes = sorted(df["archetype"].unique())

print(f"archetypes: {archetypes}")

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
        # Orden fijo de categorías
        order = ["Home_before_leaving", "Home_after_arriving", "Outside", "Home_always"]
        probs = hourly.reindex(order, fill_value=0) / total_possible
        prob_list.append(probs)

    prob_df = pd.DataFrame(prob_list, index=range(24))
    prob_df *= 100.0

    # --------------------------------------------------------
    # 4. Graficar (escala de grises)
    # --------------------------------------------------------

    # 4 tonos por si incluyes Home_always
    colors = ["#0c343d", "#134f5c", "#45818e", "#d0e0e3"]

    fig, ax = plt.subplots()

    ax.stackplot(
        prob_df.index,
        prob_df["Home_always"],
        prob_df["Home_before_leaving"],
        prob_df["Home_after_arriving"],
        prob_df["Outside"],
        
        labels=[
            "Home (always)",
            "Home (before leaving)",
            "Home (after arriving)",
            "Outside"
        ],
        colors=colors
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 24, 1))

    legend = ax.legend(loc="lower right", frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    plt.tight_layout()
    plt.show()
