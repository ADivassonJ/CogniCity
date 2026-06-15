import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------
# 1. Cargar datos y configurar ruta de guardado
# --------------------------------------------------------

download_dir = r"C:\Users\asier.divasson\Downloads"
file_path = os.path.join(download_dir, "s0_Kanaleneiland_schedule_vehicle.csv")

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df["in"] = pd.to_numeric(df["in"], errors="coerce")
df["out"] = pd.to_numeric(df["out"], errors="coerce")

# Agente único por día
df["agent"] = df["agent"] + "_" + df["day"].astype(str)

# --------------------------------------------------------
# CONFIGURACIÓN DE TAMAÑO (NUEVO)
# --------------------------------------------------------
# Convertir tamaño a pulgadas (1 pulgada = 25.4 mm)
fig_width = 368 / 34.4
fig_height = 125 / 34.4
figsize_custom = (fig_width, fig_height)

# --------------------------------------------------------
# 2. Clasificación de actividad
# --------------------------------------------------------

def map_activity_vehicle(todo):
    if pd.isna(todo) or str(todo).strip() == "":
        return "Home_always"

    todo = str(todo).strip()

    if "Home_out" in todo:
        return "Home_before_leaving"
    if "Home_in" in todo:
        return "Home_after_arriving"
    if ("WoS" in todo) or ("Dutties" in todo):
        return "Outside"

    return "Other"

df["activity"] = df["todo"].apply(map_activity_vehicle)

# Rellenar in/out para Home_always
df.loc[df["activity"] == "Home_always", "in"]  = df.loc[df["activity"] == "Home_always", "in"].fillna(0)
df.loc[df["activity"] == "Home_always", "out"] = df.loc[df["activity"] == "Home_always", "out"].fillna(1440)

keep_acts = ["Home_before_leaving", "Home_after_arriving", "Outside", "Home_always"]
df = df[df["activity"].isin(keep_acts)].copy()

# DIAGNÓSTICO
print("Actividades detectadas:")
print(df["activity"].value_counts(dropna=False))

# --------------------------------------------------------
# 3. Probabilidades por hora y arquetipo
# --------------------------------------------------------

archetypes = sorted(df["archetype"].unique())
print(f"\narchetypes: {archetypes}")

for arch in archetypes:
    df_arch = df[df["archetype"] == arch].copy()
    n_agents = df_arch["agent"].nunique()
    if n_agents == 0:
        continue

    prob_list = []

    for h in range(24):
        start = h * 60
        end   = (h + 1) * 60

        df_h = df_arch.copy()
        df_h["overlap"] = (
            np.minimum(df_h["out"], end) - np.maximum(df_h["in"], start)
        ).clip(lower=0)

        hourly = (
            df_h.groupby(["agent", "activity"])["overlap"]
                .sum()
                .clip(upper=60)
                .reset_index()
                .groupby("activity")["overlap"]
                .sum()
        )

        total_possible = n_agents * 60.0
        order = ["Home_before_leaving", "Home_after_arriving", "Outside", "Home_always"]
        probs = hourly.reindex(order, fill_value=0) / total_possible
        prob_list.append(probs)

    prob_df = pd.DataFrame(prob_list, index=range(24))
    prob_df *= 100.0

    # Diagnóstico suma por hora
    sums = prob_df.sum(axis=1)
    print(f"\n{arch} — suma por hora: min={sums.min():.1f}%  mean={sums.mean():.1f}%  max={sums.max():.1f}%")

    # --------------------------------------------------------
    # 4. Graficar y guardar (Modificado con tamaño y formato PDF)
    # --------------------------------------------------------
    colors = ["#0c343d", "#134f5c", "#45818e", "#d0e0e3"]

    # Aplicamos las dimensiones personalizadas aquí
    fig, ax = plt.subplots(figsize=figsize_custom)
    
    ax.stackplot(
        prob_df.index,
        prob_df["Home_always"],
        prob_df["Home_before_leaving"],
        prob_df["Home_after_arriving"],
        prob_df["Outside"],
        labels=["Home (always)", "Home (before leaving)", "Home (after arriving)", "Outside"],
        colors=colors
    )

    ax.set_title(" ")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 24, 1))

    legend = ax.legend(
            loc="lower right", 
            frameon=True, 
            fontsize=14,          # Tamaño del texto (puedes usar números como 12, 14 o strings como 'large', 'x-large')
            handlelength=2.0,     # Ancho de los rectángulos de color (por defecto suele ser 2.0, súbelo si quieres más)
            handleheight=1.5      # Alto de los rectángulos de color (súbelo para hacerlos más gruesos)
        )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    plt.tight_layout()
    
    # Guardar figura individual como PDF antes de mostrarla
    save_path_arch = os.path.join(download_dir, f"probabilidades_{arch}.pdf")
    fig.savefig(save_path_arch)
    print(f"Gráfico guardado en: {save_path_arch}")
    
    plt.show()
    plt.close(fig)

# --------------------------------------------------------
# 5. Análisis de solapamiento para CS_electric
# --------------------------------------------------------

arch_target = "CS_electric"
df_cs = df[df["archetype"] == arch_target].copy()

after_arriving = df_cs[df_cs["activity"] == "Home_after_arriving"].copy()
before_leaving = df_cs[df_cs["activity"] == "Home_before_leaving"].copy()

overlap_counts = []

for h in range(24):
    start = h * 60
    end   = (h + 1) * 60

    def count_active(df_group):
        ov = np.minimum(df_group["out"], end) - np.maximum(df_group["in"], start)
        return (ov.clip(lower=0) > 0).sum()

    n_after  = count_active(after_arriving)
    n_before = count_active(before_leaving)
    n_shared = min(n_after, n_before)

    overlap_counts.append({
        "hour": h,
        "Home_after_arriving": n_after,
        "Home_before_leaving": n_before,
        "Potential_reuse": n_shared
    })

overlap_df = pd.DataFrame(overlap_counts)

# --------------------------------------------------------
# 6. Graficar solapamiento y guardar (Modificado con tamaño y formato PDF)
# --------------------------------------------------------

# Aplicamos las mismas dimensiones personalizadas aquí también
fig, ax = plt.subplots(figsize=figsize_custom)

ax.bar(overlap_df["hour"] - 0.2, overlap_df["Home_after_arriving"],
       width=0.4, label="Home (after arriving)", color="#45818e")
ax.bar(overlap_df["hour"] + 0.2, overlap_df["Home_before_leaving"],
       width=0.4, label="Home (before leaving)", color="#134f5c")
ax.step(overlap_df["hour"], overlap_df["Potential_reuse"],
        where="mid", color="red", linewidth=2, linestyle="--",
        label="Potential reuse (overlap)")

ax.set_title(f"{arch_target} — Overlaping")
ax.set_xlabel("Hour")
ax.set_ylabel("Number of agents")
ax.set_xticks(range(24))
ax.legend(frameon=True)
plt.tight_layout()

# Guardar gráfico de solapamiento como PDF independiente
save_path_overlap = os.path.join(download_dir, f"solapamiento_{arch_target}.pdf")
fig.savefig(save_path_overlap)
print(f"Gráfico de solapamiento guardado en: {save_path_overlap}")

plt.show()
plt.close(fig)

# --------------------------------------------------------
# 7. Resumen
# --------------------------------------------------------

max_reuse    = overlap_df["Potential_reuse"].max()
total_agents = df_cs["agent"].nunique()

print(f"\n{'='*50}")
print(f"Archetype: {arch_target}")
print(f"Total CS_electric agents:       {total_agents}")
print(f"Max simultaneous reuse (hour):  {overlap_df.loc[overlap_df['Potential_reuse'].idxmax(), 'hour']}h")
print(f"Max potential reduction:        {max_reuse} vehicles")
print(f"Reduced fleet estimate:         {total_agents - max_reuse} vehicles")
print(f"{'='*50}")