import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, lognorm

# ----------- 1. Cargar datos crudos -----------

file_path = os.path.expanduser("~/Desktop/Kanaleneiland_schedule_citizen.xlsx")

df = pd.read_excel(
    file_path,
    decimal=",",        # por si hay comas decimales
    engine="openpyxl"
)

# Aseguramos tipos numéricos
for col in ["in", "out"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------- 2. Calcular spended y totales diarios por agente -----------

# 2.1. Tiempo invertido en cada actividad
df["spended"] = df["out"] - df["in"]

# 2.2. Agregar por agente para obtener el total diario
df_agentes = (
    df.groupby("agent")
      .agg(
          archetype=("archetype", "first"),
          suma_spended=("spended", "sum")
      )
      .reset_index()
)

# 2.3. Transformación a "tiempo total diario"
df_agentes["spended_tot"] = 1440 - df_agentes["suma_spended"]
df_agentes.drop(columns=["suma_spended"], inplace=True)

# ----------- 3. Parámetros LOGNORMALES teóricos por arquetipo -----------
# Estos son los parámetros de la NORMAL subyacente: X ~ LogNormal(mu, sigma)
# con X = exp(N(mu, sigma^2)), mu y sigma en logaritmo natural.

lognorm_params = {
    "c_arch_0": {"mu": 4.456, "sigma": 0.022},
    "c_arch_1": {"mu": 4.475, "sigma": 0.026},
    "c_arch_2": {"mu": 4.115, "sigma": 0.052},
    "c_arch_3": {"mu": 4.197, "sigma": 0.043},
    "c_arch_4": {"mu": 4.435, "sigma": 0.031},
}

# (Referencia: en Normal "a secas" serían aproximadamente
# 86.120, 87.840, 61.335, 66.540, 84.435, que encajan con exp(mu).)

# Fijamos orden de arquetipos
order = list(lognorm_params.keys())
df_agentes["archetype"] = pd.Categorical(
    df_agentes["archetype"],
    categories=order,
    ordered=True
)

# Sólo arquetipos que existan en los datos
archetypes = [
    a for a in df_agentes["archetype"].cat.categories
    if a in df_agentes["archetype"].unique()
]

# ----------- 4. Curvas de densidad empírica (totales diarios) vs LogNormales teóricas -----------

n_arch = len(archetypes)
fig, axes = plt.subplots(n_arch, 1, figsize=(8, 3 * n_arch), sharex=False)

if n_arch == 1:
    axes = [axes]

colors = plt.cm.tab10(np.linspace(0, 1, n_arch))

for ax, arch, color in zip(axes, archetypes, colors):
    subset = df_agentes[df_agentes["archetype"] == arch]["spended_tot"].dropna()

    if len(subset) < 2:
        ax.text(0.5, 0.5, f"Datos insuficientes para {arch}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Archetype: {arch}")
        continue

    # ---- 4.1. Densidad empírica (KDE) a partir de spended_tot ----
    # La LogNormal está definida sólo para valores positivos
    subset_pos = subset[subset > 0]
    if len(subset_pos) < 2:
        ax.text(0.5, 0.5, f"Valores no positivos para {arch}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Archetype: {arch}")
        continue

    kde = gaussian_kde(subset_pos)
    data_min, data_max = subset_pos.min(), subset_pos.max()

    # Rango x combinando datos y LogNormal teórica (usamos cuantiles)
    if arch in lognorm_params:
        mu = lognorm_params[arch]["mu"]
        sigma = lognorm_params[arch]["sigma"]

        s = sigma
        scale = np.exp(mu)

        # Cuantiles "extremos" para abarcar casi toda la masa
        theo_min = lognorm.ppf(0.001, s=s, scale=scale)
        theo_max = lognorm.ppf(0.999, s=s, scale=scale)

        x_min = max(1e-6, min(data_min, theo_min))
        x_max = max(data_max, theo_max)
    else:
        x_min = max(1e-6, data_min)
        x_max = data_max

    x = np.linspace(x_min, x_max, 400)
    emp_pdf = kde(x)

    # ---- 4.2. Curva LogNormal teórica ----
    if arch in lognorm_params:
        theo_pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
        ax.plot(x, theo_pdf, linestyle="--", color=color, label="LogNormal teórica")

    # ---- 4.3. Curva empírica ----
    ax.plot(x, emp_pdf, linestyle="-", color=color,
            label="Densidad empírica (totales diarios)")

    ax.set_title(f"Archetype: {arch}")
    ax.set_ylabel("Densidad")
    ax.legend()

axes[-1].set_xlabel("spended_tot (minutos por día)")
plt.tight_layout()
plt.show()
