import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------- 1. Cargar datos crudos -----------

file_path = os.path.expanduser("~/Desktop/Kanaleneiland_schedule_citizen.xlsx")

df = pd.read_excel(
    file_path,
    decimal=",",
    engine="openpyxl"
)

# Aseguramos tipos numéricos
for col in ["in", "out"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------- 2. Calcular spended y totales diarios por agente -----------

df["spended"] = df["out"] - df["in"]

df_agentes = (
    df.groupby("agent")
      .agg(
          archetype=("archetype", "first"),
          suma_spended=("spended", "sum")
      )
      .reset_index()
)

df_agentes["spended_tot"] = 1440 - df_agentes["suma_spended"]
df_agentes.drop(columns=["suma_spended"], inplace=True)

# ----------- 3. Orden de arquetipos -----------

order = [
    "c_arch_0",
    "c_arch_1",
    "c_arch_2",
    "c_arch_3",
    "c_arch_4",
]

df_agentes["archetype"] = pd.Categorical(
    df_agentes["archetype"],
    categories=order,
    ordered=True
)

# Filtramos sólo los que existen
df_agentes = df_agentes.dropna(subset=["archetype"])

# ----------- 4. Box plot por arquetipo -----------

data = [
    df_agentes[df_agentes["archetype"] == arch]["spended_tot"].dropna()
    for arch in order
    if arch in df_agentes["archetype"].unique()
]

labels = [
    arch for arch in order
    if arch in df_agentes["archetype"].unique()
]

plt.figure(figsize=(10, 5))

plt.boxplot(
    data,
    labels=labels,
    showfliers=True,      # outliers
    medianprops=dict(linewidth=2),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

plt.xlabel("Archetype")
plt.ylabel("spended_tot (minutos por día)")
plt.title("Distribución de spended_tot por arquetipo")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
