import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------- 1. Cargar datos -----------

file_path = os.path.expanduser("~/Desktop/test_results.xlsx")

df = pd.read_excel(
    file_path,
    decimal=",",        # para interpretar 63,0545 como 63.0545
    engine="openpyxl"
)

# Aseguramos tipos numéricos
num_cols = ["mean_spended_tot", "std_spended_tot"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ----------- 2. Parámetros normales teóricos por arquetipo -----------

normal_params = {
    "c_arch_0": {"mu": 86.120, "sigma": 1.852},
    "c_arch_1": {"mu": 87.840, "sigma": 2.255},
    "c_arch_2": {"mu": 61.335, "sigma": 3.179},
    "c_arch_3": {"mu": 66.540, "sigma": 2.867},
    "c_arch_4": {"mu": 84.435, "sigma": 2.630},
}

# Fijamos orden de arquetipos
order = list(normal_params.keys())
df["archetype"] = pd.Categorical(df["archetype"], categories=order, ordered=True)
archetypes = [a for a in df["archetype"].cat.categories if a in df["archetype"].unique()]

# ----------- 3. Función de densidad normal (PDF) -----------

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# ----------- 4. Comparación de curvas normales (teórica vs empírica) -----------

n_arch = len(archetypes)
fig, axes = plt.subplots(n_arch, 1, figsize=(8, 3 * n_arch), sharex=True)

if n_arch == 1:
    axes = [axes]

colors = plt.cm.tab10(np.linspace(0, 1, n_arch))

# Rango fijo de X para todos
x = np.linspace(0, 100, 500)

for ax, arch, color in zip(axes, archetypes, colors):
    subset = df[df["archetype"] == arch]["mean_spended_tot"].dropna()

    if len(subset) < 2 or arch not in normal_params:
        ax.text(0.5, 0.5, f"Datos insuficientes o sin parámetros para {arch}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(arch)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.05)
        continue

    # --- 4.1. Parámetros teóricos ---
    mu_teo = normal_params[arch]["mu"]
    sigma_teo = normal_params[arch]["sigma"]

    # --- 4.2. Parámetros empíricos (a partir de tus datos) ---
    mu_emp = subset.mean()
    sigma_emp = subset.std(ddof=1)  # desviación estándar muestral

    # --- 4.3. Curvas normales (PDF) ---
    pdf_teo = normal_pdf(x, mu_teo, sigma_teo)
    pdf_emp = normal_pdf(x, mu_emp, sigma_emp)

    # --- 4.4. Normalización de altura (máximo = 1) ---
    if pdf_teo.max() > 0:
        pdf_teo_norm = pdf_teo / pdf_teo.max()
    else:
        pdf_teo_norm = pdf_teo

    if pdf_emp.max() > 0:
        pdf_emp_norm = pdf_emp / pdf_emp.max()
    else:
        pdf_emp_norm = pdf_emp

    # --- 4.5. Cálculo de errores porcentuales (en media y sigma) ---
    mean_err_pct = 100.0 * (mu_emp - mu_teo) / mu_teo
    sigma_err_pct = 100.0 * (sigma_emp - sigma_teo) / sigma_teo

    # --- 4.6. Plot de las dos curvas normalizadas ---
    # Curva normal empírica (a partir de tus datos)
    ax.plot(x, pdf_emp_norm, linestyle="-", color=color, label="Empirical (data)")

    # Curva normal teórica (valores originales)
    ax.plot(x, pdf_teo_norm, linestyle="--", color=color, alpha=0.8, label="Theoretical (source)")

    # Configuración del subplot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized height")
    ax.legend()

    ax.set_title(
        f"{arch} | error = {mean_err_pct:+.2f}%"
    )

axes[-1].set_xlabel("Average total time spent on transportation per day")
plt.tight_layout()
plt.show()
