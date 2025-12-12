import pandas as pd
import numpy as np
from pathlib import Path

# --- RUTAS ---
path_citizens = Path(r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet")

# --- SALIDAS ---
out_perpoint_csv = Path("dist_wos_real_por_citizen.csv")
out_summary_csv  = Path("resumen_dist_wos_real_lognormal.csv")

# =========================
#   LECTURA Y FILTRO
# =========================
citizens = pd.read_parquet(path_citizens)

mask_ind  = citizens["archetype"].isin(["c_arch_0"])
mask_home = citizens["WoS_subgroup"].astype(str).str.strip() != "Home"

before = len(citizens)
citizens_f = citizens.loc[mask_ind & mask_home].copy()
after = len(citizens_f)

if citizens_f.empty:
    raise ValueError("Tras filtrar (independent_type==1 y excluir Home), no quedan citizens.")

# =========================
#   LIMPIEZA Y LOGARITMOS
# =========================
pts = citizens_f.dropna(subset=["dist_wos_real"]).copy()
pts = pts[pts["dist_wos_real"] > 0]  # solo distancias positivas
if pts.empty:
    raise ValueError("No hay ciudadanos con 'dist_wos_real' válido y positivo.")

# =========================
#   PARÁMETROS LOG-NORMALES
# =========================
log_vals = np.log(pts["dist_wos_real"].values)

mu_log = float(np.mean(log_vals))
sigma_log = float(np.std(log_vals, ddof=1))  # desviación muestral

# =========================
#   PARÁMETROS NORMALES (MU Y SIGMA SOBRE ESCALA ORIGINAL)
# =========================
mu_norm = float(np.mean(pts["dist_wos_real"]))
sigma_norm = float(np.std(pts["dist_wos_real"], ddof=1))

# Estadísticos descriptivos
min_m  = float(np.min(pts["dist_wos_real"]))
max_m  = float(np.max(pts["dist_wos_real"]))
p25, p50, p75 = np.percentile(pts["dist_wos_real"], [25, 50, 75])

# =========================
#   CONSOLA
# =========================
print(f"N citizens originales: {before}  → tras filtro: {after}")
print(f"N con dist_wos_real válido: {len(pts)}")
print("—— Log-normal parameters ——")
print(f"μ_log = {mu_log:.5f}")
print(f"σ_log = {sigma_log:.5f}")
print("—— Normal (no log) parameters ——")
print(f"μ_norm = {mu_norm:.5f}")
print(f"σ_norm = {sigma_norm:.5f}")

# =========================
#   EXPORTACIÓN
# =========================
cols_out = ["Citizen_id", "archetype", "WoS_subgroup", "dist_wos_real"]
if not all(col in pts.columns for col in cols_out):
    cols_out = [c for c in cols_out if c in pts.columns] + ["dist_wos_real"]
pts[cols_out].to_csv(out_perpoint_csv, index=False, encoding="utf-8")

summary_df = pd.DataFrame([{
    "n_total": before,
    "n_filtrado": after,
    "n_validos": len(pts),
    "mu_log": mu_log,
    "sigma_log": sigma_log,
    "mu_norm": mu_norm,
    "sigma_norm": sigma_norm,
    "min_m": min_m,
    "p25_m": p25,
    "p50_median_m": p50,
    "p75_m": p75,
    "max_m": max_m
}])
summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8")

print(f"Guardado: {out_perpoint_csv.resolve()}")
print(f"Guardado: {out_summary_csv.resolve()}")
