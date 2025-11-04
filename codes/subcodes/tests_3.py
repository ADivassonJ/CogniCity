import pandas as pd
import numpy as np
from pathlib import Path

# --- RUTAS ---
path_citizens = Path(r"C:\Users\asier\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet")

# --- SALIDAS ---
out_perpoint_csv = Path("dist_wos_por_citizen.csv")
out_summary_csv  = Path("resumen_dist_wos_lognormal.csv")

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
pts = citizens_f.dropna(subset=["dist_wos"]).copy()
pts = pts[pts["dist_wos"] > 0]  # solo distancias positivas
if pts.empty:
    raise ValueError("No hay ciudadanos con 'dist_wos' válido y positivo.")

# =========================
#   PARÁMETROS LOG-NORMALES
# =========================
# Log-transformación
log_vals = np.log(pts["dist_wos"].values)

mu_log = float(np.mean(log_vals))
sigma_log = float(np.std(log_vals, ddof=1))  # desviación muestral

# Estadísticos en la escala original
mean_m = float(np.mean(pts["dist_wos"]))
std_m  = float(np.std(pts["dist_wos"], ddof=1))
min_m  = float(np.min(pts["dist_wos"]))
max_m  = float(np.max(pts["dist_wos"]))
p25, p50, p75 = np.percentile(pts["dist_wos"], [25, 50, 75])

# =========================
#   CONSOLA
# =========================
print(f"N citizens originales: {before}  → tras filtro: {after}")
print(f"N con dist_wos válido: {len(pts)}")
print("—— Log-normal parameters ——")
print(f"μ_log = {mu_log:.5f}")
print(f"σ_log = {sigma_log:.5f}")

# =========================
#   EXPORTACIÓN
# =========================
cols_out = ["Citizen_id", "archetype", "WoS_subgroup", "dist_wos"]
if not all(col in pts.columns for col in cols_out):
    cols_out = [c for c in cols_out if c in pts.columns] + ["dist_wos"]
pts[cols_out].to_csv(out_perpoint_csv, index=False, encoding="utf-8")

summary_df = pd.DataFrame([{
    "n_total": before,
    "n_filtrado": after,
    "n_validos": len(pts),
    "mu_log": mu_log,
    "sigma_log": sigma_log,
    "mean_m": mean_m,
    "std_m_sample": std_m,
    "min_m": min_m,
    "p25_m": p25,
    "p50_median_m": p50,
    "p75_m": p75,
    "max_m": max_m
}])
summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8")

print(f"Guardado: {out_perpoint_csv.resolve()}")
print(f"Guardado: {out_summary_csv.resolve()}")
