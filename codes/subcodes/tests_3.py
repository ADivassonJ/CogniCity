import pandas as pd
import numpy as np
from pathlib import Path
from haversine import haversine, Unit

# --- RUTAS (ajusta si hace falta) ---
path_buildings = Path(r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_building.parquet")
path_citizens  = Path(r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Kanaleneiland\population\pop_citizen.parquet")

# --- SALIDAS (opcional) ---
out_perpoint_csv = Path("distancias_wos_a_home_por_citizen.csv")
out_summary_csv  = Path("resumen_distancias_wos_a_home.csv")

# =========================
#   LECTURA Y PREPARACIÓN
# =========================
# --- LEER PARQUET ---
buildings = pd.read_parquet(path_buildings)
citizens  = pd.read_parquet(path_citizens)

# --- FILTROS EN CITIZENS ---
mask_ind  = citizens["archetype"].isin(["c_arch_0", "c_arch_1"])
mask_home = citizens["WoS_subgroup"].astype(str).str.strip() != "Home"
before = len(citizens)
citizens_f = citizens.loc[mask_ind & mask_home].dropna(subset=["WoS", "Home"]).copy()
after = len(citizens_f)

if citizens_f.empty:
    raise ValueError("Tras filtrar (independent_type==1 y excluir Home), no quedan citizens.")

# --- MAPAS osm_id -> lat/lon ---
buildings_min = (
    buildings[["osm_id", "lat", "lon"]]
    .dropna(subset=["lat", "lon"])
    .drop_duplicates(subset=["osm_id"], keep="first")
)
lat_map = dict(zip(buildings_min["osm_id"], buildings_min["lat"]))
lon_map = dict(zip(buildings_min["osm_id"], buildings_min["lon"]))

# --- TRAER COORDENADAS DE WoS Y Home ---
citizens_f["lat_wos"]  = citizens_f["WoS"].map(lat_map)
citizens_f["lon_wos"]  = citizens_f["WoS"].map(lon_map)
citizens_f["lat_home"] = citizens_f["Home"].map(lat_map)
citizens_f["lon_home"] = citizens_f["Home"].map(lon_map)

# Limpiar filas sin ambas coordenadas
pts = citizens_f.dropna(subset=["lat_wos", "lon_wos", "lat_home", "lon_home"]).copy()
if pts.empty:
    raise ValueError("No hay pares WoS/Home con coordenadas válidas tras el mapeo de osm_id.")

# =========================
#   CÁLCULO DISTANCIAS (usando librería haversine)
# =========================
pts["dist_home_m"] = pts.apply(
    lambda r: haversine((r["lat_wos"], r["lon_wos"]), (r["lat_home"], r["lon_home"]), unit=Unit.METERS),
    axis=1
)

# =========================
#   RESUMEN / SALIDA
# =========================
mean_m = float(pts["dist_home_m"].mean())
std_m  = float(pts["dist_home_m"].std(ddof=1))  # desviación estándar muestral

# Consola
print(f"N citizens originales: {before}  → tras filtro: {after}")
print(f"N con WoS y Home geocodificados: {len(pts)}")
print(f"Distancia WoS→Home — promedio: {mean_m:.2f} m, desviación estándar (muestral): {std_m:.2f} m")

# CSV por punto (opcional)
cols_out = ["WoS", "Home", "lat_wos", "lon_wos", "lat_home", "lon_home", "dist_home_m"]
pts[cols_out].to_csv(out_perpoint_csv, index=False, encoding="utf-8")

# Resumen (opcional)
summary_df = pd.DataFrame([{
    "n_total": before,
    "n_filtrado": after,
    "n_geocodificados": len(pts),
    "mean_m": mean_m,
    "std_m_sample": std_m
}])
summary_df.to_csv(out_summary_csv, index=False, encoding="utf-8")

print(f"Guardado: {out_perpoint_csv.resolve()}")
print(f"Guardado: {out_summary_csv.resolve()}")

# --- Nota: si quieres la desviación poblacional, cambia ddof=1 por ddof=0 ---
