from __future__ import annotations

# estándar
import itertools
import os
import math
import random
import shutil
import sys
from pathlib import Path
from scipy.stats import norm

# terceros
import folium
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.stats import lognorm
import pyproj
from itertools import cycle
from tqdm import tqdm
from haversine import Unit, haversine
from scipy.spatial import Voronoi, cKDTree
from shapely.errors import ShapelyDeprecationWarning
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import box, MultiPolygon, Point, Polygon
from shapely.ops import clip_by_rect, transform, unary_union, voronoi_diagram

from pathlib import Path
import pandas as pd
import numpy as np

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================
METRICS = ["walk_time", "travel_time", "wait_time", "cost", "mjkm", "benefits", "emissions"]
GROUPS = ["archetype", "family_archetype", "s_class"]
TIME_COL = "time_slot"
AGENT_COL = "agent"
DAY_COL = "day"


# ============================================================
# Helpers
# ============================================================
def normalize_numeric(series: pd.Series) -> pd.Series:
    """Convierte a float manejando coma decimal."""
    if series.dtype == object:
        s = series.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def infer_resolution_from_time_slot(df: pd.DataFrame, time_col: str = TIME_COL) -> int:
    """Infere resolución (slots/día) a partir de time_slot."""
    if time_col not in df.columns:
        raise ValueError(f"No existe la columna '{time_col}' para inferir resolución.")

    times = df[time_col].dropna().astype(str).unique()
    if len(times) < 2:
        raise ValueError("No hay suficientes time_slot distintos para inferir resolución.")

    minutes = np.array(sorted(hhmm_to_minutes(t) for t in times))
    deltas = np.diff(minutes)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        raise ValueError("No se pudo inferir el paso temporal desde time_slot.")

    step = int(np.min(deltas))
    return int(round(1440 / step))


def read_quantified_excel(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet = "quantified" if "quantified" in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def ensure_required_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Faltan columnas requeridas: {missing}")


def reduce_to_daily_last_row(
    df: pd.DataFrame,
    agent_col: str = AGENT_COL,
    day_col: str = DAY_COL,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """
    Devuelve 1 fila por (day, agent): la última disponible del día (máximo time_slot).
    """
    ensure_required_columns(df, [agent_col, day_col, time_col], context="reduce_to_daily_last_row")

    out = df.copy()
    out["_tmin"] = out[time_col].astype(str).map(hhmm_to_minutes)

    # ordenar para que la "última" quede al final
    out = out.sort_values([day_col, agent_col, "_tmin"], ascending=[True, True, True])

    # quedarse con la última fila por grupo
    out = out.groupby([day_col, agent_col], as_index=False).tail(1)

    out = out.drop(columns=["_tmin"]).reset_index(drop=True)
    return out


def compute_overall_stats(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    present = [m for m in metrics if m in df.columns]
    if not present:
        raise ValueError("No hay métricas presentes para estadística global.")

    for m in present:
        df[m] = normalize_numeric(df[m])

    rows = []
    for m in present:
        s = df[m].dropna()
        rows.append({
            "metric": m,
            "count": int(s.count()),
            "mean": float(s.mean()) if len(s) else np.nan,
            "std": float(s.std()) if len(s) else np.nan,
            "min": float(s.min()) if len(s) else np.nan,
            "p25": float(s.quantile(0.25)) if len(s) else np.nan,
            "median": float(s.median()) if len(s) else np.nan,
            "p75": float(s.quantile(0.75)) if len(s) else np.nan,
            "max": float(s.max()) if len(s) else np.nan,
            "sum": float(s.sum()) if len(s) else np.nan,
        })

    return pd.DataFrame(rows)


def compute_group_stats(df: pd.DataFrame, group_col: str, metrics: list[str]) -> pd.DataFrame:
    """
    Estadísticos por grupo sobre los TOTALES DIARIOS (ya reducidos a última fila por día/agente).
    Compatible con pandas >= 2.0
    """
    present = [m for m in metrics if m in df.columns]
    if not present:
        raise ValueError(f"No hay métricas disponibles entre: {metrics}")

    for m in present:
        df[m] = normalize_numeric(df[m])

    g = df.dropna(subset=[group_col]).groupby(group_col, dropna=True)

    stats = g[present].agg([
        "count",
        "mean",
        "std",
        "min",
        lambda s: s.quantile(0.25),
        "median",
        lambda s: s.quantile(0.75),
        "max",
        "sum",
    ])

    stats = stats.rename(columns={"<lambda_0>": "p25", "<lambda_1>": "p75"}, level=1)
    stats.columns = [f"{metric}__{stat}" for metric, stat in stats.columns.to_flat_index()]

    return stats.reset_index()


# ============================================================
# Main entry
# ============================================================
def build_daily_total_stats_from_constructed_outputs(
    paths: dict,
    study_area: str,
    metrics: list[str] = METRICS,
    groups: list[str] = GROUPS,
) -> Path:
    """
    Lee cuantificados existentes y calcula estadística SOBRE TOTALES DIARIOS:
    - Primero reduce a última fila por (day, agent)
    - Luego agrupa por archetype, family_archetype, s_class
    """

    results_dir = Path(paths["results"])

    # Tomar los cuantificados existentes (si hay varios, toma el más reciente por mtime)
    cit_candidates = list(results_dir.glob(f"{study_area}_schedule_citizen_quantified_*.xlsx"))
    veh_candidates = list(results_dir.glob(f"{study_area}_schedule_vehicle_quantified_*.xlsx"))

    if not cit_candidates:
        raise FileNotFoundError(f"No encuentro: {study_area}_schedule_citizen_quantified_*.xlsx en {results_dir}")
    if not veh_candidates:
        raise FileNotFoundError(f"No encuentro: {study_area}_schedule_vehicle_quantified_*.xlsx en {results_dir}")

    citizen_q = max(cit_candidates, key=lambda p: p.stat().st_mtime)
    vehicle_q = max(veh_candidates, key=lambda p: p.stat().st_mtime)

    df_cit = read_quantified_excel(citizen_q)
    df_veh = read_quantified_excel(vehicle_q)

    # Inferir resolución (informativo)
    resolution = infer_resolution_from_time_slot(df_cit, TIME_COL)

    # Validar segmentación (debe existir en los cuantificados)
    ensure_required_columns(df_cit, groups + [AGENT_COL, DAY_COL, TIME_COL], context="citizen")
    ensure_required_columns(df_veh, groups + [AGENT_COL, DAY_COL, TIME_COL], context="vehicle")

    # Reducir a totales diarios (último slot por agente/día)
    cit_daily = reduce_to_daily_last_row(df_cit, AGENT_COL, DAY_COL, TIME_COL)
    veh_daily = reduce_to_daily_last_row(df_veh, AGENT_COL, DAY_COL, TIME_COL)

    out_path = results_dir / f"{study_area}_daily_total_stats_inferred_{resolution}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Control: los diarios
        cit_daily.to_excel(writer, index=False, sheet_name="citizen_daily_last")
        veh_daily.to_excel(writer, index=False, sheet_name="vehicle_daily_last")

        # Global diario
        compute_overall_stats(cit_daily.copy(), metrics).to_excel(writer, index=False, sheet_name="cit_overall_daily")
        compute_overall_stats(veh_daily.copy(), metrics).to_excel(writer, index=False, sheet_name="veh_overall_daily")

        # Por grupo (sobre diarios)
        for gcol in groups:
            compute_group_stats(cit_daily.copy(), gcol, metrics).to_excel(
                writer, index=False, sheet_name=f"cit_by_{gcol}"[:31]
            )
            compute_group_stats(veh_daily.copy(), gcol, metrics).to_excel(
                writer, index=False, sheet_name=f"veh_by_{gcol}"[:31]
            )

    return out_path

def paths_initialization(study_area):
    # Paths initialization
    paths = {}
    
    paths['main'] = Path(__file__).resolve().parent.parent.parent
    paths['system'] = paths['main'] / 'system'
    paths['desktop'] = Path.home() / "Desktop"
    
    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    
    file_management = system_management[['file_1', 'file_2', 'pre']]
    # Paso 2: Bucle sobre filas del mini DF
    for index, row in file_management.iterrows():
        file_1 = paths[study_area] if row['file_1'] == 'study_area' else paths[row['file_1']]
        file_2 = study_area if row['file_2'] == 'study_area' else row['file_2']
        paths[file_2] = file_1 / file_2
        if not paths[file_2].exists():
            if row['pre'] == 'y':
                print(f"[Error] Critical file not detected:")
                print(f"{paths[file_2]}")
                print(f"Please solve the mentioned issue and reestart the model.")
                sys.exit()
            elif row['pre'] == 'p':
                user_is_stupid = True
                while user_is_stupid:    
                    response = input(f"Data for the case study '{study_area}' was not found.\nDo you want to copy data from standar scenario or do you want to create your own? [Y/N]\n")
                    if response == 'Y':
                        user_is_stupid = False
                        shutil.copytree(paths['base_scenario'], paths[file_2])
                    elif response == 'N':
                        user_is_stupid = False
                        os.makedirs(paths[file_2], exist_ok=True)
                    else:
                        print(f"Your response was not valid, please respond Y (yes) or N (no).")
            else:
                os.makedirs(paths[file_2], exist_ok=True)
    return paths, system_management


# ============================================================
# Ejecución típica
# ============================================================
if __name__ == "__main__":
    study_area = "Kanaleneiland"
    paths, _ = paths_initialization(study_area)

    out = build_daily_total_stats_from_constructed_outputs(
        paths=paths,
        study_area=study_area,
    )

    print(f"[OK] Estadísticos diarios (último slot por agente/día) en: {out}")
