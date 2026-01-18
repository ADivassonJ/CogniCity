from __future__ import annotations

# estándar
import os
import shutil
import sys
from pathlib import Path

# terceros
import pandas as pd
import numpy as np


# =============================================================================
# PATHS
# =============================================================================
def paths_initialization(study_area: str):
    """
    Inicializa paths a partir de 'system_management.xlsx'.

    Requisitos esperados (según tu implementación):
    - Existe: <main>/system/system_management.xlsx
    - system_management.xlsx contiene columnas: file_1, file_2, pre
    - Si pre == 'y' y falta el path -> aborta
    - Si pre == 'p' y falta -> pregunta copiar base_scenario o crear carpeta
    """
    paths = {}

    paths["main"] = Path(__file__).resolve().parent.parent.parent
    paths["system"] = paths["main"] / "system"
    paths["desktop"] = Path.home() / "Desktop"

    system_management = pd.read_excel(paths["system"] / "system_management.xlsx")

    file_management = system_management[["file_1", "file_2", "pre"]]

    for _, row in file_management.iterrows():
        file_1 = paths[study_area] if row["file_1"] == "study_area" else paths[row["file_1"]]
        file_2 = study_area if row["file_2"] == "study_area" else row["file_2"]

        paths[file_2] = file_1 / file_2

        if not paths[file_2].exists():
            if row["pre"] == "y":
                print("[Error] Critical file not detected:")
                print(f"{paths[file_2]}")
                print("Please solve the mentioned issue and reestart the model.")
                sys.exit()

            elif row["pre"] == "p":
                user_is_stupid = True
                while user_is_stupid:
                    response = input(
                        f"Data for the case study '{study_area}' was not found.\n"
                        "Do you want to copy data from standar scenario or do you want to create your own? [Y/N]\n"
                    )
                    if response == "Y":
                        user_is_stupid = False
                        shutil.copytree(paths["base_scenario"], paths[file_2])
                    elif response == "N":
                        user_is_stupid = False
                        os.makedirs(paths[file_2], exist_ok=True)
                    else:
                        print("Your response was not valid, please respond Y (yes) or N (no).")

            else:
                os.makedirs(paths[file_2], exist_ok=True)

    return paths, system_management


# =============================================================================
# CONFIG
# =============================================================================
EXCLUDED_AGENTS = {"Public_transport", "walk"}

# Columnas que quieres eliminar del output final (después de cuantificar)
COLUMNS_TO_DROP = {
    "independent",
    "opening",
    "closing",
    "fixed",
    "time2spend",
    "trip",
    "dist",
    "in",
    "out",
    "dist_real",
}

# Columnas a acumular (cumsum) ANTES de cuantificar a horas
ACCUM_COLS = ["walk_time", "travel_time", "wait_time", "cost", "mjkm", "benefits", "emissions"]


# =============================================================================
# UTILIDADES NUMÉRICAS / TIEMPO
# =============================================================================
def normalize_numeric(series: pd.Series) -> pd.Series:
    """
    Convierte a float manejando coma decimal. Deja NaN si no puede.
    """
    if series.dtype == object:
        s = series.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def normalize_numeric_minutes(series: pd.Series) -> pd.Series:
    """
    Normaliza columnas 'in'/'out' (minutos del día) como Int64 (permite NA).
    Maneja coma decimal.
    """
    s = normalize_numeric(series)
    return s.round().astype("Int64")


def minutes_to_hhmm(mins: int) -> str:
    mins = int(mins)
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def build_slot_starts(resolution: int) -> np.ndarray:
    """
    Devuelve los minutos de inicio de cada slot.
    resolution=24 -> cada 60 min
    resolution=48 -> cada 30 min
    resolution=12 -> cada 120 min
    """
    if resolution <= 0:
        raise ValueError("resolution debe ser > 0")
    step = 1440 / resolution
    starts = np.array([int(round(i * step)) for i in range(resolution)], dtype=int)
    starts = np.clip(starts, 0, 1439)
    starts = np.unique(starts)
    return starts


# =============================================================================
# 1) ACUMULADOS ANTES DE CUANTIFICAR
# =============================================================================
def add_cumulative_metrics(
    df: pd.DataFrame,
    accum_cols: list[str] = ACCUM_COLS,
    group_cols: list[str] = ["agent", "day"],
    order_col: str = "in",
) -> pd.DataFrame:
    """
    Para cada (agent, day), ordena por 'in' y hace suma acumulada de accum_cols.
    Si 'day' no existe, se crea como 'NA'.

    Nota:
    - Asume que cada fila representa un evento/actividad en secuencia temporal.
    - Si alguna columna de accum_cols no existe, se ignora.
    """
    df = df.copy()

    if "day" not in df.columns:
        df["day"] = "NA"

    # normalizar in para ordenar
    if order_col in df.columns:
        df[order_col] = normalize_numeric(df[order_col])
    else:
        raise ValueError(f"Falta la columna '{order_col}' necesaria para ordenar y acumular.")

    # normalizar y rellenar con 0 las columnas acumulables
    cols_present = [c for c in accum_cols if c in df.columns]
    for c in cols_present:
        df[c] = normalize_numeric(df[c]).fillna(0.0)

    # ordenar + cumsum por grupo
    df = df.sort_values(group_cols + [order_col], ascending=True)
    if cols_present:
        df[cols_present] = df.groupby(group_cols, sort=False)[cols_present].cumsum()

    return df


# =============================================================================
# 2) CUANTIFICACIÓN A SLOTS (CONSERVA TODAS LAS COLUMNAS)
# =============================================================================
def quantify_schedule_keep_all_columns(
    df: pd.DataFrame,
    resolution: int = 24,
    slot_col: str = "time_slot",
    rule: str = "last_started",
) -> pd.DataFrame:
    """
    Expande un schedule con intervalos [in, out) a registros cuantificados por slots,
    conservando TODAS las columnas del input y añadiendo slot_col=HH:MM.

    - Excluye filas con agent en EXCLUDED_AGENTS
    - Usa regla [in <= T < out]
    - Si hay solapes, selecciona una fila según 'rule':
        - last_started: mayor 'in'
        - first_started: menor 'in'
        - longest_remaining: mayor (out - T)
    - No genera filas sin actividad activa
    - No genera hora_fin
    """
    required = {"agent", "in", "out"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df.copy()

    # Excluir agentes no deseados
    df = df[~df["agent"].isin(EXCLUDED_AGENTS)]

    if "day" not in df.columns:
        df["day"] = "NA"

    # Normalizar in/out
    df["in"] = normalize_numeric_minutes(df["in"])
    df["out"] = normalize_numeric_minutes(df["out"])

    # Filtrar filas inválidas
    df = df.dropna(subset=["agent", "day", "in", "out"])
    df = df[df["out"] > df["in"]]
    df = df[(df["in"] >= 0) & (df["in"] < 1440)]
    df = df[(df["out"] > 0) & (df["out"] <= 1440)]

    if df.empty:
        out_cols = list(df.columns)
        if slot_col not in out_cols:
            out_cols.insert(0, slot_col)
        return pd.DataFrame(columns=out_cols)

    slot_starts = build_slot_starts(resolution)
    slot_labels = [minutes_to_hhmm(m) for m in slot_starts]
    label_to_min = dict(zip(slot_labels, slot_starts))

    out_records = []

    for (agent, day), g in df.groupby(["agent", "day"], sort=True):
        g = g.sort_values(["in", "out"], ascending=[True, True]).reset_index(drop=True)

        in_arr = g["in"].to_numpy(dtype=float)
        out_arr = g["out"].to_numpy(dtype=float)
        g_records = g.to_dict(orient="records")

        for T, label in zip(slot_starts, slot_labels):
            mask = (in_arr <= T) & (T < out_arr)
            if not np.any(mask):
                continue

            idxs = np.where(mask)[0]

            if rule == "last_started":
                best_idx = idxs[np.argmax(in_arr[idxs])]
            elif rule == "first_started":
                best_idx = idxs[np.argmin(in_arr[idxs])]
            elif rule == "longest_remaining":
                remaining = out_arr[idxs] - T
                best_idx = idxs[np.argmax(remaining)]
            else:
                raise ValueError("rule no válida. Usa: last_started, first_started, longest_remaining")

            rec = dict(g_records[best_idx])   # copia TODAS las columnas originales
            rec[slot_col] = label
            out_records.append(rec)

    out_df = pd.DataFrame(out_records)

    # Orden final
    if not out_df.empty:
        out_df["_tmin"] = out_df[slot_col].map(label_to_min).astype(int)
        out_df = (
            out_df
            .sort_values(["day", "agent", "_tmin"])
            .drop(columns=["_tmin"])
            .reset_index(drop=True)
        )

        # poner time_slot al principio
        cols = list(out_df.columns)
        cols = [slot_col] + [c for c in cols if c != slot_col]
        out_df = out_df[cols]

    return out_df


# =============================================================================
# 3) EXPORT POR ARCHIVO (NO MEZCLA citizen/vehicle)
# =============================================================================
def quantify_one_excel_to_xlsx(
    input_path: Path,
    output_path: Path,
    resolution: int = 24,
    rule: str = "last_started",
    slot_col: str = "time_slot",
    write_summary: bool = True,
) -> Path:
    """
    Pipeline por archivo:
    1) lee excel
    2) excluye agentes Public_transport / walk
    3) aplica acumulado por agent/day ordenado por in
    4) cuantifica por slots (manteniendo TODAS las columnas)
    5) elimina columnas irrelevantes (COLUMNS_TO_DROP)
    6) exporta a xlsx
    """
    df = pd.read_excel(input_path)

    # Excluir agentes no deseados (a nivel fila de input)
    if "agent" in df.columns:
        df = df[~df["agent"].isin(EXCLUDED_AGENTS)]

    # Acumulado ANTES de cuantificar
    df = add_cumulative_metrics(
        df,
        accum_cols=ACCUM_COLS,
        group_cols=["agent", "day"],
        order_col="in",
    )

    # Cuantificar a slots (time_slot)
    quantified = quantify_schedule_keep_all_columns(
        df=df,
        resolution=resolution,
        slot_col=slot_col,
        rule=rule,
    )

    # Eliminar columnas irrelevantes ANTES de exportar
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in quantified.columns]
    if cols_to_drop:
        quantified = quantified.drop(columns=cols_to_drop)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        quantified.to_excel(writer, index=False, sheet_name="quantified")

        # Resumen opcional
        if write_summary and not quantified.empty:
            # Si existe todo en el resultado final NO (lo eliminamos), entonces resumimos por slots totales
            # (Si quieres summary por actividad, no elimines 'todo' en COLUMNS_TO_DROP)
            summary = (
                quantified.groupby(["day", "agent"], as_index=False)
                .agg(slots=(slot_col, "count"))
            )
            summary.to_excel(writer, index=False, sheet_name="summary")

    return output_path


def build_quantified_outputs_per_excel(
    paths: dict,
    study_area: str,
    resolution: int = 24,
    rule: str = "last_started",
) -> dict:
    """
    Genera un .xlsx de salida para citizen y otro para vehicle, en paths['results'].
    """
    results_dir = Path(paths["results"])

    citizen_in = results_dir / f"{study_area}_schedule_citizen.xlsx"
    vehicle_in = results_dir / f"{study_area}_schedule_vehicle.xlsx"

    if not citizen_in.exists():
        raise FileNotFoundError(f"No existe: {citizen_in}")
    if not vehicle_in.exists():
        raise FileNotFoundError(f"No existe: {vehicle_in}")

    citizen_out = results_dir / f"{study_area}_schedule_citizen_quantified_{resolution}.xlsx"
    vehicle_out = results_dir / f"{study_area}_schedule_vehicle_quantified_{resolution}.xlsx"

    quantify_one_excel_to_xlsx(
        input_path=citizen_in,
        output_path=citizen_out,
        resolution=resolution,
        rule=rule,
        slot_col="time_slot",
        write_summary=True,
    )

    quantify_one_excel_to_xlsx(
        input_path=vehicle_in,
        output_path=vehicle_out,
        resolution=resolution,
        rule=rule,
        slot_col="time_slot",
        write_summary=True,
    )

    return {"citizen": citizen_out, "vehicle": vehicle_out}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Inputs
    population = 450
    study_area = "Annelinn"

    # Paths
    paths, system_management = paths_initialization(study_area)

    # Resolución: 24=1h, 48=30min, 12=2h...
    resolution = 24

    # Selección de fila si hay solapes
    rule = "last_started"

    outputs = build_quantified_outputs_per_excel(
        paths=paths,
        study_area=study_area,
        resolution=resolution,
        rule=rule,
    )

    print("[OK] Outputs generados:")
    print(f" - Citizen: {outputs['citizen']}")
    print(f" - Vehicle: {outputs['vehicle']}")
