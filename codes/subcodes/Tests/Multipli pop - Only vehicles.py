import re
import pandas as pd
from pathlib import Path

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
PATHS = [
    r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (1)",
    r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (2)",
]

SCENARIOS  = [f"s{i}" for i in range(5)]
FILES      = [
    "Annelinn_schedule_vehicle.xlsx",
    "Aradas_schedule_vehicle.xlsx",
    "Kanaleneiland_schedule_vehicle.xlsx",
]

N_COPIES   = 10
OUTPUT_ROOT = Path(r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results")

# ── Prefijos a renumerar y en qué columnas aparecen ───────────────────────────
# Ajusta si el schedule_citizen tiene columnas distintas
ID_SPECS = [
    ("citizen_",      ["agent", "user"]),
    ("family_",       ["agent", "family"]),
    ("priv_vehicle_", ["agent"]),
]

# ── helpers ────────────────────────────────────────────────────────────────────

def get_num(value: str, prefix: str) -> int | None:
    """Extrae el número de 'citizen_42' → 42. Devuelve None si no encaja."""
    m = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(value))
    return int(m.group(1)) if m else None


def max_id(df: pd.DataFrame, prefix: str, cols: list) -> int:
    """
    Máximo numérico del prefijo dado buscando en todas las columnas indicadas.
    Usa max() sobre el conjunto completo, no la última fila.
    """
    nums = []
    for col in cols:
        if col not in df.columns:
            continue
        for v in df[col].dropna().unique():
            n = get_num(v, prefix)
            if n is not None:
                nums.append(n)
    return max(nums) if nums else -1


def shift_ids(df: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Remapeo estándar (igual que antes) ────────────────────────────────
    for prefix, cols in ID_SPECS:
        offset = offsets[prefix]

        unique_vals = sorted(
            {v for col in cols if col in df.columns
               for v in df[col].dropna().unique()
               if get_num(v, prefix) is not None},
            key=lambda v: get_num(v, prefix)
        )

        mapping = {v: f"{prefix}{offset + i + 1}" for i, v in enumerate(unique_vals)}

        for col in cols:
            if col in df.columns:
                df[col] = df[col].map(lambda v, m=mapping: m.get(v, v))

    # ── 2. FIX ESPECÍFICO: agent depende de user ─────────────────────────────
    # (esto se ejecuta DESPUÉS de actualizar user)
    if "agent" in df.columns and "user" in df.columns:
        pattern = r"(virtual_vehicle_\([A-Z0-9]+\)_citizen_)(\d+)"

        def fix_agent(row):
            agent = str(row["agent"])
            user = str(row["user"])

            if re.search(pattern, agent):
                user_num = user.split("_")[-1]

                return re.sub(
                    pattern,
                    lambda m: f"{m.group(1)}{user_num}",
                    agent
                )

            return agent

        df["agent"] = df.apply(fix_agent, axis=1)

    return df


def current_offsets(df: pd.DataFrame) -> dict:
    """Devuelve los offsets actuales (máximos) de cada prefijo en df."""
    return {prefix: max_id(df, prefix, cols) for prefix, cols in ID_SPECS}


# ── lógica principal ───────────────────────────────────────────────────────────

def merge_file(path1: Path, path2: Path, output: Path, n_copies: int) -> None:
    pop1 = pd.read_excel(path1)
    pop2 = pd.read_excel(path2)

    chunks = []
    accumulated = None

    for i in range(n_copies):
        source = pop1 if i % 2 == 0 else pop2
        label  = "pop1" if i % 2 == 0 else "pop2"

        if accumulated is None:
            # Primera copia: IDs originales sin tocar
            chunk = source.copy()
        else:
            # Calcular offsets desde el máximo real del acumulado hasta ahora
            offsets = current_offsets(accumulated)
            chunk = shift_ids(source, offsets)

        chunks.append(chunk)
        accumulated = pd.concat(chunks, ignore_index=True)
        print(f"    copia {i+1:>2} ({label}) → filas acumuladas: {len(accumulated):,}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.with_suffix(".csv")
    accumulated.to_csv(output, index=False)
    counts = count_unique_ids(accumulated)
    print(
        f"  ✓ Guardado: {output}\n"
        f"    Filas: {len(accumulated):,}\n"
        f"    Citizens: {counts['citizen_']:,}\n"
        f"    Families: {counts['family_']:,}\n"
        f"    Vehicles: {counts['priv_vehicle_']:,}\n"
    )

def count_unique_ids(df: pd.DataFrame) -> dict:
    counts = {}
    for prefix, cols in ID_SPECS:
        ids = {
            v for col in cols if col in df.columns
            for v in df[col].dropna().unique()
            if str(v).startswith(prefix)
        }
        counts[prefix] = len(ids)
    return counts

def main():
    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  Escenario: {scenario}")
        print(f"{'='*60}")
        for fname in FILES:
            p1 = Path(PATHS[0]) / scenario / fname
            p2 = Path(PATHS[1]) / scenario / fname
            if not p1.exists():
                print(f"  [SKIP] No encontrado: {p1}")
                continue
            if not p2.exists():
                print(f"  [SKIP] No encontrado: {p2}")
                continue

            out = OUTPUT_ROOT/ f"{scenario}_{fname}"
            print(f"\n  Fichero: {fname}")
            merge_file(p1, p2, out, n_copies=N_COPIES)


if __name__ == "__main__":
    main()