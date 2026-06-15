"""
Merge synthetic population Excel files across two result folders.

For each SCENARIO and each FILE, generates 10 cumulative Excel files:
  output/<scenario>/<file_base>_x01.xlsx  ← pop(1) only
  output/<scenario>/<file_base>_x02.xlsx  ← pop(1) + pop(2)
  output/<scenario>/<file_base>_x03.xlsx  ← pop(1) + pop(2) + pop(1)
  ...
  output/<scenario>/<file_base>_x10.xlsx  ← 5×pop(1) + 5×pop(2)

ID remapping: citizen_X and family_X are renumbered with cumulative offsets
to guarantee no collisions across merged blocks.
"""

import pandas as pd
import re
from pathlib import Path

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
PATHS = [
    r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (1)",
    r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (2)",
]


SCENARIOS = [f"s{i}" for i in range(5)]
FILES = [
    "Annelinn_schedule_citizen.xlsx",
    "Aradas_schedule_citizen.xlsx",
    "Kanaleneiland_schedule_citizen.xlsx",
]
N_COPIES = 20          
OUTPUT_ROOT = Path(r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results")   # Carpeta de salida (relativa al script)
# ──────────────────────────────────────────────────────────────────────────────


# ── ID columns to remap ───────────────────────────────────────────────────────
# Each entry: (column_name, prefix)  →  e.g. 'citizen_12' uses prefix 'citizen'
# Add more pairs here if your files use other ID patterns.
ID_COLUMNS = [
    ("agent",  "citizen"),
    ("family", "family"),
]


def get_max_id(df: pd.DataFrame, col: str, prefix: str) -> int:
    """Return the max numeric suffix for IDs like 'prefix_N' in column col."""
    if col not in df.columns:
        return -1
    ids = df[col].astype(str).str.extract(rf'^{re.escape(prefix)}_(\d+)$')[0]
    ids = ids.dropna().astype(int)
    return int(ids.max()) if len(ids) else -1


def remap_column(series: pd.Series, prefix: str, offset: int) -> pd.Series:
    """Add offset to all 'prefix_N' values in a Series."""
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)$')
    def shift(val):
        m = pattern.fullmatch(str(val))
        return f'{prefix}_{int(m.group(1)) + offset}' if m else val
    return series.apply(shift)


def remap_ids(df: pd.DataFrame, offsets: dict[str, int]) -> pd.DataFrame:
    """Apply all ID offsets to a copy of df."""
    df = df.copy()
    for col, prefix in ID_COLUMNS:
        if col in df.columns and offsets.get(prefix, 0) != 0:
            df[col] = remap_column(df[col], prefix, offsets[prefix])
    return df


def merge_file(path1: Path, path2: Path, out_dir: Path, base_name: str):
    """Stack N_COPIES alternating copies of path1/path2, saving cumulative files."""
    df1 = pd.read_excel(path1)
    df2 = pd.read_excel(path2)

    cumulative_offsets = {prefix: 0 for _, prefix in ID_COLUMNS}
    frames = []

    for i in range(N_COPIES):
        source_df = df1 if i % 2 == 0 else df2
        chunk = remap_ids(source_df, cumulative_offsets)
        frames.append(chunk)

        # Advance offsets by the max IDs in this source block
        for col, prefix in ID_COLUMNS:
            max_id = get_max_id(source_df, col, prefix)
            if max_id >= 0:
                cumulative_offsets[prefix] += max_id + 1

        combined = pd.concat(frames, ignore_index=True)
        out_path = out_dir / f"{base_name}_x{i+1:02d}.xlsx"
        combined.to_excel(out_path, index=False)

        # Summary
        agent_col = next((c for c, _ in ID_COLUMNS if c in combined.columns), None)
        n_agents = combined[agent_col].nunique() if agent_col else "?"
        print(f"    [{i+1:02d}/{N_COPIES}] {out_path.name}  —  {n_agents} agents, {len(combined)} rows")


def main():
    print(f"Output root: {OUTPUT_ROOT.resolve()}\n")

    for scenario in SCENARIOS:
        print(f"═══ Scenario: {scenario} ═══")
        for file_name in FILES:
            p1 = Path(PATHS[0]) / scenario / file_name
            p2 = Path(PATHS[1]) / scenario / file_name

            if not p1.exists():
                print(f"  ⚠  Missing: {p1}")
                continue
            if not p2.exists():
                print(f"  ⚠  Missing: {p2}")
                continue

            print(f"  ► {file_name}")
            out_dir = OUTPUT_ROOT
            out_dir.mkdir(parents=True, exist_ok=True)

            base_name = Path(file_name).stem
            merge_file(p1, p2, out_dir, base_name)

        print()

    print("✔ All done.")


if __name__ == "__main__":
    main()