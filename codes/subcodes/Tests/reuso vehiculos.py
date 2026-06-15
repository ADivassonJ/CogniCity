"""
analyze_cs_conflicts.py
-----------------------
Para cada fichero *_schedule_vehicle_*.xlsx / *.csv de una carpeta,
detecta si algún CS_electric sale de su nodo DESPUÉS de que otro vehículo
(del mismo nodo) ya haya vuelto a casa.

Uso:
    python analyze_cs_conflicts.py --folder /ruta/a/tu/carpeta
    python analyze_cs_conflicts.py --folder /ruta/a/tu/carpeta --output resultados.xlsx
"""

import argparse
import glob
import os
import pandas as pd

EXCLUDED_ARCH   = {'walk', 'UB_diesel'}
EXCLUDED_AGENTS = {'Public_transport', 'walk'}


def min_to_hhmm(minutes):
    if pd.isna(minutes):
        return 'N/A'
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h:02d}:{m:02d}"


def load_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext in ('.xlsx', '.xlsm'):
        return pd.read_excel(path)
    elif ext == '.csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Formato no soportado: {ext}")


def analyze_file(path: str) -> pd.DataFrame:
    df = load_file(path)
    day_tag = os.path.basename(path)

    # Filtrar vehículos privados y eléctricos (excluir walk, PT y UB)
    df_v = df[
        ~df['archetype'].isin(EXCLUDED_ARCH) &
        ~df['agent'].isin(EXCLUDED_AGENTS)
    ].copy()

    home_events = df_v[df_v['todo'].isin(['Home_out', 'Home_in'])].copy()

    home_out = (
        home_events[home_events['todo'] == 'Home_out']
        [['agent', 'archetype', 'node', 'out', 'day']]
        .rename(columns={'node': 'home_node', 'out': 'departure'})
    )

    home_in = (
        home_events[home_events['todo'] == 'Home_in']
        [['agent', 'in']]
        .rename(columns={'in': 'arrival'})
    )

    merged = home_out.merge(home_in, on='agent', how='inner')

    cs_nodes = merged[merged['archetype'] == 'CS_electric']['home_node'].unique()

    issues = []
    for node in cs_nodes:
        node_df = merged[merged['home_node'] == node]
        cs_rows = node_df[node_df['archetype'] == 'CS_electric']

        for _, cs_row in cs_rows.iterrows():
            # Otros agentes en el mismo nodo que ya llegaron antes de que el CS salga
            already_home = node_df[
                (node_df['agent'] != cs_row['agent']) &
                (node_df['arrival'] <= cs_row['departure'])
            ]
            for _, ah in already_home.iterrows():
                issues.append({
                    'file':               day_tag,
                    'day':                cs_row.get('day', '?'),
                    'node':               node,
                    'CS_agent':           cs_row['agent'],
                    'CS_departure_min':   cs_row['departure'],
                    'CS_departure_hhmm':  min_to_hhmm(cs_row['departure']),
                    'other_agent':        ah['agent'],
                    'other_archetype':    ah['archetype'],
                    'other_arrival_min':  ah['arrival'],
                    'other_arrival_hhmm': min_to_hhmm(ah['arrival']),
                    'delta_min':          cs_row['departure'] - ah['arrival'],
                })

    return pd.DataFrame(issues)


def hourly_occupancy(path: str) -> pd.DataFrame:
    """
    Para cada nodo con CS_electric, cuenta cuántos vehículos (no walk/PT)
    están 'en casa' (entre arrival y siguiente departure) por hora del día.
    """
    df = load_file(path)

    df_v = df[
        ~df['archetype'].isin(EXCLUDED_ARCH) &
        ~df['agent'].isin(EXCLUDED_AGENTS)
    ].copy()

    home_events = df_v[df_v['todo'].isin(['Home_out', 'Home_in'])].copy()

    home_out = (
        home_events[home_events['todo'] == 'Home_out']
        [['agent', 'archetype', 'node', 'out']]
        .rename(columns={'node': 'home_node', 'out': 'departure'})
    )
    home_in = (
        home_events[home_events['todo'] == 'Home_in']
        [['agent', 'in']]
        .rename(columns={'in': 'arrival'})
    )

    merged = home_out.merge(home_in, on='agent', how='inner')
    cs_nodes = set(merged[merged['archetype'] == 'CS_electric']['home_node'].unique())

    records = []
    for hour in range(24):
        t = hour * 60
        for node in cs_nodes:
            node_df = merged[merged['home_node'] == node]
            # En casa = ya volvió Y aún no ha salido de nuevo
            at_home = node_df[(node_df['arrival'] <= t) & (node_df['departure'] > t)]
            cs_at_home = at_home[at_home['archetype'] == 'CS_electric']
            records.append({
                'node': node,
                'hour': hour,
                'hhmm': f"{hour:02d}:00",
                'total_vehicles_home': len(at_home),
                'cs_electric_home': len(cs_at_home),
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='Carpeta con los ficheros schedule_vehicle')
    parser.add_argument('--output', default='cs_conflicts_report.xlsx', help='Fichero de salida')
    args = parser.parse_args()

    patterns = ['*schedule_vehicle*.xlsx', '*schedule_vehicle*.csv']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(args.folder, p)))

    if not files:
        print(f"[!] No se encontraron ficheros en: {args.folder}")
        return

    all_conflicts = []
    all_occupancy = []

    for f in sorted(files):
        print(f"  Procesando: {os.path.basename(f)}")
        try:
            conflicts = analyze_file(f)
            occupancy = hourly_occupancy(f)
            all_conflicts.append(conflicts)
            all_occupancy.append(occupancy)
        except Exception as e:
            print(f"    [ERROR] {e}")

    df_conflicts = pd.concat(all_conflicts, ignore_index=True) if all_conflicts else pd.DataFrame()
    df_occupancy = pd.concat(all_occupancy, ignore_index=True) if all_occupancy else pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"Total conflictos encontrados: {len(df_conflicts)}")
    if not df_conflicts.empty:
        print(df_conflicts.to_string(index=False))

    # Guardar Excel con dos pestañas
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df_conflicts.to_excel(writer, sheet_name='Conflictos', index=False)
        df_occupancy.to_excel(writer, sheet_name='Ocupacion_por_hora', index=False)

    print(f"\nResultados guardados en: {args.output}")


# --- Uso directo sin CLI (para pruebas rápidas) ---
if __name__ == '__main__':
    import sys
    if '--folder' in sys.argv:
        main()
    else:
        # Modo demo con el fichero de ejemplo
        TEST_FILE = r'C:\Users\asier.divasson\Documents\GitHub\CogniCity\results (0)\s0\Aradas_schedule_vehicle.xlsx'
        print(f"=== CONFLICTOS ===")
        conflicts = analyze_file(TEST_FILE)

        print(f"\n=== OCUPACIÓN POR HORA (nodos CS_electric) ===")
        occ = hourly_occupancy(TEST_FILE)
        print(occ.to_string(index=False))