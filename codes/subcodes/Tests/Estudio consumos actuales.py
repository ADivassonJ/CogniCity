import math
from pathlib import Path

import pandas as pd
import networkx as nx
import pandapower as pp


# =========================
# CONFIG DE ESTUDIO
# =========================
CASES = {
    "Annelinn": {"root": "Node_43"},
    "Aradas": {"root": "Node_20"},
    "Kanaleneiland": {"root": "Node_67"},
}

# Palabras clave para localizar archivos en Escritorio/Desktop
PATTERNS = {
    "nodes": ["node_data", "nodes", "bus", "buses", "node"],
    "lines": ["line_data", "lines", "branch", "edges", "line"],
    "excel": ["datos_extraidos", "extracted", "data", "inputs", "demand", "limits"],
}

# =========================
# UMBRALES DE ANOMALÍA
# =========================
VN_KV = 20.0
VM_PU_SET = 1.00
V_MIN_PU = 0.95
V_MAX_PU = 1.05
LINE_LOADING_MAX_PCT = 100.0

# Supuestos eléctricos por km (si no tienes impedancias reales)
R_OHM_PER_KM = 0.40
X_OHM_PER_KM = 0.10
C_NF_PER_KM = 10.0

USE_HAVERSINE_LENGTH = True
DEFAULT_LENGTH_KM = 0.3


# =========================
# UTILIDADES: DESKTOP AUTO
# =========================
def candidate_desktops():
    home = Path.home()
    cands = [
        home / "Desktop",
        home / "Escritorio",
        home / "OneDrive" / "Desktop",
        home / "OneDrive" / "Escritorio",
    ]
    return [p for p in cands if p.exists() and p.is_dir()]

def find_file_for_case(case_name: str, kind: str, exts: list[str]):
    case_low = case_name.lower()
    keywords = [k.lower() for k in PATTERNS[kind]]
    exts_low = [e.lower() for e in exts]

    hits = []
    for desk in candidate_desktops():
        for f in desk.iterdir():
            if not f.is_file():
                continue
            name = f.name.lower()
            if case_low not in name:
                continue
            if f.suffix.lower() not in exts_low:
                continue
            if not any(kw in name for kw in keywords):
                continue
            hits.append(f)

    pref = [".xlsx", ".xls", ".csv", ".txt"]
    hits.sort(key=lambda p: (pref.index(p.suffix.lower()) if p.suffix.lower() in pref else 99, len(p.name)))
    return hits[0] if hits else None


# =========================
# UTILIDADES: RED / CÁLCULOS
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def pmax_mw_to_max_i_ka(pmax_mw, vn_kv):
    # S = sqrt(3)*V*I ; asume pf ~ 1 => S(MVA) ~ P(MW)
    return float(pmax_mw) / (math.sqrt(3) * float(vn_kv))

def build_graph(nodes_df, lines_df):
    G = nx.Graph()
    G.add_nodes_from(nodes_df["i"].astype(str).tolist())
    G.add_edges_from(list(zip(lines_df["i"].astype(str), lines_df["j"].astype(str))))
    return G


# =========================
# CONSTRUCCIÓN PANDAPOWER (una sola vez)
# =========================
def build_net_static(nodes_path: Path, lines_path: Path, excel_path: Path, root_bus: str):
    nodes = pd.read_csv(nodes_path)
    lines = pd.read_csv(lines_path)
    nodes["i"] = nodes["i"].astype(str)
    lines["i"] = lines["i"].astype(str)
    lines["j"] = lines["j"].astype(str)

    G = build_graph(nodes, lines)

    xls = pd.ExcelFile(excel_path)
    if "pDemand" not in xls.sheet_names or "pPmax_line" not in xls.sheet_names:
        raise ValueError(f"Excel {excel_path.name} debe tener hojas 'pDemand' y 'pPmax_line'.")

    pDemand = pd.read_excel(xls, "pDemand")
    pPmax_line = pd.read_excel(xls, "pPmax_line")

    # Coordenadas
    coord = nodes.set_index("i")[["lat", "lon"]].to_dict("index")

    # Límites de línea (Pmax en MW)
    lim = {}
    for _, r in pPmax_line.iterrows():
        a = str(r["level_0"]); b = str(r["level_1"])
        lim[frozenset([a, b])] = float(r["values"])

    # Net
    net = pp.create_empty_network(sn_mva=10.0)
    bus_map = {bus: pp.create_bus(net, name=bus, vn_kv=VN_KV) for bus in nodes["i"]}

    if root_bus not in bus_map:
        raise ValueError(f"ROOT '{root_bus}' no existe en el dataset (nodos).")

    pp.create_ext_grid(net, bus=bus_map[root_bus], vm_pu=VM_PU_SET, name=f"GRID_{root_bus}")

    # Líneas
    for _, r in lines.iterrows():
        a, b = r["i"], r["j"]

        if USE_HAVERSINE_LENGTH:
            la1, lo1 = coord[a]["lat"], coord[a]["lon"]
            la2, lo2 = coord[b]["lat"], coord[b]["lon"]
            length_km = max(0.01, haversine_km(la1, lo1, la2, lo2))
        else:
            length_km = DEFAULT_LENGTH_KM

        pmax_mw = lim.get(frozenset([a, b]))
        max_i_ka = pmax_mw_to_max_i_ka(pmax_mw, VN_KV) if pmax_mw is not None else 0.3

        pp.create_line_from_parameters(
            net,
            from_bus=bus_map[a],
            to_bus=bus_map[b],
            length_km=length_km,
            r_ohm_per_km=R_OHM_PER_KM,
            x_ohm_per_km=X_OHM_PER_KM,
            c_nf_per_km=C_NF_PER_KM,
            max_i_ka=max_i_ka,
            name=f"{a}--{b}",
        )

    return net, G, bus_map, pDemand


def init_loads_once(net, bus_map, all_buses_in_demand):
    """
    Crea un load por bus (si no existe) para poder actualizar p_mw por hora sin reconstruir net.
    Devuelve:
      - load_idx_by_bus: dict bus_name -> load_index
    """
    load_idx_by_bus = {}
    for bus_name in all_buses_in_demand:
        if bus_name in bus_map:
            idx = pp.create_load(net, bus=bus_map[bus_name], p_mw=0.0, q_mvar=0.0, name=f"Load_{bus_name}")
            load_idx_by_bus[bus_name] = idx
    return load_idx_by_bus


# =========================
# ANÁLISIS HORARIO + REPORTE SOLO ANOMALÍAS
# =========================
def analyze_all_hours(net, pDemand, load_idx_by_bus):
    """
    Recorre todas las horas en pDemand['Time'] y devuelve lista de anomalías por hora.
    Formato: [(time, [msgs...]), ...] solo para horas con anomalías.
    """
    anomalies = []

    # Normaliza tipos
    pDemand = pDemand.copy()
    pDemand["Buses"] = pDemand["Buses"].astype(str)

    hours = sorted(pDemand["Time"].unique().tolist())

    # Para rapidez: pivot por hora -> (bus -> p)
    # (si hay buses repetidos en una misma hora, sumamos)
    grouped = pDemand.groupby(["Time", "Buses"])["values"].sum()

    for t in hours:
        # 1) poner todas las cargas a 0
        if len(load_idx_by_bus) > 0:
            net.load.loc[:, "p_mw"] = 0.0

        # 2) asignar p_mw de esta hora
        if t in grouped.index.get_level_values(0):
            slice_t = grouped.loc[t]  # Series index=Buses
            for bus_name, p_val in slice_t.items():
                if bus_name in load_idx_by_bus:
                    net.load.at[load_idx_by_bus[bus_name], "p_mw"] = float(p_val)

        # 3) runpp
        msgs = []
        try:
            pp.runpp(net, init="results", calculate_voltage_angles=False)
        except Exception as e:
            msgs.append(f"No converge (runpp): {repr(e)}")
            anomalies.append((t, msgs))
            continue

        # 4) checks
        vm = net.res_bus["vm_pu"]
        loading = net.res_line["loading_percent"]

        under = vm[vm < V_MIN_PU]
        over = vm[vm > V_MAX_PU]
        overline = loading[loading > LINE_LOADING_MAX_PCT]

        if len(under) > 0:
            worst = under.sort_values().head(3)
            msgs.append(
                f"Subtensión (<{V_MIN_PU:.2f} pu) en {len(under)} buses. Peores: "
                + ", ".join([f"{net.bus.at[i,'name']}={worst.loc[i]:.4f}" for i in worst.index])
            )

        if len(over) > 0:
            worst = over.sort_values(ascending=False).head(3)
            msgs.append(
                f"Sobretensión (>{V_MAX_PU:.2f} pu) en {len(over)} buses. Peores: "
                + ", ".join([f"{net.bus.at[i,'name']}={worst.loc[i]:.4f}" for i in worst.index])
            )

        if len(overline) > 0:
            worst = overline.sort_values(ascending=False).head(3)
            msgs.append(
                f"Sobrecarga (>100%) en {len(overline)} líneas. Peores: "
                + ", ".join([f"{net.line.at[i,'name']}={worst.loc[i]:.1f}%" for i in worst.index])
            )

        if msgs:
            anomalies.append((t, msgs))

    return anomalies


def run_case(case_name: str, root_bus: str):
    nodes_path = find_file_for_case(case_name, "nodes", exts=[".csv"])
    lines_path = find_file_for_case(case_name, "lines", exts=[".csv"])
    excel_path = find_file_for_case(case_name, "excel", exts=[".xlsx", ".xls"])

    print("\n" + "=" * 80)
    print(f"CASO: {case_name} | ROOT: {root_bus}")
    print("=" * 80)

    if not nodes_path or not lines_path or not excel_path:
        # Si no encuentra archivos, no puede analizar; informamos (esto no es “anomalía horaria”, es setup)
        print("No he podido localizar automáticamente los 3 archivos en tu Escritorio/Desktop.")
        print(f"  nodes: {nodes_path}")
        print(f"  lines: {lines_path}")
        print(f"  excel: {excel_path}")
        return

    # Construcción estática
    try:
        net, G, bus_map, pDemand = build_net_static(nodes_path, lines_path, excel_path, root_bus)
    except Exception as e:
        print("ERROR construyendo red:", repr(e))
        return

    connected = (nx.number_connected_components(G) == 1)
    radial = nx.is_tree(G)
    print(f"Topología: conectada={connected} | radial(árbol)={radial} | nodos={G.number_of_nodes()} | líneas={G.number_of_edges()}")

    # Inicializa cargas (una por bus que aparezca en pDemand)
    all_buses = sorted(pDemand["Buses"].astype(str).unique().tolist())
    load_idx_by_bus = init_loads_once(net, bus_map, all_buses)

    # Analiza todas las horas y reporta SOLO anomalías
    anomalies = analyze_all_hours(net, pDemand, load_idx_by_bus)

    if not anomalies:
        # Pedido explícito: si no hay anomalías, no digas nada
        return

    for t, msgs in anomalies:
        print(f"\nAnomalía detectada a las {int(t):02d}:00")
        for m in msgs:
            print(f"  - {m}")


def main():
    # Ejecuta 3 casos
    for case_name, cfg in CASES.items():
        run_case(case_name, cfg["root"])


if __name__ == "__main__":
    main()
