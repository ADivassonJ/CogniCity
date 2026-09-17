import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import geopandas as gpd
from shapely.geometry import Polygon, box

from collections import deque, defaultdict
import math

from matplotlib.ticker import MaxNLocator


# ========================================================
# CONFIGURACIÓN GENERAL
# ========================================================

project_root = Path(
    r"C:/Users/asier.divasson/Documents/GitHub/CogniCity"
)

# Distritos que queremos analizar
study_cases = [
    "Kanaleneiland",
    "Oog in Al",
]

# Escenarios que se suman
scenarios = [
    f"s{i:02d}"
    for i in range(1, 11)
]

# Arquetipo
archetype_folder = "PC_electric"

# Escenario utilizado para leer la geometría/topología.
# Se asume que la red y los edificios son iguales en
# todos los escenarios.
population_scenario = "s01"

# Tamaño de celda
CELL_SIZE = 50.0

# Raíz de la red
root = "Node_20"


# ========================================================
# COLORES DE LOS DISTRITOS
# ========================================================

DISTRICT_COLORS = {

    # Azul
    "Kanaleneiland": "#134f5c",

    # Magenta
    "Oog in Al": "#741b47",
}


# ========================================================
# FUNCIONES AUXILIARES
# ========================================================

CRS_WGS84 = "EPSG:4326"


def hex_to_rgb01(hex_color):
    """
    Convierte #RRGGBB a RGB en [0,1].
    """
    hex_color = hex_color.lstrip("#")

    return tuple(
        int(hex_color[i:i + 2], 16) / 255
        for i in (0, 2, 4)
    )


def lerp_rgb(c1, c2, t):
    """
    Interpolación lineal entre dos colores RGB.
    """

    t = float(
        np.clip(t, 0.0, 1.0)
    )

    return (
        c1[0] + t * (c2[0] - c1[0]),
        c1[1] + t * (c2[1] - c1[1]),
        c1[2] + t * (c2[2] - c1[2]),
    )


def to_node_label(x):
    """
    Convierte:
        43      -> Node_43
        '43'    -> Node_43
        Node_43 -> Node_43
    """

    s = str(x)

    return (
        s
        if s.startswith("Node_")
        else f"Node_{s}"
    )


def normalize_id(x):
    """
    Normaliza IDs:
        43
        43.0
        '43'
    -> '43'
    """

    if pd.isna(x):
        return None

    try:

        f = float(x)

        if f.is_integer():
            return str(int(f))

    except (
        TypeError,
        ValueError
    ):
        pass

    return str(x).strip()


def utm_epsg_for(lon, lat):
    """
    Calcula el EPSG UTM correspondiente.
    """

    zone = int(
        (lon + 180) / 6
    ) + 1

    epsg = (
        32600 + zone
        if lat >= 0
        else 32700 + zone
    )

    return f"EPSG:{epsg}"


def haversine_km(
    lon1,
    lat1,
    lon2,
    lat2
):
    """
    Distancia haversine en km.
    """

    R = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(
        lat2 - lat1
    )

    dlmb = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dphi / 2) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlmb / 2) ** 2
    )

    return (
        2
        * R
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )


# ========================================================
# BOUNDARIES
# ========================================================

def get_boundary(study_case):

    if study_case == "Kanaleneiland":

        boundary_latlon = [

            (52.07893, 5.08148),
            (52.08112, 5.09062),
            (52.06719, 5.10358),
            (52.06656, 5.10748),
            (52.06553, 5.10796),
            (52.06595, 5.10379),
            (52.06308, 5.09525),
            (52.07663, 5.08268),
            (52.07893, 5.08148),

        ]

    elif study_case == "Oog in Al":

        boundary_latlon = [

            (52.09245, 5.08429),
            (52.08747, 5.09429),
            (52.08267, 5.08766),
            (52.08337, 5.08549),
            (52.08551, 5.08354),
            (52.09003, 5.08270),
            (52.09245, 5.08429),

        ]

    else:

        raise ValueError(
            f"No hay boundary definido para {study_case}"
        )

    boundary_lonlat = [
        (lon, lat)
        for lat, lon in boundary_latlon
    ]

    return Polygon(boundary_lonlat)


# ========================================================
# PROCESAR UN DISTRITO
# ========================================================

def process_study_case(study_case):

    print()
    print("=" * 70)
    print(f"PROCESANDO: {study_case}")
    print("=" * 70)

    # ----------------------------------------------------
    # A) LEER LOS 10 ESCENARIOS
    # ----------------------------------------------------

    excel_filename = (
        f"{study_case}_schedule_vehicle_quantified_24.xlsx"
    )

    dfs = []

    for scenario in scenarios:

        results_dir = (
            project_root
            / "results"
            / scenario
        )

        excel_path = (
            results_dir
            / excel_filename
        )

        if not excel_path.exists():

            print(
                f"[AVISO] No existe: {excel_path}"
            )

            continue

        print(
            f"[INFO] Leyendo {study_case} — {scenario}"
        )

        df_scenario = pd.read_excel(
            excel_path
        )

        # Solo vehículos eléctricos
        df_scenario = df_scenario[
            df_scenario["archetype"]
            == archetype_folder
        ].copy()

        # Guardamos escenario
        df_scenario[
            "scenario"
        ] = scenario

        dfs.append(
            df_scenario
        )

    if not dfs:

        raise FileNotFoundError(
            f"No se encontró ningún Excel "
            f"para {study_case}."
        )

    # Unir escenarios
    df = pd.concat(
        dfs,
        ignore_index=True
    )

    print(
        f"[INFO] Escenarios encontrados: "
        f"{sorted(df['scenario'].unique())}"
    )

    print(
        f"[INFO] Filas totales: "
        f"{len(df)}"
    )

    # ----------------------------------------------------
    # B) ÚLTIMO TIME_SLOT
    # ----------------------------------------------------
    #
    # MUY IMPORTANTE:
    # incluimos scenario en el groupby para no mezclar
    # agentes de s01, s02, etc.
    # ----------------------------------------------------

    ts = pd.to_datetime(
        df["time_slot"],
        format="%H:%M",
        errors="coerce"
    )

    df[
        "time_slot_minutes"
    ] = (
        ts.dt.hour * 60
        + ts.dt.minute
    )

    df_last_per_agent = (

        df.sort_values(

            by=[
                "scenario",
                "agent",
                "day",
                "time_slot_minutes"
            ],

            ascending=[
                True,
                True,
                True,
                False
            ]
        )

        .groupby(
            [
                "scenario",
                "agent",
                "day"
            ],
            as_index=False
        )

        .first()

        .drop(
            columns=[
                "time_slot_minutes"
            ]
        )
    )

    print(
        f"[INFO] Registros después de "
        f"seleccionar último time_slot: "
        f"{len(df_last_per_agent)}"
    )

    # ----------------------------------------------------
    # C) MJ -> FLOAT
    # ----------------------------------------------------

    df_last_per_agent[
        "mjkm"
    ] = (

        df_last_per_agent[
            "mjkm"
        ]

        .astype(str)

        .str.replace(
            ",",
            ".",
            regex=False
        )
    )

    df_last_per_agent[
        "mjkm"
    ] = pd.to_numeric(
        df_last_per_agent["mjkm"],
        errors="coerce"
    ).fillna(0.0)

    # ----------------------------------------------------
    # D) SUMA DE LOS 10 ESCENARIOS POR NODE
    # ----------------------------------------------------

    df_mj_by_node = (

        df_last_per_agent

        .groupby(
            "node",
            as_index=False
        )["mjkm"]

        .sum()

        .rename(
            columns={
                "mjkm": "MJ_total"
            }
        )
    )

    # MJ -> kWh
    df_mj_by_node[
        "kWh_total"
    ] = (
        df_mj_by_node[
            "MJ_total"
        ] / 3.6
    )

    total_kwh = (
        df_mj_by_node[
            "kWh_total"
        ].sum()
    )

    print(
        f"[INFO] Consumo total "
        f"{study_case}, s01-s10: "
        f"{total_kwh:,.2f} kWh"
    )

    # node -> kWh
    kwh_map = dict(
        zip(
            df_mj_by_node["node"],
            df_mj_by_node["kWh_total"]
        )
    )

    # ----------------------------------------------------
    # E) LEER RED / POBLACIÓN
    # ----------------------------------------------------
    #
    # Se utiliza s01 únicamente para obtener:
    # - node_data
    # - line_data
    # - pop_building
    #
    # El consumo sí viene de s01-s10.
    # ----------------------------------------------------

    path = (
        project_root
        / "data"
        / population_scenario
        / study_case
        / "population"
    )

    nodes = pd.read_csv(
        path
        / f"node_data_{study_case}.csv"
    )

    edges = pd.read_csv(
        path
        / f"{study_case}_line_data.csv"
    )

    # ----------------------------------------------------
    # F) BOUNDARY
    # ----------------------------------------------------

    boundary_poly = get_boundary(
        study_case
    )

    centroid = (
        boundary_poly.centroid
    )

    CRS_METRIC = utm_epsg_for(
        centroid.x,
        centroid.y
    )

    boundary_gdf_wgs = (
        gpd.GeoDataFrame(
            geometry=[
                boundary_poly
            ],
            crs=CRS_WGS84
        )
    )

    boundary_gdf_m = (
        boundary_gdf_wgs
        .to_crs(CRS_METRIC)
    )

    boundary_poly_m = (
        boundary_gdf_m
        .geometry
        .iloc[0]
    )

    # ----------------------------------------------------
    # G) NODE LABELS
    # ----------------------------------------------------

    nodes = nodes.copy()

    nodes[
        "node_label"
    ] = (
        nodes["i"]
        .apply(to_node_label)
    )

    # ----------------------------------------------------
    # H) POP_BUILDING
    # ----------------------------------------------------

    pop_building_path = (
        path
        / "pop_building.parquet"
    )

    pop_building = pd.read_parquet(
        pop_building_path
    )

    print(
        f"[DEBUG] pop_building: "
        f"{len(pop_building)} filas"
    )

    OSM_ID_COL = "osm_id"
    NODE_COL = "node"
    LAT_COL = "lat"
    LON_COL = "lon"

    pop_building = (

        pop_building[
            [
                OSM_ID_COL,
                NODE_COL,
                LAT_COL,
                LON_COL
            ]
        ]

        .dropna(
            subset=[
                LAT_COL,
                LON_COL,
                NODE_COL
            ]
        )

        .rename(
            columns={
                OSM_ID_COL: "osm_id",
                NODE_COL: "node",
                LAT_COL: "lat",
                LON_COL: "lon",
            }
        )

        .copy()
    )

    pop_building[
        "node"
    ] = (
        pop_building[
            "node"
        ]
        .apply(normalize_id)
    )

    pop_building[
        "osm_id"
    ] = (
        pop_building[
            "osm_id"
        ].astype(str)
    )

    # ----------------------------------------------------
    # I) NODE -> kWh
    # ----------------------------------------------------

    kwh_by_node = pd.DataFrame({

        "node": [
            normalize_id(k)
            for k in kwh_map.keys()
        ],

        "kWh_total": list(
            kwh_map.values()
        )

    })

    print(
        f"[DEBUG] Nodes con consumo: "
        f"{len(kwh_by_node)}"
    )

    # ----------------------------------------------------
    # J) NODE -> EDIFICIOS
    # ----------------------------------------------------

    poi_df = kwh_by_node.merge(
        pop_building,
        on="node",
        how="inner"
    )

    nodes_sin_match = (
        kwh_by_node.loc[
            ~kwh_by_node["node"].isin(
                pop_building["node"]
            ),
            "node"
        ]
        .tolist()
    )

    print(
        f"[DEBUG] Edificios encontrados: "
        f"{len(poi_df)}"
    )

    if nodes_sin_match:

        print(
            "[AVISO] Nodes sin match:"
        )

        print(
            nodes_sin_match
        )

    # ----------------------------------------------------
    # K) REPARTIR KWH ENTRE EDIFICIOS
    # ----------------------------------------------------

    edificios_por_node = (

        poi_df

        .groupby("node")[
            "node"
        ]

        .transform("count")
    )

    poi_df[
        "kWh_total"
    ] = (

        poi_df[
            "kWh_total"
        ]

        / edificios_por_node
    )

    # ----------------------------------------------------
    # L) CONVERTIR A GEODATAFRAME
    # ----------------------------------------------------

    poi_gdf = (

        gpd.GeoDataFrame(

            poi_df,

            geometry=gpd.points_from_xy(
                poi_df["lon"],
                poi_df["lat"]
            ),

            crs=CRS_WGS84,
        )

        .to_crs(CRS_METRIC)
    )

    # ----------------------------------------------------
    # M) CREAR MALLA 100x100
    # ----------------------------------------------------

    minx, miny, maxx, maxy = (
        boundary_poly_m.bounds
    )

    minx = (
        np.floor(
            minx / CELL_SIZE
        )
        * CELL_SIZE
    )

    miny = (
        np.floor(
            miny / CELL_SIZE
        )
        * CELL_SIZE
    )

    maxx = (
        np.ceil(
            maxx / CELL_SIZE
        )
        * CELL_SIZE
    )

    maxy = (
        np.ceil(
            maxy / CELL_SIZE
        )
        * CELL_SIZE
    )

    x_edges = np.arange(
        minx,
        maxx + CELL_SIZE,
        CELL_SIZE
    )

    y_edges = np.arange(
        miny,
        maxy + CELL_SIZE,
        CELL_SIZE
    )

    cells = []

    for x0 in x_edges[:-1]:

        for y0 in y_edges[:-1]:

            cell = box(
                x0,
                y0,
                x0 + CELL_SIZE,
                y0 + CELL_SIZE
            )

            if cell.intersects(
                boundary_poly_m
            ):

                cells.append(
                    cell
                )

    grid_gdf = gpd.GeoDataFrame(
        geometry=cells,
        crs=CRS_METRIC
    )

    grid_gdf[
        "cell_id"
    ] = grid_gdf.index

    # Recortar al boundary
    grid_gdf[
        "geometry"
    ] = (

        grid_gdf.geometry

        .intersection(
            boundary_poly_m
        )
    )

    grid_gdf = grid_gdf[
        ~grid_gdf.geometry.is_empty
    ].copy()

    # ----------------------------------------------------
    # N) AGREGAR KWH A CADA CELDA
    # ----------------------------------------------------

    join = gpd.sjoin(

        poi_gdf[
            [
                OSM_ID_COL,
                "kWh_total",
                "geometry"
            ]
        ],

        grid_gdf[
            [
                "cell_id",
                "geometry"
            ]
        ],

        predicate="within",

        how="left",
    )

    kwh_by_cell = (

        join

        .groupby(
            "cell_id"
        )["kWh_total"]

        .sum()
    )

    grid_gdf[
        "kWh_total"
    ] = (

        grid_gdf[
            "cell_id"
        ]

        .map(
            kwh_by_cell
        )

        .fillna(0.0)
    )

    # ----------------------------------------------------
    # O) TOPOLOGÍA DE RED
    # ----------------------------------------------------

    adj = defaultdict(list)

    for _, row in edges.iterrows():

        a = to_node_label(
            row["i"]
        )

        b = to_node_label(
            row["j"]
        )

        adj[a].append(b)
        adj[b].append(a)

    # BFS
    parent = {
        root: None
    }

    children = defaultdict(list)

    order = []

    q = deque([root])

    while q:

        u = q.popleft()

        order.append(u)

        for v in adj[u]:

            if v not in parent:

                parent[v] = u

                children[u].append(v)

                q.append(v)

    # ----------------------------------------------------
    # P) COORDENADAS DE LOS NODOS
    # ----------------------------------------------------

    coord_map = dict(
        zip(
            nodes["node_label"],
            zip(
                nodes["lon"],
                nodes["lat"]
            )
        )
    )

    # ----------------------------------------------------
    # Q) LONGITUD DE LAS ARISTAS
    # ----------------------------------------------------

    edge_len = {}

    for child, p in parent.items():

        if p is None:
            continue

        if (
            p in coord_map
            and child in coord_map
        ):

            lon1, lat1 = (
                coord_map[p]
            )

            lon2, lat2 = (
                coord_map[child]
            )

            edge_len[
                (p, child)
            ] = haversine_km(
                lon1,
                lat1,
                lon2,
                lat2
            )

        else:

            edge_len[
                (p, child)
            ] = 0.0

    # ----------------------------------------------------
    # R) DISTANCIA DESDE ROOT
    # ----------------------------------------------------

    dist_from_root = {
        root: 0.0
    }

    for u in order:

        if u not in dist_from_root:
            continue

        for ch in children.get(
            u,
            []
        ):

            dist_from_root[
                ch
            ] = (

                dist_from_root[u]

                + edge_len.get(
                    (u, ch),
                    0.0
                )
            )

    reachable_nodes = list(
        dist_from_root.keys()
    )

    # ----------------------------------------------------
    # S) KWH ACUMULADO POR SUBÁRBOL
    # ----------------------------------------------------

    kwh_map_complete = {

        lab: float(
            kwh_map.get(
                lab,
                0.0
            )
        )

        for lab in parent.keys()
    }

    kwh_cum = {

        lab: kwh_map_complete.get(
            lab,
            0.0
        )

        for lab in parent.keys()
    }

    for u in reversed(order):

        for ch in children.get(
            u,
            []
        ):

            kwh_cum[u] += (
                kwh_cum.get(
                    ch,
                    0.0
                )
            )

    # ----------------------------------------------------
    # T) RESULTADOS
    # ----------------------------------------------------

    return {

        "study_case": study_case,

        "grid": grid_gdf,

        "boundary": boundary_gdf_m,

        "poi": poi_gdf,

        "nodes": nodes,

        "edges": edges,

        "parent": parent,

        "children": children,

        "order": order,

        "dist_from_root":
            dist_from_root,

        "reachable_nodes":
            reachable_nodes,

        "kwh_map":
            kwh_map,

        "kwh_cum":
            kwh_cum,

        "total_kwh":
            total_kwh,

        "crs_metric":
            CRS_METRIC,

    }


# ========================================================
# PROCESAR LOS DOS DISTRITOS
# ========================================================

results = {}

for study_case in study_cases:

    results[
        study_case
    ] = process_study_case(
        study_case
    )


# ========================================================
# ESCALA COMÚN PARA LOS DOS DISTRITOS
# ========================================================
#
# MUY IMPORTANTE:
# ambos mapas utilizan el mismo máximo.
#
# Por tanto, un mismo color representa el mismo kWh.
# ========================================================

print()
print("=" * 70)
print("RESUMEN")
print("=" * 70)

for study_case in study_cases:

    total = (
        results[
            study_case
        ]["total_kwh"]
    )

    print(
        f"{study_case}: "
        f"{total:,.2f} kWh"
    )


# ========================================================
# FIGURA 1
# MAPAS DE LOS DOS DISTRITOS
#
# Cada distrito tiene SU PROPIA escala:
#   0 -> máximo de ese distrito
# ========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 8)
)

axes = np.atleast_1d(axes)


for ax, study_case in zip(
    axes,
    study_cases
):

    result = results[study_case]

    grid_gdf = result["grid"]
    boundary_gdf_m = result["boundary"]
    poi_gdf = result["poi"]

    district_color = DISTRICT_COLORS[
        study_case
    ]

    # ----------------------------------------------------
    # MÁXIMO PROPIO DEL DISTRITO
    # ----------------------------------------------------

    vals_pos = grid_gdf.loc[
        grid_gdf["kWh_total"] > 0,
        "kWh_total"
    ].to_numpy()

    if len(vals_pos) > 0:
        district_max = float(vals_pos.max())
    else:
        district_max = 1.0

    print(
        f"[INFO] {study_case} — "
        f"máximo celda: {district_max:,.2f} kWh"
    )

    # ----------------------------------------------------
    # GRADIENTE PROPIO
    # ----------------------------------------------------

    COLOR_ZERO = hex_to_rgb01(
        "#ffffff"
    )

    COLOR_MAX = hex_to_rgb01(
        district_color
    )

    def color_from_value(v):

        if (
            v is None
            or not np.isfinite(v)
            or v <= 0
        ):
            return COLOR_ZERO

        # IMPORTANTE:
        # usamos el máximo de ESTE distrito
        t = v / district_max

        return lerp_rgb(
            COLOR_ZERO,
            COLOR_MAX,
            t
        )

    grid_gdf["face_color"] = (
        grid_gdf["kWh_total"]
        .apply(color_from_value)
    )

    # ----------------------------------------------------
    # MALLA
    # ----------------------------------------------------

    grid_gdf.plot(
        ax=ax,
        color=grid_gdf[
            "face_color"
        ].tolist(),
        edgecolor="#cccccc",
        linewidth=0.5,
        zorder=2,
    )

    # ----------------------------------------------------
    # BOUNDARY
    # ----------------------------------------------------

    boundary_gdf_m.boundary.plot(
        ax=ax,
        color=district_color,
        linewidth=1.8,
        zorder=5,
    )

    # ----------------------------------------------------
    # POIs
    # ----------------------------------------------------

    boundary_geometry = (
        boundary_gdf_m
        .geometry
        .iloc[0]
    )

    poi_dentro = poi_gdf[
        poi_gdf.within(
            boundary_geometry
        )
    ]

    ax.scatter(
        poi_dentro.geometry.x,
        poi_dentro.geometry.y,
        s=10,
        color="#1f1f1f",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.4,
        zorder=6,
    )

    # ----------------------------------------------------
    # TÍTULO
    # ----------------------------------------------------

    total_kwh = result[
        "total_kwh"
    ]

    ax.set_title(
        f"{study_case}\n"
        f"s01–s10 — "
        f"{total_kwh:,.0f} kWh",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    # ----------------------------------------------------
    # COLORBAR PROPIO DEL DISTRITO
    # ----------------------------------------------------

    cmap = (
        mpl.colors
        .LinearSegmentedColormap
        .from_list(
            f"gradient_{study_case}",
            [
                "#ffffff",
                district_color
            ]
        )
    )

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(
            vmin=0,
            vmax=district_max
        )
    )

    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "kWh por celda",
        fontsize=9
    )

    # ----------------------------------------------------
    # TICKS ADAPTADOS AL DISTRITO
    # ----------------------------------------------------

    cbar.set_ticks([
        0,
        district_max / 2,
        district_max
    ])

    cbar.set_ticklabels([
        "0",
        f"{district_max / 2:,.1f}",
        f"Max = {district_max:,.1f}"
    ])


plt.suptitle(
    "Consumo de vehículos eléctricos — "
    "escenarios s01–s10",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.show()


# ========================================================
# FIGURA 2
# PERFIL DE LA RED — DOS DISTRITOS
# ========================================================

fig, axes = plt.subplots(

    1,
    2,

    figsize=(18, 7)
)

axes = np.atleast_1d(
    axes
)


for ax, study_case in zip(
    axes,
    study_cases
):

    result = results[
        study_case
    ]

    parent = result[
        "parent"
    ]

    kwh_cum = result[
        "kwh_cum"
    ]

    dist_from_root = result[
        "dist_from_root"
    ]

    reachable_nodes = result[
        "reachable_nodes"
    ]

    children = result[
        "children"
    ]

    district_color = (
        DISTRICT_COLORS[
            study_case
        ]
    )

    # ----------------------------------------------------
    # ARISTAS
    # ----------------------------------------------------

    for child, p in parent.items():

        if p is None:
            continue

        if (
            p not in dist_from_root
            or child not in dist_from_root
        ):
            continue

        x1 = (
            dist_from_root[p]
        )

        y1 = float(
            kwh_cum.get(
                p,
                0.0
            )
        )

        x2 = (
            dist_from_root[child]
        )

        y2 = float(
            kwh_cum.get(
                child,
                0.0
            )
        )

        ax.plot(

            [x1, x2],

            [y1, y2],

            linewidth=1.5,

            color=district_color,

            alpha=0.75,
        )

    # ----------------------------------------------------
    # NODOS
    # ----------------------------------------------------

    xs = [
        dist_from_root[n]
        for n
        in reachable_nodes
    ]

    ys = [
        float(
            kwh_cum.get(
                n,
                0.0
            )
        )

        for n
        in reachable_nodes
    ]

    ax.scatter(

        xs,

        ys,

        s=25,

        color=district_color,

        edgecolor="white",

        linewidth=0.5,

        zorder=3,
    )

    # ----------------------------------------------------
    # ETIQUETAS
    # ----------------------------------------------------

    texts = []

    for n in reachable_nodes:

        x = (
            dist_from_root[n]
        )

        y = float(
            kwh_cum.get(
                n,
                0.0
            )
        )

        n_short = (
            n.replace(
                "Node_",
                ""
            )
        )

        t = ax.text(

            x,

            y,

            n_short,

            fontsize=8,

            ha="center",

            va="bottom",

            bbox=dict(

                facecolor="white",

                edgecolor="none",

                alpha=0.7,

                boxstyle=(
                    "round,pad=0.15"
                )
            ),

            zorder=4
        )

        texts.append(t)

    # ----------------------------------------------------
    # AJUSTE DE ETIQUETAS
    # ----------------------------------------------------

    plt.figure

    fig.canvas.draw()

    renderer = (
        fig.canvas
        .get_renderer()
    )

    def bbox_tuple(text_obj):

        bb = (
            text_obj
            .get_window_extent(
                renderer=renderer
            )
        )

        return [
            bb.x0,
            bb.y0,
            bb.x1,
            bb.y1
        ]

    def overlap(
        b1,
        b2,
        pad=2.0
    ):

        return not (

            b1[2] + pad < b2[0]
            or
            b2[2] + pad < b1[0]
            or
            b1[3] + pad < b2[1]
            or
            b2[3] + pad < b1[1]
        )

    ref_disp = []

    for t in texts:

        x_data, y_data = (
            t.get_position()
        )

        x_disp, y_disp = (
            ax.transData.transform(
                (
                    x_data,
                    y_data
                )
            )
        )

        ref_disp.append([
            x_disp,
            y_disp
        ])

    max_iter = 200

    step_y = 0.5
    step_x = 0.5

    pad = 1.35

    pull = 0.06

    pos_disp = [
        p.copy()
        for p in ref_disp
    ]

    for _ in range(
        max_iter
    ):

        moved = False

        for i, t in enumerate(
            texts
        ):

            x_data, y_data = (
                ax.transData
                .inverted()
                .transform(
                    pos_disp[i]
                )
            )

            t.set_position(
                (
                    x_data,
                    y_data
                )
            )

        fig.canvas.draw()

        renderer = (
            fig.canvas
            .get_renderer()
        )

        bboxes = [
            bbox_tuple(t)
            for t in texts
        ]

        for i in range(
            len(texts)
        ):

            for j in range(
                i + 1,
                len(texts)
            ):

                b1 = bboxes[i]
                b2 = bboxes[j]

                if not overlap(
                    b1,
                    b2,
                    pad=pad
                ):
                    continue

                c1x = (
                    b1[0]
                    + b1[2]
                ) * 0.5

                c1y = (
                    b1[1]
                    + b1[3]
                ) * 0.5

                c2x = (
                    b2[0]
                    + b2[2]
                ) * 0.5

                c2y = (
                    b2[1]
                    + b2[3]
                ) * 0.5

                sign_y = (
                    1.0
                    if c1y <= c2y
                    else -1.0
                )

                sign_x = (
                    1.0
                    if c1x <= c2x
                    else -1.0
                )

                pos_disp[i][1] -= (
                    sign_y
                    * step_y
                )

                pos_disp[j][1] += (
                    sign_y
                    * step_y
                )

                pos_disp[i][0] -= (
                    sign_x
                    * step_x
                )

                pos_disp[j][0] += (
                    sign_x
                    * step_x
                )

                moved = True

        # Volver ligeramente hacia la posición original
        for i in range(
            len(texts)
        ):

            pos_disp[i][0] += (

                ref_disp[i][0]
                - pos_disp[i][0]

            ) * pull

            pos_disp[i][1] += (

                ref_disp[i][1]
                - pos_disp[i][1]

            ) * pull

        if not moved:
            break

    # Aplicar posiciones finales
    for i, t in enumerate(
        texts
    ):

        x_data, y_data = (
            ax.transData
            .inverted()
            .transform(
                pos_disp[i]
            )
        )

        t.set_position(
            (
                x_data,
                y_data
            )
        )

    # ----------------------------------------------------
    # FORMATO
    # ----------------------------------------------------

    ax.set_title(

        f"{study_case} — "
        f"escenarios s01–s10",

        fontsize=13,

        fontweight="bold"
    )

    ax.set_xlabel(
        "Distancia acumulada desde la raíz [km]"
    )

    ax.set_ylabel(
        "kWh acumulado (subárbol)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.xaxis.set_major_locator(
        MaxNLocator(
            nbins=10
        )
    )


plt.suptitle(

    f"Perfil de las redes — "
    f"consumo acumulado s01–s10",

    fontsize=16,

    fontweight="bold"
)

plt.tight_layout()

plt.show()


# ========================================================
# RESUMEN FINAL
# ========================================================

print()
print("=" * 70)
print("CONSUMO TOTAL POR DISTRITO")
print("=" * 70)

for study_case in study_cases:

    total_kwh = (
        results[
            study_case
        ]["total_kwh"]
    )

    print(
        f"{study_case:20s}: "
        f"{total_kwh:,.2f} kWh"
    )

print("=" * 70)
