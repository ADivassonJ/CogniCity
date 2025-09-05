"""
Refactor highlights:
- Import hygiene + constants
- Robust CRS handling (EPSG:4326 <-> local metric UTM)
- Faster + safer OSM filtering with column existence checks
- Voronoi: duplicate-point safe, clipped to boundary, consistent naming
- Fewer side effects, clearer errors, type hints (py37-compatible)
- Small perf wins by avoiding unnecessary to_crs and merges
"""

from __future__ import annotations

# ===== Imports (clean) =====
import os
import sys
import math
import shutil
import random
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import osmnx as ox
ox.settings.timeout = 500

from haversine import haversine, Unit
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union, transform
from shapely.errors import ShapelyDeprecationWarning

from scipy.spatial import Voronoi

import pyproj

# ===== Warnings (py 3.7 / shapely) =====
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# ===== Constants =====
WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

# ===== Small utilities =====

def _ensure_polygon(geom) -> Polygon:
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    raise TypeError("Boundary geometry must be a shapely Polygon or MultiPolygon.")


def _estimate_metric_crs(gdf: gpd.GeoDataFrame) -> str:
    try:
        return str(gdf.estimate_utm_crs())
    except Exception:
        # fallback
        return WEB_MERCATOR


def _to_crs_safe(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs == crs:
        return gdf
    return gdf.to_crs(crs)


# ===== File helpers =====

def load_filter_sort_reset(filepath: Path) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    # keep non-inactive rows; do not reset index to preserve any hidden metadata columns
    return df[df['state'] != 'inactive']


# ===== Stats creation / loading =====

def add_matches_to_stats_synpop(stats_synpop: pd.DataFrame, df: pd.DataFrame, name_column: str = 'name') -> pd.DataFrame:
    for col in df.columns:
        if col == name_column:
            continue
        col_vals = df[col]
        # iteritems is py37 compatible name for Series.items
        for idx, val in col_vals.iteritems():
            if '*' in str(val):
                name_value = df.loc[idx, name_column] if pd.notna(df.loc[idx, name_column]) else "Unknown"
                stats_synpop.loc[len(stats_synpop)] = [name_value, col, None, None, None, None]
    return stats_synpop


def create_stats_synpop(archetypes_path: Path, citizen_archetypes: pd.DataFrame, family_archetypes: pd.DataFrame) -> None:
    stats_synpop = pd.DataFrame(columns=['item_1', 'item_2', 'mu', 'sigma', 'min', 'max'])
    stats_synpop = add_matches_to_stats_synpop(stats_synpop, citizen_archetypes)
    stats_synpop = add_matches_to_stats_synpop(stats_synpop, family_archetypes)
    stats_synpop.to_excel(archetypes_path / 'stats_synpop.xlsx', index=False)


def create_stats_trans(archetypes_path: Path, transport_archetypes: pd.DataFrame, family_archetypes: pd.DataFrame) -> pd.DataFrame:
    transport_names = transport_archetypes['name'].dropna().unique()
    family_names = family_archetypes['name'].dropna().unique()
    combos = list(itertools.product(family_names, transport_names))
    stats_trans = pd.DataFrame(combos, columns=['item_1', 'item_2'])
    stats_trans = stats_trans[~stats_trans['item_2'].isin(['walk', 'public'])]
    for c in ['mu', 'sigma', 'min', 'max']:
        stats_trans[c] = ''
    stats_trans.to_excel(archetypes_path / 'stats_trans.xlsx', index=False)
    return stats_trans


def load_or_create_stats(archetypes_path: Path, filename: str, creation_func, creation_args: List[pd.DataFrame]) -> pd.DataFrame:
    filepath = archetypes_path / filename
    try:
        print(f'Loading statistical data from {filename} ...')
        df_stats = pd.read_excel(filepath)
        if df_stats.isnull().sum().sum() != 0:
            raise ValueError(f"{filename} has missing values.")
    except Exception:
        creation_func(archetypes_path, *creation_args)
        raise ValueError(f"[ERROR] {filepath} is missing or incomplete. Please fill μ, σ, max, min values and rerun.")
    return df_stats


def Archetype_documentation_initialization(paths: Dict[str, Path]):
    try:
        print('Loading archetypes data ...')
        archetypes_folder = Path(paths['archetypes'])
        files = [f for f in archetypes_folder.iterdir() if f.is_file() and f.name.startswith('pop_archetypes_') and f.suffix == '.xlsx']
        pop_archetypes = {
            f.stem.replace('pop_archetypes_', ''): load_filter_sort_reset(f)
            for f in files
        }
    except Exception as e:
        raise FileNotFoundError(f"[ERROR] Archetype files are not found in {paths['archetypes']}. Please fix and restart.") from e

    stats_synpop = load_or_create_stats(
        paths['archetypes'],
        'stats_synpop.xlsx',
        create_stats_synpop,
        [pop_archetypes.get('citizens'), pop_archetypes.get('families')]
    )
    stats_trans = load_or_create_stats(
        paths['archetypes'],
        'stats_trans.xlsx',
        create_stats_trans,
        [pop_archetypes.get('transport'), pop_archetypes.get('families')]
    )
    return pop_archetypes, stats_synpop, stats_trans


# ===== OSM helpers =====

def building_type(row: pd.Series, poss_ref: Dict[str, List[str]]) -> str:
    for category, values in poss_ref.items():
        actor = row[category] if (category in row and pd.notna(row[category])) else None
        if actor is None:
            continue
        if isinstance(values, list):
            if actor in values:
                return f'{category}_{actor}'
        elif actor == values:
            return f'{category}_{actor}'
    return 'unknown'


def osmid_reform(row: pd.Series) -> Optional[str]:
    osmid = row.get('osmid')
    element_type = row.get('element_type')
    if pd.isna(osmid) or pd.isna(element_type):
        return None
    t = str(element_type).lower()
    if t == 'node':
        return f'N{osmid}'
    if t == 'relation':
        return f'R{osmid}'
    if t == 'way':
        return f'W{osmid}'
    return None


def get_osm_elements(polygon, poss_ref):
    # --- descarga ---
    gdf = ox.geometries_from_polygon(polygon, {k: True for k in poss_ref.keys()}).reset_index(drop=True)
    if gdf.empty:
        raise ValueError("OSM returned no elements for the given polygon and tags.")

    # --- filtrado robusto, sin realineaciones ambiguas ---
    import numpy as np
    mask = np.zeros(len(gdf), dtype=bool)

    def _bool_col(key, values):
        if key not in gdf.columns:
            return np.zeros(len(gdf), dtype=bool)
        s = gdf[key]
        if isinstance(values, list):
            b = s.isin(values)
        else:
            b = (s == values)
        # asegurar alineación con gdf.index y NaN -> False
        b = b.reindex(gdf.index, fill_value=False)
        return b.to_numpy()

    for key, values in poss_ref.items():
        mask |= _bool_col(key, values)

    filtered_gdf = gdf.loc[mask].copy()
    if filtered_gdf.empty:
        raise ValueError("No POIs have been detected in the study area.")

    # --- centroides precisos y resto de tu lógica ---
    projected_gdf = filtered_gdf.to_crs(filtered_gdf.estimate_utm_crs())
    centroids = projected_gdf.geometry.centroid
    centroids_geo = gpd.GeoSeries(centroids, crs=projected_gdf.crs).to_crs(epsg=4326)

    projected_gdf['lat'] = centroids_geo.y.values
    projected_gdf['lon'] = centroids_geo.x.values

    # filtra centroides dentro del polígono original (WGS84)
    inside = gpd.GeoSeries(
        [Point(xy) for xy in zip(projected_gdf['lon'], projected_gdf['lat'])],
        crs="EPSG:4326"
    )
    projected_gdf = projected_gdf[inside.within(polygon).values]
    if projected_gdf.empty:
        raise ValueError("No building has its centroid within the study area.")

    projected_gdf = projected_gdf.to_crs(epsg=4326)
    projected_gdf['building_type'] = projected_gdf.apply(lambda row: building_type(row, poss_ref), axis=1)
    projected_gdf['osm_id'] = projected_gdf.apply(osmid_reform, axis=1)

    return projected_gdf[['building_type', 'osm_id', 'geometry', 'lat', 'lon']]


# ===== Networks =====

def get_active_networks(transport_archetypes_df: pd.DataFrame) -> List[str]:
    active_maps = transport_archetypes_df.loc[
        transport_archetypes_df['state'] == 'active', 'map'
    ].dropna().unique().tolist()
    if 'walk' not in active_maps:
        active_maps.append('walk')
    return active_maps


def load_or_download_networks(study_area: str, study_area_path: Path, networks: List[str], special_areas_coords: Dict[str, List[Tuple[float, float]]]):
    networks_map = {}
    missing = []
    print('Loading maps ...')

    for net_type in networks:
        try:
            graph = ox.load_graphml(study_area_path / f"{net_type}.graphml")
            networks_map[f"{net_type}_map"] = graph
        except Exception:
            print(f'    [WARNING] {net_type} map is missing.')
            missing.append(net_type)

    if not missing:
        return networks_map

    if study_area in special_areas_coords:
        coords = special_areas_coords[study_area]
        polygon = Polygon([(lon, lat) for lat, lon in coords])
    else:
        polygon = ox.geocode_to_gdf(study_area).iloc[0].geometry

    # buffer once, reuse per network
    to_m = pyproj.Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True).transform
    to_deg = pyproj.Transformer.from_crs(WEB_MERCATOR, WGS84, always_xy=True).transform
    polygon_m = transform(to_m, polygon)

    for net_type in missing:
        try:
            print(f'        Downloading {net_type} network from {study_area} ...')
            buff = 300 if net_type == 'walk' else 1000
            poly_buff_m = polygon_m.buffer(buff)
            poly_buff = transform(to_deg, poly_buff_m)
            graph = ox.graph_from_polygon(poly_buff, network_type=net_type)
            ox.save_graphml(graph, study_area_path / f"{net_type}.graphml")
            networks_map[f"{net_type}_map"] = graph
        except Exception as e:
            print(f'        [ERROR] Failed to download {net_type} network: {e}')
            raise
    return networks_map


# ===== Electric system & Voronoi =====

def e_sys_loading(paths: Dict[str, Path], study_area: str) -> pd.DataFrame:
    try:
        return pd.read_excel(paths['maps'] / 'electric_system.xlsx')
    except Exception:
        print(f"    [WARNING] The file relating to the electrical network for the “{study_area}” case has not been found.\n       Please locate the .xlsx file on your computer desktop with the following name:")
        print(f"            electric_system_{study_area}.xlsx")
        code = 'NOT_DONE'
        while code != 'DONE':
            code = input("        Once you have completed the requested action, enter “DONE”.")
            if code != 'DONE':
                print('        Incorrect continuation code.')
            else:
                try:
                    electric_system = pd.read_excel(paths['desktop'] / f'electric_system_{study_area}.xlsx')
                    electric_system.to_excel(f"{paths['maps']}/electric_system.xlsx", index=False)
                    return electric_system
                except Exception:
                    print('        [ERROR] File not found at requested location.')
                    code = 'NOT_DONE'


def _deduplicate_nodes(df: pd.DataFrame) -> pd.DataFrame:
    # remove exact duplicate lon/lat rows (Voronoi fails with duplicates)
    before = len(df)
    df2 = df.drop_duplicates(subset=['long', 'lat']).copy()
    if len(df2) < before:
        print(f"        [INFO] Dropped {before - len(df2)} duplicate node(s) for Voronoi stability.")
    return df2


def voronoi_from_nodes(electric_system: pd.DataFrame, boundary_polygon: Polygon):
    boundary_polygon = _ensure_polygon(boundary_polygon)

    # Normalize node column name only once
    df = electric_system.rename(columns={electric_system.columns[0]: "node"}).copy()
    df = _deduplicate_nodes(df)

    nodes_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["long"], df["lat"]),
        crs=WGS84,
    )

    utm_crs = _estimate_metric_crs(nodes_gdf)
    nodes_gdf_proj = _to_crs_safe(nodes_gdf, utm_crs)
    boundary_proj = gpd.GeoSeries([boundary_polygon], crs=WGS84).to_crs(utm_crs).unary_union

    # Build Voronoi via Shapely's voronoi_diagram-like approach: add a generous envelope using boundary
    # We fallback to SciPy Voronoi if shapely.voronoi_diagram not available/unstable.
    try:
        from shapely.ops import voronoi_diagram as shp_voronoi
        mp = nodes_gdf_proj.unary_union
        vor = shp_voronoi(mp, envelope=boundary_proj)
        vor_polys, node_ids = [], []
        for idx, point in enumerate(nodes_gdf_proj.geometry):
            # pick the cell containing the point
            cell = next((c for c in vor.geoms if c.contains(point)), None)
            if cell is None:
                continue
            clipped = cell.intersection(boundary_proj)
            vor_polys.append(clipped)
            node_ids.append(nodes_gdf_proj.iloc[idx]['node'])
        vor_gdf = gpd.GeoDataFrame({'node': node_ids, 'geometry': vor_polys}, crs=utm_crs)
    except Exception:
        # SciPy fallback
        pts = np.vstack([nodes_gdf_proj.geometry.x.values, nodes_gdf_proj.geometry.y.values]).T
        if len(pts) < 2:
            raise ValueError("At least 2 unique nodes are required for Voronoi.")
        v = Voronoi(pts)
        # Build polygons from regions
        polys = []
        for region_idx in v.point_region:
            verts_idx = v.regions[region_idx]
            if -1 in verts_idx or verts_idx is None:
                polys.append(None)
                continue
            try:
                poly = Polygon(v.vertices[verts_idx])
                polys.append(poly.intersection(boundary_proj))
            except Exception:
                polys.append(None)
        vor_gdf = gpd.GeoDataFrame({'node': nodes_gdf_proj['node'], 'geometry': polys}, crs=utm_crs).dropna(subset=['geometry'])

    return nodes_gdf_proj, vor_gdf, boundary_proj


def assign_buildings_to_nodes(building_populations: pd.DataFrame, electric_system: pd.DataFrame, boundary_polygon: Polygon) -> pd.DataFrame:
    # Buildings to GDF
    buildings_gdf = gpd.GeoDataFrame(
        building_populations.copy(),
        geometry=gpd.points_from_xy(building_populations['lon'], building_populations['lat']),
        crs=WGS84,
    )
    nodes_gdf_proj, vor_gdf, _ = voronoi_from_nodes(electric_system, boundary_polygon)
    buildings_gdf_proj = _to_crs_safe(buildings_gdf, vor_gdf.crs)

    # spatial join
    joined = gpd.sjoin(buildings_gdf_proj, vor_gdf[['geometry', 'node']], how='left', predicate='within')
    joined = _to_crs_safe(joined, WGS84)
    df_out = pd.DataFrame(joined.drop(columns='geometry'))
    return df_out


def buffer_value(electric_system: pd.DataFrame) -> int:
    left = electric_system.loc[electric_system['long'].idxmin()]
    right = electric_system.loc[electric_system['long'].idxmax()]
    top = electric_system.loc[electric_system['lat'].idxmax()]
    bottom = electric_system.loc[electric_system['lat'].idxmin()]
    extremos = [left, right, top, bottom]
    max_distance = 0
    for i in range(len(extremos)):
        for j in range(i+1, len(extremos)):
            coord1 = (extremos[i]['lat'], extremos[i]['long'])
            coord2 = (extremos[j]['lat'], extremos[j]['long'])
            distance = haversine(coord1, coord2, unit=Unit.METERS)
            if distance > max_distance:
                max_distance = distance
    return int(max_distance)


def add_ebus(paths: Dict[str, Path], polygon: Polygon, building_populations: pd.DataFrame, study_area: str, buffer_m: int = 500, proj_epsg: int = 3857) -> pd.DataFrame:
    electric_system = e_sys_loading(paths, study_area)
    building_populations_with_node = assign_buildings_to_nodes(building_populations, electric_system, polygon)
    out_path = Path(paths['population']) / 'pop_building.xlsx'
    building_populations_with_node.to_excel(out_path, index=False)
    return building_populations_with_node


# ===== Services / POIs =====

def services_groups_creation(df: pd.DataFrame, to_keep: List[str]) -> Dict[str, Dict[str, List[str]]]:
    df = df[df['not considered'] != 'x'].reset_index(drop=True)
    group_dicts: Dict[str, Dict[str, List[str]]] = {}
    for group in to_keep:
        if group not in df.columns:
            continue
        group_df = df[df[group] == 'x'][['name', 'surname']].dropna()
        group_ref: Dict[str, List[str]] = {}
        for _, row in group_df.iterrows():
            nm, sn = row['name'], row['surname']
            group_ref.setdefault(nm, [])
            if sn not in group_ref[nm]:
                group_ref[nm].append(sn)
        group_dicts[group + "_list"] = group_ref
    return group_dicts


# ===== High-level init steps =====

def load_or_download_pois(study_area: str, paths: Dict[str, Path], building_archetypes_df: pd.DataFrame, special_areas_coords: Dict[str, List[Tuple[float, float]]]) -> pd.DataFrame:
    pop_path = paths['population']
    try:
        print('Loading POIs data ...')
        return pd.read_excel(f'{pop_path}/pop_building.xlsx')
    except Exception:
        print('    [WARNING] Data is missing, it needs to be downloaded.')
        return download_pois(study_area, paths, building_archetypes_df, special_areas_coords)


def building_schedule_adding(osm_elements_df: pd.DataFrame, building_archetypes_df: pd.DataFrame) -> pd.DataFrame:
    list_building_variables = [col.rsplit('_', 1)[0] for col in building_archetypes_df.columns if col.endswith('_mu')]
    for idx, row_oedf in osm_elements_df.iterrows():
        list_building_values = get_vehicle_stats(row_oedf['archetype'], building_archetypes_df, list_building_variables)
        if list_building_values == {}:
            continue
        list_building_values['Service_opening'] = list_building_values['WoS_opening'] + list_building_values['Service_opening']
        list_building_values['Service_closing'] = list_building_values['WoS_closing'] + list_building_values['Service_closing']
        for key, value in list_building_values.items():
            if not math.isinf(value):
                list_building_values[key] = int(round(value / 30.0) * 30)
        osm_elements_df = assign_data(list_building_variables, list_building_values, osm_elements_df, idx)
    return osm_elements_df


def download_pois(study_area: str, paths: Dict[str, Path], building_archetypes_df: pd.DataFrame, special_areas_coords: Dict[str, List[Tuple[float, float]]]) -> pd.DataFrame:
    study_area_path = paths[study_area]
    try:
        print(f'        Downloading services data from {study_area} ...')
        Services_Group_relationship = pd.read_excel(f'{study_area_path}/Services-Group relationship.xlsx')
    except Exception:
        print(f"    [ERROR] File 'Services-Group relationship.xlsx' is not found in the data folder ({study_area_path}).")
        raise FileNotFoundError("Required file missing.")

    print('        Processing data (it might take a while)...')
    to_keep = building_archetypes_df['name'].unique()
    services_groups = services_groups_creation(Services_Group_relationship, to_keep)

    if study_area in special_areas_coords:
        coords = special_areas_coords[study_area]
        polygon = Polygon([(lon, lat) for lat, lon in coords])
    else:
        polygon = ox.geocode_to_gdf(study_area).iloc[0].geometry

    all_osm_data: List[pd.DataFrame] = []
    for group_name, group_ref in services_groups.items():
        try:
            df_group = get_osm_elements(polygon, group_ref)
            df_group['archetype'] = group_name.replace('_list', '')
            all_osm_data.append(df_group)
            print(f"            {group_name}: {len(df_group)} elements found")
        except Exception as e:
            print(f"            [ERROR] Failed to get data for {group_name}: {e}")

    if not all_osm_data:
        raise RuntimeError("No OSM data could be downloaded for any group.")

    osm_elements_df = pd.concat(all_osm_data, ignore_index=True)
    SG_relationship = building_schedule_adding(osm_elements_df, building_archetypes_df)

    # Attach ebus nodes to these POIs (expects lat/lon present)
    buildings_populations = add_ebus(paths, polygon, SG_relationship, study_area)
    return buildings_populations


# ===== Misc remaining pieces (unchanged behavior with minor cleanups) =====

def get_active_networks_wrapper(pop_archetypes_transport: pd.DataFrame) -> List[str]:
    return get_active_networks(pop_archetypes_transport)


def get_vehicle_stats(archetype: str, transport_archetypes: pd.DataFrame, variables: List[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    row = transport_archetypes[transport_archetypes['name'] == archetype]
    if row.empty:
        return {}
    r = row.iloc[0]
    for variable in variables:
        mu = float(r.get(f'{variable}_mu', 0.0))
        sigma = float(r.get(f'{variable}_sigma', 1.0))
        max_var = float(r.get(f'{variable}_max', float('inf')))
        min_var = float(r.get(f'{variable}_min', 0.0))
        var_result = np.random.normal(mu, sigma)
        var_result = max(min(var_result, max_var), min_var)
        results[variable] = var_result
    return results


def assign_data(list_variables: List[str], list_values: Dict[str, float], df_pop: pd.DataFrame, idx: int) -> pd.DataFrame:
    for variable in list_variables:
        value = list_values.get(variable)
        if value is None:
            continue
        if variable.endswith('_type') or variable.endswith('_amount'):
            value = int(round(value))
        elif variable.endswith('_time'):
            value = int(round(value / 30.0) * 30)
        df_pop.at[idx, variable] = value
    return df_pop


def buffer_for_network(net_type: str) -> int:
    return 300 if net_type == 'walk' else 1000

def Citizen_inventory_creation(df: pd.DataFrame, population: int):
    """
    Calcula porcentaje de presencia y asigna población proporcional.
    Retorna:
      - df_dist (name, population)
      - total_presence (float)
    """
    if 'presence' not in df.columns or 'name' not in df.columns:
        raise ValueError("Citizen_inventory_creation: faltan columnas 'name' y/o 'presence'.")

    df_in = df[['name', 'presence']].copy().reset_index(drop=True)

    total_presence = float(df_in['presence'].sum())
    if total_presence <= 0:
        raise ValueError("[Error] No available presences detected in citizens_archetype's file.")

    # proporción y asignación redondeada
    df_in['presence_percentage'] = df_in['presence'] / total_presence
    df_in['population'] = (df_in['presence_percentage'] * int(population)).round().astype(int)

    # asegúrate de que la suma cuadra (por posibles redondeos)
    diff = int(population) - int(df_in['population'].sum())
    if diff != 0 and len(df_in) > 0:
        # ajusta el mayor residuo (o el mayor porcentaje)
        idx_fix = df_in['presence_percentage'].idxmax()
        df_in.loc[idx_fix, 'population'] += diff

    return df_in[['name', 'population']].copy(), total_presence

def Citizen_distribution_in_families(
    archetype_to_fill: pd.DataFrame,
    df_distribution: pd.DataFrame,
    total_presence: float,
    stats_synpop: pd.DataFrame,
    pop_archetypes: dict,
    ind_arch: str = 'f_arch_0',
    max_iterations: int = 100000
):
    """
    Genera df_families y df_citizens a partir de distribución y arquetipos.
    Mantiene la lógica original pero evita bucles infinitos y mutaciones peligrosas.
    """
    # Copias seguras
    atf = archetype_to_fill.copy().reset_index(drop=True)
    dist = df_distribution.copy().reset_index(drop=True)

    # Salidas
    df_families = pd.DataFrame(columns=['name', 'archetype', 'description', 'members'])
    df_citizens = pd.DataFrame(columns=['name', 'archetype', 'description'])
    df_part_citizens = pd.DataFrame(columns=df_citizens.columns)

    # Stats “target”
    df_stats_families = pd.DataFrame({
        'archetype': atf['name'],
        'presence': atf['presence'] if 'presence' in atf.columns else 0,
        'percentage': (atf['presence'] / max(atf['presence'].sum(), 1)) * 100 if 'presence' in atf.columns else 0,
        'stat_presence': 0,
        'stat_percentage': 0.0,
        'error': 0.0
    })

    # Iteración acotada
    steps = 0
    while steps < max_iterations:
        steps += 1

        # Filtra arquetipos imposibles por faltas en dist (como tu is_it_any_archetype)
        atf = is_it_any_archetype(atf, dist, ind_arch)
        if atf.empty:
            break

        # Recalcula estadístico observado
        counts = df_families['archetype'].value_counts()
        df_stats_families = df_stats_families[df_stats_families['archetype'].isin(atf['name'])].copy()
        df_stats_families['stat_presence'] = df_stats_families['archetype'].map(counts).fillna(0).astype(int)
        total_stat = max(int(df_stats_families['stat_presence'].sum()), 1)
        df_stats_families['stat_percentage'] = (df_stats_families['stat_presence'] / total_stat) * 100.0
        df_stats_families['error'] = df_stats_families.apply(
            lambda r: (r['stat_percentage'] - r['percentage']) / r['percentage'] if r['percentage'] else 0.0, axis=1
        )

        # Selección del arquetipo a crear
        if df_stats_families['stat_presence'].sum() == 0:
            arch_to_fill = df_stats_families.loc[df_stats_families['presence'].idxmax(), 'archetype']
        else:
            # el más infrarrepresentado (error más negativo)
            arch_to_fill = df_stats_families.loc[df_stats_families['error'].idxmin(), 'archetype']

        merged_df = process_arch_to_fill(atf, arch_to_fill, dist)

        # Caso individual
        if arch_to_fill == ind_arch:
            merged_result = (
                merged_df[merged_df['participants'].astype(str).str.contains(r'\*', na=False)]
                .merge(
                    stats_synpop[stats_synpop['item_1'] == ind_arch],
                    left_on='name', right_on='item_2', how='inner'
                )
            )
            merged_result = merged_result.dropna(subset=['population']) if 'population' in merged_result.columns else merged_result
            if merged_result.empty:
                atf = atf[atf['name'] != arch_to_fill].reset_index(drop=True)
                continue

            # probas por mu
            s = merged_result['mu'].astype(float)
            if s.sum() <= 0:
                atf = atf[atf['name'] != arch_to_fill].reset_index(drop=True)
                continue
            p = s / s.sum()
            choice = np.random.choice(merged_result['name'], p=p.values)

            merged_df.loc[:, 'participants'] = np.where(merged_df['name'] == choice, 1, 0)

            citizen_desc = pop_archetypes['citizen'].loc[
                pop_archetypes['citizen']['name'] == choice, 'description'
            ].values[0]
            df_part_citizens.loc[len(df_part_citizens)] = {
                'name': f'citizen_{len(df_part_citizens)+len(df_citizens)}',
                'archetype': choice,
                'description': citizen_desc
            }

        else:
            # Colectiva: castea '*' a stats
            for idx in merged_df.index:
                val = merged_df.at[idx, 'participants']
                if pd.isna(val):
                    continue
                stats_value = get_stats_value(val, stats_synpop, arch_to_fill, merged_df.at[idx, 'name'])
                merged_df.at[idx, 'participants'] = stats_value

                # Fit con la población disponible
                avail = merged_df.at[idx, 'population']
                need = merged_df.at[idx, 'participants']
                if need <= avail:
                    # crear N ciudadanos de ese arquetipo
                    n = int(need)
                    if n > 0:
                        cdesc = pop_archetypes['citizen'].loc[
                            pop_archetypes['citizen']['name'] == merged_df.at[idx, 'name'], 'description'
                        ].values[0]
                        base = len(df_part_citizens) + len(df_citizens)
                        rows = [{
                            'name': f'citizen_{base+i}',
                            'archetype': merged_df.at[idx, 'name'],
                            'description': cdesc
                        } for i in range(n)]
                        if rows:
                            df_part_citizens = pd.concat([df_part_citizens, pd.DataFrame(rows)], ignore_index=True)
                else:
                    # Intento de ajuste por mínimos si venía de '*'
                    fila = atf.loc[atf['name'] == arch_to_fill]
                    if not (fila.isin(['*']).any().any()):
                        atf = atf[atf['name'] != arch_to_fill].reset_index(drop=True)
                        df_part_citizens = df_part_citizens.iloc[0:0]
                        break
                    cols_star = [c for c in fila.columns if str(fila.iloc[0][c]) == '*']
                    filt = stats_synpop[(stats_synpop['item_1'] == arch_to_fill) &
                                        (stats_synpop['item_2'].isin(cols_star))][['item_2', 'min']]
                    adjusted = False
                    for _, r in filt.iterrows():
                        if int(r['min']) <= avail:
                            merged_df.loc[merged_df['name'] == r['item_2'], 'participants'] = int(r['min'])
                            n = int(r['min'])
                            if n > 0:
                                cdesc = pop_archetypes['citizen'].loc[
                                    pop_archetypes['citizen']['name'] == merged_df.at[idx, 'name'], 'description'
                                ].values[0]
                                base = len(df_part_citizens) + len(df_citizens)
                                rows = [{
                                    'name': f'citizen_{base+i}',
                                    'archetype': merged_df.at[idx, 'name'],
                                    'description': cdesc
                                } for i in range(n)]
                                if rows:
                                    df_part_citizens = pd.concat([df_part_citizens, pd.DataFrame(rows)], ignore_index=True)
                            adjusted = True
                            break
                    if not adjusted:
                        atf = atf[atf['name'] != arch_to_fill].reset_index(drop=True)
                        df_part_citizens = df_part_citizens.iloc[0:0]
                        break

        # Si llegamos aquí, consolidamos familia y descontamos población
        mask = dist['population'].notna() & merged_df['participants'].notna()
        dist.loc[mask, 'population'] = (dist.loc[mask, 'population'] - merged_df.loc[mask, 'participants']).clip(lower=0)

        # Agrega ciudadanos creados
        if not df_part_citizens.empty:
            df_citizens = pd.concat([df_citizens, df_part_citizens], ignore_index=True)

        fam_desc = pop_archetypes['family'].loc[
            pop_archetypes['family']['name'] == arch_to_fill, 'description'
        ].values[0]
        df_families.loc[len(df_families)] = {
            'name': f'family_{len(df_families)}',
            'archetype': arch_to_fill,
            'description': fam_desc,
            'members': df_part_citizens['name'].tolist()
        }
        df_part_citizens = df_part_citizens.iloc[0:0]

        if int(dist['population'].sum()) == 0:
            break

    return dist, df_citizens, df_families

def Utilities_assignment(
    df_citizens: pd.DataFrame,
    df_families: pd.DataFrame,
    pop_archetypes: dict,
    paths: dict,
    SG_relationship: pd.DataFrame,
    stats_synpop: pd.DataFrame,
    stats_trans: pd.DataFrame
):
    """
    Asigna hogares (Home) a familias, vehículos privados a familias según stats_trans,
    y variables de ciudadano; rellena WoS coherente.
    """
    # Copias seguras
    families = df_families.copy().reset_index(drop=True)
    citizens = df_citizens.copy().reset_index(drop=True)

    # Variables de transporte disponibles
    variables = [c.rsplit('_', 1)[0] for c in pop_archetypes['transport'].columns if c.endswith('_mu')]

    df_priv_vehicle = pd.DataFrame(columns=['name', 'archetype', 'family', 'ubication'] + variables)

    # IDs de Home (archetype 'Home')
    home_rows = SG_relationship[SG_relationship['archetype'] == 'Home']
    Home_ids = home_rows['osm_id'].dropna().tolist()
    if not Home_ids:
        raise ValueError("Utilities_assignment: no hay POIs de 'Home' disponibles en SG_relationship.")

    # Asignación Home round-robin (barajada)
    shuffled = random.sample(Home_ids, len(Home_ids))
    ptr = 0
    for i in range(len(families)):
        if ptr >= len(shuffled):
            shuffled = random.sample(Home_ids, len(Home_ids))
            ptr = 0
        hid = shuffled[ptr]; ptr += 1
        families.at[i, 'Home'] = hid
        families.at[i, 'Home_type'] = SG_relationship.loc[SG_relationship['osm_id'] == hid, 'building_type'].values[0]

        # Vehículos: para cada fila de stats_trans con item_1 == arquetipo de la familia
        fam_arch = families.at[i, 'archetype']
        st = stats_trans[stats_trans['item_1'] == fam_arch]
        for _, row in st.iterrows():
            n = int(computate_stats(row))
            if n <= 0:
                continue
            vars_dict = get_vehicle_stats(row['item_2'], pop_archetypes['transport'], variables)
            for k in range(n):
                rec = {'name': f'priv_vehicle_{len(df_priv_vehicle)}',
                       'archetype': row['item_2'],
                       'family': families.at[i, 'name'],
                       'ubication': hid}
                rec.update(vars_dict)
                df_priv_vehicle.loc[len(df_priv_vehicle)] = rec

    # Familia de cada ciudadano
    # Construimos un mapa {citizen -> family} a partir de 'members'
    member_map = {}
    for _, frow in families.iterrows():
        for m in frow['members']:
            member_map[m] = (frow['name'], frow['archetype'], frow['Home'])

    citizens['family'] = citizens['name'].map(lambda n: member_map.get(n, (None, None, None))[0])
    citizens['family_archetype'] = citizens['name'].map(lambda n: member_map.get(n, (None, None, None))[1])
    citizens['Home'] = citizens['name'].map(lambda n: member_map.get(n, (None, None, None))[2])

    # WoS assignment (work/study)
    work_ids = SG_relationship[SG_relationship['archetype'] == 'work']['osm_id'].dropna().tolist()
    study_ids = SG_relationship[SG_relationship['archetype'] == 'study']['osm_id'].dropna().tolist()

    # Variables de ciudadano (mu/sigma/min/max)
    list_c_vars = [c.rsplit('_', 1)[0] for c in pop_archetypes['citizen'].columns if c.endswith('_mu')]

    for i in range(len(citizens)):
        vals = get_vehicle_stats(citizens.at[i, 'archetype'], pop_archetypes['citizen'], list_c_vars)
        citizens = assign_data(list_c_vars, vals, citizens, i)

        # WoS
        fixed = int(citizens.get('WoS_fixed', pd.Series([0])).iloc[i]) if 'WoS_fixed' in citizens.columns else 0
        if fixed != 1:
            if work_ids:
                citizens.at[i, 'WoS'] = random.choice(work_ids)
        else:
            fam = citizens.at[i, 'family']
            students = citizens[(citizens['family'] == fam) & (citizens.get('WoS_fixed', 0) == 1) & (citizens['WoS'].notna())]
            if not students.empty:
                citizens.at[i, 'WoS'] = students['WoS'].iloc[0]
            elif study_ids:
                citizens.at[i, 'WoS'] = random.choice(study_ids)

        if pd.notna(citizens.at[i, 'WoS']):
            citizens.at[i, 'WoS_subgroup'] = SG_relationship.loc[
                SG_relationship['osm_id'] == citizens.at[i, 'WoS'], 'building_type'
            ].values[0]

    return families, citizens, df_priv_vehicle

# ===== Example main (kept minimal) =====

def main():
    # Example inputs (adjust to your environment)
    population = 200
    study_area = 'Kanaleneiland'

    paths: Dict[str, Path] = {}
    paths['main'] = Path(__file__).resolve().parent.parent.parent
    paths['system'] = paths['main'] / 'system'
    paths['desktop'] = Path.home() / 'Desktop'

    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    file_management = system_management[['file_1', 'file_2', 'pre']]

    for _, row in file_management.iterrows():
        file_1 = paths[study_area] if row['file_1'] == 'study_area' else paths[row['file_1']]
        file_2 = study_area if row['file_2'] == 'study_area' else row['file_2']
        paths[file_2] = file_1 / file_2
        if not paths[file_2].exists():
            if row['pre'] == 'y':
                print("[Error] Critical file not detected:")
                print(paths[file_2])
                print("Please solve the mentioned issue and reestart the model.")
                sys.exit(1)
            elif row['pre'] == 'p':
                while True:
                    response = input(f"Data for the case study '{study_area}' was not found.\nDo you want to copy data from standar scenario or do you want to create your own? [Y (copy)/N (create)]\n").strip().upper()
                    if response == 'Y':
                        shutil.copytree(paths['base_scenario'], paths[file_2])
                        break
                    if response == 'N':
                        os.makedirs(paths[file_2], exist_ok=True)
                        break
                    print('Your response was not valid, please respond Y (yes) or N (no).')
            else:
                os.makedirs(paths[file_2], exist_ok=True)

    print('#'*20, ' System initialization ', '#'*20)

    pop_archetypes, stats_synpop, stats_trans = Archetype_documentation_initialization(paths)

    # Geodata init
    def Geodata_initialization(study_area: str, paths: Dict[str, Path], pop_archetypes: Dict[str, pd.DataFrame]):
        special_areas_coords = {
            "Aradas": [(1, 1), (1, 1)],
            "Kanaleneiland": [
                (52.07892763457244, 5.081179665783377),
                (52.071082860598274, 5.087677559318499),
                (52.060700337662205, 5.097493321101714),
                (52.0589253058436, 5.111134343014198),
                (52.06371772987415, 5.113155235149382),
                (52.06713423672216, 5.112072614362676),
                (52.07698296226893, 5.109504220222101),
                (52.07814350260757, 5.108891797422314),
                (52.079586294469394, 5.107820057522688),
                (52.081311310482626, 5.106084859589962),
                (52.0818131208049, 5.105013119690336),
                (52.08520019308004, 5.09822543371076),
                (52.08291081138339, 5.094959178778566),
                (52.08102903986475, 5.090570148713432),
            ],
            "Annelinn": [(1, 1), (1, 1)],
        }
        networks = get_active_networks(pop_archetypes['transport'])
        agent_pop = {}
        agent_pop['building'] = load_or_download_pois(study_area, paths, pop_archetypes['building'], special_areas_coords)
        networks_map = load_or_download_networks(study_area, paths['maps'], networks, special_areas_coords)
        return agent_pop, networks_map

    agent_populations, networks_map = Geodata_initialization(study_area, paths, pop_archetypes)

    def Synthetic_population_initialization(agent_populations: Dict[str, pd.DataFrame], pop_archetypes: Dict[str, pd.DataFrame], population: int, stats_synpop: pd.DataFrame, paths: Dict[str, Path], SG_relationship: pd.DataFrame, study_area: str, stats_trans: pd.DataFrame):
        system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
        try:
            print('Loading synthetic population data ...')
            for type_population in system_management['archetypes'].dropna():
                agent_populations[type_population] = pd.read_excel(f"{paths['population']}/pop_{type_population}.xlsx")
        except Exception:
            print('    [WARNING] Data is missing.')
            print('        Creating synthetic population (it might take a while) ...')
            # Synthetic population generation
            archetype_to_analyze = pop_archetypes['citizen']
            archetype_to_fill = pop_archetypes['family']
            # 1. Inventario de ciudadanos
            agent_populations['distribution'], total_presence = Citizen_inventory_creation(
                archetype_to_analyze,
                population
            )
            # 2. Distribución de ciudadanos en familias
            (agent_populations['distribution'],
            agent_populations['citizen'],
            agent_populations['family']) = Citizen_distribution_in_families(
                archetype_to_fill,
                agent_populations['distribution'],
                total_presence,
                stats_synpop,
                pop_archetypes
            )
            # 3. Asignación de utilidades (vehículos, transportes, etc.)
            (agent_populations['family'],
            agent_populations['citizen'],
            agent_populations['transport']) = Utilities_assignment(
                agent_populations['citizen'],
                agent_populations['family'],
                pop_archetypes,
                paths,
                SG_relationship,
                stats_synpop,
                stats_trans
            )
            print("        Saving data ...")
            for type_population in system_management['archetypes'].dropna():
                agent_populations[type_population].to_excel(
                    f"{paths['population']}/pop_{type_population}.xlsx",
                    index=False
                )
            pass
        return agent_populations

    agent_populations = Synthetic_population_initialization(agent_populations, pop_archetypes, population, stats_synpop, paths, agent_populations['building'], study_area, stats_trans)

    print('#'*20, ' Initialization finalized ', '#'*20)


if __name__ == '__main__':
    main()
