# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

import folium
from folium.plugins import TimestampedGeoJson
import osmnx as ox


# ==========================
# CONFIGURACIÓN GLOBAL
# ==========================
UTRECHT_CENTER = [52.0907, 5.1214]  # centro fijo de Utrecht


# ==========================
# UTILIDADES GEO
# ==========================

def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_string().upper() not in ("EPSG:4326", "WGS84", "OGC:CRS84"):
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def centroid_latlon(geom):
    if geom is None or geom.is_empty:
        return None, None
    if geom.geom_type == "Point":
        return geom.y, geom.x
    c = geom.centroid
    return c.y, c.x

def parse_osm_id(osm_id):
    s = str(osm_id)
    if not s:
        raise ValueError("osm_id vacío")
    if s[0] in ("W","w"):
        return ("way", int(s[1:]))
    if s[0] in ("N","n"):
        return ("node", int(s[1:]))
    if s[0] in ("R","r"):
        return ("relation", int(s[1:]))
    return ("way", int(s))

def fetch_geoms_from_osmids(osm_ids):
    ser = pd.Series(osm_ids).dropna().astype(str).unique()
    if ser.size == 0:
        return gpd.GeoDataFrame(columns=["osm_id","geometry"], geometry="geometry", crs="EPSG:4326")

    tuples = [parse_osm_id(x) for x in ser]

    try:
        gdf = ox.features_from_osmids(tuples)   # osmnx >= 2.0
    except AttributeError:
        gdf = ox.geometries_from_osmids(tuples) # osmnx 1.x fallback

    gdf = gdf.reset_index()
    gdf = ensure_wgs84(gdf)

    if "element_type" in gdf.columns:
        pref = gdf["element_type"].map({"way":"W","node":"N","relation":"R"}).fillna("")
        gdf["osm_id"] = (pref + gdf["osmid"].astype(str)).astype(str)
    elif "osmid" in gdf.columns:
        gdf["osm_id"] = gdf["osmid"].astype(str)
    else:
        gdf["osm_id"] = gdf.index.astype(str)

    return gdf[["osm_id", "geometry"]].set_geometry("geometry")


# ==========================
# LÓGICA DE ACCIONES
# ==========================

def active_mask_for_hour(df: pd.DataFrame, hour: int) -> pd.Series:
    start0, end0 = hour*60, (hour+1)*60
    start1, end1 = start0 + 1440, end0 + 1440
    m0 = (df["in"] < end0) & (df["out"] > start0)
    m1 = (df["in"] < end1) & (df["out"] > start1)
    return m0 | m1

def normalize_actions(df_veh: pd.DataFrame, df_lvl1: pd.DataFrame) -> pd.DataFrame:
    ren = {"OSM_ID":"osm_id", "osmId":"osm_id"}
    need = ["agent","archetype","osm_id","in","out"]

    # --- Filtrar solo ciudadanos ---
    df_veh = df_veh.rename(columns=ren)
    df_lvl1 = df_lvl1.rename(columns=ren)
    if "archetype" in df_lvl1.columns:
        df_lvl1 = df_lvl1[~df_lvl1["archetype"].str.contains("vehicle", case=False, na=False)]

    for c in need:
        if c not in df_lvl1.columns:
            raise ValueError(f"df_lvl1 sin columna '{c}'")

    acts = df_lvl1[need].copy()
    acts["in"] = acts["in"].astype(int)
    acts["out"] = acts["out"].astype(int)
    acts["osm_id"] = acts["osm_id"].astype(str)
    return acts

def attach_geometries(acts: pd.DataFrame, pop_building: Optional[pd.DataFrame]) -> gpd.GeoDataFrame:
    if pop_building is not None and {"lat","lon"}.issubset(pop_building.columns):
        pb = pop_building.rename(columns={"OSM_ID":"osm_id","osmId":"osm_id"}).copy()
        pb["lat"] = pd.to_numeric(pb["lat"], errors="coerce")
        pb["lon"] = pd.to_numeric(pb["lon"], errors="coerce")
        pb = pb.dropna(subset=["lat","lon"])
        pb_g = gpd.GeoDataFrame(pb, geometry=gpd.points_from_xy(pb["lon"], pb["lat"]), crs="EPSG:4326")
        g = acts.merge(pb_g[["osm_id","geometry"]], on="osm_id", how="left")
    else:
        geoms = fetch_geoms_from_osmids(acts["osm_id"].unique())
        g = acts.merge(geoms, on="osm_id", how="left")

    latlon = g["geometry"].apply(lambda geom: centroid_latlon(geom) if geom is not None else (None, None))
    g["lat"] = [ll[0] for ll in latlon]
    g["lon"] = [ll[1] for ll in latlon]
    g = g.dropna(subset=["lat","lon"]).reset_index(drop=True)

    return gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326")


# ==========================
# MAPAS POR HORA
# ==========================

def build_maps_by_hour(df_veh: pd.DataFrame,
                       df_lvl1: pd.DataFrame,
                       pop_building: Optional[pd.DataFrame],
                       tiles="CartoDB positron",
                       zoom_start=12,
                       out_folder="maps_by_hour",
                       max_points_per_hour: int = 3000):
    """
    Genera 24 mapas HTML independientes (uno por hora).
    Sin clústeres, solo puntos individuales.
    """

    acts = normalize_actions(df_veh, df_lvl1)
    g = attach_geometries(acts, pop_building)
    g["lat"] = g["lat"].astype("float32")
    g["lon"] = g["lon"].astype("float32")

    out_path = Path(out_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Generando mapas horarios en: {out_path.resolve()}\n")

    for h in tqdm(range(24), desc="Generando mapas por hora"):
        fname = out_path / f"hour_{h:02d}.html"
        if fname.exists():
            tqdm.write(f"[SKIP] {fname.name} ya existe, se omite.")
            continue

        try:
            mask = active_mask_for_hour(g, h)
            dfh = g.loc[mask, ["lat","lon","agent","archetype","osm_id","in","out"]].copy()
            if dfh.empty:
                tqdm.write(f"[WARN] Sin datos para la hora {h:02d}.")
                continue

            if max_points_per_hour and len(dfh) > max_points_per_hour:
                dfh = dfh.sample(max_points_per_hour, random_state=0)

            m = folium.Map(location=UTRECHT_CENTER, zoom_start=zoom_start, tiles=tiles, control_scale=True)

            fg = folium.FeatureGroup(name=f"Hora {h:02d}")
            for _, r in dfh.iterrows():
                folium.CircleMarker(
                    [float(r["lat"]), float(r["lon"])],
                    radius=2, color="blue", fill=True, fill_opacity=0.6
                ).add_to(fg)
            fg.add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)

            m.save(fname)
            tqdm.write(f"[OK] Guardado: {fname.name}")

        except Exception as e:
            tqdm.write(f"[ERROR] Hora {h:02d}: {e}")
            continue

    print(f"\n[DONE] Mapas generados en: {out_path.resolve()}\n")
    return g


# ==========================
# MAPA FINAL ANIMADO
# ==========================

def build_animated_dots(g: gpd.GeoDataFrame, out_html: str, zoom_start=14, tiles="CartoDB positron"):
    """
    Crea un solo mapa con puntos (sin heatmap) que se mueven hora a hora.
    Centrado siempre en Utrecht.
    """

    m = folium.Map(location=UTRECHT_CENTER, zoom_start=zoom_start, tiles=tiles, control_scale=True)

    features = []
    for h in tqdm(range(24), desc="Preparando animación"):
        mask = active_mask_for_hour(g, h)
        dfh = g.loc[mask, ["lat","lon","agent","archetype","osm_id","in","out"]].copy()
        if dfh.empty:
            continue
        timestamp = f"2025-01-01T{h:02d}:00:00"
        for _, r in dfh.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
                "properties": {
                    "time": timestamp,
                    "style": {
                        "color": "blue",
                        "fillColor": "blue",
                        "opacity": 0.5,
                        "fillOpacity": 0.5,
                        "radius": 2
                    },
                    "icon": "circle"
                }
            })

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT1H",
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options="HH:mm",
        time_slider_drag_update=True
    ).add_to(m)

    m.save(out_html)
    print(f"[✅ OK] Mapa animado creado en: {out_html}")


# ==========================
# MAIN
# ==========================

def main():
    study_area = 'Kanaleneiland'
    paths = {}
    paths['main'] = Path(__file__).resolve().parent.parent.parent.parent
    paths['system'] = paths['main'] / 'system'

    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    file_management = system_management[['file_1', 'file_2', 'pre']]

    for _, row in file_management.iterrows():
        file_1 = paths[study_area] if row['file_1'] == 'study_area' else paths[row['file_1']]
        file_2 = study_area if row['file_2'] == 'study_area' else row['file_2']
        paths[file_2] = file_1 / file_2
        if not paths[file_2].exists():
            os.makedirs(paths[file_2], exist_ok=True)

    pop_building = pd.read_parquet(f"{paths['population']}/pop_building.parquet")
    df_veh = pd.read_excel(paths['results'] / f"{study_area}_We_vehicles.xlsx")
    df_lvl1 = pd.read_excel(paths['results'] / f"{study_area}_We_schedule.xlsx")

    out_folder = paths['results'] / f"{study_area}_hourly_maps"
    g = build_maps_by_hour(df_veh, df_lvl1, pop_building, out_folder=out_folder, max_points_per_hour=3000)

    out_html = str(paths['results'] / f"{study_area}_We_animated_citizens.html")
    build_animated_dots(g, out_html=out_html)


if __name__ == "__main__":
    main()
