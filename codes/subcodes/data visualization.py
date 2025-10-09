# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import folium
from folium.plugins import HeatMapWithTime, MarkerCluster

import osmnx as ox


# ============ UTILIDADES DE GEO ============

def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Asegura EPSG:4326."""
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_string().upper() not in ("EPSG:4326", "WGS84", "OGC:CRS84"):
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def centroid_latlon(geom):
    """Devuelve (lat, lon) desde geometría cualquiera."""
    if geom is None or geom.is_empty:
        return None, None
    if geom.geom_type == "Point":
        return geom.y, geom.x
    c = geom.centroid
    return c.y, c.x

def parse_osm_id(osm_id):
    """
    Devuelve (tipo, id_int) para OSM: 'W123', 'N456', 'R789' o '123'->('way',123).
    """
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
    """
    Descarga geometrías para una lista de osm_id.
    Devuelve GeoDataFrame ['osm_id','geometry'] en EPSG:4326.
    """
    ser = pd.Series(osm_ids).dropna().astype(str).unique()
    if ser.size == 0:
        return gpd.GeoDataFrame(columns=["osm_id","geometry"], geometry="geometry", crs="EPSG:4326")

    tuples = [parse_osm_id(x) for x in ser]

    # osmnx v2.*: features_from_osmids; v1.*: geometries_from_osmids
    try:
        gdf = ox.features_from_osmids(tuples)   # >=2.0
    except AttributeError:
        gdf = ox.geometries_from_osmids(tuples) # 1.x fallback

    gdf = gdf.reset_index()
    gdf = ensure_wgs84(gdf)

    if "element_type" in gdf.columns:
        pref = gdf["element_type"].map({"way":"W","node":"N","relation":"R"}).fillna("")
        gdf["osm_id"] = (pref + gdf["osmid"].astype(str)).astype(str)
    elif "osmid" in gdf.columns:
        gdf["osm_id"] = gdf["osmid"].astype(str)
    else:
        # fallback raro
        gdf["osm_id"] = gdf.index.astype(str)

    return gdf[["osm_id", "geometry"]].set_geometry("geometry")


# ============ LÓGICA DE ACCIONES ============

def active_mask_for_hour(df: pd.DataFrame, hour: int) -> pd.Series:
    """
    Activo si [in, out) solapa con ventana de una hora en día 0 o día 1 (hasta 2880).
    """
    start0, end0 = hour*60, (hour+1)*60
    start1, end1 = start0 + 1440, end0 + 1440
    m0 = (df["in"] < end0) & (df["out"] > start0)
    m1 = (df["in"] < end1) & (df["out"] > start1)
    return m0 | m1

def normalize_actions(df_veh: pd.DataFrame, df_lvl1: pd.DataFrame) -> pd.DataFrame:
    """
    Unifica columnas mínimas: agent, archetype, osm_id, in, out.
    """
    ren = {"OSM_ID":"osm_id", "osmId":"osm_id"}
    need = ["agent","archetype","osm_id","in","out"]

    dv = df_veh.rename(columns=ren).copy()
    dl = df_lvl1.rename(columns=ren).copy()

    for c in need:
        if c not in dv.columns:
            raise ValueError(f"df_veh sin columna '{c}'")
        if c not in dl.columns:
            raise ValueError(f"df_lvl1 sin columna '{c}'")

    acts = pd.concat([dv[need], dl[need]], ignore_index=True)
    acts["in"] = acts["in"].astype(int)
    acts["out"] = acts["out"].astype(int)
    acts["osm_id"] = acts["osm_id"].astype(str)
    return acts

def attach_geometries(acts: pd.DataFrame, pop_building: Optional[pd.DataFrame]) -> gpd.GeoDataFrame:
    """
    Adjunta geometrías a 'acts'. Si pop_building no trae geometry/lat/lon, descarga de OSM.
    """
    # 1) Si pop_building tiene geometry:
    if pop_building is not None and "geometry" in pop_building.columns:
        pb = gpd.GeoDataFrame(pop_building.rename(columns={"OSM_ID":"osm_id","osmId":"osm_id"}),
                              geometry="geometry", crs=getattr(pop_building, "crs", None))
        pb = ensure_wgs84(pb)
        g = acts.merge(pb[["osm_id","geometry"]], on="osm_id", how="left")

    # 2) Si pop_building no tiene geometry pero tiene lat/lon:
    elif pop_building is not None and {"lat","lon"}.issubset(pop_building.columns):
        pb = pop_building.rename(columns={"OSM_ID":"osm_id","osmId":"osm_id"}).copy()
        pb["lat"] = pd.to_numeric(pb["lat"], errors="coerce")
        pb["lon"] = pd.to_numeric(pb["lon"], errors="coerce")
        pb = pb.dropna(subset=["lat","lon"])
        pb_g = gpd.GeoDataFrame(pb, geometry=gpd.points_from_xy(pb["lon"], pb["lat"]), crs="EPSG:4326")
        g = acts.merge(pb_g[["osm_id","geometry"]], on="osm_id", how="left")

    # 3) Si no hay nada util en pop_building -> OSM
    else:
        geoms = fetch_geoms_from_osmids(acts["osm_id"].unique())
        g = acts.merge(geoms, on="osm_id", how="left")

    # Rellenar lat/lon desde centroides
    latlon = g["geometry"].apply(lambda geom: centroid_latlon(geom) if geom is not None else (None, None))
    g["lat"] = [ll[0] for ll in latlon]
    g["lon"] = [ll[1] for ll in latlon]
    g = g.dropna(subset=["lat","lon"]).reset_index(drop=True)

    return gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326")


# ============ MAPA ============

def build_map(df_veh: pd.DataFrame,
              df_lvl1: pd.DataFrame,
              pop_building: Optional[pd.DataFrame],
              tiles="CartoDB positron",
              zoom_start=14,
              out_html="agents_by_hour_map.html"):
    """
    Crea un HTML con:
      - HeatMap con deslizador (24h)
      - Capas de puntos por hora (clicables)
    """
    acts = normalize_actions(df_veh, df_lvl1)
    g = attach_geometries(acts, pop_building)

    center = [g["lat"].mean(), g["lon"].mean()]
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles, control_scale=True)

    # HeatMap con slider
    heat_data, labels = [], []
    for h in range(24):
        mask = active_mask_for_hour(g, h)
        dfh = g.loc[mask, ["lat","lon"]]
        heat_data.append(dfh.values.tolist())
        labels.append(f"{h:02d}:00")

    HeatMapWithTime(
        data=heat_data,
        index=labels,
        auto_play=False,
        max_opacity=0.85,
        radius=12,
        use_local_extrema=False
    ).add_to(m)

    # Puntos por hora
    for h in range(24):
        mask = active_mask_for_hour(g, h)
        dfh = g.loc[mask].copy()
        if dfh.empty:
            continue
        fg = folium.FeatureGroup(name=f"Puntos {h:02d}:00", show=False)
        mc = MarkerCluster().add_to(fg)
        for _, r in dfh.iterrows():
            popup = folium.Popup(
                (
                    f"<b>Agente:</b> {r['agent']}<br>"
                    f"<b>Arquetipo:</b> {r['archetype']}<br>"
                    f"<b>OSM:</b> {r['osm_id']}<br>"
                    f"<b>Intervalo:</b> {r['in']}–{r['out']} min"
                ),
                max_width=280
            )
            folium.CircleMarker([r["lat"], r["lon"]], radius=4, fill=True, fill_opacity=0.9).add_to(mc)
            folium.Marker([r["lat"], r["lon"]], icon=folium.DivIcon(html=""), popup=popup).add_to(mc)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    print(f"[OK] Mapa guardado en: {out_html}")


# ============ MAIN (LEE TUS DOCS) ============

def main():
    # --- Inputs de ejemplo (ajusta si quieres parametrizar) ---
    population = 450
    study_area = 'Kanaleneiland'

    # --- Paths desde system_management.xlsx ---
    paths = {}
    paths['main'] = Path(__file__).resolve().parent.parent.parent
    paths['system'] = paths['main'] / 'system'

    system_management = pd.read_excel(paths['system'] / 'system_management.xlsx')
    file_management = system_management[['file_1', 'file_2', 'pre']]

    for _, row in file_management.iterrows():
        file_1 = paths[study_area] if row['file_1'] == 'study_area' else paths[row['file_1']]
        file_2 = study_area if row['file_2'] == 'study_area' else row['file_2']
        paths[file_2] = file_1 / file_2
        if not paths[file_2].exists():
            if str(row['pre']).lower() == 'y':
                print(f"[Error] Critical file not detected:\n{paths[file_2]}")
                print("Please solve the mentioned issue and restart the model.")
                sys.exit(1)
            else:
                os.makedirs(paths[file_2], exist_ok=True)

    # (Opcional) Cargar redes si lo necesitas para otras tareas
    try:
        networks = ['drive', 'walk']
        networks_map = {}
        for net_type in networks:
            gpath = paths['maps'] / f"{net_type}.graphml"
            if gpath.exists():
                networks_map[net_type + "_map"] = ox.load_graphml(gpath)
    except Exception as e:
        print(f"[WARN] No se cargaron mapas de red: {e}")

    # --- Lee resultados / población ---
    pop_building = pd.read_parquet(f"{paths['population']}/pop_building.parquet")

    Kanaleneiland_vehicles_actions = pd.read_excel(paths['results'] / f"{study_area}_Mo_vehicles.xlsx")
    Kanaleneiland_new_level_1      = pd.read_excel(paths['results'] / f"{study_area}_Mo_schedule.xlsx")

    print("docs readed")

    # --- Generar mapa (usa pop_building si trae geometry/lat/lon; si no, descargará de OSM) ---
    out_html = str(paths['results'] / f"{study_area}_Mo.html")
    build_map(
        df_veh=Kanaleneiland_vehicles_actions,
        df_lvl1=Kanaleneiland_new_level_1,
        pop_building=pop_building,   # puede no tener geometry; el código lo maneja
        out_html=out_html
    )

if __name__ == "__main__":
    main()
