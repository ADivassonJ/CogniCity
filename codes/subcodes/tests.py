#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrae .osm.pbf por ciudad desde un .osm.pbf grande (p. ej., europe-latest.osm.pbf).
Uso en VSCode: edita las variables de la sección CONFIG y ejecuta el archivo.
"""

# =========================
# CONFIG (edita aquí)
# =========================
INPUT_PBF = r"C:\Users\asier.divasson\Downloads\utrecht-250924.osm.pbf"   # Ruta al PBF grande
CITIES = ["Utrecht"]       # Nombres tal cual los dirías en un buscador
COUNTRY_HINT = None                           # Opcional: "Netherlands", "Estonia", "Portugal", etc.
OUTDIR = None                                 # Carpeta de salida. None = crea "<carpeta del PBF>/cities"
SLEEP_BETWEEN = 1.5                           # Segundos entre consultas a Nominatim (ser amable)
USER_AGENT = "CityPBFExtractor/1.0 (contact: asierdiv@gmail.com)"  # Cambia a tu email

# =========================
# CÓDIGO (no necesitas tocar)
# =========================
import os
import sys
import time
import json
import shutil
import subprocess
from typing import Dict, List, Tuple, Optional

import requests
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def check_osmium() -> None:
    exe = shutil.which("osmium")
    if not exe:
        raise RuntimeError(
            "No se encontró 'osmium' en el PATH. Instala osmium-tool y asegúrate de que 'osmium --version' funciona."
        )


def fetch_city_geojson(city: str, country_hint: Optional[str] = None, timeout: int = 60) -> Dict:
    q = city if not country_hint else f"{city}, {country_hint}"
    params = {
        "q": q,
        "format": "jsonv2",
        "polygon_geojson": 1,
        "addressdetails": 1,
        "limit": 10,
    }
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    results = r.json()
    if not results:
        raise ValueError(f"No se encontraron resultados en Nominatim para '{q}'.")

    def score(item: Dict) -> Tuple[int, float]:
        s = 0
        if item.get("osm_type") == "relation":
            s += 3
        if item.get("class") == "boundary":
            s += 3
        if item.get("type") in {"administrative", "city", "town", "municipality"}:
            s += 2
        if "geojson" in item:
            s += 2
        imp = float(item.get("importance", 0.0))
        return (s, imp)

    results.sort(key=score, reverse=True)

    for item in results:
        gj = item.get("geojson")
        if not gj:
            continue
        geom = shape(gj)
        if isinstance(geom, (Polygon, MultiPolygon)):
            # Unir partes si multipolígono complejo
            geom = geom if isinstance(geom, Polygon) else unary_union(geom)
            return mapping(geom)

    raise ValueError(f"No se obtuvo un polígono válido para '{q}'. Prueba afinando el nombre o añadiendo COUNTRY_HINT.")


def geojson_to_poly_lines(geojson_geom: Dict, name: str) -> List[str]:
    """
    Convierte un GeoJSON Polygon/MultiPolygon en formato .poly compatible con osmium/osmosis.
    Incluye contornos interiores (holes) usando secciones con prefijo '!' según el formato .poly.
    """
    g = shape(geojson_geom)
    lines: List[str] = [name]

    def ring_to_lines(coords) -> List[str]:
        pts = list(coords)
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return [f"  {x:.7f} {y:.7f}" for x, y in pts]

    idx = 1
    if isinstance(g, Polygon):
        lines.append(str(idx))
        lines.extend(ring_to_lines(g.exterior.coords))
        lines.append("END")
        # holes
        for interior in g.interiors:
            lines.append(f"!{idx}")
            lines.extend(ring_to_lines(interior.coords))
            lines.append("END")
    elif isinstance(g, MultiPolygon):
        for poly in g.geoms:
            lines.append(str(idx))
            lines.extend(ring_to_lines(poly.exterior.coords))
            lines.append("END")
            for interior in poly.interiors:
                lines.append(f"!{idx}")
                lines.extend(ring_to_lines(interior.coords))
                lines.append("END")
            idx += 1
    else:
        raise ValueError("Geometría no soportada; se esperaba Polygon o MultiPolygon.")

    lines.append("END")
    return lines


def write_text(path: str, content_lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(content_lines))
        f.write("\n")


def sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return "_".join(name.split())


def run_osmium_extract(input_pbf: str, poly_path: str, out_pbf: str, strategy: str = "smart") -> None:
    cmd = [
        "osmium", "extract",
        "-p", poly_path,
        "--strategy", strategy,
        "--overwrite",
        "--no-progress",
        "-o", out_pbf,
        input_pbf,
    ]
    print(f"[osmium] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    # Validaciones rápidas
    if not os.path.isfile(INPUT_PBF):
        print(f"[ERROR] No se encuentra el archivo PBF: {INPUT_PBF}")
        sys.exit(1)

    if not CITIES or not isinstance(CITIES, list):
        print("[ERROR] Define CITIES como lista con al menos un nombre de ciudad.")
        sys.exit(1)

    check_osmium()

    outdir = OUTDIR
    if outdir is None:
        parent = os.path.dirname(os.path.abspath(INPUT_PBF))
        outdir = os.path.join(parent, "cities")
    os.makedirs(outdir, exist_ok=True)

    print(f"Entrada: {INPUT_PBF}")
    print(f"Salida:  {outdir}")
    print(f"Ciudades: {', '.join(CITIES)}")
    if COUNTRY_HINT:
        print(f"Pista de país: {COUNTRY_HINT}")

    for city in CITIES:
        name = city.strip()
        if not name:
            continue
        safe = sanitize_filename(name)
        print(f"\n==> {name}")

        try:
            # 1) Polígono de la ciudad
            gj = fetch_city_geojson(name, country_hint=COUNTRY_HINT)
            # 2) .poly
            poly_lines = geojson_to_poly_lines(gj, safe)
            poly_path = os.path.join(outdir, f"{safe}.poly")
            write_text(poly_path, poly_lines)
            print(f"[OK] .poly guardado: {poly_path}")

            # 3) Extracción con osmium
            out_pbf = os.path.join(outdir, f"{safe}.osm.pbf")
            run_osmium_extract(INPUT_PBF, poly_path, out_pbf, strategy="smart")
            print(f"[OK] .osm.pbf extraído: {out_pbf}")

            # 4) Respiro para Nominatim
            time.sleep(SLEEP_BETWEEN)

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] osmium extract falló (código {e.returncode}) para {name}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("\nListo ✅")


if __name__ == "__main__":
    main()
