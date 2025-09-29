import lzma
import os

# Nombre del archivo comprimido
compressed_file = "planet_5.066,52.059_5.134,52.086.osm.geojson.xz"

# Nombre del archivo de salida (sin .xz)
output_file = compressed_file.replace(".xz", "")

print(f"🗜️ Descomprimiendo {compressed_file} ...")

# Descomprimir usando lzma
with lzma.open(compressed_file, "rb") as f_in:
    with open(output_file, "wb") as f_out:
        f_out.write(f_in.read())

print(f"✅ Archivo descomprimido: {output_file}")

# (Opcional) Leer el GeoJSON
import json
with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📂 GeoJSON cargado con {len(data.get('features', []))} features.")
