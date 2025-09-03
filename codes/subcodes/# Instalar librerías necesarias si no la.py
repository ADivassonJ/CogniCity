import pandas as pd

# Ruta al CSV
ruta = r"C:\Users\asier.divasson\Downloads\Hoja de cálculo sin título - Hoja 1 (1).csv"

# Leer el archivo CSV
df = pd.read_csv(ruta)

# Exportar a LaTeX
salida = r"C:\Users\asier.divasson\Downloads\tabla.tex"
with open(salida, "w", encoding="utf-8") as f:
    f.write(df.to_latex(index=False, longtable=True))
