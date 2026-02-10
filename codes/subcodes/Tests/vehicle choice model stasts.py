import pandas as pd

# -----------------------------
# RUTAS
# -----------------------------
results_path = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results"
data_path = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\data\Annelinn\population"

excel_file = f"{results_path}\\Annelinn_schedule_vehicle.xlsx"
parquet_file = f"{data_path}\\pop_citizen.parquet"

# -----------------------------
# 1. CARGAR DATOS
# -----------------------------
df = pd.read_excel(excel_file)
pop = pd.read_parquet(parquet_file)

# -----------------------------
# 2. SIMPLIFICAR:
#    por cada user quedarse con la fila
#    con el menor valor de "in"
# -----------------------------
df_simplified = (
    df.loc[df.groupby("user")["in"].idxmin()]
    .reset_index(drop=True)
)

# -----------------------------
# 3. AÑADIR ARCHETYPE DESDE pop_citizen
#    user (df) <-> name (parquet)
# -----------------------------
pop_subset = pop[["name", "archetype"]].rename(
    columns={"name": "user", "archetype": "user_archetype"}
)

df_simplified = df_simplified.merge(
    pop_subset,
    on="user",
    how="left"
)

# -----------------------------
# 4. MATRIZ DE USO MODAL
#    Filas: user_archetype
#    Columnas: archetype (vehículo)
#    Valores: % por fila
# -----------------------------
modal_matrix = pd.crosstab(
    df_simplified["user_archetype"],
    df_simplified["archetype"],
    normalize="index"   # normaliza por filas
) * 100

# Orden opcional (si quieres consistencia visual)
modal_matrix = modal_matrix.sort_index()

print("\n=== % de uso modal por arquetipo de ciudadano ===")
print(modal_matrix.round(2))

