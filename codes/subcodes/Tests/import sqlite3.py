import sqlite3
import os

ruta_archivo = r"C:\Users\asier.divasson\Desktop\result_Annelinn_SOCP_rMIP_t1000.sqlite"

if not os.path.exists(ruta_archivo):
    print(f"Error: El archivo no existe en la ruta {ruta_archivo}")
else:
    try:
        conn = sqlite3.connect(ruta_archivo)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tablas = cursor.fetchall()

        if tablas:
            print("Tablas encontradas:")
            for tabla in tablas:
                print(f"- {tabla[0]}")

            for tabla in tablas:
                print(f"\nDatos de la tabla {tabla[0]}:")
                cursor.execute(f"SELECT * FROM {tabla[0]} LIMIT 5;")
                filas = cursor.fetchall()
                for fila in filas:
                    print(fila)
        else:
            print("No se encontraron tablas en la base de datos.")

        conn.close()
    except sqlite3.Error as e:
        print(f"Error al abrir la base de datos: {e}")

import sqlite3
import pandas as pd

# Ruta al archivo SQLite (ajusta según tu ruta)
ruta_archivo = r"C:\Users\asier.divasson\Desktop\result_Kanaleneiland_SOCP_rMIP_t1000.sqlite"

# Conectar a la base de datos
conn = sqlite3.connect(ruta_archivo)

# Leer las tablas pDemand y pPmax_line en DataFrames
try:
    df_pDemand = pd.read_sql_query("SELECT * FROM pDemand;", conn)
    df_pPmax_line = pd.read_sql_query("SELECT * FROM pPmax_line;", conn)
except Exception as e:
    print(f"Error leyendo tablas: {e}")
    conn.close()
    exit()

conn.close()

# Guardar en Excel con dos hojas
ruta_excel = r"C:\Users\asier.divasson\Desktop\datos_extraidos.xlsx"
with pd.ExcelWriter(ruta_excel, engine='openpyxl') as writer:
    df_pDemand.to_excel(writer, sheet_name='pDemand', index=False)
    df_pPmax_line.to_excel(writer, sheet_name='pPmax_line', index=False)

print(f"Datos guardados correctamente en {ruta_excel}")
