import locale
from datetime import datetime

# Definir horarios de apertura con meses en español
horarios = {
    'Edificio San Agustín': [
        {'inicio': '21-dic', 'fin': '19-jun', 'Mo': [('08:30', '14:00')], 'Tu': [('08:30', '14:00')],
         'We': [('08:30', '14:00')], 'Th': [('08:30', '14:00')], 'Fr': [('08:30', '11:30'), ('12:30', '14:00')]},
        {'inicio': '22-jun', 'fin': '20-dic', 'Mo': [('08:30', '13:00')], 'Tu': [('08:30', '13:00')],
         'We': [('08:30', '13:00')], 'Th': [('08:30', '13:00')], 'Fr': [('08:30', '13:00')]}
    ],
    'Biblioteca Bidebarrieta': [
        {'inicio': '01-sep', 'fin': '31-may', 'L-V': [('08:30', '20:30')], 'S': [('10:00', '14:00')]},
        {'inicio': '01-jun', 'fin': '15-sep', 'L-V': [('08:30', '19:30')]}
    ]
}

# Diccionario para convertir meses en español a inglés
meses_espanol_a_ingles = {
    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
    'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
}

# Función para convertir cadena de fecha a objeto datetime
def convertir_fecha(fecha_str, año):
    dia, mes = fecha_str.split('-')
    mes_ingles = meses_espanol_a_ingles[mes.lower()]  # Convertir el mes en español a inglés
    fecha_final = f"{dia}-{mes_ingles}-{año}"
    return datetime.strptime(fecha_final, '%d-%b-%Y')

# Función para verificar si el edificio está abierto
def esta_abierto(nombre_edificio, fecha_y_hora):
    año_actual = fecha_y_hora.year
    dia_semana = fecha_y_hora.strftime('%A')[:2]  # Día de la semana ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")
    hora_actual = fecha_y_hora.time()

    # Obtener horarios del edificio
    horarios_edificio = horarios.get(nombre_edificio, [])
    
    for periodo in horarios_edificio:
        inicio = convertir_fecha(periodo['inicio'], año_actual)
        fin = convertir_fecha(periodo['fin'], año_actual)
        if inicio <= fecha_y_hora <= fin:  # Si la fecha está en el rango
            # Verificar si el día tiene horarios
            if dia_semana in periodo:
                franjas_horarias = periodo[dia_semana]
                for franja in franjas_horarias:
                    horario_apertura, horario_cierre = [datetime.strptime(h, '%H:%M').time() for h in franja]
                    if horario_apertura <= hora_actual <= horario_cierre:
                        return True  # El edificio está abierto
    return False  # El edificio está cerrado

# Ejemplo de uso
fecha_y_hora = datetime(2024, 2, 26, 10, 30)  # 26 de febrero de 2024 a las 10:30
nombre_edificio = 'Edificio San Agustín'  # Nombre correcto

# Comprobar si el edificio está abierto
if esta_abierto(nombre_edificio, fecha_y_hora):
    print(f"{nombre_edificio} está abierto en {fecha_y_hora}.")
else:
    print(f"{nombre_edificio} está cerrado en {fecha_y_hora}.")
