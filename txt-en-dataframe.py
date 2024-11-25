import pandas as pd

# Leer el archivo y guardar cada línea en una lista
with open('pregunta1.txt', 'r') as archivo:
    lineas = [linea.strip() for linea in archivo]  # strip() elimina los espacios en blanco y saltos de línea adicionales

# Crear un DataFrame a partir de la lista
df = pd.DataFrame(lineas, columns=['Líneas'])

# Mostrar el DataFrame
print(df)