# Leer el archivo de etiquetas
with open("etiquetas-p1.txt", "r") as f:
    etiquetas = f.readlines()

# Diccionario de conversión
conversion = {
    '1': '1',
    '0': '1',
    '-1': '0'
}

# Convertir las etiquetas y escribir en el nuevo archivo
with open("etiquetas_convertidas2.txt", "w") as f_out:
    for etiqueta in etiquetas:
        etiqueta = etiqueta.strip()  # Quitar espacios en blanco
        f_out.write(conversion[etiqueta] + "\n")

print("Archivo 'etiquetas_convertidas2.txt' creado con éxito.")
