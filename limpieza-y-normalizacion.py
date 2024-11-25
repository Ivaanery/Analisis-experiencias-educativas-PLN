import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.es.stop_words import STOP_WORDS
import language_tool_python
from unicodedata import normalize
import re
import numpy as np

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')


#ortografia
def corregir_ortografia(texto):
    herramienta = language_tool_python.LanguageTool('es')  # Configura para español
    correcciones = herramienta.correct(texto)
    return correcciones

#acentos
def normalize_text (text):
  return normalize('NFKD', text).encode('ASCII', 'ignore') 

# Abrir el archivo en modo de lectura
with open('pregunta1.txt', 'r', encoding='utf-8') as archivo:
    # Leer línea por línea
    for linea in archivo:
        correccion1 = normalize_text(linea)
        # Eliminar las comillas dobles
        linea_sin_caracteres = linea.replace('"', '').replace(',','').replace('.','')
        #recordar que si borro los si y no, se borran literalmente todosssss fakkk.
        #linea_sin_si = linea_sin_caracteres.replace('si','').replace('Si','').replace('Sí','').replace('sí','')
        #linea_sin_no = linea_sin_si.replace('no','').replace('No','').replace('NO','')
        tokens = word_tokenize(linea_sin_caracteres)
        stop_words = set(stopwords.words('spanish'))
        tokens_sin_stopwords = [word for word in tokens if word.lower() not in stop_words]
        doc = nlp(' '.join(tokens_sin_stopwords))
        lemmas = [token.lemma_ for token in doc]
        print("\nTexto original:", linea)
        print("Texto corregido:", linea_sin_caracteres)
        print("Tokens:", tokens) #agarrar la variable tokens para la clasificación
        print("Tokens sin stopwords:", tokens_sin_stopwords)
        print("Lemmas:", lemmas)

# Abrir el archivo original en modo lectura
with open('pregunta8.txt', 'r', encoding='utf-8') as archivo:
    # Abrir el archivo de salida en modo escritura
    with open('calificaciones.txt', 'w', encoding='utf-8') as archivo_salida:
        # Leer línea por línea
        for linea in archivo:
            # Eliminar letras minúsculas
            linea_letras_Minus = linea.replace('a', '').replace('b', '').replace('c', '').replace('d', '').replace('e', '').replace('f', '').replace('g', '').replace('h', '').replace('i', '').replace('j', '').replace('k', '').replace('l', '').replace('m', '').replace('n', '').replace('ñ', '').replace('o', '').replace('p', '').replace('q', '').replace('r', '').replace('s', '').replace('t', '').replace('u', '').replace('v', '').replace('w', '').replace('x', '').replace('y', '').replace('z', '')
            # Eliminar letras mayúsculas
            linea_letras_Mayus = linea_letras_Minus.replace('A', '').replace('B', '').replace('C', '').replace('D', '').replace('E', '').replace('F', '').replace('G', '').replace('H', '').replace('I', '').replace('J', '').replace('K', '').replace('L', '').replace('M', '').replace('N', '').replace('Ñ', '').replace('O', '').replace('P', '').replace('Q', '').replace('R', '').replace('S', '').replace('T', '').replace('U', '').replace('V', '').replace('W', '').replace('X', '').replace('Y', '').replace('Z', '')
            # Eliminar vocales con acento
            linea_sin_acentos = linea_letras_Mayus.replace('á', '').replace('é', '').replace('í', '').replace('ó', '').replace('ú', '')
            #Eiminar caracteres extras
            linea_final = linea_sin_acentos.replace('"','').replace(",",'')

            # Imprimir los resultados intermedios
            print("Texto original:", linea)
            print("Texto corregido 1:", linea_letras_Minus)
            print("Texto corregido 2:", linea_letras_Mayus)
            print("Texto corregido 3:", linea_sin_acentos)
            print("\n")
            
            # Escribir la última versión corregida en el archivo de salida
            archivo_salida.write(linea_final)


# Leer el archivo de calificaciones
with open('calificaciones.txt', 'r', encoding='utf-8') as archivo:
    contenido = archivo.read()

# Extraer números válidos (incluso si son parte de rangos como "90-95")
patron = r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)"
valores_raw = re.findall(patron, contenido)

# Procesar los valores: convertir rangos y limpiar datos
valores_procesados = []
for valor in valores_raw:
    if '-' in valor:
        # Si es un rango como "90-95", calcular el promedio
        inicio, fin = map(float, valor.split('-'))
        promedio = (inicio + fin) / 2
        valores_procesados.append(promedio)
    else:
        # Convertir el valor a float
        valores_procesados.append(float(valor))

# Identificar valores que requieren normalización (mayores a 10)
max_valor = max(valores_procesados)  # Encontrar el máximo para normalizar correctamente
valores_normalizados = []
for valor in valores_procesados:
    if valor > 10:
        # Normalizar solo los valores fuera de la escala
        valor_normalizado = (valor / max_valor) * 10
        valores_normalizados.append(valor_normalizado)
    else:
        # Mantener los valores en la escala original
        valores_normalizados.append(valor)

# Escribir los valores normalizados en un nuevo archivo
with open('calificaciones_normalizadas.txt', 'w', encoding='utf-8') as archivo_salida:
    for valor in valores_normalizados:
        archivo_salida.write(f"{valor:.2f}\n")

print("Archivo 'calificaciones_normalizadas.txt' generado con éxito.")
