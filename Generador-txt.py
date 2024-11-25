import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import sent_tokenize
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.es.stop_words import STOP_WORDS
import language_tool_python
from unicodedata import normalize
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
import spacy
spacy.cli.download("es_core_news_sm")
# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')



# Leer archivo .xlsx especificando el motor 'openpyxl'
datos_excel = pd.read_excel("experiencias.xlsx", engine="openpyxl")


# Mostrar los datos
print(datos_excel)



datos_excel = datos_excel.drop('Marca temporal', axis=1)
#datos_excel = datos_excel.drop(0)
print(datos_excel)
# Seleccionar las columnas a guardar
columna1_guardar = ['¿Cómo te sentiste emocionalmente durante las clases en línea?']
# Guardar las columnas en un archivo txt
datos_excel[columna1_guardar].to_csv('pregunta1.txt', index=False, header=False)

#Pregunta 2
columna2_guardar = ['¿Experimentaste cambios en tus emociones desde que comenzaron las clases en línea? Si es así, ¿puedes describirlos?']
datos_excel[columna2_guardar].to_csv('pregunta2.txt', index=False, header=False)

#Pregunta 3
columna3_guardar = ['¿Sientes que tuviste suficiente interacción social durante las clases en línea?']
datos_excel[columna3_guardar].to_csv('pregunta3.txt', index=False, header=False)

#Pregunta 4
columna4_guardar = ['¿Cómo sentiste la relación con tus compañeros de clase y profesores?']
datos_excel[columna4_guardar].to_csv('pregunta4.txt', index=False, header=False)

#Pregunta 5
columna5_guardar = ['¿Tuviste dificultades para mantenerte concentrado/a en las clases en línea?, ¿por qué?']
datos_excel[columna5_guardar].to_csv('pregunta5.txt', index=False, header=False)

#Pregunta 6
columna6_guardar = ['¿Te sentías motivado a participar en las clases en línea?, ¿por qué?']
datos_excel[columna6_guardar].to_csv('pregunta6.txt', index=False, header=False)

#Pregunta 7
columna7_guardar = ['¿Hay algún otro aspecto relacionado con tus experiencias socioemocionales en las clases en línea que te gustaría compartir?']
datos_excel[columna7_guardar].to_csv('pregunta7.txt', index=False, header=False)

#Pregunta 8 - CALIFICACIONES
columna8_guardar = ['Menciona un aproximado de tu promedio general, obtenido durante tus clases en línea']
datos_excel[columna8_guardar].to_csv('pregunta8.txt', index=False, header=False)



#num_filas = datos_excel.shape[0]
#print(num_filas)

with open('pregunta1.txt', 'r',encoding='utf-8') as archivo:
    # Leer línea por línea
    for linea in archivo:
        print(linea)

#texto = 'hola que tal'

# Ortografía
def corregir_ortografia(texto):
    herramienta = language_tool_python.LanguageTool('es')  # Configura para español
    correcciones = herramienta.correct(texto)
    return correcciones

# Acentos
def normalize_text(text):
    return normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

# Lista para almacenar todos los textos procesados
textos_procesados = []

# Abrir el archivo en modo de lectura
with open('pregunta1.txt', 'r', encoding='utf-8') as archivo:
    # Leer línea por línea
    for linea in archivo:
        correccion1 = normalize_text(linea)
        # Eliminacion de caracteres especiales
        #aqui se borra todoooo lo que pongas
        linea_sin_caracteres = linea.replace('"', '').replace(',', '').replace('.', '')
        tokens = word_tokenize(linea_sin_caracteres)
        stop_words = set(stopwords.words('spanish'))
        tokens_sin_stopwords = [word for word in tokens if word.lower() not in stop_words]
        doc = nlp(' '.join(tokens_sin_stopwords))
        lemmas = [token.lemma_ for token in doc]
        
        # Agregar el texto lematizado a la lista de textos procesados
        textos_procesados.append(' '.join(lemmas))
        
        # Resultaos
        print("\nTexto original:", linea)
        print("Texto corregido:", linea_sin_caracteres)
        print("Tokens:", tokens)  # Agarrar la variable tokens para la clasificación
        print("Tokens sin stopwords:", tokens_sin_stopwords)
        print("Lemmas:", lemmas)


with open('pregunta1.txt', 'r', encoding='utf-8') as archivo:
    # Leer línea por línea
    for linea in archivo:
        correccion1 = normalize_text(linea)
        # Eliminacion de caracteres especiales
        #aqui se borra todoooo lo que pongas
        linea_sin_caracteres = linea.replace('"', '').replace(',', '').replace('.', '')
        tokens = word_tokenize(linea_sin_caracteres)
        stop_words = set(stopwords.words('spanish'))
        tokens_sin_stopwords = [word for word in tokens if word.lower() not in stop_words]
        doc = nlp(' '.join(tokens_sin_stopwords))
        lemmas = [token.lemma_ for token in doc]
        
        # Agregar el texto lematizado a la lista de textos procesados
        textos_procesados.append(' '.join(lemmas))
        
        # Resultaos
        print("\nTexto original:", linea)
        print("Texto corregido:", linea_sin_caracteres)
        print("Tokens:", tokens)  # Agarrar la variable tokens para la clasificación
        print("Tokens sin stopwords:", tokens_sin_stopwords)
        print("Lemmas:", lemmas)
