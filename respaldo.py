import goslate
from googletrans import Translator
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import sent_tokenize
nltk.download('punkt')
analizador = SentimentIntensityAnalyzer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
import pandas as pd


# Ruta del archivo Excel
ruta_archivo = 'experiencias.xlsx'

# Leer el archivo Excel
datos_excel = pd.read_excel(ruta_archivo)

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

#Pregunta 8
columna8_guardar = ['Menciona un aproximado de tu promedio general, obtenido durante tus clases en línea']
datos_excel[columna8_guardar].to_csv('pregunta8.txt', index=False, header=False)


num_filas = datos_excel.shape[0]
print(num_filas)

# T R A D U C C I Ó N
# Se crea una instancia del traductor
translator = Translator()
# Ruta del archivo TXT a traducir
ruta_archivo = 'pregunta1.txt'
# Leer el archivo de texto
with open('pregunta1.txt', 'r') as archivo:
    texto_original = archivo.read()

# Detectar el idioma del texto original
idioma_original = translator.detect(texto_original).lang
# Definir el idioma al que deseas traducir (ejemplo: inglés)
idioma_destino = 'en'
# Traducir el texto
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# Imprimir el resultado
#print('Texto original:')
#print(texto_original)
#print('Texto traducido:')
print(texto_traducido.text)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa1 = 0
sumaPositiva1 = 0
sumaNeutra1 = 0

sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa1 = sumaNegativa1 + scores[key]
        if(contador==1):
            sumaNeutra1 = sumaNeutra1 + scores[key]  
        if(contador==2):
            sumaPositiva1 = sumaPositiva1 + scores[key]
        contador = contador + 1

print("\n** R E U L T A D O S  F I N A L E S **\n")
print("PREGUNTA 1:")
print("La positividad de las respuestas es: ", sumaPositiva1/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa1/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra1/num_filas)


#PREGUNTA 2
# T R A D U C C I Ó N
translator = Translator()
ruta_archivo = 'pregunta2.txt'
with open('pregunta2.txt', 'r') as archivo:
    texto_original = archivo.read()
idioma_original = translator.detect(texto_original).lang
idioma_destino = 'en'
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa2 = 0
sumaPositiva2 = 0
sumaNeutra2 = 0
sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa2 = sumaNegativa2 + scores[key]
        if(contador==1):
            sumaNeutra2 = sumaNeutra2 + scores[key]  
        if(contador==2):
            sumaPositiva2 = sumaPositiva2 + scores[key]
        contador = contador + 1

print("\n\nPREGUNTA 2:")
print("La positividad de las respuestas es: ", sumaPositiva2/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa2/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra2/num_filas)


#PREGUNTA 3
# T R A D U C C I Ó N
translator = Translator()
ruta_archivo = 'pregunta3.txt'
with open('pregunta3.txt', 'r') as archivo:
    texto_original = archivo.read()
idioma_original = translator.detect(texto_original).lang
idioma_destino = 'en'
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa3 = 0
sumaPositiva3 = 0
sumaNeutra3 = 0
sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa3 = sumaNegativa3 + scores[key]
        if(contador==1):
            sumaNeutra3 = sumaNeutra3 + scores[key]  
        if(contador==2):
            sumaPositiva3 = sumaPositiva3 + scores[key]
        contador = contador + 1

print("\n\nPREGUNTA 3:")
print("La positividad de las respuestas es: ", sumaPositiva3/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa3/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra3/num_filas)



#PREGUNTA 4
# T R A D U C C I Ó N
translator = Translator()
ruta_archivo = 'pregunta4.txt'
with open('pregunta4.txt', 'r') as archivo:
    texto_original = archivo.read()
idioma_original = translator.detect(texto_original).lang
idioma_destino = 'en'
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa4 = 0
sumaPositiva4 = 0
sumaNeutra4 = 0
sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa4 = sumaNegativa4 + scores[key]
        if(contador==1):
            sumaNeutra4 = sumaNeutra4 + scores[key]  
        if(contador==2):
            sumaPositiva4 = sumaPositiva4 + scores[key]
        contador = contador + 1

print("\n\nPREGUNTA 4:")
print("La positividad de las respuestas es: ", sumaPositiva4/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa4/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra4/num_filas)



#PREGUNTA 5
# T R A D U C C I Ó N
translator = Translator()
ruta_archivo = 'pregunta5.txt'
with open('pregunta5.txt', 'r') as archivo:
    texto_original = archivo.read()
idioma_original = translator.detect(texto_original).lang
idioma_destino = 'en'
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa5 = 0
sumaPositiva5 = 0
sumaNeutra5 = 0
sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa5 = sumaNegativa5 + scores[key]
        if(contador==1):
            sumaNeutra5 = sumaNeutra5 + scores[key]  
        if(contador==2):
            sumaPositiva5 = sumaPositiva5 + scores[key]
        contador = contador + 1

print("\n\nPREGUNTA 5:")
print("La positividad de las respuestas es: ", sumaPositiva5/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa5/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra5/num_filas)

#PREGUNTA 6
# T R A D U C C I Ó N
translator = Translator()
ruta_archivo = 'pregunta6.txt'
with open('pregunta6.txt', 'r') as archivo:
    texto_original = archivo.read()
idioma_original = translator.detect(texto_original).lang
idioma_destino = 'en'
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa6 = 0
sumaPositiva6 = 0
sumaNeutra6 = 0
sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa6 = sumaNegativa6 + scores[key]
        if(contador==1):
            sumaNeutra6 = sumaNeutra6 + scores[key]  
        if(contador==2):
            sumaPositiva6 = sumaPositiva6 + scores[key]
        contador = contador + 1

print("\n\nPREGUNTA 6:")
print("La positividad de las respuestas es: ", sumaPositiva6/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa6/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra6/num_filas)


#PREGUNTA 7
# T R A D U C C I Ó N
translator = Translator()
ruta_archivo = 'pregunta7.txt'
with open('pregunta7.txt', 'r') as archivo:
    texto_original = archivo.read()
idioma_original = translator.detect(texto_original).lang
idioma_destino = 'en'
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa7 = 0
sumaPositiva7 = 0
sumaNeutra7 = 0
sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa7 = sumaNegativa7 + scores[key]
        if(contador==1):
            sumaNeutra7 = sumaNeutra7 + scores[key]  
        if(contador==2):
            sumaPositiva7 = sumaPositiva7 + scores[key]
        contador = contador + 1

print("\n\nPREGUNTA 7:")
print("La positividad de las respuestas es: ", sumaPositiva7/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa7/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra7/num_filas)


#PREGUNTA 8
# T R A D U C C I Ó N
translator = Translator()
ruta_archivo = 'pregunta8.txt'
with open('pregunta8.txt', 'r') as archivo:
    texto_original = archivo.read()
idioma_original = translator.detect(texto_original).lang
idioma_destino = 'en'
texto_traducido = translator.translate(texto_original, src=idioma_original, dest=idioma_destino)

# A N A L I Z A D O R  D E  S E N T I M I E N T O S
contador = 0
sumaNegativa8 = 0
sumaPositiva8 = 0
sumaNeutra8 = 0
sentences = tokenizer.tokenize(texto_traducido.text)
for sentence in sentences:
    #print("\n")
    #print(sentence)
    scores = analizador.polarity_scores(sentence)
    contador = 0
    for key in scores:
        #print("vuelta: ", contador)
        #print(key, ': ', scores[key])
        if(contador==0):
            sumaNegativa8 = sumaNegativa8 + scores[key]
        if(contador==1):
            sumaNeutra8 = sumaNeutra8 + scores[key]  
        if(contador==2):
            sumaPositiva8 = sumaPositiva8 + scores[key]
        contador = contador + 1

print("\n\nPREGUNTA 8:")
print("La positividad de las respuestas es: ", sumaPositiva8/num_filas)
print("\nLa negatividad de las respuestas es: ", sumaNegativa8/num_filas)
print("\nLa neutralidad de las respuestas es: ", sumaNeutra8/num_filas)
