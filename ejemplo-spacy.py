import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.es.stop_words import STOP_WORDS

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Texto de ejemplo
texto = "El procesamiento de lenguaje natural es fascinante y útil en muchas aplicaciones."

# Tokenización
tokens = word_tokenize(texto)

# Eliminación de stopwords usando NLTK
stop_words = set(stopwords.words('spanish'))
tokens_sin_stopwords = [word for word in tokens if word.lower() not in stop_words]

# Lemmatización con spaCy
doc = nlp(' '.join(tokens_sin_stopwords))
lemmas = [token.lemma_ for token in doc]

# Mostrar resultados
print("Texto original:", texto)
print("Tokens:", tokens)
print("Tokens sin stopwords:", tokens_sin_stopwords)
print("Lemmas:", lemmas)




