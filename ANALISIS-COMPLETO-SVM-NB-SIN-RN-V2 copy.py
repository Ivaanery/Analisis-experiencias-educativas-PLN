import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.es.stop_words import STOP_WORDS
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import numpy as np

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')
spacy.cli.download("es_core_news_sm")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Cargar modelo spaCy para español
nlp = spacy.load('es_core_news_sm')

# Función para normalizar texto (quitar acentos)
def normalize_text(text):
    return normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

# Lista para almacenar textos procesados y polaridades
textos_procesados = []
polaridades = []
polaridadesV = []

# Procesar cada línea del archivo de opiniones
with open('pregunta1.txt', 'r', encoding='utf-8') as archivo:
    for linea in archivo:
        correccion1 = normalize_text(linea)
        # Eliminar caracteres especiales
        linea_sin_caracteres = correccion1.replace('"', '').replace(',', '').replace('.', '')
        tokens = word_tokenize(linea_sin_caracteres)
        stop_words = set(stopwords.words('spanish'))
        tokens_sin_stopwords = [word for word in tokens if word.lower() not in stop_words]
        
        texto_procesado = ' '.join(tokens_sin_stopwords)
        #doc = nlp(' '.join(tokens_sin_stopwords))
        #lemmas = [token.lemma_ for token in doc]

        # Texto lematizado
        #texto_procesado = ' '.join(lemmas)
        textos_procesados.append(texto_procesado)

        # Calcular polaridad con TextBlob
        blob = TextBlob(texto_procesado)
        polaridad = blob.sentiment.polarity
        polaridades.append(polaridad)

        analyzer = SentimentIntensityAnalyzer()

        # Calcular polaridad VADER
        scores = analyzer.polarity_scores(texto_procesado)
        polaridadV = scores['compound']
        polaridadesV.append(polaridadV)

        #CALCULAR POLARIDAD CON TRANSFOMERS
        resultado = sentiment_pipeline(linea)
        

        # Imprimir resultados parciales
        print("\nTexto original:", linea.strip())
        print("Texto corregido:", linea_sin_caracteres.strip())
        print("Tokens sin stopwords:", tokens_sin_stopwords)
        #print("Lemmas:", lemmas)
        print("Polaridad:", polaridad)
        print("Polaridad (VADER):", polaridadV)
        print("Polaridad con transformadores:", resultado)

# Vectorización usando CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos_procesados)

# Agregar polaridad como característica adicional
X = np.hstack((X.toarray(), np.array(polaridades).reshape(-1, 1)))

# Normalizar las características al rango [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Leer etiquetas desde el archivo
with open('etiquetas_convertidas2.txt', 'r', encoding='utf-8') as archivo_etiquetas:
    labels = [int(line.strip()) for line in archivo_etiquetas]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2, random_state=42)

# Modelo 1: Máquina de Soporte Vectorial (SVM)
svm_model = SVC(kernel='linear', probability=True)  # `probability=True` para VotingClassifier
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluar SVM
print("\nResultados del modelo SVM:")
print("Precisión:", accuracy_score(y_test, y_pred_svm))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_svm))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_svm))

# Modelo 2: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Evaluar Naive Bayes
print("\nResultados del modelo Naive Bayes:")
print("Precisión:", accuracy_score(y_test, y_pred_nb))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_nb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_nb))

# Modelo combinado (VotingClassifier)
ensemble_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('naive_bayes', nb_model)
], voting='soft', weights=[2, 1])  # `soft` pondera por probabilidades
ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluar modelo combinado
print("\nResultados del modelo combinado (SVM + Naive Bayes):")
print("Precisión:", accuracy_score(y_test, y_pred_ensemble))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_ensemble))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_ensemble))