import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.es.stop_words import STOP_WORDS
import language_tool_python
from unicodedata import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

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
        # Eliminar caracteres especiales
        linea_sin_caracteres = linea.replace('"', '').replace(',', '').replace('.', '')
        tokens = word_tokenize(linea_sin_caracteres)
        stop_words = set(stopwords.words('spanish'))
        tokens_sin_stopwords = [word for word in tokens if word.lower() not in stop_words]
        doc = nlp(' '.join(tokens_sin_stopwords))
        lemmas = [token.lemma_ for token in doc]
        
        # Agregar el texto lematizado a la lista de textos procesados
        textos_procesados.append(' '.join(lemmas))
        
        # Imprimir resultados parciales
        print("\nTexto original:", linea)
        print("Texto corregido:", linea_sin_caracteres)
        print("Tokens:", tokens)  # Agarrar la variable tokens para la clasificación
        print("Tokens sin stopwords:", tokens_sin_stopwords)
        print("Lemmas:", lemmas)

# Vectorización usando CountVectorizer
vectorizer = CountVectorizer()  # Puedes cambiar a TfidfVectorizer si lo prefieres
X = vectorizer.fit_transform(textos_procesados)

# Imprimir la matriz de características y las palabras correspondientes
print("\nMatriz de características (CountVectorizer):\n", X.toarray())
print("Palabras del vocabulario:", vectorizer.get_feature_names_out())

#--------------addedddd al final

# Leer las etiquetas desde etiquetas.txt
with open('etiquetas-p1.txt', 'r', encoding='utf-8') as archivo_etiquetas:
    labels = [int(line.strip()) for line in archivo_etiquetas]


# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Modelo 1: Máquina de Soporte Vectorial (SVM)
svm_model = SVC(kernel='linear')  # Puedes probar con diferentes kernels como 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluar el modelo SVM
print("\nResultados del modelo SVM:")
print("Precisión:", accuracy_score(y_test, y_pred_svm))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_svm))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_svm))

# Modelo 2: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Evaluar el modelo Naive Bayes
print("\nResultados del modelo Naive Bayes:")
print("Precisión:", accuracy_score(y_test, y_pred_nb))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_nb))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_nb))


#---- UNIDOS AMBOS SVM Y NB

svm_model = SVC(kernel='linear', probability=True)  # `probability=True` es necesario para el VotingClassifier
nb_model = MultinomialNB()

# Combinar los modelos con VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('svm', svm_model),
    ('naive_bayes', nb_model)
], voting='soft', weights=[2, 1])  # `voting='soft'` usa las probabilidades, `voting='hard'` usa las predicciones

# Entrenar el modelo combinado
ensemble_model.fit(X_train, y_train)

# Realizar predicciones con el modelo combinado
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluar el rendimiento del modelo combinado
print("\nResultados del modelo combinado (SVM + Naive Bayes):")
print("Precisión:", accuracy_score(y_test, y_pred_ensemble))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_ensemble))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_ensemble))
