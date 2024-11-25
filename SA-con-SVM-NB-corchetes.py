import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline


ruta_archivo = 'pregunta1-etiquetada-corchete.txt'

# Lee los datos desde el archivo de texto
def cargar_datos(ruta_archivo):
    X = []  # opiniones         
    y = []  # etiquetas

    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        for line in file:
            opinion, etiqueta = line.strip().split('>')
            X.append(opinion)
            y.append(etiqueta)
            print(y)

    return X, y

# Cargar datos de ejemplo
X, y = cargar_datos('pregunta1-etiquetada-corchete.txt')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo que mezcle SVM y Naive Bayes
svm_model = SVC(kernel='linear', probability=True)
nb_model = MultinomialNB()
voting_model = VotingClassifier(estimators=[('svm', svm_model), ('nb', nb_model)], voting='soft')

# Crear un pipeline con vectorización TF-IDF
pipeline = make_pipeline(TfidfVectorizer(), voting_model)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Predecir las etiquetas en el conjunto de prueba
y_pred = pipeline.predict(X_test)

# Imprimir el reporte de clasificación
print(classification_report(y_test, y_pred))
