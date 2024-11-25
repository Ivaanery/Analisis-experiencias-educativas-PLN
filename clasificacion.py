from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Leer las etiquetas desde etiquetas.txt
with open('etiquetas.txt', 'r', encoding='utf-8') as archivo_etiquetas:
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
