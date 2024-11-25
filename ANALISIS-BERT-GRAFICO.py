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
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt_tab')
spacy.cli.download("es_core_news_sm")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


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
textos_procesados = []
polaridades = []
polaridadesV = []
polaridades_transformers = []


# Abrir el archivo en modo de lectura
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
        resultado = sentiment_pipeline(linea.strip())
        # Convertir la polaridad en un número entre -1 y 1
        label = resultado[0]['label']  # Ejemplo: '5 stars'
        score = resultado[0]['score']  # Probabilidad asociada
        if "1" in label or "2" in label:
            polaridad_transformers = -1 * score  # Sentimiento negativo
        elif "4" in label or "5" in label:
            polaridad_transformers = 1 * score  # Sentimiento positivo
        else:
            polaridad_transformers = 0  # Neutral
        polaridades_transformers.append(polaridad_transformers)

    

        # Imprimir resultados parciales
        print("\nTexto original:", linea.strip())
        print("Texto corregido:", linea_sin_caracteres.strip())
        print("Tokens sin stopwords:", tokens_sin_stopwords)
        #print("Lemmas:", lemmas)
        print("Polaridad:", polaridad)
        print("Polaridad (VADER):", polaridadV)
        print("Polaridad con transformadores:", resultado)

# Leer las calificaciones normalizadas desde el archivo
with open('calificaciones_normalizadas.txt', 'r', encoding='utf-8') as archivo_calificaciones:
    calificaciones = [float(line.strip()) for line in archivo_calificaciones]

# Asegurarse de que las listas coincidan en longitud
assert len(polaridades_transformers) == len(calificaciones), "El número de polaridades y calificaciones no coincide."

# Graficar correlación de cada respuesta
plt.figure(figsize=(10, 6))
plt.scatter(range(len(calificaciones)), calificaciones, label="Calificaciones", color='blue', alpha=0.6)
plt.scatter(range(len(polaridades_transformers)), polaridades_transformers, label="Polaridades Transformers", color='red', alpha=0.6)
plt.title("Calificaciones vs Polaridades por Respuesta")
plt.xlabel("Índice de Respuesta")
plt.ylabel("Valor")
plt.legend()
plt.show()

# Calcular y graficar la correlación general
correlacion = np.corrcoef(calificaciones, polaridades_transformers)[0, 1]
plt.figure(figsize=(8, 6))
plt.scatter(calificaciones, polaridades_transformers, alpha=0.6)
plt.title(f"Correlación General: {correlacion:.2f}")
plt.xlabel("Calificaciones")
plt.ylabel("Polaridades Transformers")
plt.grid(True)
plt.show()

# Imprimir el valor de la correlación
print(f"Correlación general entre calificaciones y polaridades (Transformers): {correlacion:.2f}")


#GENERACION DEL MODELO BERT
# 1. Cargar los datos desde los archivos de texto

with open("etiquetas_convertidas2.txt", "r", encoding="utf-8") as f:
    etiquetas = [int(line.strip()) for line in f.readlines()]

# Crear un DataFrame con los datos cargados
df = pd.DataFrame({"texto": textos_procesados, "etiqueta": etiquetas})

# 2. Dividir los datos en entrenamiento y prueba
train_texts, test_texts, train_labels, test_labels = train_test_split(df['texto'], df['etiqueta'], test_size=0.2, random_state=42)

# 3. Preparar el tokenizador y el modelo BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Para clasificación binaria

# 4. Crear el Dataset personalizado
class OpinionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        # Tokenizar el texto
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Crear los DataLoaders
train_dataset = OpinionDataset(train_texts, train_labels, tokenizer)
test_dataset = OpinionDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Configurar el optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. Entrenar el modelo
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss}")

# 7. Evaluar el modelo
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calcular la precisión y mostrar el reporte de clasificación
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(true_labels, predictions))
