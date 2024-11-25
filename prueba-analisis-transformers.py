from transformers import pipeline

# Cargar un modelo de análisis de sentimientos en español
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Prueba con un texto en español
resultado = sentiment_pipeline("Nunca había estado tan feliz en mi vida. Fue maravilloso.")
print(resultado)
