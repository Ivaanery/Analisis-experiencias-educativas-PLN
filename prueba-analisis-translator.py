from textblob import TextBlob

# Ejemplo de texto en español
texto = "Estoy muy feliz con esta experiencia."

# Traducción al inglés
blob = TextBlob(texto)
texto_traducido = blob.translate(to='en')

# Análisis de sentimientos
polaridad = texto_traducido.sentiment.polarity
print("Polaridad traducida:", polaridad)
