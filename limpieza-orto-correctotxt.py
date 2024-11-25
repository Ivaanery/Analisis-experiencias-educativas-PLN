import language_tool_python
from unicodedata import normalize

#ortografia
def corregir_ortografia(texto):
    herramienta = language_tool_python.LanguageTool('es')  # Configura para español
    correcciones = herramienta.correct(texto)
    return correcciones

#acentos
def normalize_text (text):
  return normalize('NFKD', text).encode('ASCII', 'ignore') 

# Abrir el archivo en modo de lectura
with open('pregunta1.txt', 'r', encoding='utf-8') as archivo:
    # Leer línea por línea
    for linea in archivo:
        correccion1 = normalize_text(linea)
        #print(correccion1)
        #texto_corregido = corregir_ortografia(correccion1)
        print("\nTexto anterior:", linea)
        print("Texto corregido:", correccion1)


