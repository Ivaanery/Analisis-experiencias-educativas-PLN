from unicodedata import normalize

def normalize_text (text):
  return normalize('NFKD', text).encode('ASCII', 'ignore') 

print(normalize_text('aáaá eéeé iíií oóoó ñnñn AÀAÀ'))

