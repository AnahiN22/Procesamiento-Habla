import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# DOCUMENTOS
documentos = {
    "doc1": "Los egipcios construyeron las pirámides y desarrollaron una escritura jeroglífica.",
    "doc2": "La civilización romana fue una de las más influyentes en la historia occidental.",
    "doc3": "Los mayas eran expertos astrónomos y tenían un avanzado sistema de escritura.",
    "doc4": "La antigua Grecia sentó las bases de la democracia y la filosofía moderna.",
    "doc5": "Los sumerios inventaron la escritura cuneiforme y fundaron las primeras ciudades."
}

# STOPWORDS Y TOKENIZACIÓN
stop_words = set(stopwords.words('spanish'))

def preprocesamiento(texto):
    tokens = word_tokenize(texto.lower())
    return set([word for word in tokens if word.isalnum() and word not in stop_words])

# ÍNDICE INVERTIDO
indice = {}
for doc_id, texto in documentos.items():
    palabras = preprocesamiento(texto)
    for palabra in palabras:
        if palabra not in indice:
            indice[palabra] = set()
        indice[palabra].add(doc_id)

# BÚSQUEDA BOOLEANA
def busqueda_booleana(query):
    query = query.lower().strip()
    if query == "salir":
        return "salir"

    terms = query.split()
    if not terms:
        return set()
    
    result_set = set(documentos.keys())
    i = 0

    while i < len(terms):
        term = terms[i]

        if term == "and":
            i += 1
            if i < len(terms):
                result_set &= indice.get(terms[i], set())
        elif term == "or":
            i += 1
            if i < len(terms):
                result_set |= indice.get(terms[i], set())
        elif term == "not":
            i += 1
            if i < len(terms):
                result_set -= indice.get(terms[i], set())
        else:
            result_set &= indice.get(term, set())
        i += 1

    return result_set

while True:
    consulta = input("Escriba la consulta booleana (o 'salir' para terminar): ")
    resultado = busqueda_booleana(consulta)

    if resultado == "salir":
        print("Programa finalizado.")
        break

    if resultado:
        print("Documentos encontrados:", sorted(resultado))
    else:
        print("No se encontraron documentos.")