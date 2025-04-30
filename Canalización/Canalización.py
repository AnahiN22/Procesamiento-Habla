import nltk
import pandas as pd
import string

# Tokenizar textos
from nltk import word_tokenize

# StopWords
from nltk.corpus import stopwords

# Lematizar
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Hacer gráfico de frecuencias
from nltk import FreqDist

# TF - IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# BoW
from sklearn.feature_extraction.text import CountVectorizer


lemmatizer = WordNetLemmatizer()
vectorizer_tfidf = TfidfVectorizer()
vectorizer_bow = CountVectorizer()


def quitarStopwords_eng(texto):
    ingles = stopwords.words("english")
    texto_limpio = [w.lower() for w in texto if w.lower() not in ingles
    and w not in string.punctuation
    and w not in ["'s", '|', '--', "''", "``"] ]
    return texto_limpio

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lematizar(texto):
    txt_lema = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in texto]
    return txt_lema


# Corpus lista de listas
corpus = [
lematizar(quitarStopwords_eng(word_tokenize("Python is an interpreted and high-level language, while CPlus is a compiled and low-level language."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security."))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is slower than CPlus and Rust due to its interpreted nature."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science."))),
lematizar(quitarStopwords_eng(word_tokenize("JavaScript does not require compilation, while CPlus and Rust require code compilation before execution."))),
lematizar(quitarStopwords_eng(word_tokenize("Python and JavaScript have large communities and an extensive number of available libraries."))),
lematizar(quitarStopwords_eng(word_tokenize("Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers.")))
]


# Corpus lista con palabras sepadas
new_corpus = []

for oracion in corpus:
    for palabra in oracion:
        new_corpus.append(palabra)
#print(new_corpus)


# Corpus como una lista de oraciones
document = [' '.join(doc) for doc in corpus]
print("CORPUS PREPARADO\n", document)


# Matriz TF-IDF
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.max_colwidth', None) # Mostrar todo el contenido de las columnas

corpusLista_tfidf = vectorizer_tfidf.fit_transform(document)
matriz_tfidf = pd.DataFrame(corpusLista_tfidf.toarray(), columns=vectorizer_tfidf.get_feature_names_out()) # Muestra la matriz con más decimales
matriz_tfidf.index = range(1, len(matriz_tfidf)+1)

print("\nMATRIZ TF-IDF")
print(matriz_tfidf)


# Bow - Palabra mas repetida en una oración
corpusLista_bow = vectorizer_bow.fit_transform(document)

print("\nMATRIZ BoW - FRECUENCIA DE PALABRAS")
print("\nVOCABULARIO:")
matriz_bow = pd.DataFrame(corpusLista_bow.toarray(), columns=vectorizer_bow.get_feature_names_out())
matriz_bow.index = range(1, len(matriz_bow)+1)

print(matriz_bow)

print("\nPALABRAS QUE SE REPITEN EN CADA ORACIÓN")
for i, oracion in matriz_bow.iterrows(): # .iterrows p/ iterar por filas
    repetidas = oracion[oracion > 1].index.tolist()  # Palabras con frecuencia mayor a 1
    if repetidas:
        print(f"Oración {i}: {', '.join(repetidas)}")
    else:
        print(f"Oración {i}: No hay palabras repetidas.")


# Gráfico de frecuencia
frecuencia = FreqDist(new_corpus)

print("\nLISTA DE PALABRAS MÁS FRECUENTES")
for i, (palabra, freq) in enumerate(frecuencia.most_common(6), 1):
    print(f"{i}. {palabra}: {freq}") # Paso como parametro la cantidad que quiero mostrar

frecuencia.plot(10, show = True) # gráfico de más frecuentes

# Encontrar la palabra menos usada
min_frecuencia = min(frecuencia.values())
menos_usadas = [palabra for palabra, freq in frecuencia.items() if freq == min_frecuencia] # Palabras con frec. mín.

print(f"\nLISTA DE PALABRAS MENOS USADAS")
for i, palabra in enumerate(menos_usadas, 1):
    print(f"{i}. {palabra}")

"""menos_freq = FreqDist(menos_usadas)
menos_freq.plot(10, show=True) # Gráfico de menos frecuentes"""