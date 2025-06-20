import nltk
import string
import matplotlib.pyplot as plt
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

"""
# DESCARGAS
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"""

# CARGAR CORPUS
def cargar_corpus(directorio, archivo):
    ruta = os.path.join(directorio, archivo)
    with open(ruta, encoding = "utf-8") as f:
        corpus_cargado = f.read().lower()
    return corpus_cargado

# TOKENIZAR TEXTO
def tokenizar_texto(corpus):
    corpus_tokenizado = word_tokenize(corpus)
    return corpus_tokenizado

# QUITAR STOPWORDS
def quitarStopwords_esp(corpus):
    espaniol = stopwords.words("spanish")
    corpus_limpio = [w.lower() for w in corpus if w.lower() not in espaniol and w not in string.punctuation]
    return corpus_limpio
     
# LEMATIZAR
def corpus_lematizado(corpus):
    lemmatizer = WordNetLemmatizer()
    corpus_lema = [lemmatizer.lemmatize(w) for w in corpus]
    return corpus_lema

# CORPUS PROCESADO
def corpus_procesado(corpus):
    token_corpus = tokenizar_texto(corpus)
    corpus_limpio = quitarStopwords_esp(token_corpus)
    nuevo_corpus = corpus_lematizado(corpus_limpio)
    procesado = " ".join(nuevo_corpus)
    return procesado

def grafico_ngramas(corpus):
    vectorizador_bi = CountVectorizer(ngram_range=(2, 2), min_df=1)
    vectorizador_tri = CountVectorizer(ngram_range=(3, 3), min_df=1)

    X_bi = vectorizador_bi.fit_transform([corpus])
    X_tri = vectorizador_tri.fit_transform([corpus])

    # NOMBRES Y FRECUENCIAS
    bigramas = vectorizador_bi.get_feature_names_out()
    trigramas = vectorizador_tri.get_feature_names_out()
    freqs_bi = X_bi.toarray()[0]
    freqs_tri = X_tri.toarray()[0]

    # 10 MÁS FRECUENTES
    top_bi = sorted(zip(bigramas, freqs_bi), key=lambda x: x[1], reverse=True)[:10]
    top_tri = sorted(zip(trigramas, freqs_tri), key=lambda x: x[1], reverse=True)[:10]

    # LISTAS Y POSICIONES
    etiquetas = []
    posiciones_bi = []
    valores_bi = []
    posiciones_tri = []
    valores_tri = []

    # ALTERNAR BIGRAMAS/TRIGRAMAS
    for i in range(max(len(top_bi), len(top_tri))):
        if i < len(top_bi):
            posiciones_bi.append(i * 2) # pares para bigramas
            valores_bi.append(top_bi[i][1])
            etiquetas.append(top_bi[i][0])
        if i < len(top_tri):
            posiciones_tri.append(i * 2 + 1) # impares para trigramas
            valores_tri.append(top_tri[i][1])
            etiquetas.append(top_tri[i][0])

    plt.figure(figsize=(16,6))
    plt.bar(posiciones_bi, valores_bi, width=0.8, color='skyblue', label='Bigramas')
    plt.bar(posiciones_tri, valores_tri, width=0.8, color='lightgreen', label='Trigramas')

    # ETIQUETAS CENTRADAS
    posiciones_totales = posiciones_bi + posiciones_tri
    plt.xticks(sorted(posiciones_totales), etiquetas, rotation=45, ha='right')

    plt.ylabel("Frecuencia")
    plt.title("Comparación de Frecuencias: Bigramas vs Trigramas")
    plt.legend()
    plt.tight_layout()
    plt.show()
