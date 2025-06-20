from ngrama import *

def main():
    directorio = "."
    archivo = "CorpusEducacion.txt"

    corpus = cargar_corpus(directorio, archivo)
    nuevo_corpus = corpus_procesado(corpus)
    n_gramas = grafico_ngramas(nuevo_corpus)

    print(n_gramas)


if __name__ == "__main__":
    main()