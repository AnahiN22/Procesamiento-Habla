import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documentos = {
    "doc1": "El veloz zorro marrón salta sobre el perro perezoso.",
    "doc2":"Un perro marrón persiguió al zorro.",
    "doc3": "El perro es perezoso."
}

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documentos.values())

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# DATAFRAME
df_sim = pd.DataFrame(cosine_sim, index=documentos.keys(), columns=documentos.keys())

# MOSTRAR MATRIZ
print("Matriz de similitud del coseno:\n")
print(df_sim.round(2))

# GRAFICAR MATRIZ
plt.figure(figsize=(8,6))
sns.heatmap(cosine_sim, annot=True, cmap="Blues", xticklabels=[f"Doc{i+1}" for i in range(len(documentos))], yticklabels=[f"Doc{i+1}" for i in range(len(documentos))])
plt.title("Matriz de similitud del coseno")
plt.show()