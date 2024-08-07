from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Baixar stopwords em português
nltk.download('stopwords')

# Carregar os dados
dados = pd.read_excel("magalu_com_seller.xlsx")

# TF-IDF Vectorization
stop_words = stopwords.words('portuguese')
vectorizer = TfidfVectorizer(max_features=500, stop_words=stop_words)
X = vectorizer.fit_transform(dados['Título'])

# Clusterização com KMeans
kmeans = KMeans(n_clusters=25, random_state=42)
kmeans.fit(X)

# Palavras mais frequentes em cada cluster
common_words = vectorizer.get_feature_names_out()
centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

# Criar DataFrame para armazenar os resultados
resultados = pd.DataFrame(columns=['Categoria', 'Cluster', 'Padrão'])

for i in range(len(centroids)):
    category = dados['Origem'].unique()[i % len(dados['Origem'].unique())]
    cluster_number = i + 1
    pattern = []
    for ind in centroids[i, :5]:
        pattern.append(common_words[ind])
    pattern = sorted(pattern, key=lambda x: dados[dados['Origem'] == category]['Título'].str.contains(x).sum(), reverse=True)
    resultados = pd.concat([resultados, pd.DataFrame({'Categoria': [category], 'Cluster': [cluster_number], 'Padrão': [' + '.join(pattern)]})], ignore_index=True)

resultados
