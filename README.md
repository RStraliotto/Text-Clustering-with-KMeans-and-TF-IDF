# Text Clustering with KMeans and TF-IDF

Clusterize textos com KMeans e TF-IDF para identificar padrões e temas em grandes volumes de dados textuais. O projeto inclui vetorização TF-IDF, clusterização KMeans, e extração de palavras-chave para análise detalhada de textos. Ideal para organização e categorização de conteúdos.

## Estrutura do Projeto

- **`clusterizacao_textos.py`**: Script principal que carrega os dados, aplica a vetorização TF-IDF, realiza a clusterização com KMeans e gera um DataFrame com os resultados.
- **`magalu_com_seller.xlsx`**: Arquivo de entrada contendo os dados de texto a serem clusterizados. Certifique-se de que este arquivo esteja no mesmo diretório do script.

## Dependências

Certifique-se de instalar as seguintes dependências Python:

```bash
pip install pandas scikit-learn nltk openpyxl
```

##Instruções de Uso
Prepare os Dados: Coloque o arquivo magalu_com_seller.xlsx no mesmo diretório que o script clusterizacao_textos.py. Certifique-se de que o arquivo contém uma coluna chamada Título com os textos a serem clusterizados e uma coluna chamada Origem para categorizar os textos.

Execute o Script: Execute o script Python para realizar a clusterização e gerar os resultados.
```bash
python clusterizacao_textos.py
```
Resultados: O script cria um DataFrame com os resultados da clusterização, contendo os clusters identificados, suas categorias mais frequentes e as palavras mais representativas de cada cluster.
![image](https://github.com/user-attachments/assets/770cdbda-67f4-4c4d-a878-ff07e36f333f)


## Código
Aqui está um exemplo do código usado para a clusterização:
```bash
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
    cluster_number = i + 1
    pattern = [common_words[ind] for ind in centroids[i, :5]]
    category = dados.iloc[kmeans.labels_ == i]['Origem'].mode()[0]
    resultados = resultados.append({'Categoria': category, 'Cluster': cluster_number, 'Padrão': ' + '.join(pattern)}, ignore_index=True)

resultados
```

## Contribuições
Sinta-se à vontade para contribuir com melhorias e correções. Faça um fork do repositório, crie uma branch para sua feature, e envie um pull request com suas alterações.

## Licença
Este projeto está licenciado sob a Licença MIT.

Agora você pode copiar o conteúdo inteiro diretamente para o seu arquivo `README.md`. Se precisar de mais ajustes, é só me avisar!
