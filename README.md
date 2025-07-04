# Análise de Canais Relevantes para Divulgação de Produtos Esportivos no YouTube

Com o objetivo de identificar os canais mais relevantes para a divulgação de nossos produtos esportivos, conduzimos uma análise aprofundada em uma base de dados do YouTube com aproximadamente **6 GB** de informações. A proposta era filtrar e extrair os registros mais relevantes para o nosso público-alvo, com foco em conteúdo esportivo e grande alcance.

## 1. Filtragem dos Dados

A primeira etapa consistiu na filtragem dos dados brutos. Estabelecemos critérios claros para manter apenas os registros com maior potencial de relevância:

- **Categoria do vídeo:** `Sports`  
- **País de trending:** países da Europa e Estados Unidos  
- **Visualizações do canal:** superiores a 100.000.000  
- **Número de comentários no vídeo:** superior a 5.000  

Essa filtragem permitiu reduzir significativamente o volume de dados, mantendo apenas os canais e vídeos mais engajados dentro do segmento esportivo.

## 2. Pré-processamento e Engenharia de Atributos

Após a filtragem, realizamos o pré-processamento dos dados para preparar as variáveis para o modelo de agrupamento:

- **Variáveis quantitativas utilizadas:**
  - Número de visualizações do canal  
  - Número de comentários no vídeo  
  - Número de likes no vídeo  

- **Variáveis categóricas utilizadas:**
  - País de trending  
  - País do canal  

Para incluir essas variáveis categóricas no modelo, aplicamos **tokenização** (transformação de texto em representação numérica).

Além disso, utilizamos o **StandardScaler** para normalizar os dados numéricos. Essa normalização foi essencial para garantir que as variáveis com escalas distintas (como visualizações e likes) tivessem peso equilibrado durante o treinamento.

## 3. Modelagem com K-Means

Com os dados preparados, aplicamos o algoritmo **K-means** com **3 clusters**, com o objetivo de segmentar os canais em três grupos distintos, classificados por relevância.

Utilizamos a **distância euclidiana** como métrica de similaridade para a formação dos clusters, buscando otimizar a distribuição dos pontos no espaço multidimensional.

## 4. Validação dos Resultados

Para avaliar a qualidade dos agrupamentos, usamos o **coeficiente de silhueta**, uma métrica que analisa o quão bem cada ponto se ajusta ao seu próprio cluster em comparação com os outros clusters. O resultado foi um **score médio de silhueta de 0.33**, o que indica uma separação moderadamente boa entre os grupos identificados.

## 5. Resultados Finais e Avaliação de Estratégia

Com o modelo de 3 clusters localizamos quais os 3 canais que estavam mais perto do centroide e com isso identificamos os **três canais mais relevantes** para divulgação de produtos esportivos:

- **Red Bull**
- **UFC**
- **NFL**

Esses canais se destacaram por seu grande alcance, engajamento e relevância dentro do cenário esportivo.

Buscando melhorar a qualidade da segmentação, testamos o modelo com uma quantidade maior de clusters:

- Com **5 clusters**, o **score médio de silhueta foi de 0.35**  
- Com **10 clusters**, o **score subiu para 0.37**

Apesar da leve melhora nos scores, o aumento no número de clusters trouxe mais complexidade à tomada de decisão, pois o número de canais candidatos à escolha também aumentou. Isso poderia dificultar o processo de definição de uma única parceria ou foco de divulgação por parte de empresas.

Portanto, **optamos por manter o modelo com 3 clusters e score médio de 0.33**, por apresentar um bom equilíbrio entre desempenho e aplicabilidade prática. As imagens com o resultado do coeficiente da silhueta estão dentro do projeto.
