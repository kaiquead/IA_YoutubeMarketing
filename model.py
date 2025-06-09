import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Colunas que serão utilizadas
#----------------------------------------------------------------

colums_needed = [
    'video_trending_country',
    'video_category_id',
    'video_like_count',
    'video_comment_count',
    'channel_country',
    'channel_view_count',
    'channel_title'
]

# Países da Europa e EUA (nomes completos conforme dataset)
target_countries = [
    'United States', 'United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Sweden', 'Norway', 'Denmark',
    'Finland', 'Portugal', 'Greece', 'Austria', 'Switzerland', 'Belgium', 'Ireland',
    'Estonia', 'Czechia', 'Bulgaria', 'Croatia', 'Hungary', 'Iceland', 'Lithuania', 'Luxembourg',
    'Malta', 'Poland', 'Slovakia', 'Slovenia', 'Latvia', 'Liechtenstein'
]

print("Lendo o arquivo CSV...")
chunk_size = 100000
df_chunks = pd.read_csv('database/youtube_trending_videos_global.csv', chunksize=chunk_size, usecols=colums_needed)

# Primeiro, vamos verificar os dados
print("\nVerificando os dados...")
first_chunk = next(df_chunks)
print("\nCategorias disponíveis:")
print(first_chunk['video_category_id'].unique())
print("\nPaíses disponíveis:")
print(first_chunk['video_trending_country'].unique())

# Resetar o iterador
df_chunks = pd.read_csv('database/youtube_trending_videos_global.csv', chunksize=chunk_size, usecols=colums_needed)

df_filtered = pd.DataFrame()

print("\nFiltrando os dados...")
# Filtra os dados conforme os critérios especificados
for chunk in df_chunks:
    # Convertendo para string para garantir a comparação correta
    chunk['video_category_id'] = chunk['video_category_id'].astype(str)
    filtered = chunk[
        (chunk['video_category_id'].str.contains('Sports', case=False, na=False)) &
        (chunk['video_trending_country'].isin(target_countries)) &
        (pd.to_numeric(chunk['channel_view_count'], errors='coerce') > 100000000) &
        (pd.to_numeric(chunk['video_comment_count'], errors='coerce') > 5000)
    ].fillna(0)
    df_filtered = pd.concat([df_filtered, filtered], ignore_index=True)

print(f"Total de vídeos filtrados: {len(df_filtered)}")

if len(df_filtered) == 0:
    print("\nNenhum vídeo encontrado com os critérios atuais. Ajustando os critérios...")
    # Resetar o iterador
    df_chunks = pd.read_csv('database/youtube_trending_videos_global.csv', chunksize=chunk_size, usecols=colums_needed)
    df_filtered = pd.DataFrame()
    
    for chunk in df_chunks:
        chunk['video_category_id'] = chunk['video_category_id'].astype(str)
        filtered = chunk[
            (chunk['video_category_id'].str.contains('Sports', case=False, na=False)) &
            (chunk['video_trending_country'].isin(target_countries))
        ].fillna(0)
        df_filtered = pd.concat([df_filtered, filtered], ignore_index=True)
    
    print(f"Total de vídeos após ajuste dos critérios: {len(df_filtered)}")

if len(df_filtered) > 0:
    #----------------------------------------------------------------
    # Pré-processamento para K-means
    #----------------------------------------------------------------

    print("\nPreparando os dados para o K-means...")
    features = [
        'video_trending_country',
        'video_like_count',
        'video_comment_count',
        'channel_country',
        'channel_view_count'
    ]

    # Criar cópias das colunas originais para os prints
    df_filtered['channel_title_original'] = df_filtered['channel_title']
    df_filtered['video_trending_country_original'] = df_filtered['video_trending_country']

    # Tokenização das features categóricas
    le = LabelEncoder()
    for feature in features:
        if df_filtered[feature].dtype == 'object':
            df_filtered[feature] = le.fit_transform(df_filtered[feature].astype(str))

    X = df_filtered[features].astype(float)

    # Normaliza os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #----------------------------------------------------------------
    # Aplicação do K-means
    #----------------------------------------------------------------

    print("Aplicando K-means...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered['cluster'] = kmeans.fit_predict(X_scaled)

    # Encontra os vídeos mais relevantes de cada cluster
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    most_relevant_videos = []

    for cluster in range(3):
        cluster_videos = df_filtered[df_filtered['cluster'] == cluster]
        # Calcula a distância para o centro do cluster
        distances = np.linalg.norm(X_scaled[df_filtered['cluster'] == cluster] - kmeans.cluster_centers_[cluster], axis=1)
        # Pega o vídeo mais próximo do centro
        most_relevant_idx = distances.argmin()
        most_relevant_videos.append(cluster_videos.iloc[most_relevant_idx])

    print("\nVídeos mais relevantes por cluster:")
    for i, video in enumerate(most_relevant_videos):
        print(f"\nCluster {i+1}:")
        print(f"Canal: {video['channel_title_original']}")
        print(f"País: {video['video_trending_country_original']}")
        print(f"Visualizações do canal: {video['channel_view_count']}")
        print(f"Comentários do vídeo: {video['video_comment_count']}")
        print(f"Curtidas do vídeo: {video['video_like_count']}")

    #----------------------------------------------------------------
    # Análise de Silhueta
    #----------------------------------------------------------------
    print("\nRealizando análise de silhueta...")

    # Calcula o score de silhueta
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    print(f"\nScore médio de silhueta: {silhouette_avg:.3f}")

    # Calcula os scores de silhueta para cada amostra
    sample_silhouette_values = silhouette_samples(X_scaled, kmeans.labels_)

    # Cria o gráfico de silhueta
    plt.figure(figsize=(10, 6))
    y_lower = 10

    for i in range(3):  # 3 clusters
        # Agrupa os scores de silhueta para o cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / 3)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Adiciona o número do cluster no meio do gráfico
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))
        
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.xlabel("Coeficiente de Silhueta")
    plt.ylabel("Cluster")
    plt.title("Análise de Silhueta para 3 clusters")
    plt.show()  # Mostra o gráfico
else:
    print("\nNenhum vídeo encontrado mesmo após ajuste dos critérios.")
