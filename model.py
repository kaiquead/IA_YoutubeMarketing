import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

#----------------------------------------------------------------
# Colunas que serão utilizadas
#----------------------------------------------------------------

colums_needed = [
    'video_trending__date',
    'video_trending_country',
    'video_category_id',
    'video_tags',
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
        (pd.to_numeric(chunk['channel_view_count'], errors='coerce') > 100000) &
        (pd.to_numeric(chunk['video_comment_count'], errors='coerce') > 200)
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
    # Seleciona as features para o clustering
    features = ['video_like_count', 'video_comment_count', 'channel_view_count']
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
        print(f"Canal: {video['channel_title']}")
        print(f"País: {video['video_trending_country']}")
        print(f"Visualizações do canal: {video['channel_view_count']}")
        print(f"Comentários do vídeo: {video['video_comment_count']}")
        print(f"Curtidas do vídeo: {video['video_like_count']}")
else:
    print("\nNenhum vídeo encontrado mesmo após ajuste dos critérios.")
