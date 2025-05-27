import pandas as pd

#----------------------------------------------------------------
# Lê o arquivo CSV e exibe todas as colunas da primeira linha. Util para ter um exemplo de quais informações temos
#----------------------------------------------------------------

# Lê apenas a primeira linha do CSV (linha 0)
df_example = pd.read_csv('database/youtube_trending_videos_global.csv', nrows=1)
# Obtem os nomes das colunas do DataFrame
df_example_colums = df_example.columns.tolist()

print("NOME DAS COLUNAS:\n\n")

# Exibe os nomes das colunas
for colum in df_example_colums:
    colum_name = df_example[colum]
    print('coluna:', colum, '\n Valor:', colum_name.tolist())



#----------------------------------------------------------------
# Colunas que serão utilizadas no modelo
#----------------------------------------------------------------

colums_needed = ['video_category_id']

chunk_size = 100000  # número de linhas por pedaço
df = pd.read_csv('database/youtube_trending_videos_global.csv', chunksize=chunk_size, usecols=colums_needed)

# Criar DataFrame filtrado vazio
df_filtrado = pd.DataFrame()

# Adiciona somente linhas de Sports
for chunk in df:
    filtrado = chunk[chunk['video_category_id'] == 'Sports']
    df_filtrado = pd.concat([df_filtrado, filtrado], ignore_index=True)

print(df_filtrado.count)