import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_df_train_df_test(df, reference_year, drop_year=None):
 
  # Separar os dados de treino e teste com base no ano
  df_train_year = df[df['ANO'] < reference_year]  # Dados anteriores ao ano avaliado (foi retirado o ano de 2020, pois análises mostraram que eles enviesam o treinamento)
  df_test_year = df[df['ANO'] == reference_year]  # Dados do ano que será avaliado

  if drop_year is not None:
    # Verificar se drop_year é uma lista e usar 'isin' para excluir os anos
    if isinstance(drop_year, list):
        # Filtra o DataFrame `df_train`, excluindo todas as linhas onde a coluna 'ANO' possui um valor contido na lista `drop_year`
        df_train_year = df_train_year[~df_train_year['ANO'].isin(drop_year)]
        # Filtra o DataFrame `df_test` da mesma forma, excluindo as linhas onde a coluna 'ANO' possui um valor da lista `drop_year`
        df_test_year = df_test_year[~df_test_year['ANO'].isin(drop_year)]
  
  df_train = df_train_year.drop(columns=['ANO']).dropna() # retira a coluna de ano do treino
  df_test = df_test_year.drop(columns=['ANO']).dropna() # retira a coluna de ano do teste


  # Definir as variáveis independentes (features) e dependente (target) do treino
  features_train = df_train.drop(columns=['AE'])  # Todas as colunas exceto 'AE'
  target_train = df_train['AE']  # A coluna de saída (AE)
  # Definir as variáveis de independentes (features) e dependente (target) do treino
  features_test = df_test.drop(columns=['AE'])  # Todas as colunas exceto 'AE'
  target_test = df_test['AE']  # A coluna de saída (AE)

  print("df_train shape:", df_train.shape)
  print("features_train shape:", features_train.shape)
  print("target_train shape:", target_train.shape)
            
  return df_train, df_test, features_train, target_train, features_test, target_test


def plot_data_exploration_from_categorical_features_to_drop_rate(categorical_features, df_train):

  # Configurando o estilo do seaborn
  sns.set(style='whitegrid')

  # Define o tamanho da figura para acomodar os gráficos, com 15 unidades de largura e 10 de altura
  plt.figure(figsize=(15, 10))

  # Itera sobre as variáveis categóricas usando a função enumerate para obter o índice (i) e o nome da variável (feature)
  for i, feature in enumerate(categorical_features):

      # Cria subplots, organizados em uma grade 3x3, e posiciona cada gráfico na próxima posição disponível
      plt.subplot(3, 3, i + 1)

      # Cria um gráfico de contagem (countplot) para cada variável categórica, diferenciando as barras pela variável 'AE'
      sns.countplot(data=df_train, x=feature, hue='AE', palette='Set2')

      # Define o título do gráfico como "Contagem de [nome da variável] por AE"
      plt.title(f'Contagem de {feature} por AE')

      # Define o rótulo do eixo x com o nome da variável atual
      plt.xlabel(feature)

      # Define o rótulo do eixo y como "Contagem" para indicar o número de ocorrências
      plt.ylabel('Contagem')

  plt.tight_layout()
  plt.show()


def plot_data_exploration_from_categorical_features_to_drop_rate_percent(categorical_features, df_train):

  # Configurando o estilo do seaborn
  sns.set(style='whitegrid')

  # Define o tamanho da figura para acomodar os gráficos, com 15 unidades de largura e 10 de altura
  plt.figure(figsize=(15, 10))

  # Iterando sobre as variáveis categóricas
  for i, feature in enumerate(categorical_features):
      # Criando subplots, organizados em uma grade 3x3
      plt.subplot(3, 3, i + 1)

      # Calcula as proporções (frequências relativas) de cada valor categórico por AE
      df_grouped = df_train.groupby([feature, 'AE']).size().reset_index(name='counts')
      df_grouped['percent'] = df_grouped.groupby('AE')['counts'].transform(lambda x: x / x.sum()) * 100

      # Cria o gráfico de barras com as proporções relativas
      ax = sns.barplot(data=df_grouped, x=feature, y='percent', hue='AE', palette='Set2')

      # Adiciona o percentual acima de cada barra
      #for p in ax.patches:
      #    height = p.get_height()
      #    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
      #                ha='center', va='bottom', fontsize=10, color='black')

      # Define o título e os rótulos dos eixos
      plt.title(f'Proporção de {feature} por AE')
      plt.xlabel(feature)
      plt.ylabel('Proporção (%)')

def plot_data_exploration_from_numerical_features_to_drop_rate(numerical_features, df_train):

  # Ajusta o layout para que os gráficos não se sobreponham
  plt.tight_layout()
  plt.show()

  # Define o tamanho da figura para o gráfico, com 8 unidades de largura e 6 de altura
  plt.figure(figsize=(8, 6))

    # Itera sobre as variáveis categóricas usando a função enumerate para obter o índice (i) e o nome da variável (feature)
  for i, feature in enumerate(numerical_features):

      # Cria subplots, organizados em uma grade 3x3, e posiciona cada gráfico na próxima posição disponível
      plt.subplot(3, 3, i + 1)

      # Cria um boxplot da variável numérica 'IDA-MÉD' com o eixo x sendo 'AE' (a variável categórica de interesse)
      sns.boxplot(data=df_train, x='AE', y=feature)

      # Define o título do gráfico como "Boxplot de IDA-MÉD por AE"
      plt.title(f'Boxplot de {feature} por AE')

      # Define o rótulo do eixo x como 'AE'
      plt.xlabel('AE')

      # Define o rótulo do eixo y como 'IDA-MÉD'
      plt.ylabel('IDA-MÉD')

  # Exibe o gráfico
  plt.show()

def get_description(df):
  # Descrição das variáveis numéricas
  numerical_description = df.describe()
  print("Descrição das Variáveis Numéricas:")
  print(numerical_description)

  # Descrição das variáveis categóricas

  # Seleciona as colunas do DataFrame df_train que possuem dados do tipo object, columns acessa os nomes das colunas retornadas pela seleção select_dtypes
  categorical_columns = df.select_dtypes(include=['object']).columns

  # categorical_columns conterá uma lista com os nomes das colunas que possuem dados categóricos em df_train
  print("\nDescrição das Variáveis Categóricas:")

  for column in categorical_columns:
      print(f"\nFrequência de valores em {column}:")
      print(df[column].value_counts())

def show_and_drop_missing_values(df):
  # Apresentar valores nulos
  null = df.isnull().sum()

  # Exibir colunas com valores nulos
  print("Valores nulos por coluna:")
  print(null[null > 0])
  # Eliminando valores nulos da base de treino
  df = df.dropna()
  # Exibir colunas com valores nulos
  print("Valores nulos por coluna:")
  print(null[null > 0])

def show_dataset_information(df_train, df_test):
  """
  Exibe informações sobre os conjuntos de treino e teste.

  Parâmetros:
  - df_train: DataFrame de treino.
  - df_test: DataFrame de teste.
  """
  # Exibir o número de instâncias em cada conjunto
  print("Conjunto de treino:", len(df_train))
  print("Conjunto de teste:", len(df_test))

  # Exibir o shape do conjunto de treino
  print("Shape do conjunto de treino:", df_train.shape)

  # Exibir a contagem de valores nulos por coluna
  print("Valores nulos no conjunto de treino:")
  print(df_train.isna().sum())

  # Exibir os tipos de dados de cada coluna no conjunto de treino
  print("\nTipos de dados no conjunto de treino:")
  print(df_train.dtypes)

  # Apresentar e exibir colunas com valores nulos
  null = df_train.isnull().sum()
  print("\nColunas com valores nulos:")
  print(null[null > 0])

  df_train.head(100)

def filter_dataframe_by_column(df, column_name, value):
  """
  Filtra um DataFrame com base em um valor específico de uma coluna.

  Parâmetros:
  - df: DataFrame a ser filtrado.
  - column_name: Nome da coluna a ser usada para o filtro.
  - value: Valor da coluna que será usado para filtrar os dados.

  Retorno:
  - DataFrame filtrado contendo apenas as linhas onde a coluna especificada corresponde ao valor dado.
  """
  return df[df[column_name] == value]
