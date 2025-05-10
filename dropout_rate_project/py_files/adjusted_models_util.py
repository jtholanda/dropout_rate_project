from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, recall_score, roc_curve, auc, roc_auc_score, precision_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score


def create_preprocessor(X_train):
    """
    Cria um objeto ColumnTransformer para o pré-processamento de dados.

    Parâmetros:
    X_train (DataFrame): O conjunto de dados de treinamento.

    Retorna:
    ColumnTransformer: O objeto de pré-processamento configurado.
    """
    # Seleção de colunas numéricas e categóricas
    numeric_features = X_train.select_dtypes(include=['float64', 'int']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Preprocessamento para colunas numéricas e categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # foi preciso configurá-lo handle_unknown='ignore' para lidar com categorias desconhecidas
        ]
    )
    return preprocessor






def create_model_pipeline(model, X_train, y_train):

    # Preprocessamento
    preprocessor = create_preprocessor(X_train)

    # Criando o pipeline com o modelo passado
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Ajustando o pipeline
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model_cv(pipeline, X, y, cv=10, score='recall'):
    """
    Realiza a validação cruzada para um modelo dado.

    Parameters:
    - pipeline: O pipeline do modelo a ser avaliado.
    - X: Dados de entrada.
    - y: Rótulos/target.
    - cv: Número de dobras para a validação cruzada (padrão é 10).

    Returns:
    - None (imprime os resultados da validação cruzada).
    """
    # Realizando a validação cruzada
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)  # Seed fixa na divisão dos dados
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring=score)

    # Obter as probabilidades previstas utilizando validação cruzada
    y_pred_prob = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]

    # Calculando a curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # best_threshold_tpr, best_tpr, best_threshold_accuracy, best_accuracy = find_best_threshold_max_tpr_accuracy(y, y_pred_prob)

    # Encontrando o índice do maior TPR
    # best_index_tpr = np.argmax(tpr)

    # Obtendo o melhor threshold e o TPR correspondente
    # best_threshold_tpr = thresholds[best_index_tpr]
    # best_tpr = tpr[best_index_tpr]

    # Calcular a distância para (0, 1)
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)

    # Encontrar o índice do limiar ideal
    index = np.argmin(distances)

    # Obter o limiar ideal
    optimal_threshold = thresholds[index]
    optimal_fpr = fpr[index]
    optimal_tpr = tpr[index]



    return scores, optimal_threshold

# Exemplo de uso:
# evaluate_model_cv(pipeline_lr_class_weight_l2, X, y, cv=10)

def evaluate_and_plot_roc_curve_cv(pipeline, X, y, cv=10):
    """
    Plota a curva ROC para um modelo utilizando validação cruzada.

    Parameters:
    - pipeline: O pipeline do modelo a ser avaliado.
    - X: Dados de entrada.
    - y: Rótulos/target.
    - cv: Número de dobras para a validação cruzada (padrão é 10).

    Returns:
    - None (exibe o gráfico da curva ROC).
    """

    """
    roc_curve(y, y_pred_prob): Esta função da biblioteca sklearn calcula as taxas de falso positivo (FPR - False Positive Rate) e verdadeiro positivo (TPR - True Positive Rate) para diferentes limiares de decisão, a partir dos rótulos verdadeiros (y) e das probabilidades preditas (y_pred_prob).
    Saídas:
    fpr: Um array contendo as taxas de falso positivo para cada limiar. A taxa de falso positivo é a fração de amostras negativas que foram classificadas incorretamente como positivas.
    tpr: Um array contendo as taxas de verdadeiro positivo para cada limiar. A taxa de verdadeiro positivo é a fração de amostras positivas que foram corretamente classificadas.
    _: Os limiares usados para calcular as taxas de FPR e TPR. Neste caso, não é necessário armazenar esses valores, então o símbolo _ é utilizado para ignorá-los.
    """
    # Obter as probabilidades previstas utilizando validação cruzada
    y_pred_prob = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]

    # Calculando a curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    best_threshold_tpr, best_tpr, best_threshold_accuracy, best_accuracy = find_best_threshold_max_tpr_accuracy(y, y_pred_prob)

    # Encontrando o índice do maior TPR
    # best_index_tpr = np.argmax(tpr)

    # Obtendo o melhor threshold e o TPR correspondente
    # best_threshold_tpr = thresholds[best_index_tpr]
    # best_tpr = tpr[best_index_tpr]




    # Plotando a curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    # Calcular a distância para (0, 1)
    distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)

    # Encontrar o índice do limiar ideal
    index = np.argmin(distances)

    # Obter o limiar ideal
    optimal_threshold = thresholds[index]
    optimal_fpr = fpr[index]
    optimal_tpr = tpr[index]

    # Imprimindo o limiar ideal
    # print(f"Ideal limit: {optimal_threshold:.2f}")
    # print(f"True Positive Rate (TPR): {optimal_tpr:.2f}")
    # print(f"False Positive Rate (FPR): {optimal_fpr:.2f}")

    return y_pred_prob, optimal_threshold, best_threshold_tpr, best_tpr, best_threshold_accuracy, best_accuracy



def evaluate_model(pipeline, X_test, y_test, optimal_threshold=0.5, to_print=False):
    """
    Avalia um modelo com dados de teste fornecidos.

    Parameters:
    - pipeline: O pipeline do modelo a ser avaliado.
    - X_test: Dados de entrada do teste.
    - y_test: Rótulos/target do teste.
    - optimal_threshold: Limite para decidir a classificação (opcional).

    Returns:
    - None (imprime os resultados da avaliação do modelo).
    """
    # Fazendo previsões com o modelo
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]  # Probabilidades para a classe positiva

    # Usando o threshold ideal se fornecido, caso contrário, usa 0.5
    if optimal_threshold is not None:
        y_pred = (y_pred_prob >= optimal_threshold).astype(int)
    else:
        y_pred = (y_pred_prob >= 0.5).astype(int)  # Threshold padrão de 0.5

    if to_print:
      # Imprimindo o relatório de classificação
      print(classification_report(y_test, y_pred))

    # Calculando e imprimindo AUC
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    if to_print:
      print(f'AUC: {auc:.2f}')
      plot_confusion_matrix(y_test, y_pred)
    
    return auc, f1, recall, precision, accuracy, y_test, y_pred


# Exemplo de uso:
# evaluate_model(pipeline_lr_class_weight_l2, X_test, y_test, optimal_threshold=0.6)


def plot_confusion_matrix(y_teste,y_pred):

  # Exemplo de y_teste e y_pred (valores reais e previstos)
  # y_teste: são os valores reais
  # y_pred: são os valores previstos pelo seu modelo

  # Gerando a matriz de confusão
  matriz_confusao = confusion_matrix(y_teste, y_pred)

  # Exibindo a matriz de confusão
  print(matriz_confusao)

  # Para uma visualização mais clara, podemos usar heatmap com seaborn
  sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Previsto')
  plt.ylabel('Real')
  plt.title('Matriz de Confusão')
  plt.show()

  import numpy as np

def find_best_threshold_max_tpr_accuracy(y_true, y_pred_prob):
    """
    Encontra o melhor threshold baseado no maior TPR (Taxa de Verdadeiros Positivos).

    Parameters:
    - y_true: Array contendo os rótulos verdadeiros (0 ou 1).
    - y_pred_prob: Array contendo as probabilidades previstas do modelo para a classe positiva.

    Returns:
    - best_threshold: O melhor threshold correspondente ao maior TPR.
    - best_tpr: O TPR correspondente ao melhor threshold.
    """
    # Calculando a curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # Calculando a acurácia para cada threshold
    accuracies = [accuracy_score(y_true, (y_pred_prob >= t).astype(int)) for t in thresholds]

    # Encontrando o índice do melhor threshold baseado na maior acurácia
    best_index = np.argmax(accuracies)

    # Obtendo o melhor threshold e sua acurácia correspondente
    best_threshold_accuracy = thresholds[best_index]
    best_accuracy = accuracies[best_index]


    # Encontrando o índice do maior TPR
    best_index = np.argmax(tpr)

    # Obtendo o melhor threshold e o TPR correspondente
    best_threshold_tpr = thresholds[best_index]
    best_tpr = tpr[best_index]

    return best_threshold_tpr, best_tpr, best_threshold_accuracy, best_accuracy

# Exemplo de uso:
# best_threshold, best_tpr = find_best_threshold(y, y_pred_prob)
# print(f"Melhor Threshold: {best_threshold:.2f}")
# print(f"TPR correspondente: {best_tpr:.2f}")

def get_best_hiper_params(features_train, target_train, pipeline, param_grid, cv= 10, scoring='roc_auc_ovo'):
  
  # Criando o GridSearchCV
  grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, error_score='raise')

  # Treinando o modelo
  grid_search.fit(features_train, target_train)

  # Melhor modelo
  print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
  
  # Melhor resultado
  print("Melhor score:", grid_search.best_score_)

  return grid_search.best_params_, grid_search.best_score_

def get_pipeline_created_and_evaluated(model, X, y):
  # Criando um pipeline relacionado ao modelo acima
  pipeline = create_model_pipeline(model,X,y)

  # avaliando o modelo criado com validação cruzada

  scores_roc_auc_ovo, optimal_threshold = evaluate_model_cv(pipeline,X,y,cv=10,score='roc_auc_ovo')
  scores_recall, optimal_threshold = evaluate_model_cv(pipeline,X,y,cv=10,score='recall')
  scores_accuracy, optimal_threshold = evaluate_model_cv(pipeline,X,y,cv=10,score='accuracy')

  return pipeline, scores_roc_auc_ovo, scores_recall, scores_accuracy, optimal_threshold


