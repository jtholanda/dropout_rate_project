from scipy.sparse import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier




random_state = 0

def get_base_model_logistic_regression():

  # base model
  label_pipeline = 'logreg'
  base_model = LogisticRegression(class_weight='balanced', random_state=random_state, fit_intercept=True)
  return label_pipeline, base_model

def get_params_logistic_regression():
  # Definindo os parâmetros da Logistic Regression para o GridSearch com penalty l1
  param_grid_01 = {
      'logreg__C': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10],
      'logreg__penalty': ['l1'],
      'logreg__solver': ['saga',],
      'logreg__max_iter': [100, 200, 300]
  }
  # Definindo os parâmetros para o GridSearch com penalty l2
  param_grid_02 = {
      'logreg__C': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10],
      'logreg__penalty': ['l2'],
      'logreg__solver': ['saga', 'lbfgs', 'sag', 'liblinear'],
      'logreg__max_iter': [100, 200, 300]
  }
  # Definindo os parâmetros para o GridSearch com penalty elasticnet
  param_grid_03 = {
      'logreg__C': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10],
      'logreg__penalty': ['elasticnet'],
      'logreg__solver': ['saga'],
      'logreg__max_iter': [100, 200, 300],
      'logreg__l1_ratio': [0.1, 0.5, 0.7, 1]

  }

  return [param_grid_01,param_grid_02,param_grid_03]

def get_base_model_decision_tree():

  # base model
  label_pipeline = 'decision_tree'
  base_model = DecisionTreeClassifier(class_weight='balanced', random_state=random_state)
  return label_pipeline, base_model

def get_params_decision_tree():

  param_grid_01 = {
      'decision_tree__criterion': ['gini', 'entropy'],  # Critério de divisão
      'decision_tree__max_depth': [3, 5, 10, 15 , 20],    # Profundidade da árvore
      'decision_tree__min_samples_split': [2, 3, 5, 8, 10],  # Mínimo de amostras para dividir um nó
      'decision_tree__min_samples_leaf': [1, 2, 3, 4, 5],    # Mínimo de amostras por folha
      'decision_tree__max_features': ['sqrt', 'log2', None]  # Número máximo de features consideradas
  }
  

  return [param_grid_01]

def get_base_model_random_forest():
  label_pipeline = 'random_forest'
  base_model = RandomForestClassifier(n_estimators=100, 
                                        class_weight='balanced', 
                                        random_state=random_state)
  return label_pipeline, base_model


def get_params_random_forest():
  param_grid_rf = {
    'random_forest__n_estimators': [100, 300],  # Número de árvores na floresta
    'random_forest__criterion': ['gini', 'entropy'],  # Critério de divisão
    'random_forest__max_depth': [3, 10, 20],  # Profundidade máxima das árvores
    'random_forest__min_samples_split': [2,5,10],  # Mínimo de amostras para dividir um nó
    'random_forest__min_samples_leaf': [1, 3, 5],  # Mínimo de amostras por folha
    'random_forest__max_features': ['sqrt', 'log2', None],  # Número máximo de features consideradas
    'random_forest__bootstrap': [True]  # Se as amostras são sorteadas com reposição
  }
  return [param_grid_rf]

from sklearn.svm import SVC

def get_base_model_svm():
    label_pipeline = 'svm'
    base_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=random_state)
    return label_pipeline, base_model

def get_params_svm():
    param_grid_svm = {
        'svm__C': [0.1, 0.5, 1],  # Parâmetro de regularização
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Tipo de kernel
        'svm__gamma': ['scale', 'auto'],  # Coeficiente do kernel para ‘rbf’, ‘poly’ e ‘sigmoid’
        'svm__degree': [2, 3, 4],  # Grau do polinômio (usado apenas no kernel 'poly')
    }
    return [param_grid_svm]




def get_scale_pos_weight(reference_year):
    """
    Calcula o valor de scale_pos_weight com base no ano de referência (reference_year).
    
    Parâmetros:
    reference_year (int): Ano de referência (2018 a 2023).
    
    Retorna:
    float: O valor adequado de scale_pos_weight.
    """
    # Dicionário com as fórmulas para cada ano
    scale_pos_weight_dict = {
        2018: 289 / 208,
        2019: (289 + 318) / (208 + 152),
        2020: (289 + 318 + 426) / (208 + 152 + 157),
        2021: (289 + 318 + 426 + 995) / (208 + 152 + 157 + 97),
        2022: (289 + 318 + 426 + 995 + 775) / (208 + 152 + 157 + 97 + 581),
        2023: (289 + 318 + 426 + 995 + 775 + 536) / (208 + 152 + 157 + 97 + 581 + 371)
    }
    
    # Retorna o valor do scale_pos_weight para o ano fornecido
    if reference_year in scale_pos_weight_dict:
        return scale_pos_weight_dict[reference_year]
    else:
        raise ValueError(f"Ano {reference_year} não está entre os anos válidos (2018-2023).")
    
def get_base_model_xgboost(reference_year):
    label_pipeline = 'xgboost'
    base_model = XGBClassifier(eval_metric='logloss', random_state=random_state, scale_pos_weight=get_scale_pos_weight(reference_year))
    return label_pipeline, base_model

def get_params_xgboost():
    param_grid_xgboost = {
        'xgboost__n_estimators': [100,250],  # Número de árvores
        'xgboost__max_depth': [3, 5, 7],  # Profundidade máxima das árvores
        'xgboost__learning_rate': [0.01, 0.1, 0.2],  # Taxa de aprendizado
        #'xgboost__subsample': [0.9],  # Proporção de amostras usadas para cada árvore
        #'xgboost__colsample_bytree': [0.9],  # Fração de características usadas em cada árvore
        'xgboost__gamma': [0, 0.1, 0.2],  # Regularização para divisão de nós
        'xgboost__reg_alpha': [0, 0.1, 1],  # Regularização L1
        'xgboost__reg_lambda': [1, 1.5, 2]  # Regularização L2

    }
    return [param_grid_xgboost]


def get_base_model_knn():
    label_pipeline = 'knn'
    base_model = KNeighborsClassifier()
    return label_pipeline, base_model

def get_params_knn():
    param_grid_knn = {
        'knn__n_neighbors': [3, 5, 7, 9],  # Número de vizinhos
        'knn__weights': ['uniform', 'distance'],  # Tipo de ponderação
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],  # Métrica de distância
        'knn__p': [1, 2],  # Parâmetro da métrica Minkowski (p=1 é Manhattan, p=2 é Euclidiana)
    }
    return [param_grid_knn]


