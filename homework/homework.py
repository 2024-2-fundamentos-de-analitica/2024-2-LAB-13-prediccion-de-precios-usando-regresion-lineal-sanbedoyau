#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import json
import os
import pickle
import gzip
from glob import glob
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_data(path: str) -> pd.DataFrame:
    DataFrame = pd.read_csv(path, compression = 'zip')
    return DataFrame

def clean_data(DataFrame: pd.DataFrame) -> pd.DataFrame:
    DataFrame['Age'] = 2021 - DataFrame['Year']
    DataFrame.drop(columns = ['Car_Name', 'Year'], inplace = True)
    return DataFrame

def features_target_split(DataFrame: pd.DataFrame) -> tuple:
    return DataFrame.drop(columns = 'Present_Price'), DataFrame['Present_Price']

def make_pipeline(estimator: LinearRegression, cat_features: list) -> Pipeline:
    transformer = ColumnTransformer(
        transformers = [
            ('ohe', OneHotEncoder(dtype = 'int'), cat_features),
        ],
        remainder = 'passthrough'
    )

    selectkbest = SelectKBest(f_regression)

    scaler = MinMaxScaler()

    pipeline = Pipeline(
        steps=[
            ('transformer', transformer),
            ('selectkbest', selectkbest),
            ('scaler', scaler),
            ('regressor', estimator)
        ]
    )
    
    return pipeline

def make_grid_search(estimator: Pipeline, param_grid: dict, cv = 10):
    grid_search = GridSearchCV(
        estimator = estimator,
        param_grid = param_grid,
        cv = cv,
        scoring = 'neg_mean_absolute_error'
    )
    return grid_search

def save_estimator(path: str, estimator: GridSearchCV) -> None:
    with gzip.open(path, 'wb') as file:
        pickle.dump(estimator, file)

def eval_model(estimator: GridSearchCV, features: pd.DataFrame, target: pd.Series, name: str) -> dict:
    y_pred = estimator.predict(features)
    metrics = {
        'type': 'metrics',
        'dataset': name,
        'r2': r2_score(target, y_pred),
        'mse': mean_squared_error(target, y_pred),
        'mad': mean_absolute_error(target, y_pred)
    }
    return metrics
    
def save_metrics(path: str, train_metrics: dict, test_metrics: dict) -> None:
    with open(path, 'w') as file:
        file.write(json.dumps(train_metrics) + '\n')
        file.write(json.dumps(test_metrics) + '\n')

def create_out_dir(out_dir: str) -> None:
        if os.path.exists(out_dir):
            for file in glob(f'{out_dir}/*'):
                os.remove(file)
            os.rmdir(out_dir)
        os.makedirs(out_dir)


def run():
    in_path = 'files/input'
    out_path = 'files/output'
    mod_path = 'files/models'

    train = clean_data(load_data(f'{in_path}/train_data.csv.zip'))
    test = clean_data(load_data(f'{in_path}/test_data.csv.zip'))

    X_train, y_train = features_target_split(train)
    X_test, y_test = features_target_split(test)

    cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
    estimator = make_pipeline(LinearRegression(), cat_features)

    param_grid = {
        'selectkbest__k': range(1, len(X_train.columns) + 1)
    }
    estimator = make_grid_search(
        estimator = estimator,
        param_grid = param_grid
    )
    estimator.fit(X_train, y_train)

    create_out_dir(out_path)
    create_out_dir(mod_path)

    save_estimator(f'{mod_path}/model.pkl.gz', estimator)

    train_metrics = eval_model(estimator, X_train, y_train, 'train')
    test_metrics = eval_model(estimator, X_test, y_test, 'test')
    save_metrics(f'{out_path}/metrics.json', train_metrics, test_metrics)

if __name__ == '__main__':
    run()