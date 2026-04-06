from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib
import os
import pandas as pd
from datetime import datetime


from sklearn.metrics import  accuracy_score, f1_score
from sklearn.ensemble import  GradientBoostingClassifier

import mlflow
import mlflow.sklearn

# Configurar MLflow (opcional: solo si se quiere tracking)
MLFLOW_ENABLED = True

# Detectar si estamos corriendo en Docker
import os
IN_DOCKER = os.path.exists('/.dockerenv')

if MLFLOW_ENABLED:
    try:
        if IN_DOCKER:
            mlflow.set_tracking_uri("http://mlflow:5000")
            print("=============MLflow tracking URI configurado para Docker( http://mlflow:5000 )=============")
        else:
            mlflow.set_tracking_uri("http://localhost:5000")
            print("=============MLflow tracking URI configurado para local( http://localhost:5000 )=============")
        
        mlflow.set_experiment("churn-prediction")
        print("MLflow tracking habilitado")
    except Exception as e:
        print(f"MLflow no disponible: {e}")
        MLFLOW_ENABLED = False



MODEL_PATH = "models/model_v1.pkl"

def load_data(path):
    """
    Cargar y limpiar datos de clientes.
    
    Args:
        path (str): Ruta al archivo CSV
    
    Returns:
        pd.DataFrame: DataFrame limpio con datos procesados
    """
    df = pd.read_csv(path)
    
    # Limpiar TotalCharges - convertir espacios a NaN y luego a 0
    # esto deberia estar en preprocess.py pero es minimo.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    df = df.dropna()
    return df

def build_pipeline(num_cols, cat_cols, best_params=None):
    """
    Construir pipeline completo de ML con preprocesamiento y clasificador.
    
    Args:
        num_cols (list): Lista de nombres de columnas numéricas
        cat_cols (list): Lista de nombres de columnas categóricas
        best_params (dict, optional): Hiperparámetros del modelo. 
            Por defecto usa parámetros optimizados de GradientBoostingClassifier.
    
    Returns:
        sklearn.Pipeline: Pipeline completo de preprocesamiento + clasificación
    """
    if best_params is None:
        best_params = {
            'subsample': 0.9, 
            'n_estimators': 100, 
            'max_depth': 3, 
            'learning_rate': 0.1
        }
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(**best_params))
    ])

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluar rendimiento del modelo con múltiples métricas.
    
    Args:
        model: Pipeline de sklearn entrenado
        X_test (pd.DataFrame): Características de prueba
        y_test (pd.Series): Variable objetivo de prueba
    
    Returns:
        tuple: (diccionario de métricas, probabilidades predichas)
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return metrics, y_pred_proba

def save_model_artifact(model, metrics, model_path, version="v1"):
    """
    Guardar modelo con metadatos como artifact de joblib.
    
    Args:
        model: Pipeline de sklearn entrenado
        metrics (dict): Métricas de evaluación
        model_path (str): Ruta donde guardar el modelo
        version (str): Versión del modelo
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    artifact = {
        "model": model,
        "version": version,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "model_type": "GradientBoostingClassifier"
    }
    
    joblib.dump(artifact, model_path)
    print(f"Modelo guardado en {model_path}")

def log_to_mlflow(model, metrics, params, model_path, run_name=None):
    """
    Registrar modelo, métricas y parámetros en MLflow.
    
    Args:
        model: Pipeline de sklearn entrenado
        metrics (dict): Métricas de evaluación
        params (dict): Parámetros del modelo
        model_path (str): Ruta al archivo local del modelo
        run_name (str, optional): Nombre del run en MLflow
    """
    if not MLFLOW_ENABLED:
        return
    
    try:
        with mlflow.start_run(run_name=run_name):
            # Registrar parámetros
            mlflow.log_params(params)
            
            # Registrar métricas
            mlflow.log_metrics(metrics)
            
            # Registrar modelo
            mlflow.sklearn.log_model(
                sk_model = model, 
                artifact_path = "churn_model",
                registered_model_name="churn_prediction_model"
            )
            
            # Registrar modelo local como artifact
            mlflow.log_artifact(model_path, "local_model")
            
            print("Modelo y artifacts registrados en MLflow")
    except Exception as e:
        print(f"Error al registrar en MLflow: {e}")

def main():
    """
    Pipeline principal de entrenamiento con integración MLflow.
    """
    print("Iniciando entrenamiento del modelo de predicción de churn...")
    
    # Configuración del modelo
    config = {
        'test_size'   : 0.15,
        'random_state': 777,
        'model_params': {
            'subsample'    : 0.9, 
            'n_estimators' : 100, 
            'max_depth'    : 3, 
            'learning_rate': 0.1
        }
    }
    try:
        # Cargar y preparar datos
        df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        
        # Preparar características y variable objetivo
        y = df["Churn"].map({"Yes": 1, "No": 0})
        X = df.drop(columns=["Churn", "customerID"], errors="ignore")

        # Identificar tipos de columnas
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['test_size'], 
            random_state=config['random_state'], 
            stratify=y
        )
        
        print(f"División de datos: {len(X_train)} entrenamiento, {len(X_test)} prueba")
        print(f"Características: {len(num_cols)} numéricas, {len(cat_cols)} categóricas")

        # Construir y entrenar modelo
        model = build_pipeline(num_cols, cat_cols, config['model_params'])
        print("Entrenando modelo...")
        model.fit(X_train, y_train)
        print("Entrenamiento del modelo completado")

        # Evaluar modelo
        metrics, y_pred_proba = evaluate_model(model, X_test, y_test)
        
        print(f"\nRendimiento del Modelo:")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")

        # Guardar modelo localmente
        save_model_artifact(model, metrics, MODEL_PATH)

        # Registrar en MLflow
        mlflow_params = {
            **config,
            'model_type': 'GradientBoostingClassifier',
            'num_features': len(num_cols) + len(cat_cols),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        log_to_mlflow(
            model=model,
            metrics=metrics,
            params=mlflow_params,
            model_path=MODEL_PATH,
            run_name=f"churn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("\n¡Pipeline de entrenamiento completado exitosamente!")
        return model, metrics

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()