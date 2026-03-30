import joblib
import pandas as pd

import time
from app.logger import logger

#class Predictor:
#    def __init__(self, model_path="models/model_v1.pkl"):
#        artifact     = joblib.load(model_path) #como picke pero mejor,  preserva n_jobs (esto es todo el pipeline)
#        self.model   = artifact["model"]
#        self.version = artifact["version"]
#
#    def predict(self, data: dict):
#        df          = pd.DataFrame([data])
#        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
#        proba       = self.model.predict_proba(df)[0, 1]
#
#        return {
#            "churn_probability": float(proba), 
#            "model_version": self.version
#        }

import joblib
import pandas as pd
import traceback

class Predictor:
    def __init__(self, model_path="models/model_v1.pkl"):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.version = artifact["version"]
        
        # Debug: ver las categorías que el modelo espera
        logger.info("=== Modelo cargado ===")
        preprocessor = self.model.named_steps['preprocessor']
        cat_transformer = preprocessor.named_transformers_['cat']
        logger.info("Categorías esperadas:")
        for i, cats in enumerate(cat_transformer.categories_):
            logger.info(f"  Columna {i}: {cats[:5]}...")  # Primeras 5 categorías

    def predict(self, data: dict):
        try:
            start = time.time()
            df = pd.DataFrame([data])
            logger.info(f"\n=== Datos recibidos ===")
            logger.info(f"Columnas: {df.columns.tolist()}")
            logger.info(f"Tipos:\n{df.dtypes}")
            logger.info(f"Valores:\n{df.iloc[0]}")
            
            # Preprocesamiento
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            
            logger.info(f"\n=== Después de preprocesamiento ===")
            logger.info(f"TotalCharges: {df['TotalCharges'].iloc[0]} (tipo: {df['TotalCharges'].dtype})")
            
            proba = self.model.predict_proba(df)[0, 1]

            latency = time.time() - start
            logger.info(f"Prediction: {proba:.4f} | latency: {latency:.4f}s")
            
            return {
                "churn_probability": float(proba),
                "model_version": self.version
            }
        except Exception as e:
            logger.error(f"\n=== ERROR ===")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            raise e