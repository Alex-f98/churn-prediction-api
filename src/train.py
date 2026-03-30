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



MODEL_PATH = "proyect/models/model_v1.pkl"

def load_data(path):
    df = pd.read_csv(path)
    
    # Limpiar TotalCharges - convertir espacios a NaN y luego a 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    df = df.dropna()
    return df

def build_pipeline(num_cols, cat_cols):
    best_paramns = {'subsample': 0.9, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(**best_paramns))
    ])

    return model

def main():
    # df = load_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = load_data("proyect/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=777, stratify=y
    )

    model = build_pipeline(num_cols, cat_cols)

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, model.predict(X_test))

    print(f"ROC-AUC: {roc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Guardar con metadata
    os.makedirs("models", exist_ok=True)

    artifact = {
        "model": model,
        "version": "v1",
        "metrics": {
            "roc_auc": roc,
            "accuracy": accuracy,
            "f1": f1
        },
        "timestamp": datetime.now().isoformat()
    }

    joblib.dump(artifact, MODEL_PATH) #como picke pero mejor,  preserva n_jobs

    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()