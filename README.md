# Customer Churn Prediction System

Sistema de predicción de churn de clientes construido con FastAPI, Streamlit, scikit-learn y MLflow para tracking de experimentos.

## Features

- **API REST** con FastAPI para predicciones en tiempo real
- **Dashboard interactivo** con Streamlit para análisis visual
- **Pipeline completo** de Machine Learning con preprocesamiento
- **MLflow Tracking** para experimentos y registro de modelos
- **Monitoreo de salud** y logging estructurado
- **Análisis de factores de riesgo** automáticos
- **Historial de predicciones** persistente

## Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA COMPLETO                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   STREAMLIT │    │   FASTAPI   │    │   MLFLOW    │         │
│  │   DASHBOARD │◄──►│    API      │◄──►│  TRACKING   │         │
│  │   (UI:8501) │    │ (API:8000)  │    │ (UI:5000)   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                   │                   │             │
│         │                   │                   │             │
│         ▼                   ▼                   ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 DOCKER COMPOSE                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   UI CONT.  │  │   API CONT. │  │  MLFLOW C.  │     │   │
│  │  │  streamlit  │  │  fastapi    │  │   mlflow    │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    CAPA DE SERVICIOS                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ PREDICTOR   │  │   LOGGER    │  │   SCHEMAS   │     │   │
│  │  │   SERVICE   │  │  SERVICE    │  │   PYDANTIC  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    CAPA DE DATOS                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   MODELOS   │  │    DATA     │  │   LOGS      │     │   │
│  │  │ (.pkl/.mlflow)│  │  (CSV)     │  │  (.log)     │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  PIPELINE ML                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   TRAIN.PY  │  │  PREPROCES  │  │  EVALUACIÓN │     │   │
│  │  │ (ENTRENAR)  │  │   AMIENTO   │  │   MÉTRICAS  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │         │                   │                   │         │   │
│  │         └───────────────────┼───────────────────┘         │   │
│  │                             ▼                             │   │
│  │                  ┌─────────────┐                         │   │
│  │                  │    MLFLOW   │                         │   │
│  │                  │   TRACKING  │                         │   │
│  │                  └─────────────┘                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Flujo de Datos

```
📊 DATASET → 🔄 PREPROCESAMIENTO → 🤖 MODELO ML → 📈 MLFLOW TRACKING
    │               │                   │               │
    │               │                   │               ▼
    │               │                   │         ┌─────────────┐
    │               │                   │         │ EXPERIMENTOS│
    │               │                   │         │   MÉTRICAS  │
    │               │                   │         │   MODELOS  │
    │               │                   │         └─────────────┘
    │               │                   │               │
    ▼               ▼                   ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   CLIENTE   │ │   DATOS     │ │ PREDICCIÓN │ │   LOGGING   │
│   REQUEST   │ │   PREP.     │ │   PROBA    │ │   AUDITORÍA │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
         │               │               │               │
         └───────────────┼───────────────┼───────────────┘
                         ▼               ▼
                 ┌─────────────────────────────┐
                 │        API RESPONSE         │
                 │   probability + version     │
                 └─────────────────────────────┘
```

## Arquitectura

```
proyect/
├── app/                    # API FastAPI
│   ├── main.py            # Entry point
│   ├── routes.py          # Endpoints (/predict, /health, /stats)
│   ├── schemas.py         # Pydantic models
│   └── logger.py          # Logging config
├── service/               # Lógica de negocio
│   └── predictor.py       # ML model inference (MLflow/local)
├── src/                   # Entrenamiento
│   ├── train.py           # Training pipeline con MLflow
│   └── preprocess.py      # Data preprocessing
├── streamlit_app/         # Frontend
│   └── app.py             # Streamlit dashboard
├── models/                # Modelos entrenados locales
├── data/                  # Dataset
├── Dockerfile.*           # Docker containers
├── docker-compose.yml     # Orquestación de servicios
└── tests/                 # Tests
```

## Instalación y Ejecución

### 1. Con Docker (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/Alex-f98/churn-prediction-api.git
cd churn-prediction-api

# Iniciar todos los servicios
sudo docker-compose up --build

# Acceder a los servicios:
# API: http://localhost:8000/docs
# UI: http://localhost:8501
# MLflow: http://localhost:5000
```

### 2. Entrenamiento con MLflow

```bash
# Entrar al contenedor API
sudo docker exec -it proyect-api-1 bash

# Entrenar modelo (se registra en MLflow)
python src/train.py

# Ver experimentos en: http://localhost:5000
```

### 3. Desarrollo Local

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar MLflow (terminal separado)
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Iniciar API
uvicorn app.main:app --reload

# Iniciar Dashboard
streamlit run streamlit_app/app.py
```

## MLflow Integration

### Experimentos Automáticos

El pipeline de entrenamiento registra automáticamente:

- **Parámetros**: Hiperparámetros del modelo
- **Métricas**: ROC-AUC, Accuracy, F1-Score
- **Modelos**: Modelo sklearn registrado
- **Artifacts**: Modelo local y metadata

### Registro de Modelos

```python
# Modelo registrado como: "churn_prediction_model"
# Versiones automáticas: v1, v2, v3...

# Cargar modelo desde MLflow:
mlflow.sklearn.load_model("models:/churn_prediction_model/latest")
```

### UI MLflow

- **Experiments**: http://localhost:5000/#/experiments
- **Models**: http://localhost:5000/#/models
- **Runs**: Historial completo de entrenamientos

## Modelo

- **Algoritmo**: Gradient Boosting Classifier
- **Métricas**: ROC-AUC: 0.84, F1-Score: 0.56
- **Features**: 19 variables del dataset Telco Customer Churn
- **Preprocesamiento**: StandardScaler + OneHotEncoder
- **Tracking**: MLflow con SQLite backend

## API Endpoints

### Predicción
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

### Respuesta
```json
{
  "churn_probability": 0.4234,
  "model_version": "v1"
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Estadísticas
```bash
curl http://localhost:8000/stats
```

## Métricas de Riesgo

- **Bajo riesgo**: < 30% probabilidad
- **Riesgo moderado**: 30-70% probabilidad  
- **Alto riesgo**: > 70% probabilidad

## Desarrollo

### Entrenar nuevo modelo
```bash
# Docker
sudo docker exec -it proyect-api-1 bash
python src/train.py

# Local
python src/train.py
```

### Tests
```bash
pytest tests/
```

### Logs
```bash
# Ver logs de todos los servicios
sudo docker-compose logs -f

# Logs específicos
sudo docker-compose logs -f api
sudo docker-compose logs -f mlflow
```

## Dataset

Usa el dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) de Kaggle.

## Configuración

### Variables de Entorno
```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000  # Docker
MLFLOW_TRACKING_URI=http://localhost:5000  # Local

# API
API_BASE_URL=http://api:8000  # Docker
API_BASE_URL=http://localhost:8000  # Local
```

## Contribuir

1. Fork el proyecto
2. Crear feature branch (`git checkout -b feature/amazing-feature`)
3. Commit cambios (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing-feature`)
5. Abrir Pull Request

## Licencia

MIT License - ver archivo [LICENSE](LICENSE) para detalles.

## Acknowledgments

- Dataset proveído por [IBM Watson Analytics](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Construido con [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/) y [MLflow](https://mlflow.org/)