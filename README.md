# Customer Churn Prediction System

Sistema de predicción de churn de clientes construido con FastAPI, Streamlit y scikit-learn.

## Features

- **API REST** con FastAPI para predicciones en tiempo real
- **Dashboard interactivo** con Streamlit para análisis visual
- **Pipeline completo** de Machine Learning con preprocesamiento
- **Monitoreo de salud** y logging estructurado
- **Análisis de factores de riesgo** automáticos
- **Historial de predicciones** persistente

## Arquitectura

```
proyect/
├── app/                    # API FastAPI
│   ├── main.py            # Entry point
│   ├── routes.py          # Endpoints
│   ├── schemas.py         # Pydantic models
│   └── logger.py          # Logging config
├── service/               # Lógica de negocio
│   └── predictor.py       # ML model inference
├── src/                   # Entrenamiento
│   ├── train.py           # Training pipeline
│   └── preprocess.py      # Data preprocessing
├── streamlit_app/         # Frontend
│   └── app.py             # Streamlit dashboard
├── models/                # Modelos entrenados
├── data/                  # Dataset
└── tests/                 # Tests
```

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USERNAME/churn-prediction-api.git
cd churn-prediction-api
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## Uso

### API FastAPI

```bash
# Iniciar API
uvicorn app.main:app --reload

# Documentación: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

### Dashboard Streamlit

```bash
# Iniciar dashboard
streamlit run streamlit_app/app.py

# Acceder: http://localhost:8501
```

## Modelo

- **Algoritmo**: Gradient Boosting Classifier
- **Métricas**: ROC-AUC: 0.84, F1-Score: 0.56
- **Features**: 19 variables del dataset Telco Customer Churn
- **Preprocesamiento**: StandardScaler + OneHotEncoder

## Predicción

### Via API
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

## Métricas de Riesgo

- **Bajo riesgo**: < 30% probabilidad
- **Riesgo moderado**: 30-70% probabilidad  
- **Alto riesgo**: > 70% probabilidad

## Desarrollo

### Entrenar nuevo modelo
```bash
python src/train.py
```

### Tests
```bash
pytest tests/
```

## Dataset

Usa el dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) de Kaggle.

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
- Construido con [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/) y [scikit-learn](https://scikit-learn.org/)