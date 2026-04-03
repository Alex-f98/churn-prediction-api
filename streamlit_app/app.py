import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time


API_BASE_URL = "https://churn-prediction-api-production.up.railway.app"
API_URL      = f"{API_BASE_URL}/predict"
HEALTH_URL   = f"{API_BASE_URL}/health"

st.set_page_config(
    page_title="Churn Predictor", 
    layout="wide",
    page_icon="📊"
)

# --- Sidebar ---
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("### Estado del Sistema")

# Check API Health
try:
    health_response = requests.get(HEALTH_URL, timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("🟢 API Online")
    else:
        st.sidebar.error("🔴 API Error")
except:
    st.sidebar.error("🔴 API Offline")

# --- Main Content ---
st.title("📊 Customer Churn Prediction System")
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predicción", "Análisis", "Historial"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Datos del Cliente")
        
        # Sección de Información Básica
        with st.expander("👤 Información Básica", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                gender        = st.selectbox("Gender", ["Male", "Female"])
                SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            with col_b:
                Partner    = st.selectbox("Partner", ["Yes", "No"], format_func=lambda x: "Sí" if x == "Yes" else "No")
                Dependents = st.selectbox("Dependents", ["Yes", "No"], format_func=lambda x: "Sí" if x == "Yes" else "No")
        
        # Sección de Servicios
        with st.expander("📱 Servicios", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                tenure        = st.slider("Tenure (meses)", 0, 72, 12)
                PhoneService  = st.selectbox("Phone Service", ["Yes", "No"])
                MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            with col_b:
                InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                Contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        # Sección de Servicios Adicionales
        with st.expander("🔧 Servicios Adicionales"):
            col_a, col_b = st.columns(2)
            with col_a:
                OnlineSecurity   = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                OnlineBackup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            with col_b:
                TechSupport      = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                StreamingTV      = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                StreamingMovies  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        # Sección de Pagos
        with st.expander("💳 Información de Pago", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                PaymentMethod = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check", 
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )
            with col_b:
                MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.01)
                TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0, step=0.01)
    
    with col2:
        st.markdown("### Predicción")
        
        # Botón de predicción
        if st.button("🔮 Predecir Churn", type="primary", use_container_width=True):
            with st.spinner("Analizando datos..."):
                payload = {
                    "gender":           gender,
                    "SeniorCitizen":    SeniorCitizen,
                    "Partner":          Partner,
                    "Dependents":       Dependents,
                    "tenure":           tenure,
                    "PhoneService":     PhoneService,
                    "MultipleLines":    MultipleLines,
                    "InternetService":  InternetService,
                    "OnlineSecurity":   OnlineSecurity,
                    "OnlineBackup":     OnlineBackup,
                    "DeviceProtection": DeviceProtection,
                    "TechSupport":      TechSupport,
                    "StreamingTV":      StreamingTV,
                    "StreamingMovies":  StreamingMovies,
                    "Contract":         Contract,
                    "PaperlessBilling": PaperlessBilling,
                    "PaymentMethod":    PaymentMethod,
                    "MonthlyCharges":   MonthlyCharges,
                    "TotalCharges":     TotalCharges
                }
                
                try:
                    start_time = time.time()
                    response = requests.post(API_URL, json=payload, timeout=10)
                    latency = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        proba = result["churn_probability"]
                        
                        # Guardar en historial
                        prediction_record = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "probability": f"{proba:.2%}",
                            "risk_level": "Alto" if proba > 0.7 else "Medio" if proba > 0.3 else "Bajo",
                            "contract": Contract,
                            "monthly_charges": f"${MonthlyCharges:.2f}",
                            "tenure": f"{tenure} meses"
                        }
                        st.session_state.prediction_history.append(prediction_record)
                        
                        # Visualización del resultado
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = proba * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probabilidad de Churn"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Resultado textual
                        if proba > 0.7:
                            st.error(f"🚨 **ALTO RIESGO**: {proba:.1%} de probabilidad de churn")
                            st.warning("Recomendación: Contactar al cliente urgentemente")
                        elif proba > 0.5:
                            st.warning(f"⚠️ **RIESGO MODERADO**: {proba:.1%} de probabilidad de churn")
                            st.info("Recomendación: Ofrecer incentivos o descuentos")
                        else:
                            st.success(f"✅ **BAJO RIESGO**: {proba:.1%} de probabilidad de churn")
                            st.info("Recomendación: Mantener servicio actual")
                        
                        # Métricas
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Model Version", result['model_version'])
                        with col_b:
                            st.metric("Latency", f"{latency:.3f}s")
                            
                    else:
                        st.error(f"Error en API: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏰ Timeout: La API tardó demasiado en responder")
                except Exception as e:
                    st.error(f"❌ Error de conexión: {e}")
        
        # Información adicional
        st.markdown("---")
        st.markdown("### 📊 Información del Cliente")
        
        # Resumen de datos
        info_data = {
            "Antigüedad":       f"{tenure} meses",
            "Cargo Mensual":    f"${MonthlyCharges:.2f}",
            "Cargo Total":      f"${TotalCharges:.2f}",
            "Servicio Internet": InternetService,
            "Tipo Contrato":    Contract
        }
        
        for key, value in info_data.items():
            st.write(f"**{key}**: {value}")

with tab2:
    st.markdown("### 📈 Análisis de Factores de Riesgo")
    
    # Factores de riesgo basados en el input actual
    risk_factors = []
    
    if tenure < 12:
        risk_factors.append("🔴 Cliente nuevo (< 1 año)")
    if Contract == "Month-to-month":
        risk_factors.append("🔴 Contrato mes a mes")
    if InternetService == "Fiber optic":
        risk_factors.append("🟡 Internet fibra óptica")
    if PaymentMethod == "Electronic check":
        risk_factors.append("🟡 Pago con cheque electrónico")
    if TechSupport == "No":
        risk_factors.append("🟡 Sin soporte técnico")
    
    if risk_factors:
        st.warning("### Factores de Riesgo Identificados:")
        for factor in risk_factors:
            st.write(factor)
    else:
        st.success("✅ No se identificaron factores de riesgo significativos")

with tab3:
    st.markdown("### 📋 Historial de Predicciones")
    
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.session_state.prediction_history:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_history, use_container_width=True)
    else:
        st.info("No hay predicciones previas. Realiza una predicción para ver el historial.")

# Footer
st.markdown("---")
st.markdown("Desarrollado con Streamlit + FastAPI + scikit-learn")