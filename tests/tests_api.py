import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import CustomerInput

#python3 -m pytest tests/tests_api.py -v

client = TestClient(app)

class TestHealthEndpoint:
    def test_health_check(self):
        """Test that health endpoint returns status ok"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

class TestMetadataEndpoint:
    def test_metadata_structure(self):
        """Test that metadata endpoint returns correct structure"""
        response = client.get("/metadata")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_version" in data
        assert "features" in data
        assert "target" in data
        assert "description" in data
        assert "endpoints" in data
        
        assert data["target"] == "churn"
        assert "predict" in data["endpoints"]
        assert "health" in data["endpoints"]
        assert "metadata" in data["endpoints"]
        assert "stats" in data["endpoints"]

class TestStatsEndpoint:
    def test_stats_initial(self):
        """Test that stats endpoint returns initial prediction count"""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert isinstance(data["total_predictions"], int)
        assert data["total_predictions"] >= 0

class TestPredictEndpoint:
    def test_predict_valid_data(self):
        """Test prediction with valid customer data"""
        valid_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_probability" in data
        assert isinstance(data["churn_probability"], (int, float))
        assert 0 <= data["churn_probability"] <= 1

    def test_predict_invalid_gender(self):
        """Test prediction with invalid gender value"""
        invalid_data = {
            "gender": "Invalid",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_missing_required_field(self):
        """Test prediction with missing required field"""
        incomplete_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_predict_invalid_charges(self):
        """Test prediction with invalid charge values"""
        invalid_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": -10.0,  # Invalid: negative
            "TotalCharges": 29.85
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_invalid_senior_citizen(self):
        """Test prediction with invalid SeniorCitizen value"""
        invalid_data = {
            "gender": "Male",
            "SeniorCitizen": 2,  # Invalid: should be 0 or 1
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

class TestIntegration:
    def test_prediction_increments_stats(self):
        """Test that making a prediction increments the stats counter"""
        # Get initial stats
        initial_response = client.get("/stats")
        initial_count = initial_response.json()["total_predictions"]
        
        # Make a prediction
        valid_data = {
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 89.99,
            "TotalCharges": 1079.88
        }
        
        client.post("/predict", json=valid_data)
        
        # Check stats incremented
        final_response = client.get("/stats")
        final_count = final_response.json()["total_predictions"]
        
        assert final_count == initial_count + 1

if __name__ == "__main__":
    pytest.main([__file__])