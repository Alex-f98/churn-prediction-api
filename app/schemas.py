from pydantic import BaseModel


#Esto es basicamente un contrato de datos
class CustomerInput(BaseModel):
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    tenure:           int
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    MonthlyCharges:   float
    TotalCharges:     float






#from pydantic import BaseModel, Field
#from typing import Literal
#
#class CustomerInput(BaseModel):
#    gender: Literal["Male", "Female"]
#    SeniorCitizen: int = Field(ge=0, le=1)
#
#    Partner: Literal["Yes", "No"]
#    Dependents: Literal["Yes", "No"]
#
#    tenure: int = Field(ge=0)
#
#    PhoneService: Literal["Yes", "No"]
#    MultipleLines: Literal["Yes", "No", "No phone service"]
#
#    InternetService: Literal["DSL", "Fiber optic", "No"]
#
#    OnlineSecurity: Literal["Yes", "No", "No internet service"]
#    OnlineBackup: Literal["Yes", "No", "No internet service"]
#    DeviceProtection: Literal["Yes", "No", "No internet service"]
#    TechSupport: Literal["Yes", "No", "No internet service"]
#
#    StreamingTV: Literal["Yes", "No", "No internet service"]
#    StreamingMovies: Literal["Yes", "No", "No internet service"]
#
#    Contract: Literal["Month-to-month", "One year", "Two year"]
#
#    PaperlessBilling: Literal["Yes", "No"]
#
#    PaymentMethod: Literal[
#        "Electronic check",
#        "Mailed check",
#        "Bank transfer (automatic)",
#        "Credit card (automatic)"
#    ]
#
#    MonthlyCharges: float = Field(gt=0)
#    TotalCharges: float = Field(ge=0)