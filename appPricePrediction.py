# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Municipal Cost Prediction API",
    description="API for predicting total cost of municipal waste management",
    version="1.0.0"
)

# Load model components at startup
try:
    components = joblib.load('cost_model_components.joblib')
    MODEL = components['model']
    SCALER = components['scaler']
    FEATURES = components['features']
    FEATURE_IMPORTANCE = components.get('feature_importance', None)
except Exception as e:
    raise Exception(f"Error loading model components: {str(e)}")

# Input validation model
class CostPredictionInput(BaseModel):
    pop: float = Field(..., gt=0, description="Population")
    msw: float = Field(..., gt=0, description="Total municipal solid waste (kg)")
    msw_so: float = Field(..., gt=0, description="Sorted waste (kg)")
    msw_un: float = Field(..., gt=0, description="Unsorted waste (kg)")
    sor: float = Field(..., ge=0, le=1, description="Share of sorted waste (0-1)")
    pden: float = Field(..., gt=0, description="Population density (people per km²)")
    gdp: float = Field(..., gt=0, description="GDP per capita (EUR)")
    wage: float = Field(..., gt=0, description="Average wage (EUR)")
    geo: int = Field(..., ge=1, le=3, description="Geographic location (1:South, 2:Center, 3:North)")
    urb: int = Field(..., ge=1, le=3, description="Urbanization index (1:low, 2:medium, 3:high)")
    finance: float = Field(..., gt=0, description="Municipal finances (EUR)")
    s_wteregio: float = Field(..., ge=0, le=1, description="Share of waste to energy (0-1)")
    s_landfill: float = Field(..., ge=0, le=1, description="Share of waste to landfill (0-1)")

    class Config:
        schema_extra = {
            "example": {
                "pop": 100000,
                "msw": 50000,
                "msw_so": 30000,
                "msw_un": 20000,
                "sor": 0.6,
                "pden": 500,
                "gdp": 35000,
                "wage": 30000,
                "geo": 2,
                "urb": 2,
                "finance": 3000000,
                "s_wteregio": 0.3,
                "s_landfill": 0.2
            }
        }

    @validator('msw')
    def validate_total_waste(cls, v, values):
        if 'msw_so' in values and 'msw_un' in values:
            total = values['msw_so'] + values['msw_un']
            if abs(v - total) > 0.01 * v:  # 1% tolerance
                raise ValueError("Total waste should equal sum of sorted and unsorted waste")
        return v

    @validator('sor')
    def validate_sorting_rate(cls, v, values):
        if 'msw_so' in values and 'msw' in values:
            calculated_sor = values['msw_so'] / values['msw']
            if abs(v - calculated_sor) > 0.01:  # 1% tolerance
                raise ValueError("Sorting rate should match msw_so/msw")
        return v

    @validator('s_wteregio', 's_landfill')
    def validate_waste_treatment_sum(cls, v, values):
        if 's_wteregio' in values and 's_landfill' in values:
            total = values.get('s_wteregio', 0) + v
            if total > 1:
                raise ValueError("Sum of waste treatment shares cannot exceed 1")
        return v

# Response model
class CostPredictionResponse(BaseModel):
    predicted_cost_eur: float
    input_data: CostPredictionInput
    metadata: Dict[str, Any]

def preprocess_input(data: CostPredictionInput) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    # Convert input to DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])[FEATURES]
    return input_df

@app.post("/predict/cost/", response_model=CostPredictionResponse)
async def predict_cost(data: CostPredictionInput):
    """
    Predict total cost of municipal waste management
    """
    try:
        # Preprocess input data
        input_df = preprocess_input(data)
        
        # Scale features
        input_scaled = SCALER.transform(input_df)
        
        # Make prediction
        prediction = MODEL.predict(input_scaled)[0]
        
        # Get top 5 important features if available
        top_features = None
        if FEATURE_IMPORTANCE is not None:
            top_features = FEATURE_IMPORTANCE.head().to_dict('records')
        
        # Prepare metadata
        metadata = {
            "model_type": str(type(MODEL).__name__),
            "features_used": len(FEATURES),
            "top_features": top_features,
            "input_validation": "passed",
            "cost_per_capita": prediction / data.pop,
            "cost_per_ton": prediction / (data.msw / 1000)  # Convert kg to tonnes
        }
        
        return CostPredictionResponse(
            predicted_cost_eur=float(prediction),
            input_data=data,
            metadata=metadata
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to the Municipal Cost Prediction API",
        "endpoints": {
            "predict": "/predict/cost/",
            "docs": "/docs",
            "health": "/health",
            "sample": "/sample"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "features_available": FEATURES,
        "feature_importance_available": FEATURE_IMPORTANCE is not None
    }

@app.get("/sample")
async def get_sample_data():
    """Get sample data for testing"""
    return {
        "medium_city": {
            "pop": 100000,
            "msw": 50000,
            "msw_so": 30000,
            "msw_un": 20000,
            "sor": 0.6,
            "pden": 500,
            "gdp": 35000,
            "wage": 30000,
            "geo": 2,
            "urb": 2,
            "finance": 3000000,
            "s_wteregio": 0.3,
            "s_landfill": 0.2
        },
        "small_city": {
            "pop": 25000,
            "msw": 12500,
            "msw_so": 6250,
            "msw_un": 6250,
            "sor": 0.5,
            "pden": 250,
            "gdp": 25000,
            "wage": 25000,
            "geo": 1,
            "urb": 1,
            "finance": 800000,
            "s_wteregio": 0.2,
            "s_landfill": 0.3
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)