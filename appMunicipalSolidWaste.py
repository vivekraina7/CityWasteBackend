# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
from typing import Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Municipal Solid Waste Prediction API",
    description="API for predicting municipal solid waste generation",
    version="1.0.0"
)

# Load model components at startup
try:
    components = joblib.load('msw_model_minimal.joblib')
    MODEL = components['model']
    SCALER = components['scaler']
    FEATURES = components['features']
except Exception as e:
    raise Exception(f"Error loading model components: {str(e)}")

# Input validation models
class CityData(BaseModel):
    pop: float = Field(..., gt=0, description="Population of the municipality")
    area: float = Field(..., gt=0, description="Area in km²")
    pden: float = Field(..., gt=0, description="Population density (people per km²)")
    alt: float = Field(..., description="Altitude in meters above sea level")
    urb: int = Field(..., ge=1, le=3, description="Urbanization index (1:low, 2:medium, 3:high)")
    gdp: float = Field(..., gt=0, description="GDP per capita in EUR")
    wage: float = Field(..., gt=0, description="Average wage in EUR")
    finance: float = Field(..., gt=0, description="Municipal finances in EUR")
    roads: float = Field(..., gt=0, description="Total road length in km")
    proads: float = Field(..., gt=0, description="People per km of roads")
    isle: bool = Field(..., description="Whether the municipality is on an island")
    sea: bool = Field(..., description="Whether the municipality is coastal")
    geo: int = Field(..., ge=1, le=3, description="Geographic location (1:South, 2:Center, 3:North)")

    class Config:
        schema_extra = {
            "example": {
                "pop": 100000,
                "area": 200,
                "pden": 500,
                "alt": 100,
                "urb": 2,
                "gdp": 35000,
                "wage": 30000,
                "finance": 3000000,
                "roads": 300,
                "proads": 333,
                "isle": False,
                "sea": True,
                "geo": 2
            }
        }

    @validator('pden')
    def validate_population_density(cls, v, values):
        if 'pop' in values and 'area' in values:
            calculated_density = values['pop'] / values['area']
            if abs(v - calculated_density) > calculated_density * 0.1:  # 10% tolerance
                raise ValueError("Population density should approximately match population/area")
        return v

class PredictionResponse(BaseModel):
    msw_kg: float
    msw_tonnes: float
    input_data: CityData
    confidence_metrics: Dict[str, Any]

def preprocess_data(city_data: CityData) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    # Convert input data to dictionary
    data_dict = city_data.dict()
    
    # Convert boolean values to 0/1
    data_dict['isle'] = 1 if data_dict['isle'] else 0
    data_dict['sea'] = 1 if data_dict['sea'] else 0
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([data_dict])[FEATURES]
    
    return input_df

def make_prediction(input_df: pd.DataFrame) -> float:
    """Make prediction using the loaded model"""
    # Scale features
    input_scaled = SCALER.transform(input_df)
    
    # Make prediction
    prediction = MODEL.predict(input_scaled)[0]
    
    return prediction

@app.post("/predict/", response_model=PredictionResponse)
async def predict_msw(city_data: CityData):
    """
    Predict Municipal Solid Waste generation for a city
    """
    try:
        # Preprocess input data
        input_df = preprocess_data(city_data)
        
        # Make prediction
        prediction = make_prediction(input_df)
        
        # Calculate confidence metrics (basic for now)
        confidence_metrics = {
            "model_type": str(type(MODEL).__name__),
            "features_used": len(FEATURES),
            "input_validation": "passed"
        }
        
        return PredictionResponse(
            msw_kg=float(prediction),
            msw_tonnes=float(prediction/1000),
            input_data=city_data,
            confidence_metrics=confidence_metrics
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to the MSW Prediction API",
        "endpoints": {
            "predict": "/predict/",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "features_available": len(FEATURES)
    }

# Sample cities endpoint
@app.get("/sample")
async def get_sample_city():
    """Get a sample city data for testing"""
    return {
        "sample_medium_city": {
            "pop": 100000,
            "area": 200,
            "pden": 500,
            "alt": 100,
            "urb": 2,
            "gdp": 35000,
            "wage": 30000,
            "finance": 3000000,
            "roads": 300,
            "proads": 333,
            "isle": False,
            "sea": True,
            "geo": 2
        },
        "sample_small_city": {
            "pop": 25000,
            "area": 50,
            "pden": 500,
            "alt": 150,
            "urb": 1,
            "gdp": 25000,
            "wage": 20000,
            "finance": 800000,
            "roads": 75,
            "proads": 333,
            "isle": False,
            "sea": False,
            "geo": 2
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)