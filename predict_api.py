"""FastAPI server exposing POST /api/predict using trained heart disease model

Requires:
- heart_disease_model_final.pth (trained PyTorch model)
- scalers.pkl (fitted StandardScalers for each modality)

Use train_model.py to generate these files if they don't exist.
"""
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import os
import pickle
from datetime import datetime

app = FastAPI(title="Heart Disease Predictor - Production API")

# CORS configuration - allow frontend and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://predictheartdisease.netlify.app",
        "https://heart-disease-frontend.onrender.com",  # Add your actual Render frontend URL here
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def root():
    """Root endpoint to verify API is running"""
    return {
        "message": "Heart Disease Prediction API",
        "status": "online",
        "endpoints": {
            "predict": "/api/predict",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

# Use paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "heart_disease_model_final.pth")
SCALERS_PATH = os.path.join(SCRIPT_DIR, "scalers.pkl")

# Define input schema (covers modalities used in frontend)
class PatientInput(BaseModel):
    age: float
    sex: str  # 'male' or 'female'
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    prevalent_hypertension: Optional[int] = 0
    total_cholesterol: Optional[float] = 200.0
    hdl: Optional[float] = 50.0
    ldl: Optional[float] = 120.0
    triglycerides: Optional[float] = 150.0
    fasting_glucose: Optional[float] = 95.0
    diabetes: Optional[int] = 0
    sodium: Optional[float] = 140.0
    potassium: Optional[float] = 4.2
    calcium: Optional[float] = 9.5
    creatinine: Optional[float] = 1.0
    egfr: Optional[float] = 90.0
    smoking: Optional[int] = 0
    physical_activity: Optional[str] = 'moderate'  # 'sedentary', 'light', 'moderate', 'active', 'very_active'
    family_history: Optional[int] = 0


def generate_recommendations(data: Dict, prob: float, modalities: Dict) -> List[str]:
    """Generate personalized health recommendations based on risk factors"""
    recommendations = []
    
    # Blood Pressure
    if data.get('systolic_bp', 120) > 140 or data.get('diastolic_bp', 80) > 90:
        recommendations.append("Monitor and manage blood pressure through lifestyle modifications and medical consultation")
    
    # Cholesterol
    if data.get('ldl', 120) > 130:
        recommendations.append("Consider dietary changes to reduce LDL cholesterol levels")
    if data.get('hdl', 50) < 40:
        recommendations.append("Increase physical activity to improve HDL cholesterol")
    
    # Glucose/Diabetes
    if data.get('fasting_glucose', 95) > 100 or data.get('diabetes', 0) == 1:
        recommendations.append("Maintain regular blood glucose monitoring and follow diabetic care guidelines")
    
    # Kidney Function
    if data.get('egfr', 90) < 60:
        recommendations.append("Consult with a nephrologist regarding kidney function")
    
    # Electrolytes
    if data.get('potassium', 4.2) > 5.5 or data.get('potassium', 4.2) < 3.5:
        recommendations.append("Monitor electrolyte levels and discuss with your healthcare provider")
    
    # Lifestyle Factors
    if data.get('smoking', 0) == 1:
        recommendations.append("Smoking cessation is critical for reducing cardiovascular risk")
    
    if data.get('physical_activity', 2) < 2:
        recommendations.append("Increase physical activity to at least 150 minutes of moderate exercise per week")
    
    if data.get('bmi', 27) > 30:
        recommendations.append("Weight management through balanced nutrition and regular exercise")
    
    # Overall risk-based
    if prob > 0.7:
        recommendations.append("HIGH RISK: Immediate medical consultation recommended")
        recommendations.append("Consider comprehensive cardiovascular screening")
    elif prob > 0.4:
        recommendations.append("MODERATE RISK: Schedule regular check-ups with your healthcare provider")
    else:
        recommendations.append("Continue maintaining healthy lifestyle habits")
    
    return recommendations[:8]  # Limit to top 8 recommendations


def convert_to_numeric(data: Dict) -> Dict:
    """Convert string fields to numeric values expected by the model"""
    # Convert sex: 'male' -> 1, 'female' -> 0
    if isinstance(data.get('sex'), str):
        data['sex'] = 1 if data['sex'].lower() == 'male' else 0
    
    # Convert physical_activity: map string to numeric scale
    if isinstance(data.get('physical_activity'), str):
        activity_map = {
            'sedentary': 0,
            'light': 1,
            'moderate': 2,
            'active': 3,
            'very_active': 4
        }
        data['physical_activity'] = activity_map.get(data['physical_activity'].lower(), 2)
    
    return data


@app.post('/api/predict')
async def predict(payload: PatientInput):
    data = payload.dict()
    
    # Convert string fields to numeric
    data = convert_to_numeric(data)

    # Require actual model - no demo fallback
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found at {MODEL_PATH}. Please train the model first using train_model.py"
        )
    
    if not os.path.exists(SCALERS_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Scalers file not found at {SCALERS_PATH}. Please train the model first using train_model.py"
        )

    try:
        import torch
        from train_model import MultiModalHeartDiseaseModel

        # Load scalers
        with open(SCALERS_PATH, 'rb') as f:
            scalers = pickle.load(f)

        # Build arrays matching feature groups in train_model.DataPreparator
        cardio_feats = np.array([[
            data.get('systolic_bp',120),
            data.get('diastolic_bp',80),
            data.get('heart_rate',75),
            data.get('prevalent_hypertension',0),
            data.get('age',55)
        ]], dtype=float)

        metabolic_feats = np.array([[
            data.get('total_cholesterol',200),
            data.get('hdl',50),
            data.get('ldl',120),
            data.get('triglycerides',150),
            data.get('fasting_glucose',95),
            data.get('diabetes',0)
        ]], dtype=float)

        lab_feats = np.array([[
            data.get('sodium',140),
            data.get('potassium',4.2),
            data.get('calcium',9.5),
            data.get('creatinine',1.0),
            data.get('egfr',90)
        ]], dtype=float)

        demo_feats = np.array([[
            data.get('age',55),
            data.get('bmi',27),
            data.get('smoking',0),
            data.get('family_history',0),
            data.get('sex',1),
            data.get('physical_activity',1)
        ]], dtype=float)

        # Scale using scalers
        cardio_s = scalers['cardiovascular'].transform(cardio_feats)
        metabolic_s = scalers['metabolic'].transform(metabolic_feats)
        labs_s = scalers['labs'].transform(lab_feats)
        demo_s = scalers['demographics'].transform(demo_feats)

        # Load and run model
        model = MultiModalHeartDiseaseModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()

        with torch.no_grad():
            cardio_t = torch.FloatTensor(cardio_s)
            metabolic_t = torch.FloatTensor(metabolic_s)
            labs_t = torch.FloatTensor(labs_s)
            demo_t = torch.FloatTensor(demo_s)

            final_pred, modal_outputs = model(cardio_t, metabolic_t, labs_t, demo_t)
            prob = float(final_pred.detach().cpu().flatten()[0].item())
            modalities = {k: float(v.detach().cpu().flatten()[0].item()) for k,v in modal_outputs.items()}

        # Feature importance: absolute deviation from healthy reference values
        refs = {'systolic_bp':120,'diastolic_bp':80,'ldl':100,'hdl':50,'age':50,'potassium':4.0,'egfr':90,'fasting_glucose':95}
        feats = []
        for n,ref in refs.items():
            val = float(data.get(n, ref))
            deviation = abs(val-ref)
            if deviation > 0:  # Only include features that deviate
                feats.append({'name':n,'value':deviation})
        
        # Sort by deviation magnitude
        feats = sorted(feats, key=lambda x: x['value'], reverse=True)
        
        # Generate recommendations
        recommendations = generate_recommendations(data, prob, modalities)
        
        # Determine risk category
        if prob < 0.3:
            risk_category = 'Low'
        elif prob < 0.7:
            risk_category = 'Medium'
        else:
            risk_category = 'High'

        # Return response matching frontend expectations
        return {
            'probability': prob,
            'risk_probability': prob,  # Alias for compatibility
            'risk_category': risk_category,
            'modalities': modalities,
            'modal_scores': modalities,  # Alias for compatibility
            'feature_importance': feats,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get('/api/health')
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == '__main__':
    import uvicorn
    # ensure imports / files are attempted when running directly
    print('Starting demo FastAPI server on http://127.0.0.1:8000')
    uvicorn.run('predict_api:app', host='127.0.0.1', port=8000, reload=False)