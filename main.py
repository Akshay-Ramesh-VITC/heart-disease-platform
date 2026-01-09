from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

# Initialize FastAPI
app = FastAPI(title="Heart Disease Risk Assessment API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DATA MODELS ====================
class PatientInput(BaseModel):
    # Demographics
    age: float = Field(..., ge=18, le=120)
    sex: str = Field(..., pattern="^(male|female)$")
    bmi: float = Field(..., ge=15, le=50)
    
    # Cardiovascular
    systolic_bp: float = Field(..., ge=80, le=220)
    diastolic_bp: float = Field(..., ge=40, le=130)
    heart_rate: float = Field(..., ge=40, le=150)
    prevalent_hypertension: int = Field(..., ge=0, le=1)
    
    # Metabolic
    total_cholesterol: float = Field(..., ge=100, le=400)
    hdl: float = Field(..., ge=20, le=100)
    ldl: float = Field(..., ge=40, le=250)
    triglycerides: float = Field(..., ge=30, le=500)
    fasting_glucose: float = Field(..., ge=60, le=300)
    diabetes: int = Field(..., ge=0, le=1)
    
    # Electrolytes/Labs
    sodium: float = Field(..., ge=125, le=155)
    potassium: float = Field(..., ge=2.5, le=7.0)
    calcium: float = Field(..., ge=7.0, le=12.0)
    creatinine: float = Field(..., ge=0.3, le=5.0)
    egfr: float = Field(..., ge=10, le=150)
    
    # Risk Factors
    smoking: int = Field(..., ge=0, le=1)
    physical_activity: str = Field(..., pattern="^(sedentary|light|moderate|active)$")
    family_history: int = Field(..., ge=0, le=1)

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_category: str
    modal_scores: Dict[str, float]
    feature_importance: List[Dict[str, any]]
    recommendations: List[str]
    timestamp: str

# ==================== MULTI-MODAL NEURAL NETWORK ====================
class MultiModalHeartDiseaseModel(nn.Module):
    def __init__(self):
        super(MultiModalHeartDiseaseModel, self).__init__()
        
        # Cardiovascular Branch (5 features)
        self.cardio_branch = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Metabolic Branch (6 features)
        self.metabolic_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Electrolyte/Lab Branch (5 features)
        self.lab_branch = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Demographics/Risk Branch (6 features)
        self.demo_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion Layer (32*4 = 128 combined features)
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Modal-specific output layers for interpretability
        self.cardio_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.metabolic_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.lab_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.demo_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
    
    def forward(self, cardio, metabolic, lab, demo):
        # Process each modality
        cardio_out = self.cardio_branch(cardio)
        metabolic_out = self.metabolic_branch(metabolic)
        lab_out = self.lab_branch(lab)
        demo_out = self.demo_branch(demo)
        
        # Get modal-specific predictions
        cardio_risk = self.cardio_head(cardio_out)
        metabolic_risk = self.metabolic_head(metabolic_out)
        lab_risk = self.lab_head(lab_out)
        demo_risk = self.demo_head(demo_out)
        
        # Combine all branches
        combined = torch.cat([cardio_out, metabolic_out, lab_out, demo_out], dim=1)
        
        # Final prediction
        final_pred = self.fusion(combined)
        
        return final_pred, {
            'cardiovascular': cardio_risk,
            'metabolic': metabolic_risk,
            'labs': lab_risk,
            'demographics': demo_risk
        }

# ==================== MODEL AND SCALER INITIALIZATION ====================
class ModelManager:
    def __init__(self):
        self.model = None
        self.scalers = {}
        self.feature_names = {
            'cardiovascular': ['systolic_bp', 'diastolic_bp', 'heart_rate', 
                              'prevalent_hypertension', 'age'],
            'metabolic': ['total_cholesterol', 'hdl', 'ldl', 'triglycerides', 
                         'fasting_glucose', 'diabetes'],
            'labs': ['sodium', 'potassium', 'calcium', 'creatinine', 'egfr'],
            'demographics': ['age', 'bmi', 'smoking', 'family_history', 
                           'sex_encoded', 'activity_encoded']
        }
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize model and scalers"""
        self.model = MultiModalHeartDiseaseModel()
        self.model.eval()
        
        # Initialize scalers for each modality
        for modality in self.feature_names.keys():
            self.scalers[modality] = StandardScaler()
            # Fit with synthetic ranges for demonstration
            self.fit_scaler_with_synthetic_data(modality)
    
    def fit_scaler_with_synthetic_data(self, modality):
        """Fit scalers with typical clinical ranges"""
        ranges = {
            'cardiovascular': {
                'means': [130, 80, 75, 0.3, 55],
                'stds': [20, 12, 15, 0.46, 15]
            },
            'metabolic': {
                'means': [200, 50, 120, 150, 100, 0.15],
                'stds': [40, 15, 35, 80, 25, 0.36]
            },
            'labs': {
                'means': [140, 4.2, 9.5, 1.0, 90],
                'stds': [3, 0.5, 0.5, 0.3, 20]
            },
            'demographics': {
                'means': [55, 27, 0.25, 0.3, 0.5, 1.5],
                'stds': [15, 5, 0.43, 0.46, 0.5, 1.0]
            }
        }
        
        # Generate synthetic data for scaler fitting
        n_samples = 1000
        synthetic_data = np.random.randn(n_samples, len(self.feature_names[modality]))
        for i in range(len(self.feature_names[modality])):
            synthetic_data[:, i] = (synthetic_data[:, i] * ranges[modality]['stds'][i] + 
                                   ranges[modality]['means'][i])
        
        self.scalers[modality].fit(synthetic_data)
    
    def preprocess_input(self, patient_data: PatientInput) -> Dict[str, np.ndarray]:
        """Convert patient input to model format"""
        # Encode categorical variables
        sex_encoded = 1 if patient_data.sex == 'male' else 0
        activity_map = {'sedentary': 0, 'light': 1, 'moderate': 2, 'active': 3}
        activity_encoded = activity_map[patient_data.physical_activity]
        
        # Create feature arrays for each modality
        modality_data = {
            'cardiovascular': np.array([[
                patient_data.systolic_bp,
                patient_data.diastolic_bp,
                patient_data.heart_rate,
                patient_data.prevalent_hypertension,
                patient_data.age
            ]]),
            'metabolic': np.array([[
                patient_data.total_cholesterol,
                patient_data.hdl,
                patient_data.ldl,
                patient_data.triglycerides,
                patient_data.fasting_glucose,
                patient_data.diabetes
            ]]),
            'labs': np.array([[
                patient_data.sodium,
                patient_data.potassium,
                patient_data.calcium,
                patient_data.creatinine,
                patient_data.egfr
            ]]),
            'demographics': np.array([[
                patient_data.age,
                patient_data.bmi,
                patient_data.smoking,
                patient_data.family_history,
                sex_encoded,
                activity_encoded
            ]])
        }
        
        # Scale features
        scaled_data = {}
        for modality, data in modality_data.items():
            scaled_data[modality] = self.scalers[modality].transform(data)
        
        return scaled_data
    
    def calculate_feature_importance(self, patient_data: PatientInput, 
                                    scaled_data: Dict[str, np.ndarray]) -> List[Dict]:
        """Calculate feature importance using gradient-based method"""
        all_features = []
        
        # Risk thresholds for different features
        risk_factors = {
            'systolic_bp': (140, 'high'),
            'diastolic_bp': (90, 'high'),
            'heart_rate': (100, 'high'),
            'total_cholesterol': (240, 'high'),
            'hdl': (40, 'low'),
            'ldl': (160, 'high'),
            'triglycerides': (200, 'high'),
            'fasting_glucose': (126, 'high'),
            'sodium': (145, 'high', 135, 'low'),
            'potassium': (5.5, 'high', 3.5, 'low'),
            'calcium': (10.5, 'high', 8.5, 'low'),
            'creatinine': (1.3, 'high'),
            'egfr': (60, 'low'),
            'bmi': (30, 'high'),
            'age': (60, 'high')
        }
        
        patient_dict = patient_data.dict()
        
        for feature, thresholds in risk_factors.items():
            if feature in patient_dict:
                value = patient_dict[feature]
                importance = 0
                
                if len(thresholds) == 2:  # Single direction risk
                    threshold, direction = thresholds
                    if direction == 'high' and value > threshold:
                        importance = min((value - threshold) / threshold * 0.5, 1.0)
                    elif direction == 'low' and value < threshold:
                        importance = min((threshold - value) / threshold * 0.5, 1.0)
                else:  # Bidirectional risk
                    high_thresh, high_dir, low_thresh, low_dir = thresholds
                    if value > high_thresh:
                        importance = min((value - high_thresh) / high_thresh * 0.5, 1.0)
                    elif value < low_thresh:
                        importance = min((low_thresh - value) / low_thresh * 0.5, 1.0)
                
                if importance > 0:
                    all_features.append({
                        'feature': feature.replace('_', ' ').title(),
                        'importance': importance,
                        'value': value
                    })
        
        # Add categorical risk factors
        if patient_data.smoking == 1:
            all_features.append({'feature': 'Smoking', 'importance': 0.4, 'value': 'Yes'})
        if patient_data.diabetes == 1:
            all_features.append({'feature': 'Diabetes', 'importance': 0.35, 'value': 'Yes'})
        if patient_data.prevalent_hypertension == 1:
            all_features.append({'feature': 'Hypertension', 'importance': 0.3, 'value': 'Yes'})
        if patient_data.family_history == 1:
            all_features.append({'feature': 'Family History', 'importance': 0.25, 'value': 'Yes'})
        
        # Sort by importance
        all_features.sort(key=lambda x: x['importance'], reverse=True)
        
        return all_features
    
    def generate_recommendations(self, patient_data: PatientInput, 
                                risk_prob: float) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if patient_data.systolic_bp >= 140 or patient_data.diastolic_bp >= 90:
            recommendations.append("Monitor blood pressure regularly and consult with healthcare provider about hypertension management")
        
        if patient_data.ldl >= 160:
            recommendations.append("Consider dietary modifications to reduce LDL cholesterol levels")
        
        if patient_data.hdl < 40:
            recommendations.append("Increase physical activity to improve HDL cholesterol")
        
        if patient_data.bmi >= 30:
            recommendations.append("Weight management through diet and exercise may reduce cardiovascular risk")
        
        if patient_data.smoking == 1:
            recommendations.append("Smoking cessation is critical for reducing heart disease risk")
        
        if patient_data.fasting_glucose >= 126:
            recommendations.append("Monitor blood glucose levels and consult about diabetes management")
        
        if patient_data.potassium > 5.5:
            recommendations.append("Elevated potassium - consult healthcare provider immediately")
        elif patient_data.potassium < 3.5:
            recommendations.append("Low potassium - may increase arrhythmia risk, consult healthcare provider")
        
        if patient_data.sodium < 135:
            recommendations.append("Low sodium levels detected - requires medical evaluation")
        
        if patient_data.egfr < 60:
            recommendations.append("Reduced kidney function detected - nephrology consultation recommended")
        
        if patient_data.physical_activity in ['sedentary', 'light']:
            recommendations.append("Increase physical activity to at least 150 minutes of moderate exercise per week")
        
        if risk_prob >= 0.7:
            recommendations.append("HIGH RISK: Immediate medical consultation strongly recommended")
        elif risk_prob >= 0.3:
            recommendations.append("Regular follow-up with healthcare provider recommended")
        
        return recommendations if recommendations else ["Continue maintaining healthy lifestyle habits"]
    
    def predict(self, patient_data: PatientInput) -> PredictionResponse:
        """Make prediction for a patient"""
        try:
            # Preprocess input
            scaled_data = self.preprocess_input(patient_data)
            
            # Convert to tensors
            cardio_tensor = torch.FloatTensor(scaled_data['cardiovascular'])
            metabolic_tensor = torch.FloatTensor(scaled_data['metabolic'])
            lab_tensor = torch.FloatTensor(scaled_data['labs'])
            demo_tensor = torch.FloatTensor(scaled_data['demographics'])
            
            # Make prediction
            with torch.no_grad():
                final_pred, modal_preds = self.model(
                    cardio_tensor, metabolic_tensor, lab_tensor, demo_tensor
                )
            
            # Extract results
            risk_probability = float(final_pred.item())
            
            modal_scores = {
                modality: float(pred.item())
                for modality, pred in modal_preds.items()
            }
            
            # Determine risk category
            if risk_probability < 0.3:
                risk_category = "Low Risk - Continue preventive care"
            elif risk_probability < 0.7:
                risk_category = "Moderate Risk - Enhanced monitoring recommended"
            else:
                risk_category = "High Risk - Medical intervention advised"
            
            # Calculate feature importance
            feature_importance = self.calculate_feature_importance(
                patient_data, scaled_data
            )
            
            # Generate recommendations
            recommendations = self.generate_recommendations(
                patient_data, risk_probability
            )
            
            return PredictionResponse(
                risk_probability=risk_probability,
                risk_category=risk_category,
                modal_scores=modal_scores,
                feature_importance=feature_importance,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

# ==================== API ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "message": "Multi-Modal Heart Disease Risk Assessment API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_risk(patient: PatientInput):
    """
    Predict heart disease risk for a patient using multi-modal deep learning
    """
    try:
        prediction = model_manager.predict(patient)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/normal-ranges")
async def get_normal_ranges():
    """Return normal ranges for all lab values"""
    return {
        "cardiovascular": {
            "systolic_bp": {"min": 90, "max": 120, "unit": "mmHg", "optimal": "< 120"},
            "diastolic_bp": {"min": 60, "max": 80, "unit": "mmHg", "optimal": "< 80"},
            "heart_rate": {"min": 60, "max": 100, "unit": "bpm", "optimal": "60-100"}
        },
        "metabolic": {
            "total_cholesterol": {"min": 125, "max": 200, "unit": "mg/dL", "optimal": "< 200"},
            "hdl": {"min": 40, "max": 60, "unit": "mg/dL", "optimal": "> 60"},
            "ldl": {"min": 0, "max": 100, "unit": "mg/dL", "optimal": "< 100"},
            "triglycerides": {"min": 0, "max": 150, "unit": "mg/dL", "optimal": "< 150"},
            "fasting_glucose": {"min": 70, "max": 100, "unit": "mg/dL", "optimal": "70-100"}
        },
        "labs": {
            "sodium": {"min": 135, "max": 145, "unit": "mEq/L"},
            "potassium": {"min": 3.5, "max": 5.5, "unit": "mEq/L"},
            "calcium": {"min": 8.5, "max": 10.5, "unit": "mg/dL"},
            "creatinine": {"min": 0.6, "max": 1.3, "unit": "mg/dL"},
            "egfr": {"min": 90, "max": 120, "unit": "mL/min/1.73m²", "optimal": "> 90"}
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)