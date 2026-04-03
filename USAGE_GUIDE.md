# Heart Disease Platform - Usage Guide

## Overview

This platform provides two methods for heart disease risk assessment:

### 1. Clinical Data Assessment
Use the **Demographics**, **Cardiovascular**, **Metabolic**, and **Lab Results** tabs to input patient clinical data. This uses the Framingham-based multi-modal neural network.

### 2. Cardiac Imaging Assessment
Use the **Cardiac Imaging** tab to upload medical images (X-ray, CT, MRI, ultrasound) for AI-powered structural analysis using a UNet deep learning model.

## How to Use

### For Clinical Data Assessment

1. Navigate through the tabs:
   - **Demographics**: Age, sex, BMI, physical activity, smoking, family history
   - **Cardiovascular**: Blood pressure, heart rate, hypertension status
   - **Metabolic**: Cholesterol levels, glucose, diabetes status
   - **Lab Results**: Electrolytes, kidney function markers

2. Fill in the patient's data

3. Click **"Assess Risk"** button

4. View results:
   - Overall risk probability (%)
   - Risk category (Low/Medium/High)
   - Modality contributions (which health aspects contribute most to risk)
   - Key risk factors
   - Personalized recommendations

### For Cardiac Imaging Assessment

1. Click on the **Cardiac Imaging** tab

2. Click **"Choose File"** and select a cardiac image:
   - Supported formats: JPG, PNG, JPEG, etc.
   - Image types: Cardiac X-ray, CT scan, MRI, echocardiogram, etc.

3. Preview your uploaded image

4. Click **"Analyze Cardiac Image"**

5. View results:
   - Overall risk probability based on image analysis
   - Risk category
   - Imaging-specific recommendations
   - Analysis type indicator

## Understanding Results

### Risk Categories

- **Low Risk (< 30%)**: Continue healthy lifestyle habits
- **Medium Risk (30-70%)**: Schedule regular check-ups, monitor symptoms
- **High Risk (> 70%)**: Seek immediate medical consultation

### Modality Contributions (Clinical Data Only)

Shows which aspect of health contributes most to the overall risk:
- **Cardiovascular**: Blood pressure and heart-related factors
- **Metabolic**: Cholesterol, glucose, and metabolic factors
- **Labs**: Electrolytes and kidney function
- **Demographics**: Age, BMI, lifestyle factors

### Recommendations

Each assessment provides personalized health recommendations based on:
- Identified risk factors
- Abnormal values or imaging findings
- Overall risk level
- Best practices for cardiovascular health

## API Integration

### Clinical Data Prediction

```python
import requests

data = {
    "age": 55,
    "sex": "male",
    "bmi": 27,
    "systolic_bp": 130,
    "diastolic_bp": 80,
    # ... other parameters
}

response = requests.post(
    "http://localhost:8000/api/predict",
    json=data
)
result = response.json()
print(f"Risk: {result['probability']*100:.1f}%")
```

### Image Prediction

```python
import requests

with open('cardiac_scan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        "http://localhost:8000/api/predict-image",
        files=files
    )
    
result = response.json()
print(f"Risk: {result['probability']*100:.1f}%")
print(f"Analysis Type: {result['analysis_type']}")
```

## Model Information

### Clinical Data Model
- **Architecture**: Multi-modal neural network
- **Input**: 20+ clinical features across 4 modalities
- **Output**: Risk probability + modality contributions
- **File**: `heart_disease_model_final.pth`

### Cardiac Imaging Model
- **Architecture**: UNet (encoder-decoder CNN)
- **Input**: Grayscale cardiac images (256x256)
- **Output**: Segmentation map + risk probability
- **File**: `best_cardiac_model.pth`

## Tips for Best Results

### Clinical Data
- Provide as much accurate data as possible
- Use recent lab results (within 3-6 months)
- Ensure blood pressure is measured correctly
- Consider multiple measurements over time

### Cardiac Imaging
- Use high-quality, clear images
- Ensure proper image orientation
- Standard medical imaging formats work best
- Multiple views can provide comprehensive assessment

## Important Disclaimers

⚠️ **This tool is for educational and research purposes only**

- NOT a substitute for professional medical diagnosis
- NOT approved for clinical use
- Results should be interpreted by qualified healthcare professionals
- Always consult a doctor for actual medical advice
- Do not make treatment decisions based solely on these results

## Support

For issues or questions:
- Check the README.md for setup instructions
- Review the API documentation
- Ensure all dependencies are installed
- Verify model files exist in the correct location
