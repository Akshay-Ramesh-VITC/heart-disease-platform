"""
Test script for the heart disease prediction platform
Tests both clinical data and cardiac imaging endpoints
"""
import requests
import os

# Configuration
BASE_URL = "http://localhost:8000"  # Change for deployed version

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        response.raise_for_status()
        data = response.json()
        
        print("✓ Health check successful")
        print(f"  Status: {data.get('status')}")
        print(f"  Clinical Model Loaded: {data.get('model_loaded')}")
        print(f"  Image Model Loaded: {data.get('image_model_loaded')}")
        print(f"  Timestamp: {data.get('timestamp')}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_clinical_prediction():
    """Test the clinical data prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Clinical Data Prediction Endpoint")
    print("="*60)
    
    # Sample patient data
    patient_data = {
        "age": 55,
        "sex": "male",
        "bmi": 27.5,
        "systolic_bp": 135,
        "diastolic_bp": 85,
        "heart_rate": 78,
        "prevalent_hypertension": 0,
        "total_cholesterol": 210,
        "hdl": 45,
        "ldl": 130,
        "triglycerides": 160,
        "fasting_glucose": 105,
        "diabetes": 0,
        "sodium": 142,
        "potassium": 4.1,
        "calcium": 9.4,
        "creatinine": 1.1,
        "egfr": 85,
        "smoking": 1,
        "physical_activity": "light",
        "family_history": 1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        print("✓ Clinical prediction successful")
        print(f"\nResults:")
        print(f"  Risk Probability: {data.get('probability', 0)*100:.1f}%")
        print(f"  Risk Category: {data.get('risk_category', 'Unknown')}")
        
        if 'modalities' in data:
            print(f"\n  Modality Contributions:")
            for modality, score in data['modalities'].items():
                print(f"    - {modality.capitalize()}: {score*100:.1f}%")
        
        if 'recommendations' in data and data['recommendations']:
            print(f"\n  Top Recommendations:")
            for i, rec in enumerate(data['recommendations'][:3], 1):
                print(f"    {i}. {rec}")
        
        return True
    except Exception as e:
        print(f"✗ Clinical prediction failed: {e}")
        if hasattr(e, 'response'):
            print(f"  Response: {e.response.text}")
        return False


def test_image_prediction(image_path=None):
    """Test the cardiac imaging prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Cardiac Imaging Prediction Endpoint")
    print("="*60)
    
    # If no image provided, skip test
    if not image_path:
        print("⚠ No image path provided, skipping image prediction test")
        print("  To test: python test_endpoints.py --image path/to/cardiac_image.jpg")
        return None
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{BASE_URL}/api/predict-image",
                files=files
            )
            response.raise_for_status()
            data = response.json()
        
        print("✓ Image prediction successful")
        print(f"\nResults:")
        print(f"  Risk Probability: {data.get('probability', 0)*100:.1f}%")
        print(f"  Risk Category: {data.get('risk_category', 'Unknown')}")
        print(f"  Analysis Type: {data.get('analysis_type', 'Unknown')}")
        print(f"  Predicted Class: {data.get('predicted_class', 'Unknown')}")
        print(f"  Number of Classes: {data.get('num_classes', 'Unknown')}")
        
        if 'message' in data:
            print(f"  Message: {data['message']}")
        
        if 'recommendations' in data and data['recommendations']:
            print(f"\n  Recommendations:")
            for i, rec in enumerate(data['recommendations'][:3], 1):
                print(f"    {i}. {rec}")
        
        return True
    except Exception as e:
        print(f"✗ Image prediction failed: {e}")
        if hasattr(e, 'response'):
            print(f"  Response: {e.response.text}")
        return False


def main():
    """Run all tests"""
    import sys
    
    print("\n" + "="*60)
    print("Heart Disease Platform - Endpoint Testing")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    
    # Check if image path provided
    image_path = None
    if len(sys.argv) > 1 and sys.argv[1] == '--image' and len(sys.argv) > 2:
        image_path = sys.argv[2]
    
    # Run tests
    results = {
        'health': test_health_check(),
        'clinical': test_clinical_prediction(),
        'imaging': test_image_prediction(image_path)
    }
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⚠ Skipped: {skipped}")
    
    if failed > 0:
        print("\n⚠ Some tests failed. Check the output above for details.")
        print("Ensure the backend is running: python predict_api.py")
        print("Ensure models are trained and files exist:")
        print("  - heart_disease_model_final.pth")
        print("  - best_cardiac_model.pth")
        print("  - scalers.pkl")
    else:
        print("\n✓ All executed tests passed!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
