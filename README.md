# Multi-Modal Heart Disease Risk Assessment Platform

AI-powered cardiovascular risk prediction using multi-modal clinical data with a React frontend and FastAPI backend.

## Project Structure

```
heart-disease-platform/
├── predict_api.py          # FastAPI backend with /api/predict endpoint
├── train_model.py          # PyTorch model training script
├── run_all.py             # Combined frontend + backend launcher
├── main.py                # Alternative main entry point
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── main.jsx
│   │   ├── HeartDiseaseAssessment.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js     # Proxies /api to backend:8000
│   └── index.html
├── scalers.pkl            # Generated after training (saved here)
└── heart_disease_model_final.pth  # Generated after training (saved here)
```

## Setup

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- Miniconda or Anaconda (recommended)

### Backend Setup

1. **Install Python dependencies:**

```bash
# Create conda environment (recommended)
conda create -n heart-disease python=3.11
conda activate heart-disease

# Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
pip install fastapi uvicorn pydantic numpy pandas scikit-learn matplotlib seaborn
```

2. **Train the model (optional - will use demo fallback if skipped):**

```bash
python train_model.py
```

This will generate:
- `scalers.pkl` - Feature scalers
- `heart_disease_model_final.pth` - Trained model weights
- `best_model.pth` - Best checkpoint during training
- `training_history.png` - Training curves

### Frontend Setup

1. **Install Node dependencies:**

```bash
cd frontend
npm install
cd ..
```

## Running the Application

### Option 1: Combined Server (Recommended)

Run both frontend dev server and backend with one command:

```bash
python run_all.py
```

This will:
- Start the FastAPI backend on `http://127.0.0.1:8000`
- Start the Vite dev server on `http://127.0.0.1:3000`
- Automatically open `http://127.0.0.1:3000` in your browser
- Proxy `/api/*` requests to the backend

### Option 2: Production Build

Build the frontend and serve everything from FastAPI:

```bash
# Build frontend
cd frontend
npm run build
cd ..

# Run combined server (will detect and serve built files)
python run_all.py
```

The app will be available at `http://127.0.0.1:8000`

### Option 3: Separate Servers

Run backend and frontend separately for development:

**Terminal 1 - Backend:**
```bash
python -m uvicorn predict_api:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:3000` in your browser.

## API Endpoints

### Backend (Port 8000)

- `GET /` - Root endpoint info
- `GET /api/health` - Health check
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2025-12-23T..."
  }
  ```
- `POST /api/predict` - Heart disease risk prediction
  
  **Request body:**
  ```json
  {
    "age": 55,
    "sex": "male",
    "bmi": 27,
    "systolic_bp": 130,
    "diastolic_bp": 80,
    "heart_rate": 75,
    "prevalent_hypertension": 0,
    "total_cholesterol": 200,
    "hdl": 50,
    "ldl": 120,
    "triglycerides": 150,
    "fasting_glucose": 95,
    "diabetes": 0,
    "sodium": 140,
    "potassium": 4.2,
    "calcium": 9.5,
    "creatinine": 1.0,
    "egfr": 90,
    "smoking": 0,
    "physical_activity": "moderate",
    "family_history": 0
  }
  ```

  **Response:**
  ```json
  {
    "probability": 0.234,
    "modalities": {
      "cardiovascular": 0.093,
      "metabolic": 0.082,
      "labs": 0.035,
      "demographics": 0.023
    },
    "feature_importance": [
      {"name": "systolic_bp", "value": 10},
      {"name": "ldl", "value": 20},
      ...
    ]
  }
  ```

## Testing

### Test Backend API

```bash
# Health check
curl http://127.0.0.1:8000/api/health

# Prediction (PowerShell)
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8000/api/predict `
  -ContentType "application/json" `
  -Body '{"age":55,"sex":1,"bmi":27,"systolic_bp":130,"diastolic_bp":80,"heart_rate":75,"prevalent_hypertension":0,"total_cholesterol":200,"hdl":50,"ldl":120,"triglycerides":150,"fasting_glucose":95,"diabetes":0,"sodium":140,"potassium":4.2,"calcium":9.5,"creatinine":1.0,"egfr":90,"smoking":0,"physical_activity":1,"family_history":0}'
```

## Troubleshooting

### Frontend not opening automatically

- Ensure you're running `python run_all.py` from the `heart-disease-platform` directory
- Check if port 3000 is already in use: `netstat -ano | findstr :3000`
- Try running backend and frontend separately (Option 3 above)

### Model/Scalers not found

- If you haven't trained the model, the API will use a demo fallback (heuristic-based predictions)
- Run `python train_model.py` to generate the model files
- Files will be saved in the `heart-disease-platform` directory

### CORS / Connection errors

- Ensure backend is running on port 8000
- Check `vite.config.js` has the proxy configured to `http://127.0.0.1:8000`
- Verify CORS middleware is enabled in `predict_api.py`

### Port conflicts

To use different ports:

**Backend:**
```bash
python -m uvicorn predict_api:app --host 127.0.0.1 --port 8001
```

**Frontend (update vite.config.js first):**
```bash
cd frontend
PORT=3001 npm run dev
```

## Development

### Frontend Development

The React app uses:
- **Vite** - Build tool and dev server
- **React 18** - UI framework
- **Tailwind CSS (CDN)** - Styling (currently loaded via CDN in index.html)

To add proper Tailwind support:
```bash
cd frontend
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### Backend Development

The FastAPI backend uses:
- **FastAPI** - Web framework
- **PyTorch** - Deep learning framework
- **Scikit-learn** - Data preprocessing and metrics
- **Pydantic** - Request/response validation

## Production Deployment

1. Build the frontend:
```bash
cd frontend
npm run build
```

2. Serve with Gunicorn + Uvicorn workers:
```bash
pip install gunicorn
gunicorn predict_api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

3. Or use the built-in runner:
```bash
python run_all.py
```

## License

This is a demo project for educational and research purposes.

## Notes

**⚠️ This is a demonstration project using synthetic data and should NOT be used for clinical decision-making without:**
- Validation on real clinical datasets
- Regulatory approval (FDA, CE marking, etc.)
- Clinical expert review
- IRB/ethics approval
- Proper PHI handling and security measures
- Continuous monitoring and performance evaluation
