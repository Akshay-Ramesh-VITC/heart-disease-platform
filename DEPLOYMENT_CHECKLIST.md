# Deployment Checklist - Heart Disease Platform

## Pre-Deployment Verification

### ✅ Files Check
- [ ] `predict_api.py` - Updated with image prediction endpoint
- [ ] `train_image_model.py` - Image model training script
- [ ] `frontend/src/HeartDiseaseAssessment.jsx` - Updated with imaging tab
- [ ] `README.md` - Updated documentation
- [ ] `requirements.txt` - All dependencies listed
- [ ] `test_endpoints.py` - Testing script ready
- [ ] `USAGE_GUIDE.md` - User documentation
- [ ] `INTEGRATION_SUMMARY.md` - Technical documentation

### ✅ Model Files
- [ ] `heart_disease_model_final.pth` - Clinical data model exists
- [ ] `best_cardiac_model.pth` - Cardiac imaging model exists
- [ ] `scalers.pkl` - Feature scalers exist
- [ ] Model files are in the same directory as `predict_api.py`

### ✅ Dependencies
- [ ] Backend dependencies installed (see requirements.txt)
- [ ] Frontend dependencies installed (`cd frontend && npm install`)
- [ ] PyTorch installed correctly
- [ ] OpenCV installed (`opencv-python`)
- [ ] Pillow installed for image handling
- [ ] python-multipart installed for file uploads

### ✅ Local Testing

#### Backend Tests:
```bash
# 1. Start backend
python predict_api.py

# 2. In another terminal, test health check
curl http://localhost:8000/api/health

# 3. Test clinical prediction
python test_endpoints.py

# 4. Test with an image (if available)
python test_endpoints.py --image path/to/test_image.jpg
```

#### Frontend Tests:
```bash
# 1. Navigate to frontend
cd frontend

# 2. Start dev server
npm run dev

# 3. Open http://localhost:5173

# 4. Test each tab:
# - Demographics
# - Cardiovascular
# - Metabolic
# - Lab Results
# - Cardiac Imaging (NEW)

# 5. Test form submission for clinical data
# 6. Test image upload and analysis
```

#### Integration Tests:
```bash
# Start both together
python run_all.py

# Test full workflow:
# 1. Fill clinical data → Submit → Verify results
# 2. Switch to Imaging tab → Upload image → Analyze → Verify results
# 3. Verify both result types display correctly
```

## Backend Deployment (Render/Railway/etc.)

### Configuration

1. **Environment Variables:**
   - None required for basic setup
   - Optional: Set `PORT` if needed by platform

2. **Build Command:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Command:**
   ```bash
   uvicorn predict_api:app --host 0.0.0.0 --port $PORT
   ```

4. **Upload Required Files:**
   - [ ] All `.py` files
   - [ ] `requirements.txt`
   - [ ] `heart_disease_model_final.pth`
   - [ ] `best_cardiac_model.pth`
   - [ ] `scalers.pkl`
   - [ ] `framingham.csv` (if needed)

5. **CORS Configuration:**
   - [ ] Update allowed origins in `predict_api.py`:
   ```python
   allow_origins=[
       "https://your-frontend-domain.netlify.app",
       "https://your-frontend-domain.vercel.app",
       "http://localhost:5173",  # Keep for local dev
   ]
   ```

### Post-Deployment Backend Checks:
- [ ] Health endpoint responds: `GET https://your-api.com/api/health`
- [ ] Clinical prediction works: `POST https://your-api.com/api/predict`
- [ ] Image prediction works: `POST https://your-api.com/api/predict-image`
- [ ] Both model_loaded flags are true in health check

## Frontend Deployment (Netlify/Vercel)

### Configuration

1. **Environment Variables:**
   ```
   VITE_API_URL=https://your-backend-url.com
   ```

2. **Build Settings:**
   - Base directory: `frontend`
   - Build command: `npm run build`
   - Publish directory: `frontend/dist`

3. **Update API URL:**
   - Set environment variable OR
   - Update default in `HeartDiseaseAssessment.jsx`:
   ```javascript
   const backendBase = import.meta.env.VITE_API_URL || 'https://your-backend-url.com';
   ```

### Post-Deployment Frontend Checks:
- [ ] Site loads correctly
- [ ] All 5 tabs visible (Demographics, Cardiovascular, Metabolic, Lab Results, Cardiac Imaging)
- [ ] Clinical data form works
- [ ] Image upload interface appears in Cardiac Imaging tab
- [ ] Form submissions reach backend successfully
- [ ] Results display correctly for both types

## Final Integration Testing

### Test Clinical Data Flow:
1. [ ] Open deployed frontend
2. [ ] Fill in all clinical data fields
3. [ ] Click "Assess Risk"
4. [ ] Verify risk probability displays
5. [ ] Verify modality contributions show
6. [ ] Verify recommendations appear
7. [ ] Check that "Clinical Data Analysis" badge shows

### Test Cardiac Imaging Flow:
1. [ ] Navigate to Cardiac Imaging tab
2. [ ] Select a cardiac image file
3. [ ] Verify image preview appears
4. [ ] Click "Analyze Cardiac Image"
5. [ ] Verify risk probability displays
6. [ ] Verify "Cardiac Imaging Analysis" badge shows
7. [ ] Verify imaging-specific recommendations

### Test Edge Cases:
- [ ] Submit without filling required fields
- [ ] Upload very large image (>10MB)
- [ ] Upload non-image file
- [ ] Test with slow network
- [ ] Test on mobile device
- [ ] Test on different browsers (Chrome, Firefox, Safari)

## Performance Checks

### Backend:
- [ ] Health check responds in < 100ms
- [ ] Clinical prediction responds in < 1s
- [ ] Image prediction responds in < 3s
- [ ] Memory usage stable
- [ ] No memory leaks after multiple requests

### Frontend:
- [ ] Page loads in < 2s
- [ ] Tab switching is smooth
- [ ] Image preview generates quickly
- [ ] No console errors
- [ ] Responsive on mobile

## Documentation

- [ ] README.md is up to date
- [ ] API endpoints documented
- [ ] Usage guide available
- [ ] Installation instructions clear
- [ ] Troubleshooting section complete

## Security

- [ ] CORS properly configured (not too permissive)
- [ ] No sensitive data in client-side code
- [ ] File upload size limits set
- [ ] Input validation on both frontend and backend
- [ ] HTTPS enabled on both frontend and backend
- [ ] Model files not publicly downloadable

## Monitoring

Set up monitoring for:
- [ ] API uptime
- [ ] Error rates
- [ ] Response times
- [ ] File upload failures
- [ ] Model loading failures

## Backup Plan

- [ ] Keep copy of all model files
- [ ] Document model training process
- [ ] Keep training scripts updated
- [ ] Version control all code changes
- [ ] Tag releases in git

## Post-Launch

- [ ] Monitor error logs for first 24 hours
- [ ] Test with real users
- [ ] Gather feedback
- [ ] Document common issues
- [ ] Plan for updates and improvements

## Rollback Plan

If issues occur:
1. [ ] Revert frontend to previous version
2. [ ] Revert backend to previous version
3. [ ] Check model file integrity
4. [ ] Verify environment variables
5. [ ] Check dependency versions

## Success Criteria

✅ Both prediction methods work correctly  
✅ Frontend displays results properly  
✅ No console errors  
✅ Fast response times  
✅ Mobile responsive  
✅ Clear user guidance  
✅ Error messages are helpful  
✅ Documentation is complete  

---

## Notes

- Image model requires `best_cardiac_model.pth` (trained using train_image_model.py)
- Clinical model requires `heart_disease_model_final.pth` and `scalers.pkl`
- Both can work independently if one model is missing (will show error for that endpoint)
- Consider setting up CI/CD for automatic deployments
- Plan for model updates and versioning strategy
