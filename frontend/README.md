# Frontend for Heart Disease Platform

This is a minimal static frontend to demo the multi-modal heart disease risk predictor.

Files:

- `index.html` — main UI with tabs and forms
- `styles.css` — small stylesheet
- `app.js` — client logic, posts to `/api/predict`

How to use:

1. Serve this folder statically (e.g., `python -m http.server 8000` from the `frontend` folder).
2. If the backend (FastAPI) is running at the same origin and exposes `/api/predict`, the form will call it.
3. If no backend is available, the page uses a lightweight demo response to show charts.

Integrate with FastAPI:

Implement `POST /api/predict` that accepts the JSON of form fields and returns:

```
{
  "probability": 0.23,                   # float 0..1
  "modalities": {"cardiovascular":0.2,...},
  "feature_importance": [{"name":"ldl","value":12},...]
}
```
