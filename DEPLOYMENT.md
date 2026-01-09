# 🚀 Deployment Guide - Heart Disease Platform

This guide walks you through deploying your Heart Disease Prediction Platform to **Render** (backend) and **Netlify** (frontend).

---

## 📋 Prerequisites

- GitHub account
- Render account (sign up at [render.com](https://render.com))
- Netlify account (sign up at [netlify.com](https://netlify.com))
- Git installed on your computer

---

## Part 1: Push Your Code to GitHub

### 1. Initialize Git Repository (if not already done)

```bash
cd e:\Python\heart-disease-platform
git init
```

### 2. Important: Add Model Files to Git

Since the `.gitignore` excludes `.pth` files by default, you need to force-add your trained model:

```bash
git add -f heart_disease_model_final.pth
git add -f scalers.pkl
```

Or remove `*.pth` from `.gitignore` if you want to track model files.

### 3. Commit and Push

```bash
git add .
git commit -m "Initial commit - Heart Disease Platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-platform.git
git push -u origin main
```

**Replace `YOUR_USERNAME` with your GitHub username**

---

## Part 2: Deploy Backend to Render

### 1. Create New Web Service

1. Go to [render.com/dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select the `heart-disease-platform` repository

### 2. Configure the Service

Fill in the following settings:

| Field | Value |
|-------|-------|
| **Name** | `heart-disease-api` (or your choice) |
| **Region** | Choose closest to you |
| **Branch** | `main` |
| **Root Directory** | Leave empty |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn predict_api:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` |

### 3. Deploy

1. Click **"Create Web Service"**
2. Wait for the build to complete (5-10 minutes)
3. Once deployed, you'll get a URL like: `https://heart-disease-api.onrender.com`

### 4. Test Your Backend

Visit: `https://your-backend-name.onrender.com/docs`

You should see the FastAPI interactive documentation.

**⚠️ Copy your backend URL - you'll need it for the frontend!**

---

## Part 3: Deploy Frontend to Netlify

### 1. Update Environment Variable for Production

Before deploying, you need to set the production API URL:

1. Open `netlify.toml` in your project
2. Update the redirect URL:

```toml
[[redirects]]
  from = "/api/*"
  to = "https://YOUR-BACKEND-NAME.onrender.com/api/:splat"
  status = 200
  force = true
```

Replace `YOUR-BACKEND-NAME` with your actual Render service name.

### 2. Commit the Change

```bash
git add netlify.toml
git commit -m "Update API URL for production"
git push
```

### 3. Deploy to Netlify

#### Option A: Via Netlify Dashboard (Recommended)

1. Go to [app.netlify.com](https://app.netlify.com)
2. Click **"Add new site"** → **"Import an existing project"**
3. Choose **GitHub** and select your repository
4. Configure build settings:

   | Field | Value |
   |-------|-------|
   | **Base directory** | `frontend` |
   | **Build command** | `npm install && npm run build` |
   | **Publish directory** | `frontend/dist` |

5. Click **"Deploy site"**

#### Option B: Via Netlify CLI

```bash
cd frontend
npm install -g netlify-cli
netlify login
netlify init
netlify deploy --prod
```

### 4. Set Environment Variable (Optional)

If you want to use environment variables instead of `netlify.toml` redirects:

1. In Netlify dashboard, go to **Site settings** → **Environment variables**
2. Add variable:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://your-backend-name.onrender.com`
3. Redeploy your site

### 5. Get Your Frontend URL

After deployment, Netlify will give you a URL like:
- `https://random-name-12345.netlify.app`

You can customize this in **Site settings** → **Domain management**

---

## Part 4: Final Configuration

### Update CORS in Backend (Important!)

Once you have your Netlify URL, update the CORS settings in `predict_api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-site-name.netlify.app",
        "http://localhost:5173",  # Keep for local dev
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Then commit and push:

```bash
git add predict_api.py
git commit -m "Update CORS for production"
git push
```

Render will automatically redeploy your backend.

---

## 🎉 You're Live!

Your application is now deployed:

- **Frontend**: `https://your-site-name.netlify.app`
- **Backend API**: `https://your-backend-name.onrender.com`
- **API Docs**: `https://your-backend-name.onrender.com/docs`

---

## 🔧 Troubleshooting

### Backend Issues

1. **Build fails**: Check the Render logs for missing dependencies
2. **Model not found**: Make sure you committed `heart_disease_model_final.pth` and `scalers.pkl`
3. **Port errors**: Render provides `$PORT` automatically - don't hardcode it

### Frontend Issues

1. **API calls fail**: 
   - Check CORS settings in backend
   - Verify the API URL in `netlify.toml` is correct
   - Check Network tab in browser DevTools
2. **Build fails**: 
   - Ensure `package.json` has all dependencies
   - Check Node version (should be 18+)
3. **404 errors**: Make sure the redirect rules in `netlify.toml` are correct

### Testing Locally

Backend:
```bash
uvicorn predict_api:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

---

## 💰 Cost Considerations

Both Render and Netlify offer **free tiers**:

- **Render Free**: 
  - 750 hours/month (enough for one service)
  - Spins down after 15 min of inactivity (cold starts ~30s)
  - 512 MB RAM

- **Netlify Free**:
  - 100 GB bandwidth/month
  - 300 build minutes/month
  - Automatic HTTPS

---

## 🔄 Automatic Deployments

Both platforms support automatic deployments:

1. **Push to GitHub** → Both Render and Netlify automatically detect changes
2. **Auto-build and deploy** → Your changes go live in minutes

To disable auto-deploy:
- **Render**: Settings → Auto-Deploy (toggle off)
- **Netlify**: Site settings → Build & deploy → Stop builds

---

## 📊 Monitoring

- **Render**: View logs in dashboard under "Logs" tab
- **Netlify**: View deploy logs in "Deploys" tab
- **Both**: Set up email notifications for failed deployments

---

## 🚀 Next Steps

1. **Custom Domain**: Add your own domain in Netlify/Render settings
2. **HTTPS**: Both platforms provide free SSL certificates
3. **Analytics**: Add Google Analytics or Netlify Analytics
4. **Monitoring**: Set up uptime monitoring (e.g., UptimeRobot)
5. **Performance**: Enable Netlify's Asset Optimization

---

## 📞 Need Help?

- Render Docs: https://render.com/docs
- Netlify Docs: https://docs.netlify.com
- FastAPI Docs: https://fastapi.tiangolo.com

Happy deploying! 🎊
