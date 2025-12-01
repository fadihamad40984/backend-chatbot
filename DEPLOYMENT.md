# Render Deployment Guide

## ✅ Optimizations Applied

1. **Memory Optimization**:
   - Switched to `TinyRoBERTa` (smaller QA model)
   - Forced CPU-only mode (no CUDA dependencies)
   - Removed large pre-computed embeddings
   - Added lazy loading for production

2. **Performance**:
   - Added Gunicorn for production server
   - Reduced workers to 1 (512MB limit)
   - 120s timeout for model loading

3. **Configuration**:
   - Created `Procfile` for Gunicorn
   - Updated `requirements.txt` with gunicorn
   - Modified models to use low memory mode

## 🚀 Deployment Steps

### On Render Dashboard:

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `fadihamad40984/backend-chatbot`
4. Configure:

```
Name: ai-chatbot
Region: Choose closest region
Branch: main
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 server:app
```

5. **Environment Variables**:
   - Click "Advanced"
   - Add: `RENDER` = `true`

6. Click **"Create Web Service"**

### Expected Behavior:

- ✅ Build takes ~5-10 minutes (downloading PyTorch + models)
- ✅ First request takes 30-60 seconds (model initialization)
- ✅ Subsequent requests: 2-3 seconds
- ✅ Memory usage: ~400-450MB (within free tier)

### Troubleshooting:

**If "Out of Memory" error:**
- Make sure `RENDER=true` environment variable is set
- Check that `tinyroberta` model is being used
- Verify lazy loading is enabled

**If "No open ports" warning:**
- This is normal during initialization
- Wait for "Model loaded successfully" message
- Port will open after models are loaded

**If slow responses:**
- First request is slow (model download)
- Subsequent requests should be 2-3 seconds
- Use `/admin/stats` to check if models are loaded

## 📊 Testing

After deployment, test with:

```bash
# Health check
curl https://your-app.onrender.com/

# Chat endpoint
curl -X POST https://your-app.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'

# Stats endpoint
curl https://your-app.onrender.com/admin/stats
```

## ⚠️ Important Notes

1. **Free Tier Limitations**:
   - 512MB RAM limit
   - App spins down after 15 min inactivity
   - Cold start takes 30-60 seconds

2. **First Request**:
   - Models download from Hugging Face (~200MB)
   - This happens only once
   - Cached for subsequent deploys

3. **Performance**:
   - CPU-only inference (no GPU)
   - TinyRoBERTa is smaller but still accurate
   - Optimized for low memory usage

## 🔧 Local Testing

Test the production configuration locally:

```bash
# Set environment variable
$env:RENDER="true"

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 2 --timeout 120 server:app
```

## 📝 Deployment Checklist

- [x] Updated `requirements.txt` with gunicorn
- [x] Created `Procfile` for Gunicorn
- [x] Switched to TinyRoBERTa model
- [x] Forced CPU-only mode
- [x] Added lazy loading for production
- [x] Removed large pre-computed files
- [x] Updated `.gitignore`
- [x] Updated README with deployment instructions

## 🎯 Success Criteria

Your deployment is successful when:
- ✅ Build completes without errors
- ✅ Service shows "Live" status
- ✅ Health check returns 200 OK
- ✅ `/admin/stats` shows loaded models
- ✅ `/chat` endpoint returns answers

---

**Need Help?** Check Render logs or GitHub issues.
