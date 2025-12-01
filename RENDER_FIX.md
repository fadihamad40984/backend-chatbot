# Render Deployment Fix

## المشكلة
Render يشغل `python server.py` بدل استخدام Gunicorn من الـ Procfile.

## الحل

### الطريقة 1: تحديث Start Command في Render Dashboard

1. اذهب إلى Render Dashboard
2. اختر الـ service: `backend-chatbot-bc8w`
3. اذهب إلى **Settings**
4. في **Build & Deploy** section
5. **Start Command** غيره إلى:

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 server:app
```

6. اضغط **Save Changes**
7. Redeploy

### الطريقة 2: استخدام render.yaml

قم بعمل commit و push للملف `render.yaml` الموجود في الـ repo.

```bash
cd c:\Users\fadih\backend-chatbot
git add render.yaml
git commit -m "Add render.yaml for proper Gunicorn config"
git push
```

ثم في Render Dashboard:
1. Settings → Build & Deploy
2. Enable "Auto-Deploy"
3. سيقرأ إعدادات `render.yaml` تلقائياً

### الطريقة 3: Manual Redeploy

```bash
# في Render Dashboard
1. اذهب للـ service
2. اضغط "Manual Deploy" → "Clear build cache & deploy"
3. انتظر حتى ينتهي الـ build
```

## التحقق من النجاح

بعد الـ deployment، تحقق من الـ logs. يجب أن ترى:

```
[INFO] Starting gunicorn 23.0.0
[INFO] Listening at: http://0.0.0.0:10000
[INFO] Using worker: sync
[INFO] Booting worker with pid: xxx
```

بدلاً من:

```
WARNING: This is a development server. Do not use it in a production deployment.
```

## الفرق

### Flask Development Server (حالياً):
- ⚠️ غير آمن للإنتاج
- ❌ بطيء
- ❌ لا يتحمل الضغط
- ❌ Single-threaded

### Gunicorn (المطلوب):
- ✅ آمن للإنتاج
- ✅ سريع
- ✅ يتحمل الضغط
- ✅ Multi-threaded

## Commands للـ Render

في **Settings** → **Environment** تأكد من:
- `RENDER=true` موجود

في **Settings** → **Build & Deploy**:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 server:app`

---

After fixing, the backend will use Gunicorn and work properly! 🚀
