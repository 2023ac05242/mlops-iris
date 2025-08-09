# ---- Base image ----
FROM python:3.10

# Make Python friendlier in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Workdir ----
WORKDIR /app

# ---- Install deps first (better layer caching) ----
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- Copy app code ----
# (Includes FastAPI app, src/, etc.)
COPY . .

# ---- Ensure the baked model is inside the image ----
# Your training script writes baked_models/iris_best.pkl
# We copy it explicitly so it’s present even if .dockerignore changes later.
COPY baked_models/ baked_models/

# Optional: tell app.py where to find the baked pickle
ENV PICKLE_PATH=baked_models/iris_best.pkl

# ---- Expose API port ----
EXPOSE 8000

# ---- Run server ----
# Keep a single worker so we don’t load the model multiple times
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
