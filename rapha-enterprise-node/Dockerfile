FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch to save massive amounts of RAM and disk space on Render Free
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Expose the standard port (can be overridden by Render)
EXPOSE 8000

# Use Gunicorn with 1 Uvicorn worker (to stay under 512MB limit)
CMD ["sh", "-c", "gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000}"]
