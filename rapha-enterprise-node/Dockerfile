# ──────────────────────────────────────────────────────────
# Rapha Enterprise Node — Production Docker Image
#
# Multi-stage build, non-root user, minimal attack surface.
# Deploys behind hospital firewalls for secure AI training.
# ──────────────────────────────────────────────────────────

# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt gunicorn

# Stage 2: Production runtime
FROM python:3.10-slim AS runtime

# Security: create non-root user
RUN groupadd -r rapha && useradd -r -g rapha -d /app -s /sbin/nologin rapha

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY --chown=rapha:rapha . .

# Create audit log directory with proper permissions
RUN mkdir -p /var/log/rapha && chown rapha:rapha /var/log/rapha

# Environment defaults
ENV RAPHA_ENV=production \
    RAPHA_NODE_ID=node-unnamed \
    RAPHA_AUDIT_DIR=/var/log/rapha \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check for orchestrators (Docker, K8s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Switch to non-root user
USER rapha

# Production server with limited workers (memory-constrained hospital environments)
CMD ["sh", "-c", "gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} --timeout 300"]
