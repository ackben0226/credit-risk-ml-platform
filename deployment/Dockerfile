FROM python:3.11-slim

# =====================================================
# SYSTEM DEPENDENCIES (MINIMAL + CLEAN)
# =====================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =====================================================
# NON-ROOT USER (SECURITY)
# =====================================================
RUN useradd -m appuser
WORKDIR /app

# =====================================================
# ENVIRONMENT VARIABLES
# =====================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENV=prod

# =====================================================
# INSTALL DEPENDENCIES (CACHE OPTIMIZED)
# =====================================================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================================
# COPY SOURCE CODE
# =====================================================
COPY . .

# =====================================================
# OWNERSHIP FIX (IMPORTANT FOR CONTAINERS)
# =====================================================
RUN chown -R appuser:appuser /app

USER appuser

# =====================================================
# HEALTH CHECK (CRITICAL FOR PRODUCTION)
# =====================================================
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# =====================================================
# EXPOSE PORT
# =====================================================
EXPOSE 8000

# =====================================================
# START APP
# =====================================================
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]