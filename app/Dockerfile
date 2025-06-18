# ============================================================================
# ETAPA 1: BUILDER - Instala dependencias en un entorno temporal
# ============================================================================
FROM python:3.9-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================================
# ETAPA 2: FINAL - La imagen de producción ligera
# ============================================================================
FROM python:3.9-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    KERAS_BACKEND=tensorflow \
    TF_CPP_MIN_LOG_LEVEL=2 \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

# Copia el entorno virtual con las dependencias ya instaladas
COPY --from=builder /opt/venv /opt/venv

# <<< CORRECCIÓN: Usar la ruta explícita al python del venv >>>
# Descarga los paquetes de NLTK dentro de la imagen para un arranque más rápido.
RUN /opt/venv/bin/python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab stopwords

# Copia la aplicación y los assets del modelo
COPY app.py .
COPY modelo_assets/ ./modelo_assets/

# Activa el entorno virtual para los comandos subsiguientes
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE ${PORT}

# Healthcheck para verificar que la API responde
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Comando para ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "60"]