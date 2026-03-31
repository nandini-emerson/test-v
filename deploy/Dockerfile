

FROM python:3.11-slim

# Install system dependencies for face_recognition, pillow, numpy, email-validator, cryptography, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev \
        libx11-dev \
        libgtk-3-dev \
        libboost-all-dev \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        git \
        curl \
        gcc \
        g++ \
        python3-dev \
        libsqlite3-dev \
        libjpeg-dev \
        libbz2-dev \
        liblzma-dev \
        libblas-dev \
        libatlas-base-dev \
        libmagic1 \
        && rm -rf /var/lib/apt/lists/*

# Install dlib separately to ensure face_recognition builds correctly
RUN pip install --no-cache-dir dlib

# Copy requirements to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

WORKDIR /app

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTAPI_ENV=production \
    TZ=UTC

# Expose FastAPI default port
EXPOSE 8000

# Default command to run the FastAPI app with uvicorn
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

