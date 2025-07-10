# Simple, working Dockerfile for fraud detection API
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pandas==2.1.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    xgboost==2.0.2 \
    joblib==1.3.2 \
    prometheus-client==0.19.0 \
    pydantic==2.5.0

# Copy source code
COPY src/ ./src/
COPY api/ ./api/

# Create models directory (empty but needed)
RUN mkdir -p ./models

# Create a simple health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8000/health || exit 1' > /health-check.sh && chmod +x /health-check.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /health-check.sh

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]