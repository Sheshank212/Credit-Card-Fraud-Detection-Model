# 🚀 Deployment Guide - Fraud Detection System

This guide covers various deployment options for the Credit Card Fraud Detection System, from local development to production-ready deployments.

## 📋 Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- 4GB+ RAM recommended
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🔧 Local Development Setup

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Download dataset from Kaggle and place in data/
# Rename to: data/creditcard 2.csv
```

### 3. Train Models
```bash
# Train fraud detection models
python src/fraud_detector.py

# This will generate:
# - models/*.pkl (trained models)
# - results/*.png (visualizations)
```

### 4. Start API Server
```bash
# Start development server
python api/app.py

# API available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

## 🐳 Docker Deployment

### Single Container Deployment

```bash
# Build image
docker build -t fraud-detection:latest .

# Run API container
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  -v ./models:/app/models:ro \
  -v ./data:/app/data:ro \
  fraud-detection:latest
```

### Docker Compose Deployment

```bash
# Full stack with monitoring
docker-compose --profile monitoring up -d

# API only
docker-compose up fraud-api -d

# Training service
docker-compose --profile training up fraud-trainer

# Run tests
docker-compose --profile testing up fraud-tests
```

### Available Services

| Service | Port | Description |
|---------|------|-------------|
| fraud-api | 8000 | Main API service |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboard |

## ☁️ Cloud Deployment

### AWS ECS Deployment

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t fraud-detection .
docker tag fraud-detection:latest <account>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest

# Deploy using ECS task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
aws ecs update-service --cluster fraud-detection --service fraud-api --task-definition fraud-detection:1
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection-api
  template:
    metadata:
      labels:
        app: fraud-detection-api
    spec:
      containers:
      - name: fraud-api
        image: fraud-detection:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: APP_ENV
          value: "production"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: fraud-models-pvc
```

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/
kubectl expose deployment fraud-detection-api --type=LoadBalancer --port=8000
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The project includes a comprehensive CI/CD pipeline:

- **Code Quality**: Black, Flake8, MyPy, Bandit
- **Testing**: Unit tests, API tests, coverage reporting
- **Security**: Vulnerability scanning with Trivy
- **Docker**: Multi-architecture builds
- **Deployment**: Automated releases

### Pipeline Triggers

- **Push to main**: Full pipeline including deployment
- **Pull Requests**: Testing and validation
- **Releases**: Production deployment

## 📊 Monitoring and Observability

### Prometheus Metrics

The API exposes metrics at `/metrics`:

- `api_requests_total`: Total API requests
- `fraud_predictions_total`: Fraud prediction counts
- `model_prediction_duration_seconds`: Prediction latency
- `active_models_count`: Number of loaded models

### Grafana Dashboards

Pre-configured dashboards for:
- API performance metrics
- Model prediction statistics
- System resource usage
- Fraud detection rates

### Structured Logging

JSON-formatted logs include:
- Request/response tracking
- Model performance metrics
- Error details with context
- Business metrics

## 🧪 Testing and Validation

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=html
```

### API Testing
```bash
# Test API endpoints
pytest tests/test_api.py -v

# Load testing with Locust
locust -f benchmarks/load_test.py --host=http://localhost:8000
```

### Performance Benchmarking
```bash
# Run performance benchmarks
python benchmarks/performance_benchmark.py
```

## 🔒 Security Configuration

### Production Security Checklist

- [ ] Enable HTTPS/TLS encryption
- [ ] Configure authentication middleware
- [ ] Set up rate limiting
- [ ] Enable request logging
- [ ] Configure CORS policies
- [ ] Set up API key management
- [ ] Enable container security scanning
- [ ] Configure secrets management

### Environment Variables

```bash
# Production environment variables
export APP_ENV=production
export LOG_LEVEL=INFO
export API_PORT=8000
export MODEL_PATH=/app/models
export DATA_PATH=/app/data
```

## 📈 Scaling Considerations

### Horizontal Scaling

- **Load Balancer**: Distribute requests across multiple API instances
- **Auto Scaling**: Scale based on CPU/memory usage or request rate
- **Model Caching**: Use Redis for model prediction caching

### Performance Optimization

- **Model Optimization**: Quantization, pruning for faster inference
- **Async Processing**: Use async endpoints for better concurrency
- **Connection Pooling**: Optimize database connections
- **CDN**: Cache static assets and responses

### Resource Requirements

| Deployment Type | CPU | Memory | Storage |
|----------------|-----|---------|---------|
| Development | 1 core | 2GB | 5GB |
| Production (Single) | 2 cores | 4GB | 20GB |
| Production (Scaled) | 4+ cores | 8GB+ | 50GB+ |

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model files exist
   ls -la models/
   
   # Verify model compatibility
   python -c "import joblib; joblib.load('models/xgboost.pkl')"
   ```

2. **API Performance Issues**
   ```bash
   # Check metrics
   curl http://localhost:8000/metrics
   
   # Monitor logs
   docker logs fraud-api
   ```

3. **Memory Issues**
   ```bash
   # Increase container memory limits
   docker run --memory=4g fraud-detection:latest
   ```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Model status
curl http://localhost:8000/models

# Metrics endpoint
curl http://localhost:8000/metrics
```

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [Performance Benchmarks](benchmarks/)
- [Monitoring Setup](monitoring/)
- [Security Guidelines](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)

---

## 🎯 Quick Start Commands

```bash
# Complete local setup
git clone <repo> && cd credit-card-fraud-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/fraud_detector.py
python api/app.py

# Docker deployment
docker-compose up -d

# Run tests
pytest tests/ -v

# Load testing
locust -f benchmarks/load_test.py --host=http://localhost:8000 --headless --users 50 --spawn-rate 2 --run-time 5m
```