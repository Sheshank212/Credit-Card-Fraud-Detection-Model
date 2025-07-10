"""
Load testing script using Locust for fraud detection API
"""

from locust import HttpUser, task, between
import json
import random
import numpy as np


class FraudDetectionUser(HttpUser):
    """Simulated user for load testing fraud detection API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        # Check API health
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("API health check failed")
    
    def generate_transaction(self):
        """Generate a realistic transaction for testing"""
        transaction = {}
        
        # Generate V1-V28 (PCA features) with realistic distributions
        for i in range(1, 29):
            transaction[f'V{i}'] = np.random.normal(0, 1)
        
        # Generate realistic transaction amount
        # Most transactions are small, some are large
        if random.random() < 0.8:
            amount = random.uniform(5, 200)  # Small transactions
        else:
            amount = random.uniform(200, 2000)  # Large transactions
        
        transaction['Amount'] = round(amount, 2)
        
        return transaction
    
    @task(10)
    def predict_single_transaction(self):
        """Test single transaction prediction (most common use case)"""
        transaction = self.generate_transaction()
        
        with self.client.post("/predict", 
                             json=transaction,
                             catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                # Validate response structure
                required_fields = ['is_fraud', 'fraud_probability', 'risk_level', 
                                 'algorithm_used', 'prediction_time_ms', 'recommendation']
                
                if all(field in result for field in required_fields):
                    response.success()
                    
                    # Check if prediction time is reasonable
                    if result['prediction_time_ms'] > 1000:  # > 1 second
                        response.failure("Prediction time too slow")
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def predict_with_explanation(self):
        """Test prediction with explanation (less common, more resource intensive)"""
        transaction = self.generate_transaction()
        
        with self.client.post("/predict/explain",
                             json=transaction,
                             catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                required_fields = ['is_fraud', 'fraud_probability', 'explanation', 'top_risk_factors']
                
                if all(field in result for field in required_fields):
                    response.success()
                    
                    # Check if explanation is provided
                    if not result['explanation'] or not result['top_risk_factors']:
                        response.failure("Empty explanation provided")
                else:
                    response.failure("Invalid explanation response structure")
            elif response.status_code == 400:
                # Explanation might not be available for all models
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def predict_batch_small(self):
        """Test small batch prediction"""
        batch_size = random.randint(5, 20)
        transactions = [self.generate_transaction() for _ in range(batch_size)]
        
        with self.client.post("/predict/batch",
                             json=transactions,
                             catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                expected_fields = ['predictions', 'total_transactions', 'processing_time_ms']
                
                if all(field in result for field in expected_fields):
                    if result['total_transactions'] == batch_size:
                        response.success()
                    else:
                        response.failure("Batch size mismatch")
                else:
                    response.failure("Invalid batch response structure")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def predict_batch_large(self):
        """Test larger batch prediction (less frequent)"""
        batch_size = random.randint(50, 100)
        transactions = [self.generate_transaction() for _ in range(batch_size)]
        
        with self.client.post("/predict/batch",
                             json=transactions,
                             catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                
                # Check processing time for large batches
                if result.get('processing_time_ms', 0) > 30000:  # > 30 seconds
                    response.failure("Batch processing too slow")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Test model information endpoint"""
        with self.client.get("/models", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                required_fields = ['available_models', 'model_performance', 'default_model']
                
                if all(field in result for field in required_fields):
                    response.success()
                else:
                    response.failure("Invalid models info response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure("API not healthy")
            else:
                response.failure(f"HTTP {response.status_code}")


class HighVolumeUser(HttpUser):
    """User simulating high-volume fraud detection scenarios"""
    
    wait_time = between(0.1, 0.5)  # Very frequent requests
    
    @task
    def rapid_predictions(self):
        """Rapid-fire predictions simulating real-time fraud detection"""
        transaction = {f'V{i}': np.random.normal(0, 1) for i in range(1, 29)}
        transaction['Amount'] = random.uniform(10, 500)
        
        with self.client.post("/predict", 
                             json=transaction,
                             catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                # In high-volume scenarios, speed is critical
                if result.get('prediction_time_ms', 0) > 100:  # > 100ms
                    response.failure("Too slow for high-volume scenario")
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


# Custom load test scenarios
class StressTestUser(HttpUser):
    """User for stress testing edge cases"""
    
    wait_time = between(0.5, 2)
    
    @task(5)
    def normal_prediction(self):
        """Normal prediction requests"""
        transaction = {f'V{i}': random.uniform(-3, 3) for i in range(1, 29)}
        transaction['Amount'] = random.uniform(1, 1000)
        
        self.client.post("/predict", json=transaction)
    
    @task(2)
    def edge_case_values(self):
        """Test with edge case values"""
        transaction = {f'V{i}': random.choice([-100, 0, 100]) for i in range(1, 29)}
        transaction['Amount'] = random.choice([0.01, 10000, 50000])
        
        self.client.post("/predict", json=transaction)
    
    @task(1)
    def maximum_batch(self):
        """Test maximum allowed batch size"""
        batch_size = 1000  # Maximum allowed
        transactions = []
        
        for _ in range(batch_size):
            transaction = {f'V{i}': np.random.normal(0, 1) for i in range(1, 29)}
            transaction['Amount'] = random.uniform(10, 500)
            transactions.append(transaction)
        
        with self.client.post("/predict/batch",
                             json=transactions,
                             catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Max batch failed: HTTP {response.status_code}")


# Usage instructions:
"""
To run load tests:

1. Install locust:
   pip install locust

2. Start the fraud detection API:
   python api/app.py

3. Run basic load test:
   locust -f benchmarks/load_test.py --host=http://localhost:8000

4. Run specific user class:
   locust -f benchmarks/load_test.py HighVolumeUser --host=http://localhost:8000

5. Run headless with specific parameters:
   locust -f benchmarks/load_test.py --host=http://localhost:8000 --users 50 --spawn-rate 2 --run-time 5m --headless

6. Generate reports:
   locust -f benchmarks/load_test.py --host=http://localhost:8000 --users 100 --spawn-rate 5 --run-time 10m --headless --html report.html

Load test scenarios:
- FraudDetectionUser: Normal usage patterns
- HighVolumeUser: High-frequency trading/payment processing
- StressTestUser: Edge cases and maximum load testing
"""