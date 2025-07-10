"""
Performance benchmarking suite for fraud detection system
"""

import time
import json
import requests
import asyncio
import statistics
import concurrent.futures
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from fraud_detector import FraudDetectionSystem


class FraudDetectionBenchmark:
    """Comprehensive benchmarking suite for fraud detection system"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results = {}
        
    def generate_test_transaction(self) -> Dict[str, float]:
        """Generate a synthetic transaction for testing"""
        transaction = {}
        
        # Generate V1-V28 (PCA features)
        for i in range(1, 29):
            transaction[f'V{i}'] = np.random.normal(0, 1)
        
        # Generate Amount
        transaction['Amount'] = np.random.exponential(50)
        
        return transaction
    
    def generate_test_batch(self, size: int = 100) -> List[Dict[str, float]]:
        """Generate a batch of test transactions"""
        return [self.generate_test_transaction() for _ in range(size)]
    
    def benchmark_api_single_prediction(self, num_requests: int = 1000) -> Dict[str, Any]:
        """Benchmark single prediction API endpoint"""
        print(f"Benchmarking single prediction API ({num_requests} requests)...")
        
        response_times = []
        success_count = 0
        error_count = 0
        
        transaction = self.generate_test_transaction()
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                req_start = time.time()
                response = requests.post(
                    f"{self.api_base_url}/predict",
                    json=transaction,
                    timeout=10
                )
                req_end = time.time()
                
                if response.status_code == 200:
                    success_count += 1
                    response_times.append((req_end - req_start) * 1000)  # Convert to ms
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"Request {i} failed: {e}")
        
        total_time = time.time() - start_time
        
        if response_times:
            results = {
                'total_requests': num_requests,
                'successful_requests': success_count,
                'failed_requests': error_count,
                'success_rate': success_count / num_requests,
                'total_time_seconds': total_time,
                'requests_per_second': success_count / total_time,
                'avg_response_time_ms': statistics.mean(response_times),
                'median_response_time_ms': statistics.median(response_times),
                'p95_response_time_ms': np.percentile(response_times, 95),
                'p99_response_time_ms': np.percentile(response_times, 99),
                'min_response_time_ms': min(response_times),
                'max_response_time_ms': max(response_times),
                'std_response_time_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        else:
            results = {
                'total_requests': num_requests,
                'successful_requests': 0,
                'failed_requests': error_count,
                'success_rate': 0,
                'error': 'No successful requests'
            }
        
        self.results['api_single_prediction'] = results
        return results
    
    def benchmark_api_batch_prediction(self, batch_sizes: List[int] = [10, 50, 100, 500]) -> Dict[str, Any]:
        """Benchmark batch prediction API endpoint"""
        print("Benchmarking batch prediction API...")
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            batch_transactions = self.generate_test_batch(batch_size)
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/predict/batch",
                    json=batch_transactions,
                    timeout=30
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    response_data = response.json()
                    total_time_ms = (end_time - start_time) * 1000
                    
                    batch_results[batch_size] = {
                        'batch_size': batch_size,
                        'total_time_ms': total_time_ms,
                        'time_per_transaction_ms': total_time_ms / batch_size,
                        'transactions_per_second': batch_size / (total_time_ms / 1000),
                        'fraud_detected': response_data.get('fraud_detected', 0),
                        'fraud_rate': response_data.get('fraud_detected', 0) / batch_size,
                        'success': True
                    }
                else:
                    batch_results[batch_size] = {
                        'batch_size': batch_size,
                        'success': False,
                        'error': f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                batch_results[batch_size] = {
                    'batch_size': batch_size,
                    'success': False,
                    'error': str(e)
                }
        
        self.results['api_batch_prediction'] = batch_results
        return batch_results
    
    def benchmark_concurrent_requests(self, num_workers: int = 10, requests_per_worker: int = 100) -> Dict[str, Any]:
        """Benchmark concurrent API requests"""
        print(f"Benchmarking concurrent requests ({num_workers} workers, {requests_per_worker} requests each)...")
        
        def make_requests(worker_id: int) -> Dict[str, Any]:
            """Make requests for a single worker"""
            transaction = self.generate_test_transaction()
            response_times = []
            success_count = 0
            
            for _ in range(requests_per_worker):
                try:
                    start_time = time.time()
                    response = requests.post(
                        f"{self.api_base_url}/predict",
                        json=transaction,
                        timeout=10
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        success_count += 1
                        response_times.append((end_time - start_time) * 1000)
                        
                except Exception:
                    pass
            
            return {
                'worker_id': worker_id,
                'success_count': success_count,
                'response_times': response_times
            }
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_requests, i) for i in range(num_workers)]
            worker_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Aggregate results
        total_requests = num_workers * requests_per_worker
        total_successful = sum(result['success_count'] for result in worker_results)
        all_response_times = []
        for result in worker_results:
            all_response_times.extend(result['response_times'])
        
        if all_response_times:
            concurrent_results = {
                'num_workers': num_workers,
                'requests_per_worker': requests_per_worker,
                'total_requests': total_requests,
                'successful_requests': total_successful,
                'success_rate': total_successful / total_requests,
                'total_time_seconds': total_time,
                'requests_per_second': total_successful / total_time,
                'avg_response_time_ms': statistics.mean(all_response_times),
                'p95_response_time_ms': np.percentile(all_response_times, 95),
                'p99_response_time_ms': np.percentile(all_response_times, 99)
            }
        else:
            concurrent_results = {
                'num_workers': num_workers,
                'requests_per_worker': requests_per_worker,
                'total_requests': total_requests,
                'successful_requests': 0,
                'success_rate': 0,
                'error': 'No successful requests'
            }
        
        self.results['concurrent_requests'] = concurrent_results
        return concurrent_results
    
    def benchmark_model_training(self, sample_sizes: List[int] = [1000, 5000, 10000]) -> Dict[str, Any]:
        """Benchmark model training performance"""
        print("Benchmarking model training performance...")
        
        training_results = {}
        
        for sample_size in sample_sizes:
            print(f"  Testing sample size: {sample_size}")
            
            # Generate synthetic dataset
            np.random.seed(42)
            data = {
                'Time': np.random.uniform(0, 172800, sample_size),
                'Amount': np.random.exponential(50, sample_size),
                'Class': np.random.choice([0, 1], sample_size, p=[0.998, 0.002])
            }
            
            for i in range(1, 29):
                data[f'V{i}'] = np.random.normal(0, 1, sample_size)
            
            df = pd.DataFrame(data)
            
            try:
                fraud_detector = FraudDetectionSystem(random_state=42)
                
                # Benchmark preprocessing
                preprocess_start = time.time()
                X_train, X_test, y_train, y_test = fraud_detector.preprocess_data(df)
                preprocess_time = time.time() - preprocess_start
                
                # Benchmark training
                training_start = time.time()
                models = fraud_detector.train_models(X_train, y_train)
                training_time = time.time() - training_start
                
                # Benchmark evaluation
                eval_start = time.time()
                performance = fraud_detector.evaluate_models(X_test, y_test)
                eval_time = time.time() - eval_start
                
                # Get best model performance
                best_model = max(performance.keys(), key=lambda x: performance[x]['AUC'])
                best_auc = performance[best_model]['AUC']
                
                training_results[sample_size] = {
                    'sample_size': sample_size,
                    'preprocess_time_seconds': preprocess_time,
                    'training_time_seconds': training_time,
                    'evaluation_time_seconds': eval_time,
                    'total_time_seconds': preprocess_time + training_time + eval_time,
                    'best_model': best_model,
                    'best_auc': best_auc,
                    'models_trained': list(models.keys()),
                    'success': True
                }
                
            except Exception as e:
                training_results[sample_size] = {
                    'sample_size': sample_size,
                    'success': False,
                    'error': str(e)
                }
        
        self.results['model_training'] = training_results
        return training_results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("=" * 60)
        print("FRAUD DETECTION SYSTEM PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Check if API is available
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"API health check failed: {response.status_code}")
                print("Skipping API benchmarks")
                api_available = False
            else:
                print("✓ API is available")
                api_available = True
        except Exception as e:
            print(f"API not available: {e}")
            print("Skipping API benchmarks")
            api_available = False
        
        # Run benchmarks
        if api_available:
            self.benchmark_api_single_prediction(1000)
            self.benchmark_api_batch_prediction([10, 50, 100])
            self.benchmark_concurrent_requests(5, 50)
        
        self.benchmark_model_training([1000, 5000])
        
        return self.results
    
    def generate_report(self, output_file: str = "benchmark_results.json"):
        """Generate detailed benchmark report"""
        timestamp = datetime.now().isoformat()
        
        report = {
            'timestamp': timestamp,
            'benchmark_version': '1.0.0',
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            },
            'results': self.results
        }
        
        # Save JSON report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary
        self._print_summary()
        
        return report
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        if 'api_single_prediction' in self.results:
            api_results = self.results['api_single_prediction']
            print(f"\n📊 API Single Prediction:")
            print(f"  Success Rate: {api_results.get('success_rate', 0):.1%}")
            print(f"  Requests/sec: {api_results.get('requests_per_second', 0):.1f}")
            print(f"  Avg Response: {api_results.get('avg_response_time_ms', 0):.1f}ms")
            print(f"  P95 Response: {api_results.get('p95_response_time_ms', 0):.1f}ms")
        
        if 'api_batch_prediction' in self.results:
            batch_results = self.results['api_batch_prediction']
            print(f"\n📦 API Batch Prediction:")
            for batch_size, result in batch_results.items():
                if result.get('success'):
                    print(f"  Batch {batch_size}: {result.get('time_per_transaction_ms', 0):.1f}ms/txn")
        
        if 'concurrent_requests' in self.results:
            concurrent_results = self.results['concurrent_requests']
            print(f"\n🔀 Concurrent Requests:")
            print(f"  Success Rate: {concurrent_results.get('success_rate', 0):.1%}")
            print(f"  Requests/sec: {concurrent_results.get('requests_per_second', 0):.1f}")
        
        if 'model_training' in self.results:
            training_results = self.results['model_training']
            print(f"\n🤖 Model Training:")
            for sample_size, result in training_results.items():
                if result.get('success'):
                    print(f"  {sample_size} samples: {result.get('training_time_seconds', 0):.1f}s, AUC: {result.get('best_auc', 0):.3f}")
        
        print("\n" + "=" * 60)


def main():
    """Main benchmark execution"""
    benchmark = FraudDetectionBenchmark()
    
    # Run benchmarks
    results = benchmark.run_full_benchmark()
    
    # Generate report
    report = benchmark.generate_report("benchmark_results.json")
    
    print(f"\n📄 Full report saved to: benchmark_results.json")
    
    return results


if __name__ == "__main__":
    main()