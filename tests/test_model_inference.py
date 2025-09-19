import gc
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import pynvml
import pytest
import torch
from fastapi.testclient import TestClient

from config import DATA_PATH, setup_logger
from data.utils import load_data
from main import app
from training.evaluate import inference

logger = setup_logger(__name__, "MAGENTA")


class ModelStressMetrics:
    def __init__(self):
        self.latencies = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_memory_usage = []
        self.gpu_utilization = []
        self.errors = 0
        self.total_requests = 0
        self.predictions = []
        self.confidences = []
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self.has_gpu = torch.cuda.is_available()

    def start_monitoring(self):
        self.start_time = time.time()
        gc.collect()

    def end_monitoring(self):
        self.end_time = time.time()

    def record_request(
        self, latency: float, success: bool, prediction: str = None, confidence: float = None
    ):
        self.latencies.append(latency)
        self.total_requests += 1

        if success:
            if prediction:
                self.predictions.append(prediction)
            if confidence is not None:
                self.confidences.append(confidence)
        else:
            self.errors += 1

        self.memory_usage.append(self.process.memory_info().rss / 1024 / 1024)
        self.cpu_usage.append(psutil.cpu_percent())

        if self.has_gpu:
            try:
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                self.gpu_memory_usage.append(gpu_memory)

                try:

                    if not hasattr(self, '_nvml_initialized'):
                        pynvml.nvmlInit()
                        self._nvml_initialized = True
                        self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    gpu_util = gpu_util_info.gpu
                    self.gpu_utilization.append(gpu_util)
                except (ImportError, Exception):
                    self.gpu_utilization.append(0)

            except Exception:
                self.gpu_memory_usage.append(0)
                self.gpu_utilization.append(0)

    def get_comprehensive_metrics(self) -> dict:
        if not self.latencies:
            return {"error": "No successful requests recorded"}

        duration = self.end_time - self.start_time if self.end_time and self.start_time else 1

        return {
            "total_requests": self.total_requests,
            "successful_requests": len(self.latencies),
            "error_count": self.errors,
            "error_rate": self.errors / max(self.total_requests, 1),
            "duration": duration,
            "throughput": len(self.latencies) / duration,
            "avg_latency": statistics.mean(self.latencies),
            "median_latency": statistics.median(self.latencies),
            "min_latency": min(self.latencies),
            "max_latency": max(self.latencies),
            "p50_latency": (statistics.median(self.latencies) if self.latencies else 0),
            "p95_latency": (
                statistics.quantiles(self.latencies, n=20)[18]
                if len(self.latencies) >= 20
                else max(self.latencies)
            ),
            "avg_memory_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "peak_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            "peak_cpu_percent": max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_gpu_memory_mb": (
                statistics.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0
            ),
            "peak_gpu_memory_mb": max(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            "avg_gpu_utilization": (
                statistics.mean(self.gpu_utilization) if self.gpu_utilization else 0
            ),
            "peak_gpu_utilization": max(self.gpu_utilization) if self.gpu_utilization else 0,
            "has_gpu": self.has_gpu,
            "avg_confidence": statistics.mean(self.confidences) if self.confidences else 0,
            "prediction_distribution": (
                {pred: self.predictions.count(pred) for pred in set(self.predictions)}
                if self.predictions
                else {}
            ),
        }


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def stress_test_data():
    test_data = []

    test_path = os.path.join(DATA_PATH, "test.csv")
    if os.path.exists(test_path):
        try:
            df = load_data(test_path, is_raw=False)
            real_texts = df['text'].dropna().head(100).tolist()
            test_data.extend(real_texts)
        except Exception as e:
            logger.warning(f"Could not load real test data: {e}")

    logger.info(f"Total test cases: {len(test_data)}")
    return test_data


def make_stress_request(client: TestClient, text: str) -> tuple[float, bool, str, float]:
    start_time = time.time()
    try:
        response = client.post("/evaluate?verbose=false", json={"text": text})
        latency = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            return latency, True, data.get("prediction", ""), data.get("confidence", 0.0)
        else:
            return latency, False, "", 0.0

    except Exception as e:
        latency = time.time() - start_time
        return latency, False, "", 0.0


class TestModelInference:
    def test_basic_inference(self):
        logger.info("=== Starting Basic Inference Test ===")
        test_text = "This is a test message"
        try:
            sentiment, confidence, cleaned = inference(test_text)

            assert sentiment in ["Positive", "Negative"], f"Invalid sentiment: {sentiment}"
            assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
            assert isinstance(cleaned, str), "Cleaned text should be string"

        except Exception as e:
            pytest.fail(f"Basic inference failed: {e}")

        logger.info("=== Basic Inference Test Completed ===")

    def test_model_loading(self):
        logger.info("=== Starting Model Loading Test ===")
        for i in range(3):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                test_text = "This is a test message"
                sentiment, confidence, cleaned = inference(test_text)

                assert sentiment in [
                    "Positive",
                    "Negative",
                ], f"Invalid sentiment on iteration {i+1}"
                assert 0 <= confidence <= 1, f"Invalid confidence on iteration {i+1}"

                logger.info(f"Model loading iteration {i+1}/3: SUCCESS")

            except Exception as e:
                pytest.fail(f"Model loading failed on iteration {i+1}: {e}")

        logger.info("=== Model Loading Test Completed ===")

    def test_concurrent_requests(self, client, stress_test_data):
        logger.info("=== Starting Concurrent Requests Test ===")
        if not stress_test_data:
            pytest.skip("No stress test data available")

        metrics = ModelStressMetrics()
        num_threads = 20
        requests_per_thread = 10
        test_texts = stress_test_data[:20]

        logger.info(
            f"Starting concurrent test: {num_threads} threads, {requests_per_thread} requests each"
        )
        metrics.start_monitoring()

        def worker_thread(thread_id: int):
            for i in range(requests_per_thread):
                text = test_texts[i % len(test_texts)]
                latency, success, prediction, confidence = make_stress_request(client, text)
                metrics.record_request(latency, success, prediction, confidence)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Thread execution failed: {e}")

        metrics.end_monitoring()
        results = metrics.get_comprehensive_metrics()

        logger.info("Concurrent test results:")
        logger.info(f"  Total Requests: {results['total_requests']}")
        logger.info(f"  Successful: {results['successful_requests']}")
        logger.info(f"  Error Rate: {results['error_rate']:.2%}")
        logger.info(f"  Throughput: {results['throughput']:.2f} req/s")
        logger.info(f"  Avg Latency: {results['avg_latency']:.3f}s")
        logger.info(f"  P50 Latency: {results['p50_latency']:.3f}s")
        logger.info(f"  P95 Latency: {results['p95_latency']:.3f}s")
        logger.info(f"  Peak Memory: {results['peak_memory_mb']:.1f} MB")
        logger.info(f"  Peak CPU: {results['peak_cpu_percent']:.1f}%")
        if results['has_gpu']:
            logger.info(f"  Peak GPU Memory: {results['peak_gpu_memory_mb']:.1f} MB")
            logger.info(f"  Peak GPU Utilization: {results['peak_gpu_utilization']:.1f}%")

        assert results['error_rate'] < 0.2, f"Error rate too high: {results['error_rate']:.2%}"
        assert results['throughput'] > 1.0, f"Throughput too low: {results['throughput']:.2f} req/s"

        logger.info("=== Concurrent Requests Test Completed ===")

    def test_prediction_consistency(self, client):
        logger.info("=== Starting Prediction Consistency Test ===")
        test_cases = [
            "I absolutely love this product!",
            "This is terrible, worst purchase ever.",
            "Amazing quality, highly recommend!",
        ]

        for text in test_cases:
            predictions = []
            confidences = []

            for _ in range(5):
                response = client.post("/evaluate?verbose=false", json={"text": text})
                if response.status_code == 200:
                    data = response.json()
                    predictions.append(data["prediction"])
                    confidences.append(data["confidence"])

            if predictions:
                unique_predictions = set(predictions)
                prediction_consistent = len(unique_predictions) == 1

                confidence_variance = (
                    statistics.variance(confidences) if len(confidences) > 1 else 0
                )

                assert prediction_consistent, f"Predictions inconsistent for: {text[:30]}"
                assert confidence_variance < 0.01, f"High confidence variance for: {text[:30]}"

        logger.info("=== Prediction Consistency Test Completed ===")

    def test_stress(self, client, stress_test_data):
        logger.info("=== Starting Stress Test ===")
        if not stress_test_data:
            pytest.skip("No stress test data available")

        metrics = ModelStressMetrics()
        total_requests = 1000
        target_rps = 100
        duration = total_requests / target_rps
        test_texts = stress_test_data[:500]

        logger.info(f"Starting stress test: {total_requests} requests at {target_rps} RPS")
        metrics.start_monitoring()

        def worker_batch(batch_texts):
            for text in batch_texts:
                start = time.time()
                latency, success, prediction, confidence = make_stress_request(client, text)
                metrics.record_request(latency, success, prediction, confidence)
                elapsed = time.time() - start
                sleep_time = max(0, (1.0 / target_rps) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        batch_size = 50
        batches = [test_texts[i : i + batch_size] for i in range(0, len(test_texts), batch_size)]

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(0, total_requests, batch_size):
                batch_texts = [
                    test_texts[j % len(test_texts)]
                    for j in range(i, min(i + batch_size, total_requests))
                ]
                futures.append(executor.submit(worker_batch, batch_texts))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Batch execution failed: {e}")

        metrics.end_monitoring()
        results = metrics.get_comprehensive_metrics()

        logger.info("Stress test results:")
        logger.info(f"  Total Requests: {results['total_requests']}")
        logger.info(f"  Successful: {results['successful_requests']}")
        logger.info(f"  Error Rate: {results['error_rate']:.2%}")
        logger.info(f"  Throughput: {results['throughput']:.2f} req/s")
        logger.info(f"  Avg Latency: {results['avg_latency']:.3f}s")
        logger.info(f"  P50 Latency: {results['p50_latency']:.3f}s")
        logger.info(f"  P95 Latency: {results['p95_latency']:.3f}s")
        logger.info(f"  Peak Memory: {results['peak_memory_mb']:.1f} MB")
        logger.info(f"  Peak CPU: {results['peak_cpu_percent']:.1f}%")
        if results['has_gpu']:
            logger.info(f"  Peak GPU Memory: {results['peak_gpu_memory_mb']:.1f} MB")
            logger.info(f"  Peak GPU Utilization: {results['peak_gpu_utilization']:.1f}%")

        assert results['error_rate'] < 0.1, f"Error rate too high: {results['error_rate']:.2%}"
        assert (
            results['throughput'] > 50.0
        ), f"Throughput too low: {results['throughput']:.2f} req/s"
        assert results['p95_latency'] < 2.0, f"P95 latency too high: {results['p95_latency']:.3f}s"

        logger.info("=== Stress Test Completed ===")

    def test_malformed_input(self, client):
        logger.info("=== Starting Malformed Input Test ===")
        test_cases = [
            {"text": ""},
            {"text": "   "},
            {"invalid_field": "test"},
            {},
        ]

        for i, test_case in enumerate(test_cases):
            response = client.post("/evaluate?verbose=false", json=test_case)
            assert (
                response.status_code == 422
            ), f"Test case {i+1} should return 422, got {response.status_code}"

        non_json_response = client.post("/evaluate?verbose=false", data="invalid json")
        assert non_json_response.status_code == 422, "Non-JSON input should return 422"

        logger.info("=== Malformed Input Test Completed ===")
