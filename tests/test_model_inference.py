import gc
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import psutil
import pytest
import torch
from fastapi.testclient import TestClient

from config import DATA_PATH, setup_logger
from main import app
from training.evaluate import inference

logger = setup_logger(__name__, "MAGENTA")


class ModelStressMetrics:
    def __init__(self):
        self.latencies = []
        self.memory_usage = []
        self.cpu_usage = []
        self.errors = 0
        self.total_requests = 0
        self.predictions = []
        self.confidences = []
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()

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
        self.cpu_usage.append(self.process.cpu_percent())

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
            "p95_latency": (
                statistics.quantiles(self.latencies, n=20)[18]
                if len(self.latencies) >= 20
                else max(self.latencies)
            ),
            "avg_memory_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "peak_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_percent": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            "peak_cpu_percent": max(self.cpu_usage) if self.cpu_usage else 0,
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

    # Try to load real test data
    test_path = os.path.join(DATA_PATH, "test.csv")
    if os.path.exists(test_path):
        try:
            df = pd.read_csv(
                test_path,
                names=["target", "ids", "date", "flag", "user", "text"],
                encoding="ISO-8859-1",
            )
            real_texts = df['text'].dropna().head(100).tolist()
            test_data.extend(real_texts)
        except Exception as e:
            logger.warning(f"Could not load real test data: {e}")

    # Add synthetic test cases
    synthetic_cases = [
        "I love this product!",
        "This is terrible quality.",
        "It's okay, nothing special.",
        "Amazing service, highly recommend!",
        "Worst experience ever, completely disappointed.",
        "Perfect! Exactly what I needed!",
        "Hate this so much, very disappointed",
        "Could be better but not bad overall",
    ]

    test_data.extend(synthetic_cases)
    logger.info(f"Total test cases: {len(test_data)}")
    return test_data


def make_stress_request(client: TestClient, text: str) -> tuple[float, bool, str, float]:
    start_time = time.time()
    try:
        response = client.post("/evaluate", json={"text": text})
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
        """Test basic inference functionality"""
        test_text = "This is a test message"
        try:
            sentiment, confidence, cleaned = inference(test_text)

            assert sentiment in ["Positive", "Negative"], f"Invalid sentiment: {sentiment}"
            assert 0 <= confidence <= 1, f"Invalid confidence: {confidence}"
            assert isinstance(cleaned, str), "Cleaned text should be string"

            logger.info("Basic inference test: SUCCESS")
            logger.info(f"Input: {test_text}")
            logger.info(f"Output: {sentiment} (confidence: {confidence})")

        except Exception as e:
            pytest.fail(f"Basic inference failed: {e}")

    def test_model_loading_reliability(self):
        """Test repeated model loading"""
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

    def test_concurrent_requests(self, client, stress_test_data):
        """Test concurrent API requests"""
        if not stress_test_data:
            pytest.skip("No stress test data available")

        metrics = ModelStressMetrics()
        num_threads = 10
        requests_per_thread = 5
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

        assert results['error_rate'] < 0.2, f"Error rate too high: {results['error_rate']:.2%}"
        assert results['throughput'] > 1.0, f"Throughput too low: {results['throughput']:.2f} req/s"

    def test_edge_cases(self, client):
        """Test edge case handling"""
        edge_cases = [
            "",
            " ",
            "a",
            "no",
            "yes",
            "This is a very long sentence. " * 20,
            "!@#$%^&*()",
            "   mixed   spaces   ",
            "ALLCAPS vs lowercase",
        ]

        successful_cases = 0
        total_cases = len(edge_cases)

        for i, text in enumerate(edge_cases):
            try:
                response = client.post("/evaluate", json={"text": text})

                if response.status_code == 200:
                    data = response.json()
                    assert "text" in data
                    assert "prediction" in data
                    assert "confidence" in data
                    assert data["prediction"] in ["positive", "negative"]
                    assert 0 <= data["confidence"] <= 1
                    successful_cases += 1
                elif response.status_code == 400:
                    # Bad request is acceptable for some edge cases
                    successful_cases += 1

            except Exception as e:
                logger.warning(f"Edge case {i+1} failed: {e}")

        success_rate = successful_cases / total_cases
        logger.info(f"Edge case robustness: {success_rate:.1%}")
        assert success_rate >= 0.8, f"Edge case handling too poor: {success_rate:.1%}"

    def test_prediction_consistency(self, client):
        """Test prediction consistency across multiple calls"""
        test_cases = [
            "I absolutely love this product!",
            "This is terrible, worst purchase ever.",
            "Amazing quality, highly recommend!",
        ]

        for text in test_cases:
            predictions = []
            confidences = []

            # Make multiple requests for the same text
            for _ in range(5):
                response = client.post("/evaluate", json={"text": text})
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

                logger.info(f"Text: {text[:30]}...")
                logger.info(f"  Consistent: {prediction_consistent}")
                logger.info(f"  Confidence variance: {confidence_variance:.6f}")

                assert prediction_consistent, f"Predictions inconsistent for: {text[:30]}"
                assert confidence_variance < 0.01, f"High confidence variance for: {text[:30]}"
