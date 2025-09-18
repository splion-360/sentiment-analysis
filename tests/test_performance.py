import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config import DATA_PATH, MAX_LENGTH, TEST_FILE, setup_logger
from main import app

logger = setup_logger(__name__, "CYAN")


class PerformanceMetrics:
    def __init__(self):
        self.latencies = []
        self.errors = 0
        self.total_requests = 0
        self.start_time = None
        self.end_time = None

    def add_latency(self, latency: float):
        self.latencies.append(latency)
        self.total_requests += 1

    def add_error(self):
        self.errors += 1
        self.total_requests += 1

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        self.end_time = time.time()

    def get_metrics(self) -> dict:
        if not self.latencies:
            return {
                "total_requests": self.total_requests,
                "errors": self.errors,
                "error_rate": self.errors / max(self.total_requests, 1),
                "throughput": 0,
                "p50_latency": 0,
                "p95_latency": 0,
                "avg_latency": 0,
            }

        duration = self.end_time - self.start_time if self.end_time and self.start_time else 1
        throughput = len(self.latencies) / duration

        return {
            "total_requests": self.total_requests,
            "successful_requests": len(self.latencies),
            "errors": self.errors,
            "error_rate": self.errors / max(self.total_requests, 1),
            "throughput": throughput,
            "p50_latency": statistics.median(self.latencies),
            "p95_latency": (
                statistics.quantiles(self.latencies, n=20)[18]
                if len(self.latencies) >= 20
                else max(self.latencies)
            ),
            "avg_latency": statistics.mean(self.latencies),
            "min_latency": min(self.latencies),
            "max_latency": max(self.latencies),
            "duration": duration,
        }


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_texts():
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    if not os.path.exists(test_path):
        logger.warning(f"Test data file not found: {test_path}")
        pytest.skip(f"Test data file not found: {test_path}")

    try:
        df = pd.read_csv(
            test_path,
            names=["target", "ids", "date", "flag", "user", "text"],
            encoding="ISO-8859-1",
        )
        texts = df['text'].dropna().head(100).tolist()
        logger.info(f"Loaded {len(texts)} test texts")
        return texts
    except Exception as e:
        logger.error(f"Could not load test data: {e}")
        pytest.skip(f"Could not load test data: {e}")


def make_api_request(client: TestClient, text: str) -> tuple[float, bool]:
    start_time = time.time()
    try:
        response = client.post("/evaluate", json={"text": text})
        latency = time.time() - start_time
        success = response.status_code == 200
        return latency, success
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"API request failed: {e}")
        return latency, False


def test_api_health_check(client):
    logger.info("Testing API health check")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    logger.info("Health check passed")


def test_single_request_latency(client, test_texts):
    if not test_texts:
        pytest.skip("No test texts available")

    text = test_texts[0]
    metrics = PerformanceMetrics()

    logger.info("Testing single request latency")
    for _ in range(10):
        latency, success = make_api_request(client, text)
        if success:
            metrics.add_latency(latency)
        else:
            metrics.add_error()

    results = metrics.get_metrics()

    logger.info("Single Request Performance:")
    logger.info(f"Average Latency: {results['avg_latency']:.3f}s")
    logger.info(f"P50 Latency: {results['p50_latency']:.3f}s")
    logger.info(f"P95 Latency: {results['p95_latency']:.3f}s")
    logger.info(f"Error Rate: {results['error_rate']:.1%}")

    assert results['error_rate'] < 0.1, f"Error rate too high: {results['error_rate']:.1%}"
    assert results['avg_latency'] < 2.0, f"Average latency too high: {results['avg_latency']:.3f}s"


def test_max_length_text_performance(client):
    long_text = "This is a test sentence. " * (MAX_LENGTH // 5)
    metrics = PerformanceMetrics()

    logger.info("Testing max length text performance")
    for _ in range(5):
        latency, success = make_api_request(client, long_text)
        if success:
            metrics.add_latency(latency)
        else:
            metrics.add_error()

    results = metrics.get_metrics()

    logger.info("Max Length Text Performance:")
    logger.info(f"Text length: ~{len(long_text)} characters")
    logger.info(f"Average Latency: {results['avg_latency']:.3f}s")
    logger.info(f"P50 Latency: {results['p50_latency']:.3f}s")
    logger.info(f"P95 Latency: {results['p95_latency']:.3f}s")

    assert (
        results['error_rate'] < 0.2
    ), f"Error rate too high for max length: {results['error_rate']:.1%}"


@pytest.mark.parametrize("batch_size", [8, 16])
def test_throughput_with_batches(client, test_texts, batch_size):
    if not test_texts or len(test_texts) < batch_size:
        pytest.skip(f"Not enough test texts for batch size {batch_size}")

    batch_texts = test_texts[:batch_size]
    metrics = PerformanceMetrics()

    logger.info(f"Testing throughput with batch size: {batch_size}")
    metrics.start_timer()

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(make_api_request, client, text) for text in batch_texts]

        for future in as_completed(futures):
            latency, success = future.result()
            if success:
                metrics.add_latency(latency)
            else:
                metrics.add_error()

    metrics.end_timer()
    results = metrics.get_metrics()

    logger.info(f"Throughput Test (Batch Size: {batch_size}):")
    logger.info(f"Total Requests: {results['total_requests']}")
    logger.info(f"Successful Requests: {results['successful_requests']}")
    logger.info(f"Throughput: {results['throughput']:.2f} req/s")
    logger.info(f"Average Latency: {results['avg_latency']:.3f}s")
    logger.info(f"P50 Latency: {results['p50_latency']:.3f}s")
    logger.info(f"P95 Latency: {results['p95_latency']:.3f}s")
    logger.info(f"Error Rate: {results['error_rate']:.1%}")
    logger.info(f"Duration: {results['duration']:.2f}s")

    assert results['error_rate'] < 0.1, f"Error rate too high: {results['error_rate']:.1%}"
    assert results['throughput'] > 1.0, f"Throughput too low: {results['throughput']:.2f} req/s"


def test_concurrent_load(client, test_texts):
    if not test_texts:
        pytest.skip("No test texts available")

    num_concurrent = 20
    num_requests_per_thread = 5
    test_text = test_texts[0] if test_texts else "This is a test message for sentiment analysis."

    metrics = PerformanceMetrics()
    logger.info(
        f"Testing concurrent load: {num_concurrent} users, {num_requests_per_thread} requests each"
    )
    metrics.start_timer()

    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = []
        for _ in range(num_concurrent):
            for _ in range(num_requests_per_thread):
                futures.append(executor.submit(make_api_request, client, test_text))

        for future in as_completed(futures):
            latency, success = future.result()
            if success:
                metrics.add_latency(latency)
            else:
                metrics.add_error()

    metrics.end_timer()
    results = metrics.get_metrics()

    logger.info("Concurrent Load Test:")
    logger.info(f"Concurrent Users: {num_concurrent}")
    logger.info(f"Requests per User: {num_requests_per_thread}")
    logger.info(f"Total Requests: {results['total_requests']}")
    logger.info(f"Successful Requests: {results['successful_requests']}")
    logger.info(f"Throughput: {results['throughput']:.2f} req/s")
    logger.info(f"Average Latency: {results['avg_latency']:.3f}s")
    logger.info(f"P50 Latency: {results['p50_latency']:.3f}s")
    logger.info(f"P95 Latency: {results['p95_latency']:.3f}s")
    logger.info(f"Error Rate: {results['error_rate']:.1%}")
    logger.info(f"Duration: {results['duration']:.2f}s")

    assert (
        results['error_rate'] < 0.15
    ), f"Error rate too high under load: {results['error_rate']:.1%}"
    assert (
        results['throughput'] > 5.0
    ), f"Throughput too low under load: {results['throughput']:.2f} req/s"


def test_response_accuracy(client, test_texts):
    if not test_texts:
        pytest.skip("No test texts available")

    successful_responses = 0
    total_tests = min(10, len(test_texts))

    logger.info(f"Testing response accuracy with {total_tests} requests")
    for i in range(total_tests):
        response = client.post("/evaluate", json={"text": test_texts[i]})

        if response.status_code == 200:
            data = response.json()
            assert "text" in data
            assert "prediction" in data
            assert "confidence" in data
            assert data["prediction"] in ["positive", "negative", "error"]
            assert 0 <= data["confidence"] <= 1
            successful_responses += 1

    accuracy_rate = successful_responses / total_tests
    logger.info("Response Accuracy Test:")
    logger.info(f"Successful Responses: {successful_responses}/{total_tests}")
    logger.info(f"Accuracy Rate: {accuracy_rate:.1%}")

    assert accuracy_rate >= 0.9, f"Response accuracy too low: {accuracy_rate:.1%}"
