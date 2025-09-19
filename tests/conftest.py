import os
import sys

import pytest

from config import setup_logger

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = setup_logger(__name__, "GREEN")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    logger.info("Setting up test environment...")
    yield
    logger.info("Tearing down test environment...")
