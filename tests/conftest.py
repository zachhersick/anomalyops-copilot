import os
import pytest


@pytest.fixture
def test_database_url() -> str:
    database_url = os.getenv(
        "ANOMALYOPS_TEST_DATABASE_URL"
    )
    
    if database_url is None:
        pytest.skip(
            "ANOMALYOPS_TEST_DATABASE_URL is not configured"
        )
        
    return database_url
    
    