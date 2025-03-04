import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Define test collections
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers",
                            "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
