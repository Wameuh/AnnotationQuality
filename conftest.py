import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)


# Define test collections
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers",
                            "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
