import os
import sys
import pytest

# Add root directory to PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

def pytest_configure(config):
    """Initial pytest configuration"""
    # You can add global pytest configurations here
    pass

# Define test suites with full paths from root
test_suites = {
    'logger': ['Tests/UnitTests/test_logger.py'],  # Full path from root
    'all': ['Tests/UnitTests/test_logger.py']  # Add other test files here when available
}

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--suite",
        action="store",
        default="all",
        help="Name of the test suite to run: " + ", ".join(test_suites.keys())
    )

@pytest.fixture(scope='session')
def test_suite(request):
    """Fixture to get the selected test suite"""
    suite_name = request.config.getoption("--suite")
    if suite_name not in test_suites:
        raise ValueError(f"Unknown test suite: {suite_name}")
    return test_suites[suite_name]

def pytest_collection_modifyitems(session, config, items):
    """Modify test collection based on selected suite"""
    suite_name = config.getoption("--suite")
    if suite_name == "all":
        return  # Run all tests
    
    selected_tests = test_suites[suite_name]
    selected_items = []
    deselected_items = []
    
    for item in items:
        # Get relative path from project root
        file_path = os.path.relpath(str(item.fspath), root_dir)
        file_path = file_path.replace('\\', '/')  # Normalize paths for Windows
        
        if any(test.replace('\\', '/') in file_path for test in selected_tests):
            selected_items.append(item)
        else:
            deselected_items.append(item)
    
    items[:] = selected_items
    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items) 