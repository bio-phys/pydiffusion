import pytest

from pydiffusion.util.testing import TestDataDir


@pytest.fixture
def data(request):
    """access test directory in a pytest. This works independent of where tests are
    started"""
    return TestDataDir(request.fspath.dirname, "data")
