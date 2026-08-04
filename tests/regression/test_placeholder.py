import pytest


@pytest.mark.regression
def test_regression_suite_is_discoverable() -> None:
    """Protect the intended location and marker for future real-data tests."""

    assert True
