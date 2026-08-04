import pytest


@pytest.mark.synthetic
def test_synthetic_suite_is_discoverable() -> None:
    """Protect the intended location and marker for generated-data tests."""

    assert True
