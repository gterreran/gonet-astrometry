import pytest


@pytest.mark.integration
def test_integration_suite_is_discoverable() -> None:
    """Protect the intended location and marker for component-level tests."""

    assert True
