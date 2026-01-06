"""Module for the utility functions."""

from os import getcwd, path

from dotenv import load_dotenv
from torch import cuda, device
from utils import get_data, get_device


def test_get_device() -> None:
    """Test the get_device function.

    This makes sure that if the function is changed, the tests verifies that
    the function returns the expected device.
    """
    expected_device = (
        device(device="cuda") if cuda.is_available() else device(device="cpu")
    )
    assert get_device() == expected_device


def test_get_data() -> None:
    """Test the get_data function.

    This makes sure that if the function is changed, the tests verifies that
    the function returns the expected dataset location.
    """
    load_dotenv()

    dataset = get_data()

    expected_location = path.join(getcwd(), "data")

    assert path.normpath(dataset.location) == path.normpath(expected_location), (
        "The dataset location is not correct."
    )
