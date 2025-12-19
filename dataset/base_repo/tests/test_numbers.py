"""Tests for number utilities."""

import pytest
from pyutils.numbers import is_prime, factorial


def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(17) == True
    assert is_prime(1) == False
    assert is_prime(4) == False


def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120
    with pytest.raises(ValueError):
        factorial(-1)
