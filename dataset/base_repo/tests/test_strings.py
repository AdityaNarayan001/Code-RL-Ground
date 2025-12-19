"""Tests for string utilities."""

import pytest
from pyutils.strings import reverse_string, capitalize_words


def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"


def test_capitalize_words():
    assert capitalize_words("hello world") == "Hello World"
    assert capitalize_words("HELLO") == "Hello"
