"""Tests for file utilities."""

import pytest
from pyutils.files import read_file, write_file


def test_read_file(tmp_path):
    filepath = tmp_path / "sample.txt"
    filepath.write_text("hello world")
    assert read_file(str(filepath)) == "hello world"


def test_read_file_empty(tmp_path):
    filepath = tmp_path / "empty.txt"
    filepath.write_text("")
    assert read_file(str(filepath)) == ""


def test_write_file(tmp_path):
    filepath = tmp_path / "output.txt"
    write_file(str(filepath), "some content")
    assert filepath.read_text() == "some content"


def test_write_file_overwrites(tmp_path):
    filepath = tmp_path / "output.txt"
    write_file(str(filepath), "first")
    write_file(str(filepath), "second")
    assert filepath.read_text() == "second"


def test_write_then_read_roundtrip(tmp_path):
    filepath = tmp_path / "roundtrip.txt"
    write_file(str(filepath), "line1\nline2\n")
    assert read_file(str(filepath)) == "line1\nline2\n"
