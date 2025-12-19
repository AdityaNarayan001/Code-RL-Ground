"""File handling utilities."""

import os
from pathlib import Path


def read_file(filepath: str) -> str:
    """Read entire file contents.
    
    Args:
        filepath: Path to file
        
    Returns:
        File contents as string
    """
    with open(filepath, 'r') as f:
        return f.read()


def write_file(filepath: str, content: str) -> None:
    """Write content to file.
    
    Args:
        filepath: Path to file
        content: Content to write
    """
    with open(filepath, 'w') as f:
        f.write(content)
