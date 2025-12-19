"""Input validation functions."""

import re


def is_valid_email(email: str) -> bool:
    """Validate email format.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
