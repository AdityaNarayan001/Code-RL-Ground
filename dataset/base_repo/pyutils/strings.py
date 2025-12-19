"""String manipulation utilities."""


def reverse_string(s: str) -> str:
    """Reverse a string.
    
    Args:
        s: Input string to reverse
        
    Returns:
        Reversed string
    """
    return s[::-1]


def capitalize_words(s: str) -> str:
    """Capitalize first letter of each word.
    
    Args:
        s: Input string
        
    Returns:
        String with capitalized words
    """
    return ' '.join(word.capitalize() for word in s.split())
