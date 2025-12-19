"""Collection helper utilities."""

from typing import List, Any, Dict


def flatten(nested_list: List) -> List:
    """Flatten a nested list one level.
    
    Args:
        nested_list: List potentially containing sublists
        
    Returns:
        Flattened list
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def unique(items: List) -> List:
    """Return unique items preserving order.
    
    Args:
        items: List with potential duplicates
        
    Returns:
        List with duplicates removed
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
