"""
Utility functions for Unified Field Theory

Provides safe conversion utilities for JSON serialization and other common operations.
"""

import numpy as np
from typing import Any, Union, Dict, List

def to_python(obj: Any) -> Any:
    """
    Convert NumPy types to Python types for safe JSON serialization
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with all NumPy types converted to Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    else:
        return obj

def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON, handling NumPy types
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string representation
    """
    import json
    
    # Convert NumPy types to Python types
    python_obj = to_python(obj)
    
    # Serialize to JSON
    return json.dumps(python_obj, sort_keys=True, indent=2)

def validate_field_state(field_state: Any) -> bool:
    """
    Validate that a field state can be safely serialized
    
    Args:
        field_state: Field state object to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Try to serialize
        _ = safe_json_serialize(field_state)
        return True
    except Exception:
        return False 