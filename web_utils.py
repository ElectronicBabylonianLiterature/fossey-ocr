#!/usr/bin/env python3
"""
Web Application Utilities
Helper functions for file handling, data conversion, and web-specific operations.
"""

import os
import re
import json
import numpy as np
from typing import Dict, Any, Union
from werkzeug.utils import secure_filename


# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


def allowed_file(filename: str) -> bool:
    """
    Check if a filename has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types recursively into Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (can be nested dict, list, etc.)
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def extract_page_info_from_filename(filename: str) -> tuple:
    """
    Extract page number and suffix from filename using regex.
    
    Args:
        filename: Input filename (e.g., "Fossey_00009b.png")
        
    Returns:
        Tuple of (page_number, suffix) where page_number is int and suffix is str
    """
    page_number = ""
    suffix = ""
    
    base_filename = os.path.basename(filename)
    match = re.match(r"Fossey_(\d+)([a-zA-Z]*)", base_filename)
    if match:
        page_number = int(match.group(1))
        suffix = match.group(2)
    
    return page_number, suffix


def create_session_data(filename: str, timestamp: str, result: Dict) -> Dict:
    """
    Create session data structure for web interface.
    
    Args:
        filename: Original filename
        timestamp: Processing timestamp
        result: Processing result dictionary
        
    Returns:
        Session data dictionary
    """
    result_serializable = convert_numpy_types(result)
    return {
        'filename': filename,
        'timestamp': timestamp,
        'result': result_serializable
    }


def save_session_file(session_data: Dict, output_dir: str) -> str:
    """
    Save session data to JSON file.
    
    Args:
        session_data: Session data dictionary
        output_dir: Output directory
        
    Returns:
        Path to saved session file
    """
    session_file = os.path.join(output_dir, 'session.json')
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    return session_file


def get_suffix_from_filename(filename: str) -> str:
    """
    Determine suffix ('a' or 'b') from filename.
    
    Args:
        filename: Input filename
        
    Returns:
        Suffix string ('a' or 'b')
    """
    return 'b' if '_b.' in filename.lower() else 'a'


def prepare_upload_filename(original_filename: str, timestamp: str) -> str:
    """
    Create safe upload filename with timestamp.
    
    Args:
        original_filename: Original uploaded filename
        timestamp: Processing timestamp
        
    Returns:
        Safe filename with timestamp prefix
    """
    safe_filename = secure_filename(original_filename)
    return f"{timestamp}_{safe_filename}"


def validate_file_upload(file) -> tuple:
    """
    Validate uploaded file.
    
    Args:
        file: Flask file upload object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file or file.filename == '':
        return False, 'No file selected'
    
    if not allowed_file(file.filename):
        return False, 'Invalid file format'
    
    return True, None