"""
OCR Utilities Module
Contains all OCR-related functionality for the cuneiform line splitting system.
"""

import easyocr
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Union


class OCRManager:
    """Centralized OCR management for the cuneiform system."""
    
    def __init__(self, languages: List[str] = None, use_google_vision: bool = None):
        """Initialize OCR manager with specified languages and OCR engine."""
        self.languages = languages or ['en']
        self._reader = None
        
        # Check for Google Vision preference from environment or parameter
        if use_google_vision is None:
            use_google_vision = os.environ.get('USE_GOOGLE_VISION', '').lower() == 'true'
        
        self.use_google_vision = use_google_vision
    
    @property
    def reader(self):
        """Lazy-load the appropriate OCR reader."""
        if self._reader is None:
            if self.use_google_vision:
                try:
                    from google_vision_ocr import GoogleVisionReader
                    self._reader = GoogleVisionReader(self.languages)
                    print("✅ Google Cloud Vision OCR initialized")
                except ImportError as e:
                    print(f"❌ Google Cloud Vision not available: {e}")
                    print("🔄 Falling back to EasyOCR")
                    self._reader = easyocr.Reader(self.languages)
            else:
                self._reader = easyocr.Reader(self.languages)
        return self._reader
    
    def read_text(self, image: np.ndarray, allowlist: str = None) -> List[Tuple]:
        """
        Read text from image using EasyOCR.
        
        Args:
            image: Image array
            allowlist: String of allowed characters (e.g., '0123456789' for digits only)
            
        Returns:
            List of (bbox, text, confidence) tuples
        """
        try:
            if allowlist:
                results = self.reader.readtext(image, allowlist=allowlist)
            else:
                results = self.reader.readtext(image)
            return results
        except Exception as e:
            print(f"❌ OCR error: {e}")
            return []
    
    def read_numbers_only(self, image: np.ndarray) -> List[Tuple]:
        """Read only numbers from image."""
        return self.read_text(image, allowlist='0123456789')
    
    def extract_line_numbers(self, image: np.ndarray, left_region_width: float = 0.15, 
                           right_region_width: float = 0.15) -> Tuple[str, List[Tuple], np.ndarray]:
        """
        Extract line numbers from left or right edge of image.
        
        Args:
            image: Input image
            left_region_width: Width ratio for left region (0.0-1.0)
            right_region_width: Width ratio for right region (0.0-1.0)
            
        Returns:
            Tuple of (region_type, ocr_results, region_image)
        """
        height, width = image.shape[:2]
        
        # Calculate region boundaries
        left_bound = int(width * left_region_width)
        right_bound = int(width * (1.0 - right_region_width))
        
        # Extract regions
        left_section = image[:, :left_bound]
        right_section = image[:, right_bound:]
        
        # OCR on both regions
        left_results = self.read_numbers_only(left_section)
        right_results = self.read_numbers_only(right_section)
        
        # Return the region with more detected numbers
        if len(left_results) > len(right_results):
            return "left", left_results, left_section
        else:
            return "right", right_results, right_section


def fix_sequential_line_number(detected_number: int, previous_number: int, 
                              position: int, used_numbers: set) -> int:
    """
    Fix detected line numbers using sequential expectations and heuristics.
    
    This function applies predefined corrections for common OCR misreads,
    then falls back to logical sequential corrections while ensuring uniqueness.
    
    Args:
        detected_number: Number detected by OCR
        previous_number: Previous line number in sequence
        position: Position in the sequence
        used_numbers: Set of already used numbers
        
    Returns:
        Corrected line number
    """
    expected_number = previous_number + 1
    
    # If detected number matches expected and is unused, accept it
    if detected_number == expected_number and detected_number not in used_numbers:
        return detected_number
    
    # Common OCR error corrections
    corrections = {
        # Extra digit errors (2701 → 270)
        2701: 270, 2702: 270, 2703: 270, 2704: 270, 2705: 270,
        2706: 270, 2707: 270, 2708: 270, 2709: 270, 2710: 270,
        # Wrong digit errors (366 → 266)
        366: 266, 367: 267, 368: 268, 369: 269, 370: 270,
        # 4 → 1 errors (4259 → 1259)
        4259: 1259, 4258: 1258, 4257: 1257, 4256: 1256,
        4255: 1255, 4254: 1254, 4253: 1253, 4252: 1252,
        4251: 1251, 4250: 1250, 4249: 1249, 4248: 1248,
        # Missing digit errors
        126: 1260, 125: 1259, 127: 1270, 128: 1280,
        # Other common errors
        239: 269, 213: 273, 274: 274, 275: 275, 276: 276, 277: 277, 278: 278, 279: 279
    }
    
    # Apply common error corrections first
    if detected_number in corrections:
        corrected = corrections[detected_number]
        if corrected not in used_numbers:
            print(f"    🔧 OCR error corrected: {detected_number} → {corrected}")
            return corrected
        else:
            print(f"    🔧 OCR error corrected but number already used: {detected_number} → {expected_number}")
            return expected_number
    
    # Logical correction - use expected number
    if expected_number not in used_numbers:
        print(f"    🔧 Logical correction: {detected_number} → {expected_number}")
        return expected_number
    else:
        # Find next available number
        next_number = expected_number + 1
        while next_number in used_numbers:
            next_number += 1
        print(f"    🔧 Unique correction: {detected_number} → {next_number}")
        return next_number


def preprocess_for_ocr(image: np.ndarray, target: str = "general") -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image: Input image
        target: Type of text to optimize for ("general", "numbers", "roman")
        
    Returns:
        Preprocessed image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if target == "numbers":
        # Optimize for number recognition
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    elif target == "roman":
        # Optimize for Roman numerals and punctuation
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 15, 10)
        
        return binary
    
    else:
        # General preprocessing
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced


def fix_roman_numerals_and_punctuation(text: str) -> str:
    """
    Fix common OCR errors in Roman numerals and punctuation.
    
    Args:
        text: OCR output text
        
    Returns:
        Corrected text
    """
    if not text:
        return text
    
    # Common OCR corrections for Roman numerals
    corrections = {
        # Roman numeral fixes
        'VIII': ['Vlll', 'VIll', 'VIII', 'vlll', 'viii'],
        'VII': ['Vll', 'VlI', 'vii', 'vil'],
        'VI': ['Vl', 'vi', 'vl'],
        'IV': ['lV', 'iv', 'lv'],
        'IX': ['lX', 'ix', 'lx'],
        'XIV': ['XlV', 'XIV', 'xiv'],
        'XV': ['XV', 'xv'],
        'XVI': ['XVl', 'xvi'],
        'XVII': ['XVll', 'xvii'],
        'XVIII': ['XVlll', 'xviii'],
        'XIX': ['XlX', 'xix'],
        'XX': ['XX', 'xx'],
        
        # Punctuation fixes
        ',': ['.', ';', ':', '!'],
        '.': [',', ';', ':', '!'],
        ':': [';', '.', ',', '!'],
        
        # Letter substitutions
        'H': ['ll', '11', 'II'],
        'R': ['R', 'r'],
        'h': ['II', 'll', 'n'],
        '8': ['8', 'B', 'g'],
        '5': ['5', 'S', 's']
    }
    
    corrected_text = text
    
    # Apply corrections
    for correct, variants in corrections.items():
        for variant in variants:
            corrected_text = corrected_text.replace(variant, correct)
    
    return corrected_text.strip()


# Global OCR manager instance
_ocr_manager = None

def get_ocr_manager(languages: List[str] = None, use_google_vision: bool = None) -> OCRManager:
    """Get singleton OCR manager instance."""
    global _ocr_manager
    if _ocr_manager is None:
        _ocr_manager = OCRManager(languages, use_google_vision)
    return _ocr_manager

def reset_ocr_manager():
    """Reset the OCR manager (useful when switching OCR engines)."""
    global _ocr_manager
    _ocr_manager = None