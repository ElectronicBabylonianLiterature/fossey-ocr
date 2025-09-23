"""
Image Processing Utilities Module
Contains low-level image processing functions for the cuneiform line splitting system.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union


def enhance_image_contrast(image: np.ndarray, clip_limit: float = 2.0, 
                          tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image (grayscale or color)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of neighborhood for local histogram equalization
        
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), 
                       sigma_x: float = 0) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                          sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter to reduce noise while preserving edges."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_adaptive_threshold(image: np.ndarray, max_value: int = 255, 
                           adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                           threshold_type: int = cv2.THRESH_BINARY_INV,
                           block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding to convert image to binary.
    
    Args:
        image: Input grayscale image
        max_value: Maximum value assigned to pixel
        adaptive_method: Adaptive method (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
        threshold_type: Threshold type
        block_size: Size of neighborhood area
        c: Constant subtracted from mean
        
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    return cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, c)


def apply_otsu_threshold(image: np.ndarray, threshold_type: int = cv2.THRESH_BINARY_INV) -> Tuple[np.ndarray, float]:
    """
    Apply Otsu's thresholding method.
    
    Args:
        image: Input grayscale image
        threshold_type: Type of thresholding
        
    Returns:
        Tuple of (binary_image, threshold_value)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    threshold_value, binary = cv2.threshold(gray, 0, 255, threshold_type + cv2.THRESH_OTSU)
    return binary, threshold_value


def find_contours(binary_image: np.ndarray, retrieval_mode: int = cv2.RETR_EXTERNAL,
                 approximation_method: int = cv2.CHAIN_APPROX_SIMPLE) -> List[np.ndarray]:
    """
    Find contours in binary image.
    
    Args:
        binary_image: Binary input image
        retrieval_mode: Contour retrieval mode
        approximation_method: Contour approximation method
        
    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(binary_image, retrieval_mode, approximation_method)
    return contours


def get_largest_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Get the largest contour by area.
    
    Args:
        contours: List of contours
        
    Returns:
        Largest contour or None if no contours
    """
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def get_bounding_rect(contour: np.ndarray, padding: int = 0, 
                     image_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """
    Get bounding rectangle for contour with optional padding.
    
    Args:
        contour: Input contour
        padding: Padding to add around bounding box
        image_shape: Shape of the image (height, width) for bounds checking
        
    Returns:
        Tuple of (x, y, width, height)
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    # Apply padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding
    
    # Ensure bounds don't exceed image dimensions
    if image_shape:
        height, width = image_shape[:2]
        w = min(width - x, w)
        h = min(height - y, h)
    
    return x, y, w, h


def create_white_background(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Create a white background image."""
    return np.ones((height, width, channels), dtype=np.uint8) * 255


def enhance_cuneiform_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance contrast specifically for cuneiform signs.
    
    Args:
        image: Input cuneiform image
        
    Returns:
        Enhanced image with better contrast for cuneiform signs
    """
    # Create white background
    height, width = image.shape[:2]
    channels = 3 if len(image.shape) == 3 else 1
    
    if channels == 3:
        white_background = create_white_background(height, width, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        white_background = np.ones((height, width), dtype=np.uint8) * 255
        gray = image
    
    # Apply adaptive thresholding to identify dark pixels (cuneiform)
    binary = apply_adaptive_threshold(gray)
    
    # Enhance dark pixels on white background
    if channels == 3:
        for i in range(height):
            for j in range(width):
                if binary[i, j] > 0:  # Dark pixel (cuneiform sign)
                    original_pixel = image[i, j]
                    
                    # Make dark pixels darker
                    enhanced_pixel = np.zeros(3, dtype=np.uint8)
                    for k in range(3):  # BGR channels
                        if original_pixel[k] < 80:
                            enhanced_pixel[k] = 0  # Pure black
                        else:
                            enhanced_pixel[k] = max(0, original_pixel[k] - 40)
                    
                    white_background[i, j] = enhanced_pixel
    else:
        # Grayscale enhancement
        for i in range(height):
            for j in range(width):
                if binary[i, j] > 0:  # Dark pixel
                    if gray[i, j] < 80:
                        white_background[i, j] = 0
                    else:
                        white_background[i, j] = max(0, gray[i, j] - 40)
    
    return white_background


def apply_morphological_operations(binary_image: np.ndarray, operation: int, 
                                  kernel_size: Tuple[int, int] = (3, 3),
                                  kernel_shape: int = cv2.MORPH_RECT) -> np.ndarray:
    """
    Apply morphological operations to binary image.
    
    Args:
        binary_image: Input binary image
        operation: Morphological operation (MORPH_OPEN, MORPH_CLOSE, etc.)
        kernel_size: Size of the morphological kernel
        kernel_shape: Shape of the kernel
        
    Returns:
        Processed binary image
    """
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    return cv2.morphologyEx(binary_image, operation, kernel)


def crop_image_region(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop a region from the image.
    
    Args:
        image: Input image
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of the region
        height: Height of the region
        
    Returns:
        Cropped image region
    """
    return image[y:y+height, x:x+width]


def normalize_image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if it's in color."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image