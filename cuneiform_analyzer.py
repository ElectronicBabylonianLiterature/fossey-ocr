"""
Cuneiform Analyzer Module
Contains functionality for analyzing cuneiform signs and extracting reference text.
"""

import cv2
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

from ocr_utils import get_ocr_manager
from image_utils import (
    apply_otsu_threshold, find_contours, get_largest_contour, 
    get_bounding_rect, enhance_cuneiform_contrast, crop_image_region
)
from utils import match_reference_with_catalog, load_cuneiform_catalog


class CuneiformSign:
    """Represents a detected cuneiform sign with its properties."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 area: int, aspect_ratio: float, image_path: str = "",
                 vector_info: Dict = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = area
        self.aspect_ratio = aspect_ratio
        self.image_path = image_path
        self.vector_info = vector_info or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'aspect_ratio': self.aspect_ratio,
            'position_x': self.x,
            'position_y': self.y,
            'image_path': self.image_path,
            'vector_info': self.vector_info
        }


class ReferenceInfo:
    """Represents reference text information."""
    
    def __init__(self, text: str, x: int, y: int, width: int, height: int):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'position_x': self.x,
            'position_y': self.y,
            'width': self.width,
            'height': self.height
        }


class LineNumberInfo:
    """Represents line number information."""
    
    def __init__(self, number: int, position: str, bbox: Optional[Tuple] = None):
        self.number = number
        self.position = position  # 'left' or 'right'
        self.bbox = bbox
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'number': self.number,
            'position': self.position,
            'bbox': self.bbox
        }


class CuneiformAnalyzer:
    """Main class for analyzing cuneiform signs and extracting information."""
    
    def __init__(self, ocr_languages: List[str] = None):
        """Initialize analyzer with OCR manager."""
        self.ocr_manager = get_ocr_manager(ocr_languages)
        self.catalog = load_cuneiform_catalog()
    
    def detect_cuneiform_sign(self, image: np.ndarray) -> Optional[CuneiformSign]:
        """
        Detect the main cuneiform sign in an image using improved segmentation.
        
        Args:
            image: Input line image
            
        Returns:
            CuneiformSign object or None if no sign detected
        """
        try:
            # Convert to grayscale and apply OTSU thresholding
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Close small gaps from scanner noise to prevent artificial breaks in the sign
            kernel = np.ones((4, 4), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get the largest contour (cuneiform sign)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add 2 pixels padding on all sides to ensure no part is cut off
            x = max(0, x - 2)
            y = max(0, y - 2)
            w = min(image.shape[1] - x, w + 4)  # +4 because we add 2 pixels on both sides
            h = min(image.shape[0] - y, h + 4)  # +4 because we add 2 pixels on both sides
            
            # Calculate properties
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            return CuneiformSign(x, y, w, h, area, aspect_ratio)
            
        except Exception as e:
            print(f"❌ Error detecting cuneiform sign: {e}")
            return None
    
    def extract_cuneiform_image(self, line_image: np.ndarray, sign: CuneiformSign, 
                               output_path: str) -> str:
        """
        Extract and enhance cuneiform sign image with improved processing.
        
        Args:
            line_image: Full line image
            sign: Detected cuneiform sign
            output_path: Path to save the extracted sign
            
        Returns:
            Path to saved cuneiform image
        """
        # Extract sign region from the line image
        cuneiform_sign = line_image[sign.y:sign.y+sign.height, sign.x:sign.x+sign.width]
        
        # Create white background
        white_background = np.ones((sign.height, sign.width, 3), dtype=np.uint8) * 255
        
        # Process the cuneiform sign to enhance contrast and copy to white background
        gray = cv2.cvtColor(cuneiform_sign, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Copy black pixels (cuneiform sign) to white background with contrast enhancement
        for i in range(sign.height):
            for j in range(sign.width):
                if binary[i, j] > 0:  # Black pixel (cuneiform sign)
                    # Get original pixel
                    original_pixel = cuneiform_sign[i, j]
                    
                    # Enhance contrast - make black pixels darker
                    enhanced_pixel = np.zeros(3, dtype=np.uint8)
                    for k in range(3):  # BGR channels
                        # Make dark pixels (0-80 range) completely black
                        if original_pixel[k] < 80:
                            enhanced_pixel[k] = 0  # Pure black
                        else:
                            # Darken other values slightly
                            enhanced_pixel[k] = max(0, original_pixel[k] - 40)
                    
                    white_background[i, j] = enhanced_pixel
        
        # Save enhanced image
        cv2.imwrite(output_path, white_background)
        
        return output_path

    def create_vector_representation(self, cuneiform_image: np.ndarray, 
                                   line_number: int, timestamp: str,
                                   output_dir: str) -> Dict:
        """
        Create vector (SVG) representation.
        
        Args:
            cuneiform_image: Cuneiform sign image
            line_number: Line number
            timestamp: Processing timestamp
            output_dir: Output directory
            
        Returns:
            Dictionary with vector information
        """
        try:
            # Enhance and threshold the image with improved preprocessing
            enhanced_image = enhance_cuneiform_contrast(cuneiform_image)
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY) if len(enhanced_image.shape) == 3 else enhanced_image
            
            # Apply Gaussian blur to reduce noise before thresholding
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            binary, _ = apply_otsu_threshold(blurred)
            
            # Apply morphological operations to clean up the binary image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Close small gaps
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            # Remove small noise
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours for SVG creation with hierarchical detection
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                return {}
            
            # Filter out very small contours (noise) and organize by hierarchy
            min_contour_area = 30  # Reduced minimum area to capture more detail
            filtered_contours = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > min_contour_area:
                    # Check if this is an outer contour (not a hole)
                    if hierarchy[0][i][3] == -1:  # No parent (outer contour)
                        filtered_contours.append(contour)
                    elif area > min_contour_area * 2:  # Include larger inner contours (holes)
                        filtered_contours.append(contour)
            
            if not filtered_contours:
                return {}
            
            # Create SVG file
            h, w = cuneiform_image.shape[:2]
            svg_filename = f"cuneiform_vector_{line_number}_{timestamp}.svg"
            svg_path = os.path.join(output_dir, svg_filename)
            
            # Generate SVG content
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write(f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" style="background-color: white;">\n')
                f.write('  <rect width="100%" height="100%" fill="white" stroke="none"/>\n')
                f.write('  <defs>\n')
                f.write('    <style>\n')
                f.write('      .cuneiform-path {\n')
                f.write('        fill: #000000 !important;\n')
                f.write('        stroke: #000000 !important;\n')
                f.write('        stroke-width: 1;\n')
                f.write('        stroke-linejoin: round;\n')
                f.write('        stroke-linecap: round;\n')
                f.write('      }\n')
                f.write('    </style>\n')
                f.write('  </defs>\n')
                
                # Convert all significant contours to SVG paths
                total_contour_points = 0

                # Helper: Chaikin subdivision for smoothing while preserving detail
                def chaikin_subdivision(pts, iterations=2, closed=True):
                    # pts: list or Nx2 numpy array of points
                    pts = [tuple(p) for p in pts]
                    for _ in range(iterations):
                        new_pts = []
                        n = len(pts)
                        for i in range(n):
                            p0 = pts[i]
                            p1 = pts[(i + 1) % n] if closed else (pts[i + 1] if i + 1 < n else None)
                            if p1 is None:
                                new_pts.append(p0)
                                break
                            # Q and R points
                            qx = 0.75 * p0[0] + 0.25 * p1[0]
                            qy = 0.75 * p0[1] + 0.25 * p1[1]
                            rx = 0.25 * p0[0] + 0.75 * p1[0]
                            ry = 0.25 * p0[1] + 0.75 * p1[1]
                            new_pts.append((qx, qy))
                            new_pts.append((rx, ry))
                        pts = new_pts
                    return pts

                for i, contour in enumerate(filtered_contours):
                    # Use the full contour points for higher fidelity
                    pts = contour.reshape(-1, 2)

                    if pts.shape[0] < 3:
                        continue

                    # Downsample overly dense contours to keep file size reasonable
                    max_points = 2500
                    if pts.shape[0] > max_points:
                        step = max(1, pts.shape[0] // max_points)
                        pts = pts[::step]

                    # Apply light smoothing: Chaikin subdivision produces smoother curves
                    smooth_iters = 2 if pts.shape[0] < 800 else 1
                    smoothed = chaikin_subdivision(pts, iterations=smooth_iters, closed=True)

                    total_contour_points += len(smoothed)
                    f.write(f'  <path class="cuneiform-path" d="')

                    # Emit path as a dense polyline (M + many L commands) for accuracy
                    first_point = smoothed[0]
                    f.write(f"M {first_point[0]:.2f} {first_point[1]:.2f} ")
                    for pt in smoothed[1:]:
                        f.write(f"L {pt[0]:.2f} {pt[1]:.2f} ")

                    # Close path
                    f.write('Z"/>\n')
                
                f.write('</svg>\n')
            
            return {
                'svg_path': svg_filename,
                'original_size': f"{w}x{h}",
                'contour_points': total_contour_points,
                'contour_count': len(filtered_contours)
            }
            
        except Exception as e:
            print(f"❌ Vector creation error: {e}")
            return {}
    
    def extract_reference_text(self, line_image: np.ndarray, sign: CuneiformSign,
                              line_number_bbox: Optional[Tuple] = None) -> ReferenceInfo:
        """
        Extract reference text from the third part of the line using improved segmentation.
        
        Args:
            line_image: Full line image
            sign: Detected cuneiform sign
            line_number_bbox: Bounding box of line number if detected
            
        Returns:
            ReferenceInfo object
        """
        # Calculate reference text region (right of cuneiform sign, excluding line number)
        third_part_x = sign.x + sign.width + 10  # 10px gap after cuneiform sign
        third_part_width = line_image.shape[1] - third_part_x
        
        # If line number is on the right side, exclude it from reference area
        if line_number_bbox and line_number_bbox[0] > sign.x + sign.width:
            third_part_width = line_number_bbox[0] - third_part_x - 10  # 10px gap before line number
        
        if third_part_width <= 0:
            return ReferenceInfo("", third_part_x, sign.y, 0, sign.height)
        
        # Crop the third part (reference text area) using same height as cuneiform sign
        third_part = line_image[sign.y:sign.y+sign.height, third_part_x:third_part_x+third_part_width]
        
        if third_part.size == 0:
            print(f"⚠️ Warning: Reference text part is empty")
            final_text = ""
        else:
            # OCR on the reference part
            third_part_results = self.ocr_manager.read_text(third_part)
            
            # Join recognized text fragments
            third_part_texts = [res[1].strip() for res in third_part_results if res[1].strip()]
            final_text = " ".join(third_part_texts)
        
        return ReferenceInfo(final_text, third_part_x, sign.y, third_part_width, sign.height)
    
    def detect_line_number(self, line_image: np.ndarray) -> Optional[LineNumberInfo]:
        """
        Detect line number in the image.
        
        Args:
            line_image: Input line image
            
        Returns:
            LineNumberInfo object or None if no line number detected
        """
        # OCR on full image
        results = self.ocr_manager.read_text(line_image)
        
        for (bbox, text, confidence) in results:
            clean_text = text.strip()
            
            # Check if it's a digit in line number range
            if clean_text.isdigit():
                num = int(clean_text)
                if 1000 <= num <= 2000:  # Line number range
                    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                    x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                    position = 'left' if x1 < line_image.shape[1] // 2 else 'right'
                    return LineNumberInfo(num, position, (x1, y1, x2, y2))
        
        return None
    
    def analyze_line(self, line_data: Union[Dict, str], line_number: int, 
                    timestamp: str, output_dir: str, suffix: str = 'a') -> Optional[Dict]:
        """
        Analyze a complete line image using the original working segmentation logic.
        
        Args:
            line_data: Line data dictionary or image path
            line_number: Line number
            timestamp: Processing timestamp
            output_dir: Output directory
            suffix: File suffix
            
        Returns:
            Analysis result dictionary or None if failed
        """
        try:
            # Resolve image path
            if isinstance(line_data, dict):
                filename = line_data.get('filename')
                if not filename:
                    print("❌ No filename provided in line_data")
                    return None
            else:
                filename = line_data
            
            # Try different path resolution strategies
            line_image_path = self._resolve_image_path(filename, output_dir, timestamp)
            if not line_image_path:
                print(f"❌ Could not resolve image path for {filename}")
                return None
            
            # Load image
            line_image = cv2.imread(line_image_path)
            if line_image is None:
                print(f"❌ Could not load image: {line_image_path}")
                return None
            
            # === ORIGINAL WORKING SEGMENTATION LOGIC ===
            
            # Step 1: OCR to find line number
            results = self.ocr_manager.read_text(line_image)
            line_number_bbox = None
            
            for (bbox, text, confidence) in results:
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                
                clean_text = text.strip()
                
                # Check if it's a line number (digit in range 1000-2000)
                if clean_text.isdigit():
                    num = int(clean_text)
                    if 1000 <= num <= 2000:  # Line number range
                        line_number_bbox = (x1, y1, x2, y2)
                        break
            
            # Step 2: Find cuneiform sign (largest contour) with EXACT original logic
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Close small gaps from scanner noise to prevent artificial breaks in the sign
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"⚠️ No contours found in line {line_number}")
                return None
            
            # Get largest contour (cuneiform sign)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add 2 pixels padding on all sides - EXACTLY as in original
            x = max(0, x - 2)
            y = max(0, y - 2)
            w = min(line_image.shape[1] - x, w + 4)  # +4 because both sides get 2 pixels
            h = min(line_image.shape[0] - y, h + 4)  # +4 because both sides get 2 pixels
            
            # Step 3: Define reference text region using SAME Y coordinates
            third_part_x = x + w + 10
            third_part_width = line_image.shape[1] - third_part_x
            
            # If line number is on the right, exclude it from reference area
            if line_number_bbox and line_number_bbox[0] > x + w:
                third_part_width = line_number_bbox[0] - third_part_x - 10
            
            # Extract reference text using SAME Y coordinates as cuneiform sign
            if third_part_width > 0:
                third_part = line_image[y:y+h, third_part_x:third_part_x+third_part_width]
                
                if third_part.size > 0:
                    # OCR on the reference part
                    third_part_results = self.ocr_manager.read_text(third_part)
                    
                    # Join recognized text fragments
                    third_part_texts = [res[1].strip() for res in third_part_results if res[1].strip()]
                    final_text = " ".join(third_part_texts)
                else:
                    final_text = ""
            else:
                final_text = ""
            
            # Step 4: Extract cuneiform sign image with white background
            cuneiform_sign = line_image[y:y+h, x:x+w]
            
            # Create white background
            white_background = np.ones((h, w, 3), dtype=np.uint8) * 255
            
            # Process cuneiform sign onto white background with contrast enhancement
            gray_sign = cv2.cvtColor(cuneiform_sign, cv2.COLOR_BGR2GRAY)
            _, binary_sign = cv2.threshold(gray_sign, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Copy black pixels (cuneiform sign) to white background with contrast enhancement
            for i in range(h):
                for j in range(w):
                    if binary_sign[i, j] > 0:  # Black pixel (cuneiform sign)
                        # Get original pixel
                        original_pixel = cuneiform_sign[i, j]
                        
                        # Enhance contrast - make black pixels darker
                        enhanced_pixel = np.zeros(3, dtype=np.uint8)
                        for k in range(3):  # BGR channels
                            # Make dark pixels (0-80 range) completely black
                            if original_pixel[k] < 80:
                                enhanced_pixel[k] = 0  # Pure black
                            else:
                                # Darken other values slightly
                                enhanced_pixel[k] = max(0, original_pixel[k] - 40)
                        
                        white_background[i, j] = enhanced_pixel
            
            # Save cuneiform sign image
            cuneiform_filename = f"cuneiform_{line_number}_{timestamp}.jpg"
            cuneiform_path = os.path.join(output_dir, cuneiform_filename)
            cv2.imwrite(cuneiform_path, white_background)
            
            # Create vector representation
            vector_info = self.create_vector_representation(cuneiform_sign, line_number, timestamp, output_dir)
            
            # Match with catalog
            catalog_match = match_reference_with_catalog(final_text, self.catalog)
            
            # Return result in expected format
            line_position = 'left' if line_number_bbox and line_number_bbox[0] < line_image.shape[1]//2 else 'right'
            
            return {
                'line_number': {
                    'number': line_number,
                    'position': line_position,
                    'bbox': line_number_bbox
                },
                'cuneiform_sign': {
                    'width': w,
                    'height': h,
                    'area': w * h,
                    'aspect_ratio': w / h if h > 0 else 0,
                    'position_x': x,
                    'position_y': y,
                    'image_path': cuneiform_filename,
                    'vector_info': vector_info
                },
                'reference_info': {
                    'text': final_text,
                    'position_x': third_part_x,
                    'position_y': y,  # SAME Y as cuneiform sign
                    'width': third_part_width,
                    'height': h  # SAME height as cuneiform sign
                },
                'catalog_match': catalog_match
            }
            
        except Exception as e:
            print(f"❌ Line analysis error for line {line_number}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _resolve_image_path(self, filename: str, output_dir: str, timestamp: str) -> Optional[str]:
        """Resolve the full path to a line image file."""
        # Try absolute path first
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        
        # Try output directory with timestamp
        candidate = os.path.join(output_dir, filename)
        if os.path.exists(candidate):
            return candidate
        
        # Try web outputs folder structure
        candidate = os.path.join('web_outputs', timestamp, filename)
        if os.path.exists(candidate):
            return candidate
        
        # Try current directory
        candidate = os.path.join(os.getcwd(), filename)
        if os.path.exists(candidate):
            return candidate
        
        # Try recursive search
        search_roots = [output_dir, 'web_outputs', os.getcwd()]
        for root in search_roots:
            if os.path.exists(root):
                pattern = os.path.join(root, '**', filename)
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    return matches[0]
        
        return None


def create_cuneiform_analyzer(ocr_languages: List[str] = None) -> CuneiformAnalyzer:
    """
    Factory function to create a CuneiformAnalyzer instance.
    
    Args:
        ocr_languages: List of languages for OCR
        
    Returns:
        CuneiformAnalyzer instance
    """
    return CuneiformAnalyzer(ocr_languages)