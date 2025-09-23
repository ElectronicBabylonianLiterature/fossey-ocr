"""
Line Detection Module
Contains functionality for detecting and splitting cuneiform tablet lines.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import os

from ocr_utils import get_ocr_manager, fix_sequential_line_number
from image_utils import normalize_image_to_grayscale


class LineRegion:
    """Represents a detected line region in a cuneiform tablet."""
    
    def __init__(self, line_number: int, original_number: int, top_y: int, bottom_y: int, 
                 bbox: List, confidence: float, padding_type: str = ""):
        self.line_number = line_number
        self.original_number = original_number
        self.top_y = top_y
        self.bottom_y = bottom_y
        self.bbox = bbox
        self.confidence = confidence
        self.padding_type = padding_type
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'line_number': self.line_number,
            'original_number': self.original_number,
            'top_y': self.top_y,
            'bottom_y': self.bottom_y,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'padding_type': self.padding_type
        }
    
    @property
    def height(self) -> int:
        """Get line region height."""
        return self.bottom_y - self.top_y


class LineDetector:
    """Main class for detecting and splitting lines in cuneiform tablets."""
    
    def __init__(self, ocr_languages: List[str] = None):
        """Initialize line detector with OCR manager."""
        self.ocr_manager = get_ocr_manager(ocr_languages)
        self.min_line_spacing = 20  # Minimum spacing between lines
    
    def detect_line_numbers_region(self, image: np.ndarray) -> Tuple[str, List[Tuple], np.ndarray]:
        """
        Detect whether line numbers appear on the left or right edge of the page.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (region_type, ocr_results, region_image)
        """
        return self.ocr_manager.extract_line_numbers(image)
    
    def apply_padding_system(self, line_regions: List[LineRegion], image_height: int) -> List[LineRegion]:
        """
        Apply special padding system to line regions.
        
        Args:
            line_regions: List of detected line regions
            image_height: Height of the source image
            
        Returns:
            List of line regions with applied padding
        """
        for i, region in enumerate(line_regions):
            if i == 0:  # Top line
                # Top padding: 20 pixels up (less)
                top_point = max(0, region.top_y - 20)
                # Bottom padding: 150 pixels down (more)
                bottom_point = min(image_height, region.bottom_y + 150)
                padding_type = "Top Line"
            elif i == len(line_regions) - 1:  # Bottom line
                # Top padding: 150 pixels up (more)
                top_point = max(0, region.top_y - 150)
                # Bottom padding: 20 pixels down (less)
                bottom_point = min(image_height, region.bottom_y + 20)
                padding_type = "Bottom Line"
            else:  # Middle lines
                # Top padding: 150 pixels up (much more)
                top_point = max(0, region.top_y - 150)
                # Bottom padding: 150 pixels down (much more)
                bottom_point = min(image_height, region.bottom_y + 150)
                padding_type = "Middle Line"
            
            # Update region
            region.top_y = top_point
            region.bottom_y = bottom_point
            region.padding_type = padding_type
            
            # Check for overlap with previous region
            if i > 0:
                prev_region = line_regions[i-1]
                if top_point - prev_region.bottom_y < self.min_line_spacing:
                    # Find midpoint
                    mid_point = (prev_region.bottom_y + top_point) // 2
                    # Adjust boundaries
                    prev_region.bottom_y = mid_point - (self.min_line_spacing // 2)
                    region.top_y = mid_point + (self.min_line_spacing // 2)
                    print(f"    Overlap corrected between line {prev_region.line_number} and {region.line_number}")
        
        return line_regions
    
    def split_image_into_lines(self, image_path: str, output_dir: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Split cuneiform image into individual line images.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output files
            
        Returns:
            Tuple of (result_dict, error_message)
        """
        print(f"🔍 Starting line splitting process: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Image could not be loaded"
        
        height, width = image.shape[:2]
        print(f"📷 Image size: {width}x{height}")
        
        # Detect line numbers
        region_type, results, region_section = self.detect_line_numbers_region(image)
        
        if not results:
            return None, "Line numbers not found"
        
        print(f"✅ {len(results)} line numbers detected ({region_type} edge)")
        
        # Sort results by Y coordinate
        results.sort(key=lambda x: x[0][0][1])
        
        # Process line regions
        line_regions = []
        corrected_numbers = []
        used_numbers = set()
        
        for i, (bbox, text, conf) in enumerate(results):
            try:
                original_number = int(text)
                
                # Apply logical correction
                if i == 0:
                    corrected_number = original_number
                else:
                    prev_number = corrected_numbers[i-1] if corrected_numbers else original_number
                    corrected_number = fix_sequential_line_number(original_number, prev_number, i, used_numbers)
                
                corrected_numbers.append(corrected_number)
                used_numbers.add(corrected_number)
                
                print(f"  Line {original_number} → {corrected_number}: Y={bbox[0][1]:.0f}, Confidence={conf:.2f}")
                
                # Extract bbox coordinates (EasyOCR format)
                top_left = bbox[0]
                bottom_right = bbox[2]
                
                # Get Y coordinates
                top_y = int(top_left[1])
                bottom_y = int(bottom_right[1])
                
                # Adjust coordinates for full image if region was cropped
                if region_type == "right":
                    # Adjust X coordinates back to full image
                    for point in bbox:
                        point[0] += int(width * 0.85)
                
                region = LineRegion(
                    line_number=corrected_number,
                    original_number=original_number,
                    top_y=top_y,
                    bottom_y=bottom_y,
                    bbox=bbox,
                    confidence=conf
                )
                
                line_regions.append(region)
                
            except ValueError:
                print(f"⚠️ Invalid line number: {text}")
                continue
        
        # Apply padding system
        line_regions = self.apply_padding_system(line_regions, height)
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main visualization
        vis_image = image.copy()
        for i, region in enumerate(line_regions):
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 255)
            cv2.rectangle(vis_image, (0, region.top_y), (width, region.bottom_y), color, 2)
            cv2.putText(vis_image, f"Line {region.line_number}", 
                       (10, region.top_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis_image, f"H:{region.height}px", 
                       (10, region.top_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Summary visualization
        summary_image = image.copy()
        for region in line_regions:
            # Top boundary (red)
            cv2.line(summary_image, (0, region.top_y), (width, region.top_y), (0, 0, 255), 2)
            # Bottom boundary (blue)
            cv2.line(summary_image, (0, region.bottom_y), (width, region.bottom_y), (255, 0, 0), 2)
            # Line number
            cv2.putText(summary_image, f"{region.line_number}", 
                       (10, region.top_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save visualization files
        vis_path = os.path.join(output_dir, f"line_visualization_{timestamp}.jpg")
        summary_path = os.path.join(output_dir, f"summary_visualization_{timestamp}.jpg")
        cv2.imwrite(vis_path, vis_image)
        cv2.imwrite(summary_path, summary_image)
        
        # Save individual line images
        line_files = []
        for region in line_regions:
            line_image = image[region.top_y:region.bottom_y, :]
            line_filename = f"line_{region.line_number}_{timestamp}.jpg"
            line_path = os.path.join(output_dir, line_filename)
            cv2.imwrite(line_path, line_image)
            
            line_files.append({
                'filename': line_filename,
                'line_number': region.line_number,
                'height': region.height,
                'confidence': region.confidence
            })
        
        # Prepare result
        result = {
            'success': True,
            'total_lines': len(line_regions),
            'line_regions': [region.to_dict() for region in line_regions],
            'line_files': line_files,
            'visualization_path': vis_path,
            'summary_path': summary_path,
            'timestamp': timestamp
        }
        
        print(f"✅ Line splitting completed: {len(line_regions)} lines")
        return result, None


def create_line_detector(ocr_languages: List[str] = None) -> LineDetector:
    """
    Factory function to create a LineDetector instance.
    
    Args:
        ocr_languages: List of languages for OCR
        
    Returns:
        LineDetector instance
    """
    return LineDetector(ocr_languages)