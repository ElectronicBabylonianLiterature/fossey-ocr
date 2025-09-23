"""
Refactored Image Processing Module
Main orchestration for cuneiform line processing using modular components.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union

from line_detection import create_line_detector
from cuneiform_analyzer import create_cuneiform_analyzer
from utils import match_reference_with_catalog, load_cuneiform_catalog


# Load catalog once at module level
CUNEIFORM_CATALOG = load_cuneiform_catalog()


def split_image_into_lines(image_path: str, output_dir: str) -> tuple:
    """
    Split cuneiform image into individual line images.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output files
        
    Returns:
        Tuple of (result_dict, error_message)
    """
    line_detector = create_line_detector()
    return line_detector.split_image_into_lines(image_path, output_dir)


def analyze_line_three_parts(line_data: Union[Dict, str], line_number: int, 
                           timestamp: str, app_config: Union[Dict, str], 
                           base_output_dir: Optional[str] = None, 
                           suffix: str = 'a') -> Optional[Dict]:
    """
    Analyze the three logical parts of a line image.
    
    Args:
        line_data: Line data dictionary or image path
        line_number: Line number
        timestamp: Processing timestamp
        app_config: App configuration or output directory path
        base_output_dir: Base output directory
        suffix: File suffix
        
    Returns:
        Analysis result dictionary or None if failed
    """
    # Determine output directory
    if base_output_dir:
        output_dir = base_output_dir
    elif isinstance(app_config, dict) and 'OUTPUT_FOLDER' in app_config:
        output_dir = os.path.join(app_config['OUTPUT_FOLDER'], timestamp)
    else:
        # CLI mode: app_config is the output directory
        output_dir = app_config
    
    # Create analyzer and perform analysis
    analyzer = create_cuneiform_analyzer()
    return analyzer.analyze_line(line_data, line_number, timestamp, output_dir, suffix)


def create_detailed_analysis_json(result_data: Dict, timestamp: str, output_dir: str, 
                                suffix: str = 'a', app_config: Optional[Dict] = None) -> None:
    """
    Create detailed analysis JSON file with line information.
    
    Args:
        result_data: Processing result data
        timestamp: Processing timestamp
        output_dir: Output directory
        suffix: File suffix
        app_config: App configuration
    """
    try:
        print(f"🔍 Creating detailed analysis JSON for {len(result_data.get('line_regions', []))} lines")
        
        # Create analyzer for processing
        analyzer = create_cuneiform_analyzer()
        
        # Process each line
        detailed_analysis = []
        line_regions = result_data.get('line_regions', [])
        line_files = result_data.get('line_files', [])
        
        for i, line_region in enumerate(line_regions):
            line_number = line_region.get('line_number')
            if not line_number:
                continue
            
            # Find corresponding line file
            line_file_data = None
            for line_file in line_files:
                if line_file.get('line_number') == line_number:
                    line_file_data = line_file
                    break
            
            if not line_file_data:
                print(f"⚠️ No file data found for line {line_number}")
                continue
            
            # Analyze line
            analysis = analyzer.analyze_line(line_file_data, line_number, timestamp, output_dir, suffix)
            
            if analysis:
                detailed_analysis.append({
                    'line_number': line_number,
                    'analysis': analysis,
                    'original_data': line_region
                })
            else:
                # Add basic info even if analysis failed
                detailed_analysis.append({
                    'line_number': line_number,
                    'analysis': None,
                    'original_data': line_region
                })
        
        # Create simplified output structure
        analysis_data = {'lines': []}
        
        for line in detailed_analysis:
            line_number = line.get('line_number')
            analysis = line.get('analysis')
            
            if analysis and analysis.get('cuneiform_sign'):
                cuneiform_data = analysis['cuneiform_sign']
                reference_text = analysis.get('reference_info', {}).get('text', '')
                catalog_match = match_reference_with_catalog(reference_text, CUNEIFORM_CATALOG)
                
                line_data = {
                    'line_number': line_number,
                    'cuneiform_sign': {
                        'width': cuneiform_data.get('width', 0),
                        'height': cuneiform_data.get('height', 0),
                        'position_x': cuneiform_data.get('position_x', 0),
                        'position_y': cuneiform_data.get('position_y', 0)
                    },
                    'reference_text': reference_text,
                    'catalog_match': catalog_match
                }
            else:
                # Basic structure for failed analysis
                line_data = {
                    'line_number': line_number,
                    'cuneiform_sign': {
                        'width': 0,
                        'height': 0,
                        'position_x': 0,
                        'position_y': 0
                    },
                    'reference_text': '',
                    'catalog_match': None
                }
            
            analysis_data['lines'].append(line_data)
        
        # Save JSON file
        json_file = os.path.join(output_dir, f'analysis_{timestamp}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Analysis JSON created: {json_file}")
        print(f"📊 {len(detailed_analysis)} line analyses processed")
        
    except Exception as e:
        print(f"❌ JSON creation error: {e}")
        import traceback
        traceback.print_exc()


# Backward compatibility functions (deprecated)
def fix_sequential_line_number(detected_number, previous_number, position, used_numbers):
    """Deprecated: Use ocr_utils.fix_sequential_line_number instead."""
    from ocr_utils import fix_sequential_line_number as new_fix
    return new_fix(detected_number, previous_number, position, used_numbers)


def detect_line_numbers_region(image):
    """Deprecated: Use line_detection.LineDetector.detect_line_numbers_region instead."""
    from line_detection import create_line_detector
    detector = create_line_detector()
    return detector.detect_line_numbers_region(image)


def apply_special_padding_system(line_regions, height):
    """Deprecated: Use line_detection.LineDetector.apply_padding_system instead."""
    from line_detection import create_line_detector, LineRegion
    detector = create_line_detector()
    
    # Convert to LineRegion objects if needed
    if line_regions and not isinstance(line_regions[0], LineRegion):
        regions = []
        for region in line_regions:
            lr = LineRegion(
                region.get('line_number', 0),
                region.get('original_number', 0),
                region.get('top_y', 0),
                region.get('bottom_y', 0),
                region.get('bbox', []),
                region.get('confidence', 0.0)
            )
            regions.append(lr)
        line_regions = regions
    
    processed_regions = detector.apply_padding_system(line_regions, height)
    
    # Convert back to dict format for compatibility
    return [region.to_dict() for region in processed_regions]


def create_vector_cuneiform(cuneiform_image, line_number, timestamp, app_config, base_output_dir=None):
    """Deprecated: Use cuneiform_analyzer.CuneiformAnalyzer.create_vector_representation instead."""
    from cuneiform_analyzer import create_cuneiform_analyzer
    
    # Determine output directory
    if base_output_dir:
        output_dir = base_output_dir
    elif isinstance(app_config, dict) and 'OUTPUT_FOLDER' in app_config:
        output_dir = os.path.join(app_config['OUTPUT_FOLDER'], timestamp)
    else:
        output_dir = app_config
    
    # Create analyzer and use its method
    analyzer = create_cuneiform_analyzer()
    return analyzer.create_vector_representation(cuneiform_image, line_number, timestamp, output_dir)
