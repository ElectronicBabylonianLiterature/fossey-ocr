#!/usr/bin/env python3
"""
CLI Processing Utilities
Command-line interface specific processing functions.
"""

import os
import json
import re
from typing import Dict, List, Any
from datetime import datetime

from image_processing import split_image_into_lines, analyze_line_three_parts
from utils import match_reference_with_catalog, load_cuneiform_catalog
from web_utils import extract_page_info_from_filename
from ocr_utils import reset_ocr_manager


def process_image_cli(image_path: str, output_dir: str) -> Dict:
    """
    Run the full processing pipeline for CLI mode.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        
    Returns:
        Dictionary with processing results and metadata
    """
    print(f"⚙️ CLI mode: processing image {image_path} -> {output_dir}")
    
    # Split image into lines
    result, error = split_image_into_lines(image_path, output_dir)
    if error:
        print(f"❌ Error during line splitting: {error}")
        return {'success': False, 'error': error}

    print(f"✅ Completed splitting: {result.get('total_lines', 0)} lines found.")
    
    # Extract page information from filename
    page_number, suffix = extract_page_info_from_filename(image_path)
    print(f"📄 Extracted from filename: Page={page_number}, Suffix='{suffix}'")
    
    # Process each line
    final_json_output = process_lines_for_cli(result, output_dir, page_number, suffix)
    
    # Output and save results
    output_results_cli(final_json_output, output_dir)
    
    return {
        'success': True,
        'total_lines': len(final_json_output),
        'page_number': page_number,
        'suffix': suffix,
        'output_dir': output_dir
    }


def process_lines_for_cli(result: Dict, output_dir: str, page_number: int, suffix: str) -> List[Dict]:
    """
    Process individual lines and create CLI output format.
    
    Args:
        result: Line splitting result
        output_dir: Output directory
        page_number: Extracted page number
        suffix: Extracted suffix
        
    Returns:
        List of line analysis dictionaries
    """
    final_json_output = []
    line_regions = result.get('line_regions', [])
    timestamp = result.get('timestamp')
    catalog = load_cuneiform_catalog()

    for line_region in line_regions:
        line_number = line_region.get('line_number')
        
        # Find corresponding line file data
        line_file_data = next(
            (f for f in result.get('line_files', []) if f['line_number'] == line_number), 
            None
        )
        if not line_file_data:
            print(f"⚠️ Could not find file data for line {line_number}. Skipping.")
            continue

        # Analyze the three parts of the line
        analysis = analyze_line_parts_cli(line_file_data, line_number, timestamp, output_dir, suffix)
        if not analysis:
            print(f"⚠️ Analysis failed for line {line_number}. Skipping.")
            continue

        # Extract SVG content
        svg_content = extract_svg_content(analysis, output_dir)
        
        # Get reference text and catalog match
        reference_text = analysis.get('reference_info', {}).get('text', '')
        catalog_match = match_reference_with_catalog(reference_text, catalog)
        
        # Create CLI output format
        line_output = create_cli_line_output(
            page_number, line_number, suffix, reference_text, 
            catalog_match, svg_content
        )
        final_json_output.append(line_output)

    return final_json_output


def analyze_line_parts_cli(line_file_data: Dict, line_number: int, timestamp: str, 
                          output_dir: str, suffix: str) -> Dict:
    """
    Analyze line parts for CLI mode.
    
    Args:
        line_file_data: Line file data
        line_number: Line number
        timestamp: Processing timestamp
        output_dir: Output directory
        suffix: File suffix
        
    Returns:
        Analysis result dictionary
    """
    # Create mock app_config for CLI mode
    mock_app_config = {'OUTPUT_FOLDER': output_dir}
    
    return analyze_line_three_parts(
        line_file_data, line_number, timestamp, 
        mock_app_config, output_dir, suffix=suffix
    )


def extract_svg_content(analysis: Dict, output_dir: str) -> str:
    """
    Extract SVG content from analysis results.
    
    Args:
        analysis: Line analysis results
        output_dir: Output directory
        
    Returns:
        SVG content string (empty if SVG export is disabled)
    """
    # Only extract SVG content if SVG export is enabled
    if os.environ.get('EXPORT_SVG', '').lower() != 'true':
        return ""
    
    svg_content = ""
    vector_info = analysis.get('cuneiform_sign', {}).get('vector_info', {})
    
    if vector_info and 'svg_path' in vector_info:
        svg_path = os.path.join(output_dir, vector_info['svg_path'])
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                # Extract only the <svg> element itself
                full_svg = f.read()
                svg_match = re.search(r'<svg.*?</svg>', full_svg, re.DOTALL)
                if svg_match:
                    svg_content = svg_match.group(0)
        except FileNotFoundError:
            print(f"⚠️ SVG file not found: {svg_path}")
        except Exception as e:
            print(f"⚠️ Error reading SVG: {e}")
    
    return svg_content


def create_cli_line_output(page_number: int, line_number: int, suffix: str, 
                          reference_text: str, catalog_match: Any, svg_content: str) -> Dict:
    """
    Create CLI output format for a single line.
    
    Args:
        page_number: Page number
        line_number: Line number
        suffix: File suffix
        reference_text: Extracted reference text
        catalog_match: Catalog match result
        svg_content: SVG content string
        
    Returns:
        CLI output dictionary for the line
    """
    return {
        "page": page_number,
        "number": line_number,
        "suffix": suffix,
        "reference": reference_text,
        "newEdition": "",
        "secondaryLiterature": "",
        "cdliNumber": "",
        "museumNumber": None,
        "externalProject": "",
        "notes": "",
        "date": catalog_match if catalog_match else "",
        "transliteration": "",
        "sign": svg_content
    }


def output_results_cli(final_json_output: List[Dict], output_dir: str) -> None:
    """
    Output CLI results to stdout and save to file.
    
    Args:
        final_json_output: List of line output dictionaries
        output_dir: Output directory
    """
    # Print to stdout
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(final_json_output, indent=4))
    print("--- END JSON OUTPUT ---\n")

    # Save to file
    summary_path = os.path.join(output_dir, 'cli_output.json')
    try:
        with open(summary_path, 'w', encoding='utf-8') as sf:
            json.dump(final_json_output, sf, ensure_ascii=False, indent=4)
        print(f"📄 Full JSON output saved to: {summary_path}")
    except Exception as e:
        print(f"⚠️ Could not write full JSON output file: {e}")


def run_cli_mode(image_path: str, output_dir: str = None, use_google_vision: bool = False, export_svg: bool = False) -> int:
    """
    Main entry point for CLI mode processing.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory (optional, will be auto-generated if not provided)
        use_google_vision: Whether to use Google Cloud Vision OCR instead of EasyOCR
        export_svg: Whether to export SVG vector files
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Set OCR engine choice globally before processing
        if use_google_vision:
            print("🔧 Using Google Cloud Vision OCR")
            os.environ['USE_GOOGLE_VISION'] = 'true'
        else:
            print("🔧 Using EasyOCR")
            os.environ.pop('USE_GOOGLE_VISION', None)
        
        # Set SVG export flag
        if export_svg:
            print("🎨 SVG export enabled")
            os.environ['EXPORT_SVG'] = 'true'
        else:
            print("🎨 SVG export disabled")
            os.environ.pop('EXPORT_SVG', None)
        
        # Create output directory if not provided
        if not output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join('cli_outputs', timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the image
        result = process_image_cli(image_path, output_dir)
        
        if result['success']:
            print(f"🎉 CLI processing completed successfully!")
            return 0
        else:
            print(f"❌ CLI processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error during CLI processing: {e}")
        return 1