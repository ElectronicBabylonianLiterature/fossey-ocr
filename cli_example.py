#!/usr/bin/env python3
"""
Command-line example for the refactored cuneiform line splitting system.

This script demonstrates how to use the new modular components without the web interface.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Import the refactored modules
from line_detection import create_line_detector
from ocr_utils import get_ocr_manager
from cuneiform_analyzer import create_cuneiform_analyzer
from image_utils import enhance_image_contrast, create_white_background
from utils import load_cuneiform_catalog, match_reference_with_catalog


def main():
    """Main CLI function to process a cuneiform image."""
    parser = argparse.ArgumentParser(
        description="Split cuneiform tablet images into individual lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_example.py input.jpg output_dir
  python cli_example.py --help
        """
    )
    
    parser.add_argument(
        'input_image',
        help='Path to the input cuneiform tablet image'
    )
    
    parser.add_argument(
        'output_dir',
        help='Directory to save the processed line images and analysis'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en'],
        help='OCR languages to use (default: en)'
    )
    
    parser.add_argument(
        '--googlecv',
        action='store_true',
        help='Use Google Cloud Vision OCR instead of EasyOCR'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input_image):
        print(f"❌ Error: Input image '{args.input_image}' not found")
        return 1
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🔍 Processing cuneiform image: {args.input_image}")
    print(f"📁 Output directory: {output_path}")
    print(f"🕒 Timestamp: {timestamp}")
    
    try:
        # Step 1: Initialize components
        if args.verbose:
            print("\n📋 Initializing components...")
        
        if args.googlecv:
            print("🔧 Using Google Cloud Vision OCR")
        else:
            print("🔧 Using EasyOCR")
        
        ocr_manager = get_ocr_manager(args.languages, args.googlecv)
        line_detector = create_line_detector()
        analyzer = create_cuneiform_analyzer()
        catalog = load_cuneiform_catalog()
        
        if args.verbose:
            print("   ✅ OCR Manager initialized")
            print("   ✅ Line Detector initialized")
            print("   ✅ Cuneiform Analyzer initialized")
            print("   ✅ Catalog loaded")
        
        # Step 2: Split image into lines
        print(f"\n🔄 Splitting image into lines...")
        result = line_detector.split_image_into_lines(args.input_image, str(output_path))
        
        if not result['success']:
            print(f"❌ Error: Failed to split image into lines")
            return 1
        
        print(f"   ✅ Found {result['total_lines']} lines")
        
        # Step 3: Analyze each line
        print(f"\n🔬 Analyzing individual lines...")
        analysis_results = []
        
        for i, line_region in enumerate(result['line_regions']):
            line_num = line_region['line_number']
            
            if args.verbose:
                print(f"   📏 Processing line {line_num}...")
            
            # Analyze the line using the cuneiform analyzer
            line_image_path = os.path.join(str(output_path), f"line_{line_num}_{timestamp}.jpg")
            
            if os.path.exists(line_image_path):
                line_analysis = analyzer.analyze_line_three_parts(
                    {'filename': line_image_path},
                    line_num,
                    timestamp,
                    str(output_path)
                )
                
                if line_analysis:
                    analysis_results.append(line_analysis)
                    
                    # Try to match with catalog
                    if line_analysis['reference_info']['text']:
                        catalog_match = match_reference_with_catalog(
                            line_analysis['reference_info']['text'],
                            catalog
                        )
                        line_analysis['catalog_match'] = catalog_match
                    
                    if args.verbose:
                        ref_text = line_analysis['reference_info']['text']
                        print(f"     📝 Reference text: '{ref_text}'")
                        if 'catalog_match' in line_analysis and line_analysis['catalog_match']:
                            print(f"     📚 Catalog match: {line_analysis['catalog_match']['text']}")
        
        # Step 4: Save final analysis
        print(f"\n💾 Saving analysis results...")
        
        import json
        analysis_file = output_path / f"analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'input_image': args.input_image,
                'total_lines': len(analysis_results),
                'lines': analysis_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Analysis saved to: {analysis_file}")
        print(f"   ✅ Line images saved to: {output_path}")
        print(f"   ✅ Visualization saved to: {result.get('visualization_path', 'N/A')}")
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📊 Summary: {len(analysis_results)} lines processed")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())