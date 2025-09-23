#!/usr/bin/env python3
"""
Cuneiform Line Splitting Web Application
This Flask application uses methods (inspired from splitting-1.txt)
to split cuneiform photograph pages into individual lines.
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import json
from datetime import datetime
import traceback
import io
import zipfile
import glob
import re
import shutil
from image_processing import split_image_into_lines, create_detailed_analysis_json, analyze_line_three_parts
from utils import load_cuneiform_catalog, match_reference_with_catalog
from web_utils import (allowed_file, convert_numpy_types, prepare_upload_filename, 
                      validate_file_upload, create_session_data, save_session_file, 
                      get_suffix_from_filename)
from cli_utils import run_cli_mode

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cuneiform_line_splitter_2025'
app.config['UPLOAD_FOLDER'] = 'web_uploads'
app.config['OUTPUT_FOLDER'] = 'web_outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create folders used by the web app
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Cuneiform catalog is loaded once at startup
CUNEIFORM_CATALOG = load_cuneiform_catalog()

@app.route('/')
def index():
    """Main web page (index)."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle a file upload and run the line-splitting pipeline."""
    # Enable SVG export for web mode
    os.environ['EXPORT_SVG'] = 'true'
    
    # Validate file upload
    file = request.files.get('file')
    is_valid, error_message = validate_file_upload(file)
    
    if not is_valid:
        flash(error_message)
        return redirect(request.url)

    # Prepare filename and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = prepare_upload_filename(file.filename, timestamp)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Run the core splitting function
    result, error = split_image_into_lines(file_path, output_dir)
    if error:
        flash(f'Error: {error}')
        return redirect(url_for('index'))

    # Create and save session data
    session_data = create_session_data(filename, timestamp, result)
    save_session_file(session_data, output_dir)

    # Create detailed analysis JSON
    suffix = get_suffix_from_filename(filename)
    create_detailed_analysis_json(session_data['result'], timestamp, output_dir, suffix, app.config)

    return redirect(url_for('results', timestamp=timestamp))

@app.route('/results/<timestamp>')
def results(timestamp):
    """Display results for a specific timestamped run."""
    session_file = os.path.join(app.config['OUTPUT_FOLDER'], timestamp, 'session.json')
    
    if not os.path.exists(session_file):
        flash('Result not found')
        return redirect(url_for('index'))
    
    with open(session_file, 'r') as f:
        session_data = json.load(f)
    
    return render_template('results.html', data=session_data)

@app.route('/download/<timestamp>/<filename>')
def download_file(timestamp, filename):
    """Download a single output file from an output directory."""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], timestamp, filename)
    
    if not os.path.exists(file_path):
        flash('File not found')
        return redirect(url_for('index'))
    
    return send_file(file_path, as_attachment=True)

@app.route('/download_json/<timestamp>')
def download_json(timestamp):
    """Download the complete analysis JSON for a run."""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], timestamp)
    
    if not os.path.exists(output_dir):
        flash('Result not found')
        return redirect(url_for('index'))
    
    # JSON dosyasını oku
    json_file = os.path.join(output_dir, f'analysis_{timestamp}.json')
    
    if not os.path.exists(json_file):
        flash('JSON analysis file not found')
        return redirect(url_for('index'))
    
    return send_file(
        json_file,
        mimetype='application/json',
        as_attachment=True,
        download_name=f'cuneiform_analysis_{timestamp}.json'
    )

@app.route('/download_all/<timestamp>')
def download_all(timestamp):
    """Download all output files for a run as a ZIP archive."""
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], timestamp)
    
    if not os.path.exists(output_dir):
        flash('Result not found')
        return redirect(url_for('index'))
    
    # ZIP dosyası oluştur
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    
    return send_file(
        io.BytesIO(memory_file.read()),
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'cuneiform_lines_{timestamp}.zip'
    )

@app.route('/line_detail/<int:line_number>')
def line_detail(line_number):
    """Show detailed analysis page for a single line (three-part view)."""
    # Find session files
    session_files = glob.glob('web_outputs/*/session.json')
    if not session_files:
        return "❌ Session not found!", 404
    
    # En son session'ı al
    latest_session = max(session_files, key=os.path.getctime)
    
    with open(latest_session, 'r') as f:
        session_data = json.load(f)
    
    result = session_data.get('result', {})
    lines = result.get('line_files', [])
    
    # İlgili satırı bul
    target_line = None
    for line in lines:
        if line.get('line_number') == line_number:
            target_line = line
            break
    
    if not target_line:
        return f"❌ Line {line_number} not found!", 404
    
    # Satırın 3 kısmını analiz et
    base_output = os.path.join(app.config['OUTPUT_FOLDER'], session_data['timestamp'])
    
    # Determine suffix from original filename
    original_filename = session_data.get('filename', '')
    suffix = 'b' if '_b.' in original_filename.lower() else 'a'
    
    line_analysis = analyze_line_three_parts(target_line, line_number, session_data['timestamp'], app.config, base_output, suffix)
    
    return render_template('line_detail.html', 
                         line_number=line_number,
                         line_data=target_line,
                         analysis=line_analysis)

@app.route('/api/process', methods=['POST'])
def api_process():
    """API endpoint that accepts an image and returns a JSON result."""
    # Enable SVG export for web mode
    os.environ['EXPORT_SVG'] = 'true'
    
    # Validate file upload
    file = request.files.get('file')
    is_valid, error_message = validate_file_upload(file)
    
    if not is_valid:
        return jsonify({'error': error_message}), 400

    # Process file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = prepare_upload_filename(file.filename, timestamp)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run line splitting
    result, error = split_image_into_lines(file_path, output_dir)
    
    if error:
        return jsonify({'error': error}), 400
    
    # Convert result to JSON serializable format
    result_serializable = convert_numpy_types(result)
    return jsonify(result_serializable)

if __name__ == '__main__':
    # Allow running either as the web app (default) or as a CLI tool.
    import argparse

    parser = argparse.ArgumentParser(description='Cuneiform line splitting web + CLI')
    parser.add_argument('--image', help='Path to an image file to process (CLI mode)')
    parser.add_argument('--output-dir', help='Output directory for CLI mode')
    parser.add_argument('--exportAsSVG', action='store_true', help='Export cuneiform signs as SVG vector files (CLI mode only)')
    parser.add_argument('--googlecv', action='store_true', help='Use Google Cloud Vision OCR instead of EasyOCR')
    parser.add_argument('--host', default='0.0.0.0', help='Flask host')
    parser.add_argument('--port', type=int, default=3001, help='Flask port')
    args = parser.parse_args()

    if args.image:
        # Run in CLI mode using the dedicated utility
        exit_code = run_cli_mode(args.image, args.output_dir, args.googlecv, args.exportAsSVG)
        raise SystemExit(exit_code)

    print("🚀 Cuneiform Line Splitting Web Application Starting...")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("📁 Output folder:", app.config['OUTPUT_FOLDER'])
    print(f"🌐 Website: http://localhost:{args.port}")
    app.run(debug=True, host=args.host, port=args.port)
