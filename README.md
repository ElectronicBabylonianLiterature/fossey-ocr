# 🎯 Cuneiform Line Splitting System

A sophisticated web application that automatically splits cuneiform tablet photos into individual lines using advanced OCR and image processing techniques.

## ✨ Features

- **🔍 Automatic Line Detection**: Uses EasyOCR to detect line numbers on left/right edges
- **🎯 Special Padding System**: Smart padding with fixed boundaries for top/bottom lines and expanded boundaries for middle lines
- **📏 Precise Spacing**: Maintains minimum 20px distance between lines (almost touching)
- **🔧 Auto-Correction**: Automatically corrects OCR errors and overlapping regions
- **📊 3-Part Analysis**: Analyzes line number, cuneiform sign, and reference information
- **🖼️ Vector Output**: Generates SVG vector formats
- **📁 Multiple Downloads**: Individual JPG files, ZIP archives, and JSON analysis data
- **🌐 Web Interface**: Modern, responsive Flask web application
- **📈 High Success Rate**: 43%-100% OCR accuracy range

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd splttt
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python3 app.py
   ```

5. **Open in browser**
   ```
   http://localhost:3001
   ```

## 💻 Command Line Interface (CLI)

The system also supports command-line processing for batch operations and integration with other tools.

### CLI Usage

```bash
# Basic usage
python app.py --image path/to/cuneiform_image.jpg --output-dir output_folder

# Export SVG vector files (optional)
python app.py --image path/to/cuneiform_image.jpg --output-dir output_folder --exportAsSVG

# Use Google Cloud Vision OCR
python app.py --image path/to/cuneiform_image.jpg --output-dir output_folder --googlecv

# Combined options
python app.py --image path/to/cuneiform_image.jpg --output-dir output_folder --exportAsSVG --googlecv
```

### CLI Options

- `--image`: Path to the input cuneiform tablet image (required for CLI mode)
- `--output-dir`: Directory to save processed results (optional, auto-generated if not provided)
- `--exportAsSVG`: Export cuneiform signs as SVG vector files (optional, disabled by default)
- `--googlecv`: Use Google Cloud Vision OCR instead of EasyOCR (optional)

### CLI Output

The CLI mode generates:
- Individual line images (JPG format)
- JSON analysis file (`cli_output.json`)
- SVG vector files (only if `--exportAsSVG` is specified)
- Console output with processing status

## 📖 How It Works

### 1. Line Number Detection
- Scans left or right 15% area of the image
- Uses EasyOCR to detect line numbers
- Automatically determines the best edge for detection

### 2. Special Padding System
- **Top Line**: Fixed top boundary (20px), expanded bottom boundary (150px)
- **Middle Lines**: Both boundaries expanded (150px)
- **Bottom Line**: Expanded top boundary (150px), fixed bottom boundary (20px)

### 3. Collision Control
- Ensures minimum 20px distance between lines
- Automatically corrects overlapping regions
- Maintains optimal spacing for analysis

### 4. 3-Part Analysis
Each line is analyzed for:
- **Part 1**: Line number and position
- **Part 2**: Cuneiform sign extraction and vectorization
- **Part 3**: Reference text and metadata

## 📁 Project Structure

```
splttt/
├── app.py                 # Main Flask application
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── results.html      # Results page
│   └── line_detail.html  # Line detail page
├── static/               # CSS and JavaScript files
├── web_uploads/          # Uploaded images (created automatically)
├── web_outputs/          # Processed results (created automatically)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 API Endpoints

- `GET /` - Home page
- `POST /upload` - Upload and process image
- `GET /results/<timestamp>` - View results
- `GET /line_detail/<line_number>` - View line details
- `GET /download/<timestamp>/<filename>` - Download files
- `GET /download_json/<timestamp>` - Download JSON analysis
- `GET /download_all/<timestamp>` - Download ZIP archive
- `POST /api/process` - API endpoint for programmatic access

## 📊 Technical Specifications

- **OCR Engine**: EasyOCR + PyTorch
- **Image Processing**: OpenCV + scikit-image
- **Web Framework**: Flask
- **Frontend**: Bootstrap 5 + Font Awesome
- **Output Formats**: JPG, PNG, SVG, JSON, ZIP
- **Supported Input**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Maximum File Size**: 16MB

## 🎨 Sample Usage

1. **Upload Image**: Select a cuneiform tablet photo with visible line numbers
2. **Automatic Processing**: The system detects and splits lines automatically
3. **View Results**: Browse individual lines and analysis data
4. **Download**: Get individual files, ZIP archives, or JSON data

## 🔍 Analysis Output

The system generates comprehensive analysis including:
- Line coordinates and dimensions
- Cuneiform sign extraction
- Vector representations (SVG)
- Reference text detection
- Confidence scores
- Processing metadata

## 🛠️ Development

### Running in Development Mode
```bash
export FLASK_ENV=development
python3 app.py
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Contributors

The parser was initially developed by **Abdullah Bakla** ([abdullahbakla7323](https://github.com/abdullahbakla7323)). Some refinement, including a CLI implementation, refinements for the OCR and further parsing, was done by **Enrique Jiménez** ([ejimsan](https://github.com/ejimsan)).

## 📧 Support

For support, please open an issue in the GitHub repository.

---

**Version**: v2.0  
**Status**: Production Ready ✅  
**Last Updated**: September 2025
