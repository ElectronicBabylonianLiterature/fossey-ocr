#!/usr/bin/env python3
"""
Google Cloud Vision OCR Helper

This module provides a wrapper around the Google Cloud Vision API to mimic the
behavior of easyocr, allowing it to be used as a drop-in replacement for text
detection in the cuneiform line splitting application.
"""
from google.cloud import vision
import cv2
import numpy as np
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/ocrafo-f5dd311467bc.json"

def readtext(image_np, allowlist=None):
    """
    Encodes a numpy image to bytes and sends it to Google Vision OCR.
    This function is a drop-in replacement for easyocr.Reader.readtext.
    
    Args:
        image_np (np.ndarray): The image in numpy array format (from cv2.imread).
        allowlist (str, optional): Unused, kept for compatibility.

    Returns:
        list: A list of (bounding_box, text, confidence) tuples, mimicking
              easyocr's output format. Returns an empty list on failure.
    """
    # Note: The GOOGLE_APPLICATION_CREDENTIALS environment variable must be set.
    
    # Encode image to bytes
    success, encoded_image = cv2.imencode('.jpg', image_np)
    if not success:
        print("❌ Failed to encode image to JPG format.")
        return []
    image_bytes = encoded_image.tobytes()

    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        
        # Use document_text_detection for dense text.
        response = client.document_text_detection(image=image)

        if response.error.message:
            raise Exception(
                f'{response.error.message}\\nFor more info on error messages, '
                'check: https://cloud.google.com/apis/design/errors')

        results = []
        if response.full_text_annotation:
            # Process based on paragraphs to get line-like structures
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        paragraph_text = ""
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            paragraph_text += word_text + " "
                        
                        # The bounding box from Google Vision is a list of 4 vertices (x, y objects).
                        # easyocr's format is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
                        vertices = paragraph.bounding_box.vertices
                        box_points = [[vertices[0].x, vertices[0].y], [vertices[1].x, vertices[1].y], [vertices[2].x, vertices[2].y], [vertices[3].x, vertices[3].y]]
                        
                        # Filter out results with empty text
                        if paragraph_text.strip():
                            results.append((box_points, paragraph_text.strip(), paragraph.confidence))
        return results

    except Exception as e:
        print(f"❌ Google Cloud Vision API error: {e}")
        return []

class GoogleVisionReader:
    """
    A mock reader class that mimics easyocr.Reader.
    Its readtext method calls the Google Vision API.
    """
    def __init__(self, languages):
        # Languages are not used by Google Vision in this context but kept for compatibility.
        pass

    def readtext(self, image_np, allowlist=None, detail=1, paragraph=False):
        # Add detail and paragraph parameters for full compatibility with easyocr call signature
        return readtext(image_np, allowlist=allowlist)
