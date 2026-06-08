#!/bin/bash

echo "========================================"
echo "AI SURVEILLANCE SYSTEM - SETUP"
echo "========================================"
echo ""

echo "Step 1: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.8+"
    exit 1
fi
echo "✓ Virtual environment created"
echo ""

echo "Step 2: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

echo "Step 3: Upgrading pip..."
python -m pip install --upgrade pip
echo ""

echo "Step 4: Installing requirements..."
pip install -r requirements.txt
echo ""

echo "Step 5: Installing the AI surveillance model package..."
pip install -e .
echo ""

echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo ""
echo "To run the application:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run: streamlit run ui_streamlit.py"
echo ""
echo "OR for command line: python complete_detector.py"
echo ""