@echo off
echo ========================================
echo AI SURVEILLANCE SYSTEM - SETUP
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 4: Installing requirements...
pip install -r requirements.txt
echo.

echo Step 5: Installing the AI surveillance model package...
pip install -e .
echo.

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo To run the application:
echo 1. Activate environment: venv\Scripts\activate
echo 2. Run: streamlit run ui_streamlit.py
echo.
echo OR for command line: python complete_detector.py
echo.
pause