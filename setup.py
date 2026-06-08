# setup.py
from setuptools import setup, find_packages

setup(
    name="ai_surveillance_model",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "torch>=1.9.0",
    ],
    author="Your Name",
    description="Reusable AI surveillance model for detection, tracking, and event analysis",
    python_requires=">=3.8",
)