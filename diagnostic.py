import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}\n")

print("Checking imports...")

try:
    import ultralytics
    print(f"✓ Ultralytics {ultralytics.__version__}")
except Exception as e:
    print(f"✗ Ultralytics: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import fastapi
    print(f"✓ FastAPI {fastapi.__version__}")
except Exception as e:
    print(f"✗ FastAPI: {e}")

try:
    from ai_surveillance_model import Detector, SimpleTracker, EventAnalyzer
    print("✓ AI Surveillance Model - imports successful")
    
    # Test initialization
    print("\nTesting model initialization...")
    detector = Detector()
    print("✓ Detector initialized")
    
    tracker = SimpleTracker()
    print("✓ Tracker initialized")
    
    analyzer = EventAnalyzer()
    print("✓ Event Analyzer initialized")
    
    print("\n✅ All tests passed! The model is ready to use.")
    
except Exception as e:
    print(f"✗ AI Surveillance Model error: {e}")
    import traceback
    traceback.print_exc()