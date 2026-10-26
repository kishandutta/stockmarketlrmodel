import os
import sys
import runpy

# Ensure model directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

target_script = os.path.join(MODEL_DIR, "app.py")
runpy.run_path(target_script, run_name="__main__")
