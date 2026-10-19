"""Makes `src/` importable from tests without installing the project."""
import os
import sys

# Add the project's src/ folder to Python's import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
