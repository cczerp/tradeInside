"""pytest bootstrap: put the project root on sys.path so tests can import
top-level modules (database.py, normalize.py, logging_config.py, ...)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
