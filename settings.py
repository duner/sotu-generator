import os
import sys
import urlparse

PROJECT_ROOT = os.path.realpath(os.path.dirname(__file__))

# Speech Generator
LANG_MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
TEXT_DIR = os.path.join(PROJECT_ROOT, 'data', 'sotus')
