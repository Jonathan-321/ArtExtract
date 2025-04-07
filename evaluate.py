#!/usr/bin/env python3
"""
Wrapper script for evaluating the CNN-RNN classifier.
Run this script from the root directory of the project.
"""

import sys
from pathlib import Path

# Add the scripts directory to the path
script_dir = Path(__file__).parent / "scripts"
sys.path.append(str(script_dir))

# Import and run the main function from the evaluation script
from evaluate_cnn_rnn_classifier import main

if __name__ == '__main__':
    main() 