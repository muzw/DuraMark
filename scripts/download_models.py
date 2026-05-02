#!/usr/bin/env python3
"""Standalone script to download DuraMark pre-trained model weights.

Can also be invoked via: duramark download-models
"""

import sys
import os

# Add parent directory to path so we can import duramark
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duramark.cli.download import main

if __name__ == "__main__":
    main()
