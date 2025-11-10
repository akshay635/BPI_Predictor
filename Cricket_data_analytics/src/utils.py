# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:57:46 2025

@author: admin
"""


import os

def safe_divide(a, b):
    """Avoid division by zero."""
    return a / b if b != 0 else 0


def ensure_dir_exists(path):
    """Create folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


import logging

def setup_logger():
    """Setup a simple project-wide logger."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("IPLProject")
 