# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:12:29 2025

@author: admin
"""

from src.utils import safe_divide

def test_safe_divide():
    assert safe_divide(10, 2) == 5
    assert safe_divide(10, 0) == 0
