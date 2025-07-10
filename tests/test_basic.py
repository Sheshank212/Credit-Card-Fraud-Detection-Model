"""
Basic smoke tests to ensure CI/CD pipeline works
"""

import pytest
import numpy as np


def test_basic_imports():
    """Test that basic imports work"""
    import pandas as pd
    import sklearn
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None


def test_numpy_functionality():
    """Test basic numpy functionality"""
    arr = np.array([1, 2, 3, 4, 5])
    assert len(arr) == 5
    assert np.mean(arr) == 3.0


def test_basic_math():
    """Test basic mathematical operations"""
    assert 2 + 2 == 4
    assert 3 * 3 == 9
    assert 10 / 2 == 5.0


class TestBasicFunctionality:
    """Basic test class for CI/CD validation"""

    def test_list_operations(self):
        """Test basic list operations"""
        test_list = [1, 2, 3]
        test_list.append(4)
        assert len(test_list) == 4
        assert test_list[-1] == 4

    def test_dictionary_operations(self):
        """Test basic dictionary operations"""
        test_dict = {"a": 1, "b": 2}
        test_dict["c"] = 3
        assert len(test_dict) == 3
        assert test_dict["c"] == 3

    def test_string_operations(self):
        """Test basic string operations"""
        test_string = "hello world"
        assert test_string.upper() == "HELLO WORLD"
        assert "world" in test_string