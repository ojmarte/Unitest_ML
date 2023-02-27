import pandas as pd
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import warnings

from data_preprocessing import (remove_symbols, fill_empty_values)

def test_remove_symbols():
    col = pd.Series(['foo$', 'bar@', 'baz%'])
    expected = pd.Series(['foo', 'bar', 'baz'])
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The default value of regex will change")
            result, null_count = remove_symbols(col, symbol='[$@%]')
    assert result.equals(expected)
    assert null_count == 0

def test_fill_empty_values():
    col = pd.Series([1, np.nan, 2, np.nan])
    fill_values = [0, 1, 2]
    probabilities = [0.5, 0.3, 0.2]
    type_inference = int
    expected = pd.Series([1, 1, 2, 1])
    num_trials = 1000  # Increase number of trials for more accurate results
    results = []
    null_counts = []
    for i in range(num_trials):
        result, null_count = fill_empty_values(col, fill_values, probabilities, type_inference)
        results.append(result)
        null_counts.append(null_count)
    # Check that the mode of the results matches the expected values
    assert pd.concat(results).mode()[0] == expected.mode()[0]
    assert pd.Series(null_counts).sum() == 0

