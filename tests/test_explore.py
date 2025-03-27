"""Tests for the explore module."""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import statsaid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from statsaid.explore import (
    load_data,
    explore,
    suggest_preprocessing,
    suggest_models,
)


class TestExplore(unittest.TestCase):
    """Test class for the explore module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test dataframe
        self.df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, np.nan],
            'categorical1': ['A', 'B', 'A', 'C', np.nan],
            'categorical2': ['X', 'Y', 'Z', 'X', 'Y'],
        })
    
    def test_explore(self):
        """Test the explore function."""
        results = explore(self.df)
        
        # Check that the results have the expected structure
        self.assertIn('shape', results)
        self.assertIn('columns', results)
        self.assertIn('dtypes', results)
        self.assertIn('summary', results)
        self.assertIn('missing', results)
        self.assertIn('variable_types', results)
        
        # Check that the results have the expected values
        self.assertEqual(results['shape'], (5, 4))
        self.assertEqual(set(results['columns']), {'numeric1', 'numeric2', 'categorical1', 'categorical2'})
        
        # Check that the variable types are correctly identified
        self.assertIn('numeric1', results['variable_types']['numeric'])
        self.assertIn('numeric2', results['variable_types']['numeric'])
        self.assertIn('categorical1', results['variable_types']['categorical'])
        self.assertIn('categorical2', results['variable_types']['categorical'])
        
        # Check that missing values are correctly identified
        self.assertEqual(results['missing']['total']['numeric2'], 1)
        self.assertEqual(results['missing']['total']['categorical1'], 1)
    
    def test_suggest_preprocessing(self):
        """Test the suggest_preprocessing function."""
        suggestions = suggest_preprocessing(self.df)
        
        # Check that the suggestions have the expected structure
        self.assertIn('missing_values', suggestions)
        self.assertIn('normalization', suggestions)
        self.assertIn('encoding', suggestions)
        
        # Check that there are suggestions for columns with missing values
        self.assertIn('numeric2', suggestions['missing_values'])
        self.assertIn('categorical1', suggestions['missing_values'])
        
        # Check that there are normalization suggestions for numeric columns
        self.assertIn('numeric1', suggestions['normalization'])
        self.assertIn('numeric2', suggestions['normalization'])
        
        # Check that there are encoding suggestions for categorical columns
        self.assertIn('categorical1', suggestions['encoding'])
        self.assertIn('categorical2', suggestions['encoding'])
    
    def test_suggest_models(self):
        """Test the suggest_models function."""
        # Test with no study design
        models1 = suggest_models(self.df)
        
        # Check that the results have the expected structure
        self.assertIn('models', models1)
        self.assertIn('packages', models1)
        self.assertIn('python', models1['packages'])
        self.assertIn('r', models1['packages'])
        
        # Test with a study design
        models2 = suggest_models(self.df, study_design='cross_sectional')
        
        # Check that the results have the expected structure
        self.assertIn('models', models2)
        self.assertIn('packages', models2)
        
        # Check that the study design affects the recommendations
        # (cross_sectional should include chi-square and Fisher's exact test)
        model_names = ' '.join(models2['models'])
        self.assertIn('Chi-square', model_names)
        self.assertIn('Fisher', model_names)


if __name__ == '__main__':
    unittest.main()