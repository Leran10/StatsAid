#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example demonstrating the advanced statistical test selection.

This example shows how to:
1. Generate sample datasets with different characteristics
2. Get appropriate test recommendations based on data structure and study design
3. Visualize the test selection process with a flowchart
4. Calculate and interpret effect sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import statsaid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import statsaid as sa
from statsaid.stats import select_statistical_test, calculate_effect_size, plot_test_selection_flowchart, plot_effect_sizes

def generate_crosssectional_data(n_samples=200):
    """Generate a sample cross-sectional dataset."""
    np.random.seed(42)
    
    data = pd.DataFrame()
    
    # Continuous outcome
    data['blood_pressure'] = np.random.normal(120, 15, n_samples)
    
    # Categorical predictors
    data['treatment'] = np.random.choice(['Control', 'Treatment A', 'Treatment B'], n_samples)
    data['sex'] = np.random.choice(['Male', 'Female'], n_samples)
    
    # Continuous predictors
    data['age'] = np.random.normal(45, 15, n_samples)
    data['bmi'] = np.random.normal(25, 5, n_samples)
    
    # Add some effect of treatment on blood pressure
    data.loc[data['treatment'] == 'Treatment A', 'blood_pressure'] += 10
    data.loc[data['treatment'] == 'Treatment B', 'blood_pressure'] += 5
    
    # Add relationship between age and blood pressure
    data['blood_pressure'] += (data['age'] - 45) * 0.3
    
    # Binary outcome for classification examples
    data['hypertension'] = (data['blood_pressure'] > 140).astype(int)
    
    return data

def generate_longitudinal_data(n_subjects=50, n_timepoints=3):
    """Generate a sample longitudinal dataset."""
    np.random.seed(42)
    
    data = pd.DataFrame()
    
    subject_ids = []
    timepoints = []
    treatment = []
    outcome = []
    age = []
    sex = []
    
    for subject in range(n_subjects):
        subject_treatment = np.random.choice(['Control', 'Treatment'])
        subject_age = np.random.normal(45, 15)
        subject_sex = np.random.choice(['Male', 'Female'])
        
        # Generate base outcome with individual variation
        base_outcome = np.random.normal(100, 10)
        
        for time in range(n_timepoints):
            subject_ids.append(subject)
            timepoints.append(time)
            treatment.append(subject_treatment)
            age.append(subject_age)
            sex.append(subject_sex)
            
            # Treatment effect increases over time
            treatment_effect = 0
            if subject_treatment == 'Treatment':
                treatment_effect = 5 * time
                
            # Add time effect and noise
            current_outcome = base_outcome + treatment_effect + time * 2 + np.random.normal(0, 5)
            outcome.append(current_outcome)
    
    data['subject_id'] = subject_ids
    data['timepoint'] = timepoints
    data['treatment'] = treatment
    data['age'] = age
    data['sex'] = sex
    data['outcome'] = outcome
    
    # Binary outcome for classification examples
    data['high_outcome'] = (data['outcome'] > 110).astype(int)
    
    return data

def main():
    """Main function to demonstrate test selection capabilities."""
    # Generate datasets
    print("Generating sample datasets...")
    cross_data = generate_crosssectional_data()
    long_data = generate_longitudinal_data()
    
    # Create output directory if it doesn't exist
    os.makedirs("examples/output", exist_ok=True)
    
    # Case 1: Cross-sectional data with continuous outcome and one categorical predictor
    print("\n=== Case 1: Two-group comparison (t-test scenario) ===")
    # Filter to just two treatment groups for this example
    two_group_data = cross_data[cross_data['treatment'].isin(['Control', 'Treatment A'])].copy()
    
    # Get recommended tests
    test_rec1 = select_statistical_test(
        data=two_group_data,
        outcome='blood_pressure',
        predictors=['treatment'],
        study_design='cross_sectional'
    )
    
    # Print recommendations
    print("Data characteristics:")
    for key, value in test_rec1['data_characteristics'].items():
        if isinstance(value, dict):
            continue  # Skip nested dictionaries for clarity
        print(f"  - {key}: {value}")
    
    print("\nRecommended primary tests:")
    for test in test_rec1['recommended_tests']['primary']:
        print(f"  - {test}")
    
    print("\nRecommended secondary tests:")
    for test in test_rec1['recommended_tests']['secondary']:
        print(f"  - {test}")
    
    # Calculate effect size
    effect1 = calculate_effect_size(
        data=two_group_data,
        group_col='treatment',
        value_col='blood_pressure'
    )
    
    print("\nEffect size:")
    for metric, value in effect1['effect_sizes'].items():
        print(f"  - {metric}: {value:.3f}")
    
    print("\nInterpretation:")
    for metric, interpretation in effect1['interpretation'].items():
        print(f"  - {interpretation}")
    
    # Visualize test selection
    flowchart1 = plot_test_selection_flowchart(test_rec1['data_characteristics'])
    flowchart1.savefig('examples/output/test_selection_flowchart_1.png')
    print("Test selection flowchart saved as 'examples/output/test_selection_flowchart_1.png'")
    
    # Visualize effect sizes
    effect_plot1 = plot_effect_sizes(effect1)
    effect_plot1.savefig('examples/output/effect_sizes_1.png')
    print("Effect size plot saved as 'examples/output/effect_sizes_1.png'")
    
    # Case 2: Cross-sectional data with multiple predictors
    print("\n=== Case 2: Multiple regression scenario ===")
    
    # Get recommended tests
    test_rec2 = select_statistical_test(
        data=cross_data,
        outcome='blood_pressure',
        predictors=['treatment', 'sex', 'age', 'bmi'],
        study_design='cross_sectional'
    )
    
    # Print recommendations
    print("Data characteristics:")
    print(f"  - outcome_type: {test_rec2['data_characteristics']['outcome_type']}")
    print(f"  - sample_size: {test_rec2['data_characteristics']['sample_size']}")
    print(f"  - number of predictors: {len(test_rec2['data_characteristics']['predictor_variables'])}")
    
    print("\nRecommended primary tests:")
    for test in test_rec2['recommended_tests']['primary']:
        print(f"  - {test}")
    
    # Case 3: Longitudinal data
    print("\n=== Case 3: Longitudinal data analysis ===")
    
    # Get recommended tests
    test_rec3 = select_statistical_test(
        data=long_data,
        outcome='outcome',
        predictors=['treatment', 'age', 'sex', 'timepoint'],
        study_design='longitudinal'
    )
    
    # Print recommendations
    print("Data characteristics:")
    print(f"  - outcome_type: {test_rec3['data_characteristics']['outcome_type']}")
    print(f"  - has_repeated_measures: {test_rec3['data_characteristics']['has_repeated_measures']}")
    print(f"  - study_design: {test_rec3['data_characteristics']['study_design']}")
    
    print("\nRecommended primary tests:")
    for test in test_rec3['recommended_tests']['primary']:
        print(f"  - {test}")
    
    # Case 4: Binary outcome (classification)
    print("\n=== Case 4: Binary outcome (logistic regression scenario) ===")
    
    # Get recommended tests
    test_rec4 = select_statistical_test(
        data=cross_data,
        outcome='hypertension',
        predictors=['treatment', 'age', 'bmi'],
        study_design='cross_sectional'
    )
    
    # Print recommendations
    print("Data characteristics:")
    print(f"  - outcome_type: {test_rec4['data_characteristics']['outcome_type']}")
    
    print("\nRecommended primary tests:")
    for test in test_rec4['recommended_tests']['primary']:
        print(f"  - {test}")
    
    # Visualize test selection
    flowchart4 = plot_test_selection_flowchart(test_rec4['data_characteristics'])
    flowchart4.savefig('examples/output/test_selection_flowchart_4.png')
    print("Test selection flowchart saved as 'examples/output/test_selection_flowchart_4.png'")
    
    # Generate a summary report
    print("\nGenerating summary report...")
    
    with open("examples/output/test_selection_report.md", "w") as f:
        f.write("# Statistical Test Selection Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report demonstrates StatsAid's ability to recommend appropriate statistical tests based on:\n\n")
        f.write("- Data characteristics (variable types, sample size, distribution)\n")
        f.write("- Study design (cross-sectional, longitudinal, etc.)\n")
        f.write("- Research question structure\n\n")
        
        # Case 1
        f.write("## Case 1: Two-group Comparison\n\n")
        f.write("### Data Characteristics\n\n")
        f.write("- **Outcome**: Continuous (blood pressure)\n")
        f.write("- **Predictor**: Categorical with 2 groups (treatment)\n")
        f.write("- **Study Design**: Cross-sectional\n")
        f.write("- **Sample Size**: {}\n\n".format(test_rec1['data_characteristics']['sample_size']))
        
        f.write("### Recommended Tests\n\n")
        f.write("Primary: {}\n\n".format(", ".join(test_rec1['recommended_tests']['primary'])))
        f.write("Secondary: {}\n\n".format(", ".join(test_rec1['recommended_tests']['secondary'])))
        
        f.write("### Effect Size\n\n")
        for metric, interp in effect1['interpretation'].items():
            f.write("- {}\n".format(interp))
        f.write("\n")
        
        f.write("![Test Selection Flowchart](test_selection_flowchart_1.png)\n\n")
        f.write("![Effect Size Analysis](effect_sizes_1.png)\n\n")
        
        # Case 2
        f.write("## Case 2: Multiple Regression\n\n")
        f.write("### Data Characteristics\n\n")
        f.write("- **Outcome**: Continuous (blood pressure)\n")
        f.write("- **Predictors**: Multiple (treatment, sex, age, bmi)\n")
        f.write("- **Study Design**: Cross-sectional\n\n")
        
        f.write("### Recommended Tests\n\n")
        f.write("Primary: {}\n\n".format(", ".join(test_rec2['recommended_tests']['primary'])))
        
        # Case 3
        f.write("## Case 3: Longitudinal Data Analysis\n\n")
        f.write("### Data Characteristics\n\n")
        f.write("- **Outcome**: Continuous (outcome measure)\n")
        f.write("- **Predictors**: Multiple, including time\n")
        f.write("- **Study Design**: Longitudinal with repeated measures\n\n")
        
        f.write("### Recommended Tests\n\n")
        f.write("Primary: {}\n\n".format(", ".join(test_rec3['recommended_tests']['primary'])))
        
        # Case 4
        f.write("## Case 4: Binary Outcome (Classification)\n\n")
        f.write("### Data Characteristics\n\n")
        f.write("- **Outcome**: Binary (hypertension status)\n")
        f.write("- **Predictors**: Multiple (treatment, age, bmi)\n")
        f.write("- **Study Design**: Cross-sectional\n\n")
        
        f.write("### Recommended Tests\n\n")
        f.write("Primary: {}\n\n".format(", ".join(test_rec4['recommended_tests']['primary'])))
        
        f.write("![Test Selection Flowchart](test_selection_flowchart_4.png)\n\n")
        
        # Summary
        f.write("## Conclusion\n\n")
        f.write("StatsAid provides context-aware recommendations for statistical analysis by considering:\n\n")
        f.write("1. **The nature of your data** (distributions, types, sample size)\n")
        f.write("2. **Your research design** (cross-sectional, longitudinal, etc.)\n")
        f.write("3. **Appropriate statistical assumptions** (normality, etc.)\n")
        f.write("4. **Best practices for your field** (effect size reporting, etc.)\n\n")
        
        f.write("This helps researchers select the most appropriate statistical approaches for their specific research questions.")
    
    print("Summary report saved as 'examples/output/test_selection_report.md'")
    print("\nExample completed.")

if __name__ == "__main__":
    main()