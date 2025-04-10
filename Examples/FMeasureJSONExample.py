#!/usr/bin/env python3
"""
Example demonstrating how to use IAAWrapper directly to calculate F-measure
and export the results to JSON.

This example shows:
1. How to create sample binary annotation data
2. How to configure IAAWrapper for F-measure
3. How to calculate F-measure with a specific positive class
4. How to export results to JSON
5. How to read and parse the JSON results
"""

import os
import sys
import json
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import after path modification
from src.iaa_wrapper import IAAWrapper  # noqa: E402


def create_sample_data():
    """Create a sample DataFrame with binary annotations from 3 annotators."""
    data = {
        'id': range(1, 11),  # 10 samples
        'text': [
            "Hate speech example",
            "Normal comment",
            "Offensive content",
            "Friendly message",
            "Hate speech detected",
            "Nice weather today",
            "Harmful content here",
            "Great job everyone",
            "This is offensive",
            "Hello world"
        ],
        # Binary annotations: 1 for hate speech/offensive, 0 for normal
        'Annotator1_score': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'Annotator2_score': [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        'Annotator3_score': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


def main():
    """Main function demonstrating the use of IAAWrapper for F-measure."""
    # Create sample data
    df = create_sample_data()

    # Get the example directory path
    example_dir = os.path.dirname(__file__)

    # Save the sample data to a CSV file
    input_file = os.path.join(example_dir, "binary_annotations.csv")
    df.to_csv(input_file, index=False)
    print(f"Created sample data file: {input_file}")

    # Configure the wrapper
    args = {
        'input_file': input_file,
        'output': os.path.join(example_dir, 'f_measure_results.json'),
        'output_format': 'json',
        'log_level': 'info',
        'all': False,
        'f_measure': True,  # Calculate F-measure
        # Specify positive class (1 = hate speech/offensive)
        'positive_class': 1,
        'confidence_interval': 0.95,
        'confidence_method': 'wilson',
        'bootstrap_samples': 1000
    }

    # Create and run the wrapper
    wrapper = IAAWrapper(args)

    try:
        # Run the complete analysis
        wrapper.run()
        print(f"Results have been saved to: {args['output']}")

        # Read and parse the JSON results
        with open(args['output'], 'r') as f:
            results = json.load(f)

        # Display the results in a readable format
        print("\nF-measure Results:")
        if 'f_measure' in results:
            f_results = results['f_measure']

            # Show overall results
            if 'overall' in f_results:
                print(f"\nOverall F-measure: {f_results['overall']:.4f}")

            # Show pairwise results
            if 'pairwise' in f_results:
                print("\nPairwise F-measures:")
                for pair, value in f_results['pairwise'].items():
                    print(f"{pair}: {value:.4f}")

            # Show confidence intervals if available
            if 'confidence_intervals' in results:
                ci = results['confidence_intervals']['f_measure']
                if 'overall' in ci:
                    overall_ci = ci['overall']
                    print("\nConfidence Intervals:")
                    print(f"Overall: {overall_ci['ci_lower']:.4f} - "
                          f"{overall_ci['ci_upper']:.4f}")

            # Show interpretations if available
            if 'interpretations' in results:
                if 'f_measure' in results['interpretations']:
                    print("\nInterpretation:")
                    interp = results['interpretations']['f_measure']
                    print(interp)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
