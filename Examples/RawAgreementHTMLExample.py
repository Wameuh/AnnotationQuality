#!/usr/bin/env python3
"""
Example demonstrating how to use IAAWrapper directly to calculate raw agreement
and export the results to HTML.

This example shows:
1. How to create sample annotation data
2. How to configure IAAWrapper
3. How to calculate raw agreement
4. How to export results to HTML
"""

import os
import sys
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import after path modification
from src.iaa_wrapper import IAAWrapper  # noqa: E402


def create_sample_data():
    """Create a sample DataFrame with annotations from 3 annotators."""
    data = {
        'id': [1, 2, 3, 4, 5],  # Added ID column
        'text': [
            "This is a positive review",
            "This is a negative review",
            "This is a neutral review",
            "Another positive review",
            "Another negative review"
        ],
        'Annotator1_score': [5, 1, 3, 5, 2],
        'Annotator2_score': [5, 2, 3, 4, 1],
        'Annotator3_score': [4, 1, 3, 5, 1]
    }
    return pd.DataFrame(data)


def main():
    """Main function demonstrating the use of IAAWrapper."""
    # Create sample data
    df = create_sample_data()

    # Get the example directory path
    example_dir = os.path.dirname(__file__)

    # Save the sample data to a CSV file
    input_file = os.path.join(example_dir, "sample_annotations.csv")
    df.to_csv(input_file, index=False)
    print(f"Created sample data file: {input_file}")

    # Configure the wrapper
    args = {
        'input_file': input_file,
        'output': os.path.join(example_dir, 'raw_agreement_results.html'),
        'output_format': 'html',
        'log_level': 'info',
        'all': False,
        'raw': True,  # We want to calculate raw agreement
        'confidence_interval': 0.95,  # 95% confidence interval
        'confidence_method': 'wilson',
        'bootstrap_samples': 1000
    }

    # Create and run the wrapper
    wrapper = IAAWrapper(args)

    try:
        # Run the complete analysis
        wrapper.run()
        print(f"Results have been saved to: {args['output']}")

        # You can also access the results programmatically
        if hasattr(wrapper, 'results') and 'raw' in wrapper.results:
            raw_results = wrapper.results['raw']
            print("\nRaw Agreement Results:")

            if 'overall' in raw_results:
                print(f"Overall agreement: {raw_results['overall']:.4f}")

            if 'pairwise' in raw_results:
                print("\nPairwise agreements:")
                for pair, agreement in raw_results['pairwise'].items():
                    print(f"{pair[0]} - {pair[1]}: {agreement:.4f}")

            # Access confidence intervals if available
            has_ci = (hasattr(wrapper, 'confidence_intervals') and
                      'raw' in wrapper.confidence_intervals)
            if has_ci:
                ci = wrapper.confidence_intervals['raw']
                if 'overall' in ci:
                    overall_ci = ci['overall']
                    ci_str = (f"({overall_ci['ci_lower']:.4f} - "
                              f"{overall_ci['ci_upper']:.4f})")
                    print(f"\nOverall agreement 95% CI: {ci_str}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
