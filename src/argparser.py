import argparse
import os
import sys
from typing import Dict, Any


class CustomHelpFormatter(argparse.HelpFormatter):
    """
    Custom help formatter that provides more detailed and organized help.
    """
    def _format_action(self, action):
        # For the help action, we'll add our custom help after the standard
        # help
        if action.dest == 'help':
            result = super()._format_action(action)
            # We'll add a note about --show-options
            result += ("\n\nFor a more user-friendly overview of options with "
                       "examples, use --show-options\n")
            return result
        return super()._format_action(action)


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments for the IAA-Eval tool.

    Returns:
        Dict[str, Any]: Dictionary containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "IAA-Eval: A tool for Inter-Annotator Agreement evaluation"),
        formatter_class=CustomHelpFormatter,
        add_help=True
    )

    # Special option to show all available options
    parser.add_argument(
        "--show-options",
        action="store_true",
        help=(
            "Show all available options and examples in a user-friendly "
            "format")
    )

    # Input file arguments
    parser.add_argument(
        "input_file",
        nargs="?",  # Make input_file optional when --show-options is used
        help="Path to the input file containing annotation data (CSV format)"
    )

    # Output arguments
    parser.add_argument(
        "--output",
        help="Path to save the results (default: print to console)"
    )

    parser.add_argument(
        "--output-format",
        choices=["text", "csv", "json", "html"],
        default="text",
        help="Format of the output file"
    )

    # Logging arguments - use -v with a number
    parser.add_argument(
        "-v",
        dest="verbose",  # Explicitly set the destination attribute name
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Set verbosity level (0=error, 1=warning, 2=info, 3=debug)"
    )

    # Agreement measures to calculate
    measures_group = parser.add_argument_group("Agreement Measures")

    measures_group.add_argument(
        "--all",
        action="store_true",
        help="Calculate all applicable agreement measures"
    )

    measures_group.add_argument(
        "--raw",
        action="store_true",
        help="Calculate raw agreement (percentage agreement)"
    )

    measures_group.add_argument(
        "--cohen-kappa",
        action="store_true",
        help="Calculate Cohen's Kappa (for two annotators)"
    )

    measures_group.add_argument(
        "--fleiss-kappa",
        action="store_true",
        help="Calculate Fleiss' Kappa (for three or more annotators)"
    )

    measures_group.add_argument(
        "--krippendorff-alpha",
        action="store_true",
        help="Calculate Krippendorff's Alpha"
    )

    measures_group.add_argument(
        "--f-measure",
        action="store_true",
        help="Calculate F-measure"
    )

    measures_group.add_argument(
        "--icc",
        action="store_true",
        help="Calculate Intraclass Correlation Coefficient (ICC)"
    )

    measures_group.add_argument(
        "--bwfk",
        action="store_true",
        help=(
            "Calculate Boundary-Weighted Fleiss' Kappa "
            "(for binary segmentation)")
    )

    measures_group.add_argument(
        "--dbcaa",
        action="store_true",
        help=(
            "Calculate Distance-Based Cell Agreement Algorithm "
            "(for cell annotations)")
    )

    measures_group.add_argument(
        "--iou",
        action="store_true",
        help="Calculate Intersection over Union (for binary segmentation)"
    )

    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")

    advanced_group.add_argument(
        "--confidence-interval",
        type=float,
        default=0.95,
        help="Confidence level for interval calculations (between 0 and 1)"
    )

    advanced_group.add_argument(
        "--confidence-method",
        choices=["bootstrap", "normal", "wilson", "agresti-coull"],
        default="bootstrap",
        help="Method to calculate confidence intervals"
    )

    advanced_group.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals"
    )

    advanced_group.add_argument(
        "--positive-class",
        help="Specify the positive class for F-measure calculation"
    )

    advanced_group.add_argument(
        "--distance-threshold",
        type=float,
        default=10.0,
        help="Distance threshold for DBCAA calculation"
    )

    advanced_group.add_argument(
        "--bwfk-width",
        type=int,
        default=5,
        help="Width parameter for BWFK calculation"
    )

    advanced_group.add_argument(
        "--icc-form",
        choices=["1,1", "2,1", "3,1", "1,k", "2,k", "3,k"],
        default="2,1",
        help="ICC form to use as the primary result"
    )

    # Parse the arguments
    if '--help' in sys.argv or '-h' in sys.argv:
        # If help is requested, show options after the standard help
        parser.print_help()
        print("\n")
        print_available_options()
        sys.exit(0)

    args = parser.parse_args()

    # Convert verbosity to log_level
    args.log_level = _verbosity_to_log_level(args.verbose)

    # If --show-options is specified, print options and exit
    if args.show_options:
        print_available_options()
        sys.exit(0)

    # Validate input file (only if not showing options)
    if not args.input_file:
        parser.error("Input file is required")

    if not os.path.exists(args.input_file):
        parser.error(f"Input file does not exist: {args.input_file}")

    if args.confidence_interval <= 0 or args.confidence_interval >= 1:
        parser.error("Confidence interval must be between 0 and 1")

    if args.bootstrap_samples <= 0:
        parser.error("Number of bootstrap samples must be positive")

    # If no specific measures are selected, use --all
    measure_args = [
        args.raw, args.cohen_kappa, args.fleiss_kappa, args.krippendorff_alpha,
        args.f_measure, args.icc, args.bwfk, args.dbcaa, args.iou
    ]

    if not any(measure_args) and not args.all:
        args.all = True

    # Convert namespace to dictionary
    return vars(args)


def print_available_options():
    """
    Print all available options for the IAA-Eval tool in a user-friendly
    format.
    This function provides a quick overview of the tool's capabilities.
    """
    print("\n" + "=" * 80)
    print("IAA-EVAL: AVAILABLE OPTIONS".center(80))
    print("=" * 80)

    # Basic usage
    print("\nBASIC USAGE:")
    print("  iaa-eval input_file.csv [options]")

    # Input options
    print("\nINPUT OPTIONS:")
    print("  input_file.csv              CSV file containing annotation data")

    # Output options
    print("\nOUTPUT OPTIONS:")
    print(
        "  --output FILE                Path to save results "
        "(default: print to console)")
    print(
        "  --output-format FORMAT       Format for output: text, csv, "
        "json, html (default: text)")

    # Logging options
    print("\nLOGGING OPTIONS:")
    print("  -v LEVEL                     Set verbosity level:")
    print("                               0=error, 1=warning, 2=info, "
          "3=debug (default: 0)")

    # Agreement measures
    print("\nAGREEMENT MEASURES:")
    print(
        "  --all                        Calculate all applicable agreement "
        "measures")
    print(
        "  --raw                        Calculate raw agreement "
        "(percentage agreement)")
    print(
        "  --cohen-kappa                Calculate Cohen's Kappa "
        "(for two annotators)")
    print(
        "  --fleiss-kappa               Calculate Fleiss' Kappa "
        "(for three or more annotators)")
    print(
        "  --krippendorff-alpha         Calculate Krippendorff's Alpha")
    print(
        "  --f-measure                  Calculate F-measure")
    print(
        "  --icc                        Calculate Intraclass Correlation "
        "Coefficient")
    print(
        "  --bwfk                       Calculate Boundary-Weighted Fleiss' "
        "Kappa")
    print(
        "  --dbcaa                      Calculate Distance-Based Cell "
        "Agreement Algorithm")
    print(
        "  --iou                        Calculate Intersection over Union")

    # Advanced options
    print("\nADVANCED OPTIONS:")
    print(
        "  --confidence-interval VALUE  Confidence level (0-1) for intervals "
        "(default: 0.95)")
    print(
        "  --confidence-method METHOD   Method for confidence intervals: "
        "bootstrap, normal, wilson, agresti-coull (default: bootstrap)")
    print(
        "  --bootstrap-samples N        Number of bootstrap samples "
        "(default: 1000)")
    print(
        "  --positive-class CLASS       Specify positive class for F-measure")
    print(
        "  --distance-threshold VALUE   Distance threshold for DBCAA "
        "(default: 10.0)")
    print(
        "  --bwfk-width N               Width parameter for BWFK "
        "(default: 5)")
    print(
        "  --icc-form FORM              ICC form to use: 1,1|2,1|3,1|1,"
        "k|2,k|3,k (default: 2,1)")

    # Examples
    print("\nEXAMPLES:")
    print("  # Calculate all agreement measures")
    print("  iaa-eval annotations.csv --all")
    print()
    print("  # Calculate only Cohen's Kappa and Fleiss' Kappa")
    print("  iaa-eval annotations.csv --cohen-kappa --fleiss-kappa")
    print()
    print("  # Calculate ICC with specific form and save results to CSV")
    print(
        "  iaa-eval annotations.csv --icc --icc-form 3,1 --output "
        "results.csv --output-format csv")
    print()
    print("  # Calculate F-measure with a specific positive class")
    print("  iaa-eval annotations.csv --f-measure --positive-class 1")

    print("\n" + "=" * 80)
    print("For more detailed help, use: iaa-eval --help")
    print("=" * 80 + "\n")


def _verbosity_to_log_level(verbosity: int) -> str:
    """
    Convert verbosity level to log level.

    Args:
        verbosity (int): Verbosity level (0-3)

    Returns:
        str: Log level name
    """
    if verbosity == 0:
        return "error"
    elif verbosity == 1:
        return "warning"
    elif verbosity == 2:
        return "info"
    else:  # verbosity == 3
        return "debug"


if __name__ == "__main__":
    # Example usage
    args = parse_arguments()
    print("Parsed arguments:")
    for key, value in args.items():
        print(f"  {key}: {value}")
