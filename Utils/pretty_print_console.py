from typing import Dict, Tuple
from tabulate import tabulate
from Utils.agreement_utils import (
    get_unique_annotators, create_agreement_matrix,
    get_confidence_method_display, get_confidence_level
)


def print_agreement_table(
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None,
        max_width: int = 120,
        file=None
) -> None:
    """
    Print a pretty table of agreement values between annotators.

    Args:
        agreements: Dictionary with annotator pairs as keys and agreement
        values.
        confidence_intervals: Optional dictionary with confidence interval
        info.
        max_width: Maximum width of the table in characters.
        file: File-like object to write to. Defaults to None (stdout).
    """
    # Get unique annotators
    annotators = get_unique_annotators(agreements)
    n_annotators = len(annotators)

    # Truncate annotator names if table would be too wide
    name_width = max(len(ann) for ann in annotators)
    cell_width = 8  # Minimum width for agreement value
    if confidence_intervals:
        cell_width = 16  # Width with CI

    total_width = name_width + (n_annotators * (cell_width + 3))

    # Create a mapping from original names to display names
    display_names = {}
    if total_width > max_width and n_annotators > 3:
        # Truncate names to make table fit
        max_name = (max_width - (n_annotators * (cell_width + 3))) // 2

        # Check if max_name is negative or too small
        if max_name <= 3:
            raise ValueError(
                f"Table width ({total_width}) exceeds maximum width "
                f"({max_width}). Cannot truncate names to fit. "
                "Please increase max_width or use fewer annotators.")

        for i, ann in enumerate(annotators):
            if len(ann) > max_name:
                display_names[ann] = ann[:max_name] + '...' + str(i)
            else:
                display_names[ann] = ann
    else:
        # No truncation needed
        for ann in annotators:
            display_names[ann] = ann

    # Create matrix of results
    matrix = create_agreement_matrix(
        agreements, annotators, confidence_intervals,
        format_func=lambda val: f"{val:.1%}"
    )

    # Print the table
    print(f"\nInter-annotator Agreement Matrix ({n_annotators} annotators)",
          file=file)
    if confidence_intervals:
        confidence_level = get_confidence_level(confidence_intervals)
        confidence_percent = int(confidence_level * 100)
        print(f"Values shown as: Agreement (CI lower-CI upper) with"
              f" p={confidence_percent}%", file=file)

    # Create header with display names
    header = [""] + [display_names[ann] for ann in annotators]

    # Print the table using tabulate
    print(tabulate(
        [[display_names[annotators[i]]] + row for i, row in enumerate(matrix)],
        headers=header,
        tablefmt="simple"
    ), file=file)

    # Print summary
    if confidence_intervals:
        confidence_level = get_confidence_level(confidence_intervals)
        confidence_percent = int(confidence_level * 100)
        method_display = get_confidence_method_display(confidence_intervals)

        print(f"\nSummary of Agreement Values with Confidence "
              f"Intervals (p={confidence_percent}%, {method_display}):",
              file=file)
    else:
        print("\nSummary of Agreement Values:", file=file)

    # Print each pair's agreement
    for (ann1, ann2), value in sorted(agreements.items()):
        if confidence_intervals and (ann1, ann2) in confidence_intervals:
            ci = confidence_intervals[(ann1, ann2)]
            print(f"  {ann1} & {ann2}: {value:.1%} "
                  f"[{ci['ci_lower']:.1%}-{ci['ci_upper']:.1%}]", file=file)
        else:
            print(f"  {ann1} & {ann2}: {value:.1%}", file=file)
