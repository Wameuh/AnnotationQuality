from typing import Dict, Tuple
from tabulate import tabulate
import csv


def print_agreement_table(
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None,
        max_width: int = 120
        ) -> None:
    """
    Print a pretty table of agreement values between annotators.

    Args:
        agreements: Dictionary with annotator pairs as keys and agreement
            values as values.
        confidence_intervals: Optional dictionary with annotator pairs as keys
            and confidence interval info as values.
        max_width: Maximum width of the table in characters. If exceeded,
            annotator names will be truncated.
    """
    # Get unique annotators
    annotators = sorted(
        list({ann for pair in agreements.keys() for ann in pair}))
    n_annotators = len(annotators)

    # Truncate annotator names if table would be too wide
    name_width = max(len(ann) for ann in annotators)
    cell_width = 8  # Minimum width for agreement value
    if confidence_intervals:
        cell_width = 16  # Width with CI

    total_width = name_width + (n_annotators * (cell_width + 3))
    if total_width > max_width and n_annotators > 3:
        # Truncate names to make table fit
        max_name = (max_width - (n_annotators * (cell_width + 3))) // 2
        annotators = [ann[:max_name] + '...' if len(ann) > max_name else ann
                      for ann in annotators]

    # Create matrix of results
    matrix = []
    for ann1 in annotators:
        row = []
        for ann2 in annotators:
            if ann1 == ann2:
                row.append("---")
            else:
                # Try both orderings of the pair
                pair1 = (ann1, ann2)
                pair2 = (ann2, ann1)

                if pair1 in agreements:
                    value = agreements[pair1]
                    pair = pair1
                elif pair2 in agreements:
                    value = agreements[pair2]
                    pair = pair2
                else:
                    row.append("N/A")
                    continue

                if confidence_intervals and pair in confidence_intervals:
                    ci = confidence_intervals[pair]
                    cell = (f"{value:.1%}\n"
                            f"[{ci['ci_lower']:.1%}-{ci['ci_upper']:.1%}]")
                else:
                    cell = f"{value:.1%}"
                row.append(cell)
        matrix.append(row)

    # Print table
    print(f"\nInter-annotator Agreement Matrix ({n_annotators} annotators):")
    print(tabulate(
        matrix,
        headers=annotators,
        showindex=annotators,
        tablefmt="grid",
        numalign="center",
        stralign="center"
    ))


def save_agreement_csv(
        filename: str,
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None
        ) -> None:
    """
    Save agreement values and confidence intervals to a CSV file.

    Args:
        filename: Path to output CSV file.
        agreements: Dictionary with annotator pairs as keys and agreement
            values as values.
        confidence_intervals: Optional dictionary with annotator pairs as keys
            and confidence interval info as values.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'Annotator 1',
            'Annotator 2',
            'Agreement',
            'CI Lower',
            'CI Upper'
        ])

        # Sort pairs for consistent output
        pairs = sorted(agreements.keys())

        # Write data rows
        for ann1, ann2 in pairs:
            value = agreements[(ann1, ann2)]
            row = [ann1, ann2, f"{value:.3%}"]
            if confidence_intervals and (ann1, ann2) in confidence_intervals:
                ci = confidence_intervals[(ann1, ann2)]
                row.extend([
                    f"{ci['ci_lower']:.3%}",
                    f"{ci['ci_upper']:.3%}"
                ])
            else:
                row.extend(['', ''])
            writer.writerow(row)
