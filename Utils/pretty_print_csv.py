from typing import Dict, Tuple
import csv
from Utils.agreement_utils import (
    get_unique_annotators, create_agreement_matrix,
    get_confidence_method_display, get_confidence_level
)


def save_agreement_csv(
        filename: str,
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None,
        agreement_name: str = "Agreement"
) -> None:
    """
    Save agreement results to a CSV file.

    Args:
        filename: Path to the output CSV file.
        agreements: Dictionary with annotator pairs as keys and agreement
        values.
        confidence_intervals: Optional dictionary with confidence interval
        info.
        agreement_name: Name of the agreement method to use in column headers.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        if confidence_intervals:
            writer.writerow([
                'Annotator1_name', 'Annotator2_name', agreement_name,
                'Lower_Bound_interval', 'Upper_bound_interval', 'p'
            ])
        else:
            writer.writerow(['Annotator1_name',
                             'Annotator2_name',
                             agreement_name])

        # Write data
        for pair, value in sorted(agreements.items()):
            if confidence_intervals and pair in confidence_intervals:
                ci = confidence_intervals[pair]
                writer.writerow([
                    pair[0], pair[1], f"{value:.4f}",
                    f"{ci['ci_lower']:.4f}", f"{ci['ci_upper']:.4f}",
                    f"{1 - ci.get('confidence_level', 0.95):.2f}"
                ])
            else:
                writer.writerow([pair[0], pair[1], f"{value:.4f}"])


def export_agreement_csv(
        filename: str,
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None,
        include_matrix: bool = True
        ) -> None:
    """
    Export agreement values and confidence intervals to a CSV file.

    Args:
        filename: Path to output CSV file.
        agreements: Dictionary with annotator pairs as keys and agreement
        values.
        confidence_intervals: Optional dictionary with confidence interval
        info.
        include_matrix: Whether to include the full agreement matrix in the
        CSV.
    """
    # Get unique annotators
    annotators = get_unique_annotators(agreements)

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header row with the exact format requested
        if confidence_intervals:
            writer.writerow([
                            'Annotator1_name',
                            'Annotator2_name',
                            'Agreement',
                            'Lower_Bound_interval',
                            'Upper_bound_interval',
                            'p'
                            ])
        else:
            writer.writerow([
                'Annotator1_name',
                'Annotator2_name',
                'Agreement'
            ])

        # Write agreement values and confidence intervals
        for (ann1, ann2), value in sorted(agreements.items()):
            if confidence_intervals and (ann1, ann2) in confidence_intervals:
                ci = confidence_intervals[(ann1, ann2)]
                confidence_level = ci.get('confidence_level', 0.95)
                writer.writerow([
                    ann1,
                    ann2,
                    f"{value:.4f}",
                    f"{ci['ci_lower']:.4f}",
                    f"{ci['ci_upper']:.4f}",
                    f"{1-confidence_level:.2f}"
                ])
            else:
                if confidence_intervals:
                    writer.writerow([
                        ann1,
                        ann2,
                        f"{value:.4f}",
                        '',
                        '',
                        ''
                    ])
                else:
                    writer.writerow([
                        ann1,
                        ann2,
                        f"{value:.4f}"
                    ])

        # If we don't want to include the matrix, stop here
        if not include_matrix:
            return

        # Add a blank row as separator
        writer.writerow([])

        # Include the full agreement matrix if requested
        if include_matrix:
            # Write header for the matrix section
            writer.writerow(['Agreement Matrix'])

            # Write the header row with annotator names
            header_row = ['']
            header_row.extend(annotators)
            writer.writerow(header_row)

            # Create and write matrix of results
            matrix = create_agreement_matrix(
                agreements, annotators, confidence_intervals,
                format_func=lambda val: f"{val:.4f}"
            )

            for i, row in enumerate(matrix):
                writer.writerow([annotators[i]] + row)

            # Add metadata if confidence intervals are available
            if confidence_intervals:
                writer.writerow([])
                confidence_level = get_confidence_level(confidence_intervals)
                method = get_confidence_method_display(confidence_intervals)
                writer.writerow([
                    f"Confidence Level: {confidence_level:.2f}",
                    f"Method: {method}"
                ])


def export_multi_agreement_csv(
        filename: str,
        agreements_dict: Dict[str, Dict[Tuple[str, str], float]],
        confidence_intervals_dict: Dict[str,
                                        Dict[Tuple[str, str],
                                             Dict[str, float]]] = None,
        use_method_names: bool = True
) -> None:
    """
    Export multiple agreement results to a single CSV file.

    Args:
        filename: Path to the output CSV file.
        agreements_dict: Dictionary with method names as keys and agreement
        dictionaries as values.
        confidence_intervals_dict: Dictionary with method names as keys and
        confidence interval dictionaries as values.
        use_method_names: If True, use method names as column headers.
                          If False, use 'Agreement' as column header.
    """
    # Get all unique annotator pairs across all methods
    all_pairs = set()
    for agreements in agreements_dict.values():
        all_pairs.update(agreements.keys())

    # Sort pairs for consistent output
    sorted_pairs = sorted(all_pairs)

    # Create header row
    header = ['Annotator1_name', 'Annotator2_name']

    # Add columns for each method
    for method_name in agreements_dict.keys():
        if (confidence_intervals_dict and
                method_name in confidence_intervals_dict):
            header.extend([
                f'{method_name}' if use_method_names else 'Agreement',
                'Lower_Bound_interval',
                'Upper_bound_interval',
                'p'
            ])
        else:
            header.append(
                f'{method_name}' if use_method_names else 'Agreement')

    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Write data for each pair
        for pair in sorted_pairs:
            row = [pair[0], pair[1]]

            # Add data for each method
            for method_name, agreements in agreements_dict.items():
                if pair in agreements:
                    value = agreements[pair]
                    row.append(f"{value:.4f}")

                    # Add confidence intervals if available
                    if (confidence_intervals_dict and
                            method_name in confidence_intervals_dict and
                            pair in confidence_intervals_dict[method_name]):

                        ci = confidence_intervals_dict[method_name][pair]
                        row.append(f"{ci['ci_lower']:.4f}")
                        row.append(f"{ci['ci_upper']:.4f}")
                        row.append(
                            f"{1 - ci.get('confidence_level', 0.95):.2f}")
                else:
                    row.append("N/A")
                    if (confidence_intervals_dict and
                            method_name in confidence_intervals_dict):
                        row.extend(["N/A", "N/A", "N/A"])

            writer.writerow(row)

    print(f"Multiple agreement results exported to {filename}")
