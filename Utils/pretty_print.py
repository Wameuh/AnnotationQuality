from typing import Dict, Tuple, List, Any
from tabulate import tabulate
import csv
import io
import base64
import numpy as np
from datetime import datetime


def get_unique_annotators(agreements: Dict[Tuple[str, str],
                                           float]) -> List[str]:
    """
    Extract unique annotators from agreement dictionary.

    Args:
        agreements: Dictionary with annotator pairs as keys and agreement
        values.

    Returns:
        List of unique annotator names, sorted alphabetically.
    """
    return sorted(list({ann for pair in agreements.keys() for ann in pair}))


def create_agreement_matrix(
        agreements: Dict[Tuple[str, str], float],
        annotators: List[str],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None,
        format_func=None
) -> List[List[str]]:
    """
    Create a matrix of agreement values.

    Args:
        agreements: Dictionary with annotator pairs as keys and agreement
        values.
        annotators: List of unique annotator names.
        confidence_intervals: Optional dictionary with confidence interval
        info.
        format_func: Optional function to format agreement values.

    Returns:
        Matrix of agreement values as a list of lists.
    """
    if format_func is None:
        # Default format function for percentage display
        def default_format(val):
            return f"{val:.1%}"
        format_func = default_format

    matrix = []
    for i, ann1 in enumerate(annotators):
        row = []
        for j, ann2 in enumerate(annotators):
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

                # Format the cell based on whether we have confidence intervals
                if confidence_intervals and pair in confidence_intervals:
                    ci = confidence_intervals[pair]
                    cell = format_with_confidence_interval(value,
                                                           ci,
                                                           format_func)
                else:
                    cell = format_func(value)
                row.append(cell)
        matrix.append(row)
    return matrix


def format_with_confidence_interval(
        value: float,
        ci: Dict[str, float],
        format_func=None
) -> str:
    """
    Format a value with its confidence interval.

    Args:
        value: The agreement value.
        ci: Dictionary with confidence interval information.
        format_func: Function to format the values.

    Returns:
        Formatted string with value and confidence interval.
    """
    if format_func is None:
        def default_format(val):
            return f"{val:.1%}"
        format_func = default_format

    return (f"{format_func(value)} ({format_func(ci['ci_lower'])}"
            f" - {format_func(ci['ci_upper'])})")


def get_confidence_method_display(
        confidence_intervals: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    """
    Get the display name for the confidence interval method.

    Args:
        confidence_intervals: Dictionary with confidence interval information.

    Returns:
        Display name for the confidence interval method.
    """
    method = "wilson"  # Default method
    if confidence_intervals and len(confidence_intervals) > 0:
        first_ci = next(iter(confidence_intervals.values()))
        if 'method' in first_ci:
            method = first_ci['method']

    # Format the method name for display
    method_display = {
        "wilson": "Wilson score",
        "normal": "Normal approximation",
        "agresti_coull": "Agresti-Coull",
        "clopper_pearson": "Clopper-Pearson (exact)",
        "bootstrap": "Bootstrap",
        "standard": "Standard",
        "fallback": "Estimated"
    }.get(method, method)

    return method_display


def get_confidence_level(
        confidence_intervals: Dict[Tuple[str, str], Dict[str, Any]]) -> float:
    """
    Get the confidence level from confidence intervals.

    Args:
        confidence_intervals: Dictionary with confidence interval information.

    Returns:
        Confidence level as a float (0-1).
    """
    if not confidence_intervals or len(confidence_intervals) == 0:
        return 0.95  # Default confidence level

    first_ci = next(iter(confidence_intervals.values()))
    return first_ci.get('confidence_level', 0.95)


def print_agreement_table(
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None,
        max_width: int = 120
) -> None:
    """
    Print a pretty table of agreement values between annotators.

    Args:
        agreements: Dictionary with annotator pairs as keys and agreement
        values.
        confidence_intervals: Optional dictionary with confidence interval
        info.
        max_width: Maximum width of the table in characters.
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
    print(f"\nInter-annotator Agreement Matrix ({n_annotators} annotators)")
    if confidence_intervals:
        confidence_level = get_confidence_level(confidence_intervals)
        confidence_percent = int(confidence_level * 100)
        print(f"Values shown as: Agreement (CI lower-CI upper) with"
              f" p={confidence_percent}%")

    # Create header with display names
    header = [""] + [display_names[ann] for ann in annotators]

    # Print the table using tabulate
    print(tabulate(
        [[display_names[annotators[i]]] + row for i, row in enumerate(matrix)],
        headers=header,
        tablefmt="simple"
    ))

    # Print summary
    if confidence_intervals:
        confidence_level = get_confidence_level(confidence_intervals)
        confidence_percent = int(confidence_level * 100)
        method_display = get_confidence_method_display(confidence_intervals)

        print(f"\nSummary of Agreement Values with Confidence "
              f"Intervals (p={confidence_percent}%, {method_display}):")
    else:
        print("\nSummary of Agreement Values:")

    # Print each pair's agreement
    for (ann1, ann2), value in sorted(agreements.items()):
        if confidence_intervals and (ann1, ann2) in confidence_intervals:
            ci = confidence_intervals[(ann1, ann2)]
            print(f"  {ann1} & {ann2}: {value:.1%} "
                  f"[{ci['ci_lower']:.1%}-{ci['ci_upper']:.1%}]")
        else:
            print(f"  {ann1} & {ann2}: {value:.1%}")


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
    import csv

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


def save_agreement_html(
        filename: str,
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Dict[Tuple[str, str], Dict[str, float]] = None,
        title: str = "Inter-annotator Agreement Results",
        include_heatmap: bool = True
) -> None:
    """
    Save agreement results to an HTML file with visualization.

    Args:
        filename: Path to output HTML file.
        agreements: Dictionary with annotator pairs as keys and agreement
        values.
        confidence_intervals: Optional dictionary with confidence interval
        info.
        title: Title for the HTML page.
        include_heatmap: Whether to include the heatmap visualization.
    """
    # Get unique annotators
    annotators = get_unique_annotators(agreements)
    n_annotators = len(annotators)

    # Create matrix of results
    matrix = create_agreement_matrix(
        agreements, annotators, confidence_intervals,
        format_func=lambda val: f"{val:.1%}")

    # Create heatmap data
    heatmap_data = []
    for i, ann1 in enumerate(annotators):
        row = []
        for j, ann2 in enumerate(annotators):
            if ann1 == ann2:
                row.append(1.0)  # Perfect agreement with self
            else:
                # Try both orderings of the pair
                pair1 = (ann1, ann2)
                pair2 = (ann2, ann1)

                if pair1 in agreements:
                    row.append(agreements[pair1])
                else:
                    row.append(agreements[pair2])
        heatmap_data.append(row)

    # Get confidence interval method display name if available
    method_display = ""
    significance_level = 0.95
    if confidence_intervals and len(confidence_intervals) > 0:
        method_display = get_confidence_method_display(confidence_intervals)
        significance_level = get_confidence_level(confidence_intervals)

    # Create summary table
    summary_rows = []
    for (ann1, ann2), value in sorted(agreements.items()):
        if confidence_intervals and (ann1, ann2) in confidence_intervals:
            ci = confidence_intervals[(ann1, ann2)]
            ci_lower = ci['ci_lower']
            ci_upper = ci['ci_upper']
            ci_class = get_confidence_interval_class(ci_lower, ci_upper)
            summary_rows.append(
                f'<tr><td>{ann1} & {ann2}</td>'
                f'<td class="{get_agreement_class(value)}">{value:.1%} '
                f'<span class="{ci_class}">[{ci_lower:.1%}-{ci_upper:.1%}]'
                f'</span></td></tr>'
            )
        else:
            summary_rows.append(
                f'<tr><td>{ann1} & {ann2}</td>'
                f'<td class="{get_agreement_class(value)}"'
                f'>{value:.1%}</td></tr>'
            )

    summary_table = f"""
    <h2>Summary of Agreement Values</h2>
    <table class="summary">
        <tr>
            <th>Annotator Pair</th>
            <th>Agreement</th>
        </tr>
        {''.join(summary_rows)}
    </table>
    """

    # Generate heatmap image if requested
    heatmap_img = ""
    if include_heatmap:
        # Import matplotlib only when needed
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        # Create a masked array for the heatmap
        masked_data = np.ma.masked_invalid(heatmap_data)

        # Create a custom colormap with 6 colors
        colors = [
            '#000000',  # black for 0.0 (no agreement)
            '#8b0000',  # dark red for 0.2 (very low agreement)
            '#ff0000',  # red for 0.4 (low agreement)
            '#ffa500',  # orange for 0.6 (medium agreement)
            '#ffff00',  # yellow for 0.8 (good agreement)
            '#008000',  # green for 1.0 (excellent agreement)
        ]
        # 6 positions corresponding to the 6 colors
        positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", list(zip(positions, colors)))

        # Create heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(label='Agreement')

        # Add labels
        plt.xticks(range(n_annotators), annotators, rotation=45)
        plt.yticks(range(n_annotators), annotators)

        # Add values in cells
        for i in range(n_annotators):
            for j in range(n_annotators):
                if not np.ma.is_masked(masked_data[i, j]):
                    color = "black" if masked_data[i, j] > 0.5 else "black"
                    plt.text(j, i, f'{masked_data[i, j]:.2f}',
                             ha="center", va="center", color=color)

        plt.title('Agreement Heatmap')
        plt.tight_layout()

        # Save to base64 for embedding in HTML
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        plt.close()

        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.read()).decode('utf-8')
        heatmap_img = (f'<img src="data:image/png;base64,{img_base64}"'
                       f' alt="Agreement Heatmap">')

    # Create HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: center;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .high-agreement {{
            background-color: #d4edda;
            color: #155724;
        }}
        .medium-agreement {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .low-agreement {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .wide-interval {{
            color: #721c24;
        }}
        .medium-interval {{
            color: #856404;
        }}
        .narrow-interval {{
            color: #155724;
        }}
        .summary {{
            width: auto;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 0.8em;
            color: #777;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .legend {{
            font-style: italic;
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>

    <h2>Agreement Matrix ({n_annotators} annotators)</h2>
    <table>
        <tr>
            <th></th>
            {' '.join(f'<th>{ann}</th>' for ann in annotators)}
        </tr>
    {''.join(
        f'<tr><th>{annotators[i]}</th>'
        f'{" ".join(
            get_cell_html(cell,
                          val) for cell, val in zip(row,
                                                    heatmap_data[i]))}</tr>'
        for i, row in enumerate(matrix)
    )}
</table>
    {confidence_intervals and
     f'<p class="legend">Note: Confidence intervals are calculated using the '
     f'{method_display} method with p = {1 - significance_level:.2f}</p>' or ''
     }

    {summary_table}

    {heatmap_img}

    <div class="footer">
        <p>Generated by <a href="https://github.com/Wameuh/AnnotationQuality"
           target="_blank">IAA-Eval</a> on
           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""

    # Write HTML to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)


def get_cell_html(cell_content, value):
    """Helper function to generate HTML for a table cell with color coding."""
    if cell_content == "---":
        return f'<td>{cell_content}</td>'
    elif cell_content == "N/A":
        return f'<td>{cell_content}</td>'
    else:
        # Color code based on agreement value
        css_class = get_agreement_class(value)
        return f'<td class="{css_class}">{cell_content}</td>'


def get_agreement_class(value):
    """Get CSS class based on agreement value."""
    if value >= 0.8:
        return "high-agreement"
    elif value >= 0.6:
        return "medium-agreement"
    else:
        return "low-agreement"


def get_confidence_interval_class(lower, upper):
    """Get CSS class based on confidence interval width."""
    width = upper - lower
    if width > 0.2:
        return "wide-interval"
    elif width > 0.1:
        return "medium-interval"
    else:
        return "narrow-interval"


def export_multi_agreement_csv(
        filename: str,
        agreements_dict: Dict[str, Dict[Tuple[str, str], float]],
        confidence_intervals_dict: Dict[str,
                                        Dict[Tuple[str, str],
                                             Dict[str, float]]] = None
) -> None:
    """
    Export multiple agreement results to a single CSV file.

    Args:
        filename: Path to the output CSV file.
        agreements_dict: Dictionary with method names as keys and agreement
        dictionaries as values.
        confidence_intervals_dict: Dictionary with method names as keys and
        confidence interval dictionaries as values.
    """
    import csv

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
                f'{method_name}',
                'Lower_Bound_interval',
                'Upper_bound_interval',
                'p'
            ])
        else:
            header.append(f'{method_name}')

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
