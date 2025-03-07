from typing import Dict, Tuple, Optional
from tabulate import tabulate
import csv
import io
import base64
import numpy as np
from datetime import datetime


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

    # Create a mapping from original names to display names
    display_names = {}
    if total_width > max_width and n_annotators > 3:
        # Truncate names to make table fit
        max_name = (max_width - (n_annotators * (cell_width + 3))) // 2

        # Check if max_name is negative or too small
        if max_name <= 3:
            raise ValueError(
                f"Table width ({total_width}) exceeds maximum width"
                f" ({max_width}). "
                f"Cannot truncate names to fit. "
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
                    cell = (f"{value:.1%} ({ci['ci_lower']:.1%}"
                            f"-{ci['ci_upper']:.1%})")
                else:
                    cell = f"{value:.1%}"
                row.append(cell)
        matrix.append(row)

    # Print the table
    print(f"\nInter-annotator Agreement Matrix ({n_annotators} annotators)")
    if confidence_intervals:
        confidence_level = next(
            iter(confidence_intervals.values()))['confidence_level']
        confidence_percent = int(confidence_level * 100)
        print(f"Values shown as: Agreement (CI lower-CI upper)"
              f" with p={confidence_percent}%")

    # Use display names for the table headers
    display_annotators = [display_names[ann] for ann in annotators]

    print(tabulate(
        matrix,
        headers=display_annotators,
        showindex=display_annotators,
        tablefmt="grid"
    ))

    # Print summary
    if confidence_intervals:
        print(f"\nSummary of Agreement Values with Confidence "
              f"Intervals (p={confidence_percent}%):")
        for (ann1, ann2), value in sorted(agreements.items()):
            if (ann1, ann2) in confidence_intervals:
                ci = confidence_intervals[(ann1, ann2)]
                print(f"{ann1} & {ann2}: {value:.1%} ({ci['ci_lower']:.1%}"
                      f"-{ci['ci_upper']:.1%})")
            else:
                print(f"{ann1} & {ann2}: {value:.1%}")
    else:
        print("\nSummary of Agreement Values:")
        for (ann1, ann2), value in sorted(agreements.items()):
            print(f"{ann1} & {ann2}: {value:.1%}")


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


def save_agreement_html(
        filename: str,
        agreements: Dict[Tuple[str, str], float],
        confidence_intervals: Optional[Dict[Tuple[str, str],
                                            Dict[str, float]]] = None,
        title: str = "Inter-Annotator Agreement Results",
        include_heatmap: bool = True
        ) -> None:
    """
    Save agreement values and confidence intervals to an HTML file.

    Args:
        filename: Path to output HTML file.
        agreements: Dictionary with annotator pairs as keys and agreement
            values as values.
        confidence_intervals: Optional dictionary with annotator pairs as keys
            and confidence interval info as values.
        title: Title for the HTML report.
        include_heatmap: Whether to include a heatmap visualization.
    """
    # Get unique annotators
    annotators = sorted(
        list({ann for pair in agreements.keys() for ann in pair}))
    n_annotators = len(annotators)

    # Create matrix of results
    matrix = []
    heatmap_data = np.zeros((n_annotators, n_annotators))

    for i, ann1 in enumerate(annotators):
        row = []
        for j, ann2 in enumerate(annotators):
            if ann1 == ann2:
                row.append("---")
                heatmap_data[i, j] = 1.0  # Perfect agreement with self
            else:
                # Try both orderings of the pair
                pair1 = (ann1, ann2)
                pair2 = (ann2, ann1)

                if pair1 in agreements:
                    value = agreements[pair1]
                    pair = pair1
                else:
                    value = agreements[pair2]
                    pair = pair2

                heatmap_data[i, j] = value

                if confidence_intervals and pair in confidence_intervals:
                    ci = confidence_intervals[pair]
                    cell = (f"{value:.1%}<br>"
                            f"[{ci['ci_lower']:.1%}-{ci['ci_upper']:.1%}]")
                else:
                    cell = f"{value:.1%}"
                row.append(cell)
        matrix.append(row)

    # Generate heatmap image if requested
    heatmap_img = ""
    if include_heatmap:
        # Use Agg backend which doesn't require a GUI
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        plt.figure(figsize=(8, 6))
        # Create a masked array for NaN values
        masked_data = np.ma.masked_invalid(heatmap_data)

        # Create a custom colormap according to preferences:
        # 1.0 green, 0.8 yellow, 0.6 orange, 0.4 red, 0.2 purple, 0.0 black
        colors = [
            (0, 0, 0),       # 0.0: black
            (0.5, 0, 0.5),   # 0.2: purple
            (1, 0, 0),       # 0.4: red
            (1, 0.5, 0),     # 0.6: orange
            (1, 1, 0),       # 0.8: yellow
            (0, 0.8, 0)      # 1.0: green
        ]
        positions = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cmap = LinearSegmentedColormap.from_list('agreement_cmap',
                                                 list(zip(positions, colors)))

        plt.imshow(masked_data, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(label='Agreement Value')
        plt.xticks(range(n_annotators), annotators, rotation=45, ha='right')
        plt.yticks(range(n_annotators), annotators)
        plt.title('Pairwise Agreement Heatmap')
        plt.tight_layout()

        # Convert plot to base64 for embedding in HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()

        heatmap_img = (f"""
        <div class="visualization">
            <h2>Agreement Heatmap</h2>
            <img src="data:image/png;base64,{img_str}" """
                       f"""alt="Pairwise Agreement Heatmap">
        </div>
        """)

    # Generate summary table with confidence intervals
    summary_table = ""
    if confidence_intervals:
        confidence_level = getattr(
            confidence_intervals.get(next(iter(confidence_intervals)), {}),
            'get',
            lambda x, y: 0.95)('confidence_level', 0.95)
        significance_level = 1 - confidence_level

        # Determine the method used for confidence interval calculation
        method = "wilson"  # Default method
        if len(confidence_intervals) > 0:
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

        summary_rows = []
        for (ann1, ann2), value in sorted(agreements.items()):
            if (ann1, ann2) in confidence_intervals:
                ci = confidence_intervals[(ann1, ann2)]
                ci_class = get_confidence_interval_class(ci['ci_lower'],
                                                         ci['ci_upper'])
                agreement_with_ci = (f"{value:.1%} <span class='{ci_class}'>("
                                     f"{ci['ci_lower']:.1%} - "
                                     f"{ci['ci_upper']:.1%})</span>")
                summary_rows.append(
                    f"""<tr>
                        <td>{ann1} & {ann2}</td>
                        <td class="{get_agreement_class(value)}">"""
                    f"""{agreement_with_ci}</td>
                    </tr>"""
                )
            else:
                summary_rows.append(
                    f"""<tr>
                        <td>{ann1} & {ann2}</td>
                        <td class="{get_agreement_class(value)}" """
                    f""">{value:.1%}</td>
                    </tr>"""
                )

        summary_table = (f"""
        <h2>Summary of Agreement Values</h2>
        <table>
            <tr>
                <th>Annotator Pair</th>
                <th>Agreement Value with Confidence Interval</th>
            </tr>
            {"".join(summary_rows)}
        </table>
        <p class="legend">Note: Confidence intervals are calculated using """
                         f"""the {method_display} method with p ="""
                         f""" {significance_level:.2f}</p>
        """)

    # Create HTML content
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
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 50px;
            border-top: 1px solid #eee;
            padding-top: 10px;
            font-size: 0.8em;
            color: #7f8c8d;
        }}
        .high-agreement {{
            background-color: #e6ffe6;
        }}
        .medium-agreement {{
            background-color: #ffffcc;
        }}
        .low-agreement {{
            background-color: #ffe6e6;
        }}
        .wide-interval {{
            color: #cc0000;
        }}
        .narrow-interval {{
            color: #006600;
        }}
        .medium-interval {{
            color: #cc6600;
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
     f'{method_display} method with p = {significance_level:.2f}</p>' or ''}

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
