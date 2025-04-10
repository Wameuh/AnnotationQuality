from typing import Dict, Tuple
import io
import base64
import numpy as np
from datetime import datetime
from Utils.agreement_utils import (
    get_unique_annotators, create_agreement_matrix,
    get_confidence_method_display, get_confidence_level
)


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
    html_parts = []

    # Head section
    html_parts.append(f"""<!DOCTYPE html>
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
""")

    # Construct the header part of the HTML table
    header_row = "    <tr>\n        <th></th>\n" + \
                 "        " + \
                 " ".join(f'<th>{ann}</th>' for ann in annotators) + "\n" + \
                 "    </tr>\n"

    # Construct the body part of the HTML table
    body_rows = "".join(
        f'    <tr><th>{annotators[i]}</th>' +
        " ".join(
            get_cell_html(cell, val) for cell, val in zip(row, heatmap_data[i])
        ) +
        "</tr>\n"
        for i, row in enumerate(matrix)
    )

    # Combine everything into the final HTML string
    matrix_html = (
        f"<h2>Agreement Matrix ({n_annotators} annotators)</h2>\n"
        "<table>\n"
        f"{header_row}"
        f"{body_rows}"
        "</table>\n"
    )
    html_parts.append(matrix_html)

    # Confidence interval note
    if confidence_intervals:
        ci_p_value = f"{1 - significance_level:.2f}"
        confidence_note = (
            f"    <p class=\"legend\">Note: Confidence intervals are "
            f"calculated using the {method_display} method with p = "
            f"{ci_p_value}</p>\n"
        )
        html_parts.append(confidence_note)

    # Summary table
    html_parts.append(summary_table)

    # Heatmap
    html_parts.append(heatmap_img)

    # Footer
    footer_html = f"""    <div class="footer">
        <p>Generated by <a href="https://github.com/Wameuh/AnnotationQuality"
           target="_blank">IAA-Eval</a> on
           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
    html_parts.append(footer_html)

    # Join all parts
    html = "".join(html_parts)

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
