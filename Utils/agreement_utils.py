from typing import Dict, Tuple, List, Any


def get_unique_annotators(
        agreements: Dict[Tuple[str, str], float]) -> List[str]:
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
