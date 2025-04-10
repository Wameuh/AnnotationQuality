# This file re-exports tests from the specialized test modules to maintain
# backwards compatibility. The actual implementations have been moved to
# separate files for better organization.

# Re-export all test functions and fixtures
from Tests.UnitTests.test_pretty_print_console import (
            sample_agreements,
            sample_confidence_intervals,
            test_print_agreement_table_basic,
            test_print_agreement_table_with_ci,
            test_print_agreement_table_different_order,
            test_print_agreement_table_truncate_names,
            test_print_agreement_table_missing_pair,
            test_print_agreement_table_with_empty_agreements,
            test_print_agreement_table_with_single_annotator,
            test_print_agreement_table_with_many_long_names,
            test_print_agreement_table_with_extreme_truncation,
            test_print_agreement_table_with_partial_ci
)

from Tests.UnitTests.test_pretty_print_csv import (
    test_save_agreement_csv,
    test_save_agreement_csv_without_ci,
    test_export_agreement_csv,
    test_export_agreement_csv_without_ci,
    test_export_agreement_csv_without_matrix,
    test_export_multi_agreement_csv,
    test_export_multi_agreement_csv_without_ci,
    test_export_multi_agreement_csv_with_missing_pairs,
    test_export_multi_agreement_csv_with_missing_pairs_and_ci
)

from Tests.UnitTests.test_pretty_print_html import (
    test_save_agreement_html,
    test_save_agreement_html_no_heatmap,
    test_get_cell_html_with_different_values,
    test_get_cell_html_with_special_values,
    test_get_agreement_class,
    test_get_confidence_interval_class,
    test_save_agreement_html_with_partial_ci
)

# Define __all__ to explicitly indicate we're re-exporting these symbols
__all__ = [
    # Fixtures
    'sample_agreements',
    'sample_confidence_intervals',

    # Console tests
    'test_print_agreement_table_basic',
    'test_print_agreement_table_with_ci',
    'test_print_agreement_table_different_order',
    'test_print_agreement_table_truncate_names',
    'test_print_agreement_table_missing_pair',
    'test_print_agreement_table_with_empty_agreements',
    'test_print_agreement_table_with_single_annotator',
    'test_print_agreement_table_with_many_long_names',
    'test_print_agreement_table_with_extreme_truncation',
    'test_print_agreement_table_with_partial_ci',

    # CSV tests
    'test_save_agreement_csv',
    'test_save_agreement_csv_without_ci',
    'test_export_agreement_csv',
    'test_export_agreement_csv_without_ci',
    'test_export_agreement_csv_without_matrix',
    'test_export_multi_agreement_csv',
    'test_export_multi_agreement_csv_without_ci',
    'test_export_multi_agreement_csv_with_missing_pairs',
    'test_export_multi_agreement_csv_with_missing_pairs_and_ci',

    # HTML tests
    'test_save_agreement_html',
    'test_save_agreement_html_no_heatmap',
    'test_get_cell_html_with_different_values',
    'test_get_cell_html_with_special_values',
    'test_get_agreement_class',
    'test_get_confidence_interval_class',
    'test_save_agreement_html_with_partial_ci'
]

# This file now serves as a coordinator for the individual test modules
# and is maintained for backwards compatibility.
# The actual tests have been moved to:
#  - test_pretty_print_console.py - Tests for console output functions
#  - test_pretty_print_csv.py - Tests for CSV output functions
#  - test_pretty_print_html.py - Tests for HTML output functions
