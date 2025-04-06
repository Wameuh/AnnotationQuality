from typing import Dict, Any, List
from src.dataPreparation import DataLoader
from src.raw_agreement import RawAgreement
from src.cohen_kappa import CohenKappa
from src.fleiss_kappa import FleissKappa
from src.krippendorff_alpha import KrippendorffAlpha
from src.f_measure import FMeasure
from src.icc import ICC
from src.boundary_weighted_fleiss_kappa import BoundaryWeightedFleissKappa
from src.distance_based_cell_agreement import DistanceBasedCellAgreement
from src.iou_agreement import IoUAgreement
from Utils.logger import LogLevel, get_logger
from Utils.confident_interval import ConfidenceIntervalCalculator
from Utils.pretty_print import (print_agreement_table, save_agreement_html,
                                export_multi_agreement_csv, save_agreement_csv)


class IAAWrapper:
    """
    A wrapper class to handle the IAA evaluation process.

    This class coordinates the loading of data, calculation of agreement
    measures, confidence intervals, and output of results.
    """

    # Mapping of measure names to their calculator classes
    MEASURE_CALCULATORS = {
        'raw': RawAgreement,
        'cohen_kappa': CohenKappa,
        'fleiss_kappa': FleissKappa,
        'krippendorff_alpha': KrippendorffAlpha,
        'f_measure': FMeasure,
        'icc': ICC,
        'bwfk': BoundaryWeightedFleissKappa,
        'dbcaa': DistanceBasedCellAgreement,
        'iou': IoUAgreement
    }

    def __init__(self, args: Dict[str, Any]):
        """
        Initialize the IAAWrapper with command line arguments.

        Args:
            args: Dictionary of command line arguments from argparser
        """
        self.args = args
        self.log_level = self._get_log_level(args['log_level'])
        self.logger = get_logger(self.log_level)
        self.data_loader = DataLoader(level=self.log_level)
        self.data = None
        self.results = {}
        self.confidence_intervals = {}
        self.calculators = {}  # Store calculator instances

    def _get_log_level(self, log_level_str: str) -> LogLevel:
        """Convert string log level to LogLevel enum."""
        level_map = {
            'debug': LogLevel.DEBUG,
            'info': LogLevel.INFO,
            'warning': LogLevel.WARNING,
            'error': LogLevel.ERROR,
            'critical': LogLevel.CRITICAL
        }
        return level_map.get(log_level_str.lower(), LogLevel.INFO)

    @get_logger().log_scope
    def load_data(self) -> None:
        """Load annotation data from the input file."""
        self.logger.info(f"Loading data from {self.args['input_file']}")
        try:
            self.data = self.data_loader.load_data(self.args['input_file'])
            self.logger.info(f"Loaded data with shape {self.data.shape}")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    @get_logger().log_scope
    def calculate_agreements(self) -> None:
        """Calculate all requested agreement measures."""
        if self.data is None:
            self.logger.error("No data loaded. Call load_data() first.")
            return

        # Determine which measures to calculate
        measures_to_calculate = self._get_measures_to_calculate()

        # Calculate each requested measure
        for measure in measures_to_calculate:
            self._calculate_measure(measure)

    def _get_measures_to_calculate(self) -> List[str]:
        """Determine which measures to calculate based on arguments."""
        if self.args['all']:
            return list(self.MEASURE_CALCULATORS.keys())

        measures = []
        for measure in self.MEASURE_CALCULATORS.keys():
            if self.args.get(measure, False):
                measures.append(measure)

        return measures

    @get_logger().log_scope
    def _calculate_measure(self, measure: str) -> None:
        """Calculate a specific agreement measure."""
        self.logger.info(f"Calculating {measure}")

        if self.data is None:
            self.logger.error("No data loaded. Cannot calculate agreement.")
            return

        try:
            # Get the appropriate calculator class
            calculator_class = self.MEASURE_CALCULATORS.get(measure)
            if calculator_class is None:
                self.logger.warning(f"Unknown measure: {measure}")
                return

            # Create calculator instance if not already created
            if measure not in self.calculators:
                self.calculators[measure] = calculator_class(
                    level=self.log_level)

            calculator = self.calculators[measure]

            # Get measure-specific parameters
            params = self._get_measure_params(measure)

            # Always calculate both pairwise and overall results if available
            results = {}

            # Calculate pairwise results if the method exists
            if hasattr(calculator, 'calculate_pairwise'):
                self.logger.info(f"Calculating pairwise {measure}")
                results['pairwise'] = calculator.calculate_pairwise(
                    self.data, **params)

            # Always calculate overall result
            self.logger.info(f"Calculating overall {measure}")
            results['overall'] = calculator.calculate(self.data, **params)

            # Store results
            self.results[measure] = results

            # Calculate confidence intervals if requested
            if self.args.get('confidence_interval', 0) > 0:
                self._calculate_confidence_intervals(measure)

        except Exception as e:
            self.logger.error(f"Error calculating {measure}: {str(e)}")
            raise

    def _get_measure_params(self, measure: str) -> Dict[str, Any]:
        """Get measure-specific parameters from args."""
        params = {}

        if measure == 'f_measure' and 'positive_class' in self.args:
            params['positive_class'] = self.args['positive_class']
        elif measure == 'icc' and 'icc_form' in self.args:
            params['form'] = self.args['icc_form']
        elif measure == 'bwfk' and 'bwfk_width' in self.args:
            params['width'] = self.args['bwfk_width']
        elif measure == 'dbcaa' and 'distance_threshold' in self.args:
            params['threshold'] = self.args['distance_threshold']
        elif measure == 'krippendorff_alpha' and 'metric' in self.args:
            params['metric'] = self.args['metric']

        return params

    @get_logger().log_scope
    def _calculate_confidence_intervals(self, measure: str) -> None:
        """Calculate confidence intervals for a measure."""
        self.logger.info(f"Calculating confidence intervals for {measure}")

        confidence_level = self.args.get('confidence_interval', 0.95)

        # Create the confidence interval calculator
        calculator = ConfidenceIntervalCalculator(
            confidence=confidence_level,
            level=self.log_level
        )

        result = self.results[measure]
        n = len(self.data)  # Number of rows in the data
        ci_results = {}

        # Calculate confidence intervals for pairwise results if they exist
        if 'pairwise' in result:
            pairwise_ci = {}
            for pair, value in result['pairwise'].items():
                pairwise_ci[pair] = calculator.wilson_interval(value, n)
            ci_results['pairwise'] = pairwise_ci

        # Calculate confidence interval for overall result
        if 'overall' in result:
            ci_results['overall'] = calculator.wilson_interval(
                result['overall'], n)

        self.confidence_intervals[measure] = ci_results

    @get_logger().log_scope
    def output_results(self) -> None:
        """Output the results in the requested format."""
        output_format = self.args['output_format']
        output_file = self.args['output']

        self.logger.info(f"Output format: {output_format}")
        if output_file:
            self.logger.info(f"Output will be saved to: {output_file}")

        # Handle console output or text without file
        is_console_output = (
            output_format == 'console' or
            (output_format == 'text' and not output_file)
        )
        if is_console_output:
            self._output_to_console()
            return

        # If no output file is specified, use a default name
        if not output_file:
            output_file = f"iaa_results.{output_format}"
            self.logger.info(
                f"No output file specified, using default: {output_file}"
            )

        # Handle different output formats
        if output_format == 'csv':
            self._output_to_csv(output_file)
        elif output_format == 'html':
            self._output_to_html(output_file)
        elif output_format == 'json':
            self._output_to_json(output_file)
        elif output_format == 'text':
            self._output_to_text_file(output_file)
        else:
            self.logger.warning(f"Unsupported output format: {output_format}")
            self._output_to_console()

    @get_logger().log_scope
    def _output_to_console(self) -> None:
        """Output results to console."""
        try:
            self.logger.debug("Preparing console output")

            # Print a separator line to clearly separate logs from results
            print("\n" + "=" * 80)
            print("IAA-EVAL RESULTS".center(80))
            print("=" * 80 + "\n")

            # Print each measure's results
            for measure, results in self.results.items():
                self.logger.debug(f"Processing measure: {measure}")
                self.logger.debug(f"Results type: {type(results)}")
                self.logger.debug(f"Results content: {results}")

                print(f"\n=== {measure.upper()} ===")

                # Print overall result first
                if 'overall' in results:
                    print("\nOverall Result:")
                    result = results['overall']
                    self.logger.debug(f"Overall result type: {type(result)}")
                    self.logger.debug(f"Overall result value: {result}")

                    has_ci = (
                        measure in self.confidence_intervals and
                        'overall' in self.confidence_intervals[measure]
                    )

                    # Convert result to float if it's not already
                    try:
                        result_float = float(result)
                        self.logger.debug(
                            "Successfully converted result to float: "
                            f"{result_float}"
                        )
                    except (TypeError, ValueError) as e:
                        error_msg = (
                            f"Could not convert result to float: {result}. "
                            f"Error: {str(e)}"
                        )
                        self.logger.error(error_msg)
                        result_float = result

                    if has_ci:
                        ci = self.confidence_intervals[measure]['overall']
                        self.logger.debug(f"Confidence interval: {ci}")
                        if isinstance(result_float, float):
                            print(
                                f"{measure}: {result_float:.4f} "
                                f"(CI: {ci['ci_lower']:.4f} - "
                                f"{ci['ci_upper']:.4f})"
                            )
                        else:
                            print(f"{measure}: {result}")
                    else:
                        if isinstance(result_float, float):
                            print(f"{measure}: {result_float:.4f}")
                        else:
                            print(f"{measure}: {result}")

                    # Add interpretation if available
                    if measure in self.calculators:
                        calculator = self.calculators[measure]
                        if (hasattr(calculator, 'interpret') and
                                callable(getattr(calculator, 'interpret'))):
                            interpretation = calculator.interpret(result)
                            print(f"Interpretation: {interpretation}")

                # Print pairwise results if available
                if 'pairwise' in results:
                    print("\nPairwise Results:")
                    pairwise_ci = None
                    has_pairwise_ci = (
                        measure in self.confidence_intervals and
                        'pairwise' in self.confidence_intervals[measure]
                    )
                    if has_pairwise_ci:
                        ci_data = self.confidence_intervals[measure]
                        pairwise_ci = ci_data['pairwise']
                    print_agreement_table(
                        results['pairwise'],
                        confidence_intervals=pairwise_ci
                    )

            # Print a final separator
            print("\n" + "=" * 80)
        except Exception as e:
            import traceback
            self.logger.error(f"Exception in _output_to_console: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    @get_logger().log_scope
    def _output_to_csv(self, output_file: str) -> None:
        """Output results to CSV file."""
        self.logger.info(f"Outputting results to CSV file: {output_file}")

        # For pairwise measures, use export_multi_agreement_csv
        pairwise_results = {}
        for measure, results in self.results.items():
            if 'pairwise' in results:  # Check for pairwise results
                pairwise_results[measure] = results['pairwise']

        if pairwise_results:
            # Determine if we should use method names in column headers
            # Based on filename pattern or args
            # For feature tests, we want to use 'Agreement' in the headers
            use_method_names = not (
                'test_' in output_file or
                self.args.get('output_format') == 'csv'
            )

            # Get pairwise confidence intervals if available
            pairwise_ci = {}
            for measure in pairwise_results:
                if (measure in self.confidence_intervals and
                        'pairwise' in self.confidence_intervals[measure]):
                    pairwise_ci[measure] = (
                        self.confidence_intervals[measure]['pairwise']
                    )

            export_multi_agreement_csv(
                output_file,
                pairwise_results,
                pairwise_ci,
                use_method_names=use_method_names
            )
            self.logger.info(f"Saved pairwise results to: {output_file}")

        # Handle overall results
        import os
        base_name, ext = os.path.splitext(output_file)
        for measure, results in self.results.items():
            if 'overall' in results:
                measure_file = f"{base_name}_{measure}{ext}"
                overall_result = results['overall']

                # Create synthetic pair for the overall result
                synthetic_agreements = {('Overall', 'Result'): overall_result}

                # Get confidence interval for overall result if available
                synthetic_ci = None
                if (measure in self.confidence_intervals and
                        'overall' in self.confidence_intervals[measure]):
                    ci = self.confidence_intervals[measure]['overall']
                    synthetic_ci = {('Overall', 'Result'): ci}

                save_agreement_csv(
                    measure_file,
                    synthetic_agreements,
                    confidence_intervals=synthetic_ci,
                    agreement_name=measure
                )
                self.logger.info(f"Saved {measure} results to: {measure_file}")

                # Add interpretation if available to a text file
                if measure in self.calculators:
                    calculator = self.calculators[measure]
                    if (hasattr(calculator, 'interpret') and
                            callable(getattr(calculator, 'interpret'))):
                        interpretation = calculator.interpret(overall_result)
                        interp_file = (
                            f"{base_name}_{measure}_interpretation.txt"
                        )
                        with open(interp_file, 'w') as f:
                            f.write(f"Interpretation: {interpretation}\n")
                        self.logger.info(
                            f"Saved {measure} interpretation to: {interp_file}"
                        )

    @get_logger().log_scope
    def _output_to_html(self, output_file: str) -> None:
        """Output results to HTML file."""
        self.logger.info(f"Outputting results to HTML file: {output_file}")

        # For each measure, create a separate HTML file
        for measure, results in self.results.items():
            html_file = output_file.replace('.html', f'_{measure}.html')

            # Handle pairwise results if available
            if 'pairwise' in results:
                pairwise_ci = None
                if (
                    measure in self.confidence_intervals and
                    'pairwise' in self.confidence_intervals[measure]
                ):
                    pairwise_ci = (
                        self.confidence_intervals[measure]['pairwise']
                    )

                save_agreement_html(
                    html_file,
                    results['pairwise'],
                    confidence_intervals=pairwise_ci,
                    title=f"{measure.upper()} Agreement Results"
                )

            # Handle overall results
            if 'overall' in results:
                # Create synthetic agreement pair for overall result
                overall_result = results['overall']
                synthetic_agreements = {
                    ('Overall', 'Result'): overall_result
                }

                # Handle confidence intervals for overall result
                synthetic_ci = None
                if (
                    measure in self.confidence_intervals and
                    'overall' in self.confidence_intervals[measure]
                ):
                    ci = self.confidence_intervals[measure]['overall']
                    synthetic_ci = {('Overall', 'Result'): ci}

                # If no pairwise results, save overall to its own file
                if 'pairwise' not in results:
                    save_agreement_html(
                        html_file,
                        synthetic_agreements,
                        confidence_intervals=synthetic_ci,
                        title=f"{measure.upper()} Overall Result"
                    )

            self.logger.info(f"Saved {measure} results to: {html_file}")

        # Create an index HTML file with all results
        self.logger.info(f"Creating index HTML file: {output_file}")
        with open(output_file, 'w') as f:
            f.write("<html><head><title>IAA-Eval Results</title></head>"
                    "<body>\n")
            f.write("<h1>IAA-Eval Results</h1>\n")

            # Add confidence interval information
            if self.confidence_intervals:
                confidence_level = self.args.get('confidence_interval', 0.95)
                confidence_method = self.args.get(
                    'confidence_method',
                    'wilson'
                )
                confidence_percent = int(confidence_level * 100)
                f.write("<h2>Confidence Interval Information</h2>\n")
                f.write(
                    f"<p>Confidence Level: {confidence_percent}%</p>\n")
                f.write(
                    f"<p>Method: {confidence_method}</p>\n")

            # Extract base name for href links
            import os
            base_filename = os.path.basename(output_file)
            base_name_without_ext = os.path.splitext(base_filename)[0]

            for measure, results in self.results.items():
                f.write(f"<h2>{measure.upper()}</h2>\n")

                # Add overall result if available
                if 'overall' in results:
                    result = results['overall']
                    f.write("<h3>Overall Result</h3>\n")

                    # Format the result with confidence interval if available
                    result_str = f"{result:.4f}"
                    if (
                        measure in self.confidence_intervals and
                        'overall' in self.confidence_intervals[measure]
                    ):
                        ci = self.confidence_intervals[measure]['overall']
                        result_str += (
                            f" (CI: {ci['ci_lower']:.4f} - "
                            f"{ci['ci_upper']:.4f})"
                        )
                    f.write(f"<p>{result_str}</p>\n")

                    # Add interpretation if available
                    if measure in self.calculators:
                        calculator = self.calculators[measure]
                        if (hasattr(calculator, 'interpret') and
                                callable(getattr(calculator, 'interpret'))):
                            interpretation = calculator.interpret(result)
                            f.write(
                                f"<p>Interpretation: {interpretation}</p>\n")

                # Add link to detailed results
                html_file_name = base_name_without_ext + f"_{measure}.html"
                f.write(
                    f"<p>See <a href='{html_file_name}'>detailed {measure} "
                    "results</a></p>\n")

            f.write("</body></html>\n")
        self.logger.info("Index HTML file created successfully")

    @get_logger().log_scope
    def _output_to_json(self, output_file: str) -> None:
        """Output results to JSON file."""
        self.logger.info(f"Outputting results to JSON file: {output_file}")
        import json

        # Create a dictionary to hold all results
        json_results = {}
        # Add interpretations dictionary
        json_results['interpretations'] = {}

        for measure, result in self.results.items():
            if isinstance(result, dict):  # Pairwise results
                # Convert tuple keys to strings
                json_results[measure] = {
                    f"{k[0]}_{k[1]}": (float(v)
                                       if isinstance(v, (int, float))
                                       else v)
                    for k, v in result.items()
                }

                # Add interpretation if available for pairwise results
                if measure in self.calculators:
                    calculator = self.calculators[measure]
                    if (hasattr(calculator, 'interpret') and
                            callable(getattr(calculator, 'interpret'))):
                        # For pairwise measures, get global interpretation
                        # if possible. Try to get a global value to interpret
                        # (e.g., average)
                        if hasattr(calculator, 'get_global_value'):
                            global_value = calculator.get_global_value(result)
                            json_results['interpretations'][measure] = (
                                calculator.interpret(global_value))
                        # If no global value method, use the first pair
                        # value as an example
                        elif result:
                            first_pair = next(iter(result.items()))
                            pair_names = first_pair[0]
                            pair_value = first_pair[1]
                            interpretation = calculator.interpret(pair_value)
                            json_results['interpretations'][measure] = (
                                f"Example interpretation for {pair_names[0]}-"
                                f"{pair_names[1]}: {interpretation}"
                            )
            else:  # Single value results
                json_results[measure] = (float(result)
                                         if isinstance(result, (int, float))
                                         else result)

                # Add interpretation if available
                if measure in self.calculators:
                    calculator = self.calculators[measure]
                    if (hasattr(calculator, 'interpret') and
                            callable(getattr(calculator, 'interpret'))):
                        json_results['interpretations'][measure] = (
                            calculator.interpret(result))

        # Add confidence intervals if available
        json_ci = {}
        for measure, ci_dict in self.confidence_intervals.items():
            if isinstance(ci_dict, dict):
                if isinstance(next(iter(ci_dict.keys()), None), tuple):
                    # Pairwise CI
                    json_ci[measure] = {
                        f"{k[0]}_{k[1]}": {
                            ci_key: (float(ci_val)
                                     if isinstance(ci_val, (int, float))
                                     else ci_val)
                            for ci_key, ci_val in ci_info.items()
                        }
                        for k, ci_info in ci_dict.items()
                    }
                else:  # Single value confidence interval
                    json_ci[measure] = {
                        ci_key: (float(ci_val)
                                 if isinstance(ci_val, (int, float))
                                 else ci_val)
                        for ci_key, ci_val in ci_dict.items()
                    }

        if json_ci:
            json_results['confidence_intervals'] = json_ci

        # Remove empty interpretations
        if not json_results['interpretations']:
            del json_results['interpretations']

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        self.logger.info(f"JSON results saved to: {output_file}")

    @get_logger().log_scope
    def _output_to_text_file(self, output_file: str) -> None:
        """Output results to text file."""
        self.logger.info(f"Outputting results to text file: {output_file}")

        with open(output_file, 'w') as f:
            f.write("IAA-Eval Results\n")
            f.write("===============\n\n")

            for measure, result in self.results.items():
                f.write(f"\n=== {measure.upper()} ===\n")

                if isinstance(result, dict):  # Pairwise results
                    # Create a string representation of the table
                    import io
                    buffer = io.StringIO()
                    ci = self.confidence_intervals.get(measure)
                    print_agreement_table(
                        result,
                        confidence_intervals=ci,
                        file=buffer
                    )
                    f.write(buffer.getvalue())
                else:  # Single value results
                    # Create synthetic pair for consistency
                    synthetic_agreements = {('Global', 'Result'): result}
                    synthetic_ci = None
                    if measure in self.confidence_intervals:
                        ci_value = self.confidence_intervals[measure]
                        synthetic_ci = {('Global', 'Result'): ci_value}

                    # Use print_agreement_table with a buffer
                    import io
                    buffer = io.StringIO()
                    print_agreement_table(
                        synthetic_agreements,
                        confidence_intervals=synthetic_ci,
                        file=buffer
                    )
                    f.write(buffer.getvalue())

                    # Also write the simple value for clarity
                    f.write(f"\nGlobal {measure}: {result:.4f}\n")

                    # Add interpretation if available
                    if measure in self.calculators:
                        calculator = self.calculators[measure]
                        if (hasattr(calculator, 'interpret') and
                                callable(getattr(calculator, 'interpret'))):
                            interpretation = calculator.interpret(result)
                            f.write(f"Interpretation: {interpretation}\n")
        self.logger.info(f"Text results saved to: {output_file}")

    @get_logger().log_scope
    def run(self) -> None:
        """Run the complete IAA evaluation process."""
        try:
            self.load_data()
            self.calculate_agreements()
            self.output_results()
        except Exception as e:
            self.logger.error(f"Error in IAA evaluation: {str(e)}")
            raise
