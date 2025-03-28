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
                                export_multi_agreement_csv)


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

            # Determine if we should use calculate_pairwise or calculate
            # Check if the calculator has a calculate_pairwise method and
            # if it's explicitly requested
            if hasattr(calculator, 'calculate_pairwise') and self.args.get(
                    'pairwise', False):
                self.results[measure] = calculator.calculate_pairwise(
                    self.data, **params)
            else:
                # Default to the standard calculate method
                self.logger.info(
                    f"Calculating {measure} with calculate method")
                self.results[measure] = calculator.calculate(
                    self.data, **params)

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

        # Calculate confidence intervals
        if isinstance(result, dict):  # Pairwise results
            # For each pair, calculate the confidence interval
            ci_dict = {}
            for pair, value in result.items():
                # Use the wilson_interval method to calculate the confidence
                # interval.
                # We need to estimate the sample size
                n = len(self.data)  # Number of rows in the data
                ci_dict[pair] = calculator.wilson_interval(value, n)

            self.confidence_intervals[measure] = ci_dict
        else:  # Global result
            # Calculate the confidence interval for the global result
            n = len(self.data)  # Number of rows in the data
            self.confidence_intervals[measure] = calculator.wilson_interval(
                result, n)

    @get_logger().log_scope
    def output_results(self) -> None:
        """Output the results in the requested format."""
        output_format = self.args['output_format']
        output_file = self.args['output']
        if (output_format == 'console' or
                (output_format == 'text' and not output_file)):
            self._output_to_console()
        elif output_file:
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
        self.logger.info("Outputting results to console")

        # Print each measure's results
        for measure, result in self.results.items():
            print(f"\n=== {measure.upper()} ===")

            if isinstance(result, dict):  # Pairwise results
                print_agreement_table(
                    result,
                    confidence_intervals=self.confidence_intervals.get(measure)
                )
            else:  # Single value results
                print(f"{measure}: {result:.4f}")

                # Add interpretation if available
                if measure in self.calculators:
                    calculator = self.calculators[measure]
                    if (hasattr(calculator, 'interpret') and
                            callable(getattr(calculator, 'interpret'))):
                        interpretation = calculator.interpret(result)
                        print(f"Interpretation: {interpretation}")

    @get_logger().log_scope
    def _output_to_csv(self, output_file: str) -> None:
        """Output results to CSV file."""
        self.logger.info(f"Outputting results to CSV file: {output_file}")

        # For pairwise measures, use export_multi_agreement_csv
        pairwise_results = {}
        for measure, result in self.results.items():
            if isinstance(result, dict):  # Pairwise results
                pairwise_results[measure] = result

        if pairwise_results:
            export_multi_agreement_csv(
                output_file,
                pairwise_results,
                {k: v for k, v in self.confidence_intervals.items()
                 if k in pairwise_results}
            )
        else:
            # For single value measures, create a simple CSV
            with open(output_file, 'w') as f:
                f.write("Measure,Value,Interpretation\n")
                for measure, result in self.results.items():
                    interpretation = ""
                    if measure in self.calculators:
                        calculator = self.calculators[measure]
                        if (hasattr(calculator, 'interpret') and
                                callable(getattr(calculator, 'interpret'))):
                            interpretation = calculator.interpret(result)
                    f.write(f"{measure},{result:.4f},\"{interpretation}\"\n")

    @get_logger().log_scope
    def _output_to_html(self, output_file: str) -> None:
        """Output results to HTML file."""
        self.logger.info(f"Outputting results to HTML file: {output_file}")

        # For pairwise measures, use save_agreement_html
        for measure, result in self.results.items():
            if isinstance(result, dict):  # Pairwise results
                save_agreement_html(
                    result,
                    output_file.replace('.html', f'_{measure}.html'),
                    confidence_intervals=self.confidence_intervals.get(measure)
                )

        # Create an index HTML file with all results
        with open(output_file, 'w') as f:
            f.write("<html><head><title>IAA-Eval Results</title></head>"
                    "<body>\n")
            f.write("<h1>IAA-Eval Results</h1>\n")

            for measure, result in self.results.items():
                f.write(f"<h2>{measure.upper()}</h2>\n")

                if isinstance(result, dict):  # Pairwise results
                    f.write(f"<p>See <a href='{measure}_results.html'>"
                            f"{measure} results</a></p>\n")
                else:  # Single value results
                    f.write(f"<p>{measure}: {result:.4f}</p>\n")

                    # Add interpretation if available
                    if measure in self.calculators:
                        calculator = self.calculators[measure]
                        if (hasattr(calculator, 'interpret') and
                                callable(getattr(calculator, 'interpret'))):
                            interpretation = calculator.interpret(result)
                            f.write(
                                f"<p>Interpretation: {interpretation}</p>\n")

            f.write("</body></html>\n")

    @get_logger().log_scope
    def _output_to_json(self, output_file: str) -> None:
        """Output results to JSON file."""
        self.logger.info(f"Outputting results to JSON file: {output_file}")
        import json

        # Create a dictionary to hold all results
        json_results = {}
        for measure, result in self.results.items():
            if isinstance(result, dict):  # Pairwise results
                # Convert tuple keys to strings
                json_results[measure] = {
                    f"{k[0]}_{k[1]}": (float(v)
                                       if isinstance(v, (int, float))
                                       else v)
                    for k, v in result.items()
                }
            else:  # Single value results
                json_results[measure] = (float(result)
                                         if isinstance(result, (int, float))
                                         else result)

                # Add interpretation if available
                if measure in self.calculators:
                    calculator = self.calculators[measure]
                    if (hasattr(calculator, 'interpret') and
                            callable(getattr(calculator, 'interpret'))):
                        if 'interpretations' not in json_results:
                            json_results['interpretations'] = {}
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

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

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
                    print_agreement_table(
                        result,
                        confidence_intervals=self.confidence_intervals.get(
                            measure),
                        file=buffer
                    )
                    f.write(buffer.getvalue())
                else:  # Single value results
                    f.write(f"{measure}: {result:.4f}\n")

                    # Add interpretation if available
                    if measure in self.calculators:
                        calculator = self.calculators[measure]
                        if (hasattr(calculator, 'interpret') and
                                callable(getattr(calculator, 'interpret'))):
                            interpretation = calculator.interpret(result)
                            f.write(f"Interpretation: {interpretation}\n")

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
