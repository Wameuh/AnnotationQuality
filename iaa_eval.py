#!/usr/bin/env python3
"""
IAA-Eval: A Command-Line Tool for Inter-Annotator Agreement Evaluation

This script serves as the main entry point for the IAA-Eval tool.
It parses command line arguments and delegates the processing to the
IAAWrapper class.
"""

import sys
from src.argparser import parse_arguments
from src.iaa_wrapper import IAAWrapper
from Utils.logger import get_logger, LogLevel


def main():
    """Main entry point for the IAA-Eval tool."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Set up logging
        log_level_str = args.get('log_level', 'info')
        log_level_map = {
            'debug': LogLevel.DEBUG,
            'info': LogLevel.INFO,
            'warning': LogLevel.WARNING,
            'error': LogLevel.ERROR,
            'critical': LogLevel.CRITICAL
        }
        log_level = log_level_map.get(log_level_str.lower(), LogLevel.INFO)
        logger = get_logger(log_level)

        logger.info("Starting IAA-Eval")

        try:
            # Create and run the IAA wrapper
            wrapper = IAAWrapper(args)
            wrapper.run()

            logger.info("IAA-Eval completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Error in IAA-Eval: {str(e)}")
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
