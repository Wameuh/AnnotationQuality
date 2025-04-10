# IAA-Eval: A Command-Line Tool for Inter-Annotator Agreement Evaluation

## Introduction

**Inter-Annotator Agreement (IAA) is a crucial method for evaluating the quality of annotations** , as it measures the reliability and reproducibility of the annotation process [1-2]. A high IAA suggests that the annotations are consistent and reliable, while a low IAA indicates potential problems with the annotation guidelines or the annotators' understanding of the data [1]. IAA is not merely an indicator of reliability; it's a tool for understanding sources of variation in the data and for improving the annotations [1].

Here are the key interests of inter-annotator agreement and different ways to calculate it, according to the sources:

* **Reliability Assessment:** The main purpose of IAA is to determine if the annotation process is reliable, meaning it produces consistent results. If annotators disagree, this highlights a problem in the process or in how guidelines are interpreted. Reliable annotation is a necessary but not sufficient condition for correct annotations [1-2].
* **Ambiguity Identification:** Disagreements between annotators reveal ambiguities or difficulties in the data or annotation guidelines. This helps to identify areas that require clarification and improvement [2, 3].
* **Improvement of Annotation Guidelines:** Analyzing disagreements can lead to adjustments and refinements of the annotation guidelines, thus enhancing the consistency of annotations. The process of improving guidelines is iterative, with regular reliability tests and adjustments until an acceptable level of agreement is achieved [2 3].
* **Understanding Data Variation:** IAA helps to identify parts of the data or types of annotations that are more difficult or less reliable than others. By segmenting the data and calculating agreement for each part, a better understanding of the annotation complexity can be achieved [1].
* **Annotator Selection:** IAA analysis can help identify annotators who are better suited to a specific annotation task. Some annotators may be more consistent in certain types of annotations than others [2].
* **Measuring Impact of Semantic Context:** By introducing semantic contexts during annotation, for example, for hate speech, IAA helps measure the impact of this information on convergence between annotators, especially between those belonging to the target population and those who do not [4].

Different methods exist for calculating IAA:

* **Raw Agreement (Observed Agreement):** This is the simplest measure, which calculates the percentage of items on which the annotators agree. However, it doesn't account for agreement that could be obtained by chance [1].
* **Chance-Corrected Measures:** These measures are used to correct the raw agreement by taking into account the probability of random agreement. The most common include:
* **Cohen's Kappa:** This measure is often used for categorical annotations between two annotators. It calculates the observed agreement beyond the agreement expected by chance, taking into account potential annotator biases. Several sources recommend choosing a metric based on the annotation task [1].
* **Fleiss' Kappa:** This coefficient is used to measure agreement between multiple annotators and is particularly suitable for classification evaluation [5].
* **Krippendorff's Alpha (α):** Similar to Fleiss' Kappa but more flexible as it can take into account different levels of disagreement and is suitable for incomplete data or with a variable number of annotators [6]. The sources show a range of alpha scores based on different annotation tasks, with some scores showing substantial agreement [1].
* **F-measure:** This measure calculates agreement using precision, recall, and their harmonic mean. It is particularly useful for tasks such as named entity recognition or classification, where the ability of a model to identify correct entities and its ability to identify all relevant entities must be balanced.
* **Boundary-Weighted Fleiss' Kappa (BWFK):** This method is used to evaluate agreement in tissue segmentation, particularly on tissue boundaries. It reduces the impact of minor disagreements that often occur along these boundaries [10]. This method can be used only on binary segmentation masks.
* **Distance-Based Cell Agreement Algorithm (DBCAA):** An algorithm that measures agreement in cell detection without relying on ground truth [10].
* **Intersection over Union (IoU)-Based Measures:** These measures are used for regional segmentation but are limited for assessments involving more than two observers [1].
* **Intra-class correlation coefficient (ICC):** This coefficient measures the degree of agreement between multiple annotators by comparing the variance between annotators to the total variance [11].

It is important to note that **the choice of IAA measure depends on the nature of the data and the annotation task** [1]. Additionally, IAA should not be considered an end in itself, but rather as a tool to understand the data and improve the annotation process. Disagreements between annotators can provide valuable insights [1, 9].

## IAA-Eval: Features and Usage

**IAA-Eval** is designed to be a user-friendly CLI tool with the following key features:

* **Data Loading and Preprocessing:** Reads annotation data from various file formats, including CSV and JSON. Supports different annotation types (categorical, numerical, etc.)
* **Test Data Note:** For testing purposes, the file `Tests/Assets/Reviews_annotated.csv` has been generated using the program available at [LLM-Tests](https://github.com/Wameuh/LLM-Tests), which provides sentiment annotations for Amazon reviews.
* **IAA Calculation:** Computes various IAA metrics, including Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha, and F-measure. Enables pairwise and overall IAA calculation [1].
* **Data Filtering:** Allows for data filtering by annotator or item to assess agreement subsets.
* **Visualization:**  Generates graphs and tables to present IAA results clearly, aiding in interpretation. Provides options for saving these visualizations to files.
* **Disagreement Analysis:**  Identifies and analyzes specific disagreements between annotators, including detailed reports and statistics. Allows for exporting disagreements for further analysis.
* **Configuration Options:**  Provides flexible options for input data formats, IAA metrics, output directories, and more.

### Basic Usage

The basic command for running IAA-Eval is:

```bash
python iaa_eval.py input_file.csv [options]
```

#### Common Options:

- **Input/Output:**
  - `input_file.csv`: Path to the CSV file containing annotation data
  - `--output <file>`: Path to save the results (default: print to console)
  - `--output-format <format>`: Format for output: text, csv, json, html, console (default: text)

- **Agreement Measures:**
  - `--all`: Calculate all applicable agreement measures
  - `--raw`: Calculate raw agreement (percentage agreement)
  - `--cohen-kappa`: Calculate Cohen's Kappa
  - `--fleiss-kappa`: Calculate Fleiss' Kappa
  - `--krippendorff-alpha`: Calculate Krippendorff's Alpha
  - `--f-measure`: Calculate F-measure
  - `--icc`: Calculate Intraclass Correlation Coefficient
  - `--bwfk`: Calculate Boundary-Weighted Fleiss' Kappa
  - `--dbcaa`: Calculate Distance-Based Cell Agreement Algorithm
  - `--iou`: Calculate Intersection over Union

- **Advanced Options:**
  - `--confidence-interval <value>`: Confidence level (0-1) for intervals (default: 0.95)
  - `--confidence-method <method>`: Method for confidence intervals: bootstrap, normal, wilson, agresti-coull (default: wilson)
  - `--bootstrap-samples <n>`: Number of bootstrap samples (default: 1000)
  - `--positive-class <class>`: Specify positive class for F-measure
  - `--distance-threshold <value>`: Distance threshold for DBCAA (default: 10.0)
  - `--bwfk-width <n>`: Width parameter for BWFK (default: 5)
  - `--icc-form <form>`: ICC form to use: 1,1|2,1|3,1|1,k|2,k|3,k (default: 2,1)

- **Logging:**
  - `-v <level>`: Set verbosity level (0=error, 1=warning, 2=info, 3=debug)

#### Examples:

```bash
# Calculate all agreement measures
python iaa_eval.py annotations.csv --all

# Calculate specific measures with confidence intervals
python iaa_eval.py annotations.csv --cohen-kappa --fleiss-kappa --confidence-interval 0.95

# Calculate ICC with specific form and save results to CSV
python iaa_eval.py annotations.csv --icc --icc-form 3,1 --output results.csv --output-format csv

# Calculate F-measure with a specific positive class
python iaa_eval.py annotations.csv --f-measure --positive-class 1

# Get detailed help and options
python iaa_eval.py --help
python iaa_eval.py --show-options
```

For a complete overview of all options and their descriptions, use:
```bash
python iaa_eval.py --show-options
```

### Data Format

The tool accepts CSV files with the following format:

- Each row represents an annotated item (e.g., a review)
- Columns should include scores from each annotator with the suffix `_score`
- For example: `Annotator1_score`, `Annotator2_score`, etc.

Example of accepted CSV format:

```
id,text,Annotator1_score,Annotator2_score,Annotator3_score
1,"Text to annotate 1",5,4,5
2,"Text to annotate 2",3,3,4
3,"Text to annotate 3",1,2,1
```

Missing values are allowed and will be ignored when calculating agreement.

### Core Modules

#### raw_agreement.py

This module calculates raw agreement between annotators. It provides the following functionalities:

- `calculate_pairwise()`: Calculates pairwise agreement between all annotators
- `calculate_overall()`: Calculates overall agreement across all annotators
- `get_agreement_statistics()`: Provides agreement statistics (overall, average, min, max)

Agreement is calculated as the proportion of items for which annotators gave the same score.

#### dataPreparation.py

This module handles loading and preprocessing annotation data.

#### confident_interval.py

This module calculates confidence intervals for agreement measures. Confidence intervals provide a range of values that likely contain the true agreement value, helping to assess the reliability of the calculated agreement.

The module supports several methods for calculating confidence intervals:

1. **Bootstrap Method**: This non-parametric approach generates multiple resampled datasets from the original data, calculates agreement for each sample, and determines the confidence interval from the distribution of these values. It's robust and doesn't require assumptions about the underlying distribution.

2. **Normal Approximation**: This method assumes that the sampling distribution of agreement follows a normal distribution. It uses the standard error of the agreement measure to calculate confidence intervals. This approach is computationally efficient but assumes normality.

3. **Wilson Score Interval**: Particularly useful for proportions (like agreement scores), this method provides better coverage than normal approximation, especially for extreme agreement values (near 0 or 1) or small sample sizes.

4. **Agresti-Coull Interval**: An improved version of the Wilson Score method that adds "pseudo-observations" to the sample, resulting in intervals with better coverage properties.

The choice of method depends on factors such as sample size, agreement values, and computational constraints:

- For small samples or extreme agreement values, bootstrap or Wilson Score methods are recommended
- For large samples with moderate agreement values, normal approximation is computationally efficient
- When in doubt, the bootstrap method provides robust estimates without distributional assumptions

The confidence level (typically 95%) indicates the probability that the true agreement value falls within the calculated interval.

**Advanced Features**

• **Semantic Information:** The application can potentially incorporate semantic information to analyze how it affects IAA, building on research which suggests that providing background semantics increases inter-annotator agreement [1-2].

• **Support for diverse data types:** The application can be extended to support data coming from different annotation tasks (e.g., segmentation, cell detection), and metrics (e.g., BWFK, DBCAA) [2].

• **Annotation variation analysis :** The application can incorporate a tool to analyze how the introduction of additional information influences agreement, for example, by calculating changes in the metrics before and after using semantics [1-2].

• **Integration with other tools:** The application can easily export results in common data format, for use in external data analysis tools.

## Bibliography

[1] Artstein, R. (2017). Inter-annotator Agreement. In: Ide, N., Pustejovsky, J. (eds) Handbook of Linguistic Annotation. Springer, Dordrecht. https://doi.org/10.1007/978-94-024-0881-2_11

[2] Vámos, Csilla et al. 'Ontology of Active and Passive Environmental Exposure'. 1 Jan. 2024 : 1733 – 1761

[3] Cheng, Xiang, Raveesh Mayya, and João Sedoc. "From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research." *arXiv preprint arXiv:2412.14461* (2024).

[4] Reyero Lobo, Paula, et al. "Enhancing Hate Speech Annotations with Background Semantics."  *ECAI 2024* . IOS Press, 2024. 3923-3930.

[5] McHugh ML. Interrater reliability: the kappa statistic. Biochem Med (Zagreb). 2012;22(3):276-82. PMID: 23092060; PMCID: PMC3900052.

[6] Hayes, A.F., Krippendorff, K.: Answering the call for a standard reliability measure for coding data. Commun. Methods Meas. 1(1), 77–89 (2007)

[7] Zhang, Ziqi, Sam Chapman, and Fabio Ciravegna. "A methodology towards effective and efficient manual document annotation: addressing annotator discrepancy and annotation quality."  *Knowledge Engineering and Management by the Masses: 17th International Conference, EKAW 2010, Lisbon, Portugal, October 11-15, 2010. Proceedings 17* . Springer Berlin Heidelberg, 2010.

[8] Reyero Lobo, Paula, et al. "Enhancing Hate Speech Annotations with Background Semantics."  *ECAI 2024* . IOS Press, 2024. 3923-3930.

[9] Krippendorff, K.: Reliability in content analysis: some common misconceptions and recom-mendations. Hum. Commun. Res. 30(3), 411–433 (2004)

[10] Capar, Abdulkerim & Ekinci, Dursun & Ertano, Mucahit & Niazi, M. & Balaban, Erva & Aloglu, Ibrahim & Dogan, Meryem & Su, Ziyu & Aker, Fugen & Gurcan, Metin. (2024). An interpretable framework for inter-observer agreement measurements in TILs scoring on histopathological breast images: A proof-of-principle study. PLOS ONE. 19. 10.1371/journal.pone.0314450.

[11] Koo TK, Li MY. A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research. J Chiropr Med. 2016 Jun;15(2):155-63. doi: 10.1016/j.jcm.2016.02.012. Epub 2016 Mar 31. Erratum in: J Chiropr Med. 2017 Dec;16(4):346. doi: 10.1016/j.jcm.2017.10.001. PMID: 27330520; PMCID: PMC4913118.

## Contributing

We welcome contributions to **AnnotationQuality**! Whether you're fixing bugs, adding new features, improving documentation, or reporting issues, your help is appreciated. Here's how you can contribute:

### Getting Started

1. **Fork the Repository**
   - Fork the repository on GitHub
   - Clone your fork locally: `git clone https://github.com/yourusername/AnnotationQuality.git`
   - Add the upstream repository: `git remote add upstream https://github.com/Wameuh/AnnotationQuality.git`

2. **Set Up Development Environment**
   - Create a virtual environment: `python -m venv venv`
   - Activate the virtual environment:
     - Windows: `.\venv\Scripts\activate`
     - Unix/MacOS: `source venv/bin/activate`
   - Install dependencies: `pip install -r requirements.txt`
   - Install development dependencies: `pip install -r requirements-dev.txt`

### Development Guidelines

1. **Code Style**
   - Follow PEP 8 guidelines
   - Use 4 spaces for indentation
   - Limit line length to 79 characters
   - Use descriptive variable and function names in snake_case
   - Use PascalCase for class names
   - Add docstrings following Google Python Style Guide

2. **Testing**
   - Write tests for all new functionality
   - Aim for 100% test coverage
   - Include both unit and integration tests
   - Run tests using:
     ```bash
     pytest                                           # Run all tests
     pytest Tests/UnitTests/test_file.py -v          # Test specific file
     pytest --cov=src --cov-report=term-missing      # Check coverage
     ```

3. **Documentation**
   - Update documentation for any changes
   - Include docstrings for all functions and classes
   - Update README.md if adding new features
   - Add examples for new functionality

4. **Version Control**
   - Create a new branch for each feature/fix
   - Follow the GitHub Flow branching strategy
   - Use Conventional Commits for commit messages
   - Keep commits focused and atomic

### Submitting Changes

1. **Before Submitting**
   - Ensure all tests pass
   - Run linting checks: `flake8`
   - Update documentation if needed
   - Add your changes to CHANGELOG.md

2. **Pull Request Process**
   - Push your changes to your fork
   - Create a Pull Request (PR) to the main repository
   - Fill out the PR template completely
   - Link any related issues
   - Wait for review and address any feedback

3. **Code Review**
   - All contributions require review
   - Address review comments promptly
   - Keep discussions professional and constructive
   - Be open to suggestions and improvements

### Security

- Report security vulnerabilities privately
- Use HTTPS for secure connections
- Never commit sensitive information
- Use environment variables for secrets

### Need Help?

- Check existing issues and documentation
- Ask for clarification on unclear tasks
- Join our community discussions
- Reach out to maintainers

Thank you for contributing to AnnotationQuality! Your efforts help make this tool better for everyone.

## License

This project is licensed under the MIT license.

## Available Agreement Measures

The following agreement measures are available in IAA-Eval:

- **Raw Agreement (Percentage Agreement)**: The simplest measure, calculating the percentage of items on which annotators agree. While easy to interpret, it doesn't account for chance agreement.

- **Cohen's Kappa**: Designed for two annotators, this measure calculates agreement while accounting for chance agreement. Particularly useful for categorical annotations.

- **Fleiss' Kappa**: An extension of Cohen's Kappa for three or more annotators. Suitable for measuring agreement in classification tasks with multiple annotators.

- **Krippendorff's Alpha**: A versatile measure that handles missing data and works with any number of annotators. It can accommodate different types of data (nominal, ordinal, interval, ratio).

- **F-measure**: Calculates agreement using precision, recall, and their harmonic mean. Particularly useful for tasks like named entity recognition or classification where both precision and recall are important.

- **Intraclass Correlation Coefficient (ICC)**: Measures the degree of correlation and agreement between measurements made by different annotators. Supports different forms (1,1 | 2,1 | 3,1 | 1,k | 2,k | 3,k) for various study designs.

- **Boundary-Weighted Fleiss' Kappa (BWFK)**: Specialized measure for evaluating agreement in binary segmentation tasks, particularly useful for tissue boundaries where minor disagreements are common.

- **Distance-Based Cell Agreement Algorithm (DBCAA)**: Designed specifically for cell annotations, this algorithm measures agreement based on spatial relationships without requiring ground truth.

- **Intersection over Union (IoU)**: Measures agreement in segmentation tasks by calculating the overlap between annotated regions. Particularly useful for evaluating spatial annotations.

Each measure can be calculated with confidence intervals using various methods (bootstrap, normal approximation, Wilson score, or Agresti-Coull), providing statistical bounds for the agreement values.

