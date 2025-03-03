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
* **F-measure:** This measure calculates agreement using precision, recall, and their harmonic mean. It is particularly useful for tasks such as named entity recognition or classification, where the ability of a model to identify correct entities and its ability to identify all relevant entities must be balanced [].
* **Boundary-Weighted Fleiss' Kappa (BWFK):** This method is used to evaluate agreement in tissue segmentation, particularly on tissue boundaries. It reduces the impact of minor disagreements that often occur along these boundaries [].
* **Distance-Based Cell Agreement Algorithm (DBCAA):** An algorithm that measures agreement in cell detection without relying on ground truth [7].
* **Intersection over Union (IoU)-Based Measures:** These measures are used for regional segmentation but are limited for assessments involving more than two observers [1].
* **Analysis of Annotation Variance:** For a deeper understanding, an analysis of the variance in annotations can be performed to study how annotations change before and after the introduction of semantics, in order to better understand cases where agreement decreases despite the addition of semantic information [4, 8].

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

**The primary command for calculating IAA is**:

```
python iaa-eval.py --input <file> --format <format> --annotator_col <col> --item_col <col> --metric <metric> --output <output_dir> [--viz]
```

• `--input <file>`: Path to the annotation data file.

• `--format <format>`: Format of the input file (**csv**, **json**).

• `--annotator_col <col>`: Name of the column containing annotator IDs.

• `--item_col <col>`: Name of the column containing item IDs.

• `--metric <metric>`: IAA metric to use (**cohen_kappa**, **fleiss_kappa**, **krippendorff_alpha**, **f_measure**).

• `--output <output_dir>`: Directory to save the output results.

• `--viz`: Optional flag to generate visualizations.

• `--disagreements` : flag to activate disagreement analysis.

### For help and options:

```
iaa-eval --help
```

### Example

To calculate Cohen's Kappa on a CSV file named **annotations.csv** with annotator IDs in the **annotator_id** column and item IDs in the **item_id** column, saving the output to a directory named **results**:

```
iaa-eval --input annotations.csv --format csv --annotator_col annotator_id --item_col item_id --metric cohen_kappa --output results --viz
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

## Contributing

We welcome contributions to **IAA-Eval**. Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT license.

## Available Agreement Measures

- Raw agreement (percentage of agreement)
- Cohen's Kappa (for two annotators)
- Fleiss' Kappa (for three or more annotators)
- Krippendorff's Alpha (for any number of annotators, handles missing data)

