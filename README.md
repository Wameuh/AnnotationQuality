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
* **Boundary-Weighted Fleiss’ Kappa (BWFK):** This method is used to evaluate agreement in tissue segmentation, particularly on tissue boundaries. It reduces the impact of minor disagreements that often occur along these boundaries [].
* **Distance-Based Cell Agreement Algorithm (DBCAA):** An algorithm that measures agreement in cell detection without relying on ground truth [7].
* **Intersection over Union (IoU)-Based Measures:** These measures are used for regional segmentation but are limited for assessments involving more than two observers [1].
* **Analysis of Annotation Variance:** For a deeper understanding, an analysis of the variance in annotations can be performed to study how annotations change before and after the introduction of semantics, in order to better understand cases where agreement decreases despite the addition of semantic information [4, 8].

It is important to note that **the choice of IAA measure depends on the nature of the data and the annotation task** [1]. Additionally, IAA should not be considered an end in itself, but rather as a tool to understand the data and improve the annotation process. Disagreements between annotators can provide valuable insights [1, 9].


## IAA-Eval: Features and Usage

**IAA-Eval** is designed to be a user-friendly CLI tool with the following key features:

* **Data Loading and Preprocessing:** Reads annotation data from various file formats, including CSV and JSON. Supports different annotation types (categorical, numerical, etc.)
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

**Advanced Features**

• **Semantic Information:** The application can potentially incorporate semantic information to analyze how it affects IAA, building on research which suggests that providing background semantics increases inter-annotator agreement [1-2].

• **Support for diverse data types:** The application can be extended to support data coming from different annotation tasks (e.g., segmentation, cell detection), and metrics (e.g., BWFK, DBCAA) [2].

• **Annotation variation analysis :** The application can incorporate a tool to analyze how the introduction of additional information influences agreement, for example, by calculating changes in the metrics before and after using semantics [1-2].

• **Integration with other tools:** The application can easily export results in common data format, for use in external data analysis tools.

## Bibliography

[1] Artstein, R. (2017). Inter-annotator Agreement. In: Ide, N., Pustejovsky, J. (eds) Handbook of Linguistic Annotation. Springer, Dordrecht. https://doi.org/10.1007/978-94-024-0881-2_11
[2] Vámos, Csilla et al. ‘Ontology of Active and Passive Environmental Exposure’. 1 Jan. 2024 : 1733 – 1761
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

