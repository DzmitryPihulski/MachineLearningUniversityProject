# Research of an Upper Bound on the Fraction of Training Errors for Different Kernels in One-Class Classification with OneClassSVM of the Text Embeddings of Political and Non-Political Texts

This repository contains code and data for a research project focused on evaluating different kernels used in the One-Class SVM (OC-SVM) algorithm for one-class classification problems. The objective of this study is to assess the upper bound on the fraction of training errors based on text embeddings of political and non-political content.


**Reproducibility**: To ensure reproducibility, we recommend downloading the pre-collected datasets (links provided in the data section below) rather than re-scraping the data, as websites may have changed or become unavailable since data collection. The experiments can then be reproduced by running files 01-04 with the provided datasets.


## 1. List and Description of Files

### 1.1 Data

The `data` folder doesn't have `datasets` folder, but it can be downloaded via the following link:
[Google Drive - Datasets Folder](https://drive.google.com/drive/folders/11AmxtSDsoE1bGa2_vg7HkEt6N4ORRL4Z?usp=sharing). 

This folder contains structured datasets used throughout the experimentation process, including both raw web-scraped data and pre-processed text embeddings. It is **highly recommended** to download it firstly for reproducibility

---

### 1.2 Experiments

#### 1.2.1 Preparation

The data-gathering pipeline consists of the following steps:

1. **Politician Selection**  
   The file `data/datasets/politicians_by_country.json` contains 10 countries, each with a list of 10 prominent politicians. Countries were chosen to ensure linguistic diversity (e.g., USA, Germany, Poland, Japan, etc.).

2. **Search Result Collection**  
   The [Serper.dev](https://serper.dev/) Google Search API was used to collect search results for each individual politician. For each country, the search parameters (location and language) were set specifically to match the local context and language.

3. **Scraping and Text Extraction**  
   Web scraping and processing of text fragments were conducted using the `scraping.py` script.


##### `scraping.py`

This script automates the acquisition of textual data from webpages corresponding to political figures. It operates in two stages:

**Input Parsing**:
   The script begins by loading a JSON file (`politicians_search_results.json`) that contains structured search results from Google, including organic search links and associated search queries (e.g., politician names).

**Asynchronous Web Scraping**:
   For each query (e.g., a politician’s name), the script:

   * Extracts all valid URLs from the organic search results.
   * Initiates asynchronous HTTP requests to download the contents of each linked page using `aiohttp`.
   * Parses the HTML content using `BeautifulSoup` to extract clean textual data from the `<body>` of the webpage.
   * Locates the first occurrence of any query term within the page content.
   * Extracts a fixed-size textual window (500 characters) centered around the first occurrence of Politician name or surname using the `extract_text_window()` function to ensure relevant context is captured.
   * Filters out invalid or non-textual responses and aggregates valid entries under each query.

**Output**:
   The script writes the final output to a JSON file (`extracted_text_results.json`), where each key is a query (e.g., politician’s name) and the value is a list of dictionaries containing `link` and `text` pairs.

This extracted text serves as the raw material for the embedding stage, where each sample is transformed into a high-dimensional vector representation using a sentence embedding model prior to classification.


4. **Negative Sample Collection**  
   Steps 2 and 3 were repeated to collect `negative` (non-political) samples. Five non-political topics were selected:  
   - "Renewable energy storage solutions"  
   - "Michelin-star restaurants in New York"  
   - "Effects of microplastics on marine life"  
   - "Latest advances in AI-generated art"  
   - "History of jazz music in New Orleans"  
   
   These topics were adapted and translated for each country to ensure meaningful and contextually relevant results. Search results for these topics were gathered per country and then scraped. From each page, 10 text fragments of 500 characters were extracted.

5. **Text Embedding and Dataset Splitting**  
   All collected text fragments (both positive and negative) were embedded using Kaggle notebooks for [positive samples](https://www.kaggle.com/code/dzmitrypihulski/processing-texts-um) and [negative samples](https://www.kaggle.com/code/dzmitrypihulski/fork-of-processing-texts-um-zbi-r-testowy).  
   The resulting dataset was split as follows:
   - `data/datasets/train.jsonl`: Contains **only positive** samples (5,395 in total).
   - `data/datasets/test.jsonl`: Contains an **equal number of positive and negative** samples (2,698 total).

6. **Dataset Loading**  
   The datasets can be loaded using the `data_loader.py` module.




##### `data_loader.py`

This module defines the `CustomDataLoader` class, which provides a interface for loading precomputed text embeddings into memory.

**Key functionalities include:**

* **Model Flexibility**:
  The class is initialized with a list of supported model names (e.g., `"xlm-roberta"`, `"labse"`, `"distill_bert"`).

* **Training Data Loader (`load_train_embeddings`)**:
  It parses the JSON object and extracts the vector embedding associated with the specified model name.

* **Testing Data Loader (`load_test_embeddings`)**:
  Similar to the training loader

This loader is used in the experimentation phase, enabling consistent and memory-efficient access to the embedded data throughout training, validation, and testing workflows.

#### 1.2.3 Experiments Conducted

##### `01_analysis_of_threshold.py`

The script conducts the core evaluation of model performance across various kernel types and values of the hyperparameter `ν` (nu), which in the context of One-Class SVM represents an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.

The experiment evaluates the F1-score of OC-SVM classifiers for combinations of:

* Three different text embedding models: `xlm-roberta`, `labse`, `distill_bert`
* Four SVM kernel types: `linear`, `rbf`, `poly`, `sigmoid`
* Nine values of `ν`: \[0.05, 0.1, ..., 0.8]

The experiment is structured as follows:

1. **Evaluation Strategy**:

   * The training data is split into 10 folds using K-Fold for training the OC-SVM model.
   * The test data is similarly split into 10 corresponding folds.
   * Each model-kernel-ν combination is evaluated over these folds, resulting in robust estimates of performance.

2. **Standardization**:
   For each fold, training and testing embeddings are scaled using `StandardScaler` to ensure consistent feature ranges, which is critical for kernel-based methods.

3. **Model Training and Evaluation**:

   * An OC-SVM model is trained with the given kernel and `ν` value.
   * Predictions on the test fold are evaluated using the F1-score against the ground truth binary labels.
   * The number of iterations to converge (`n_iter_`) and the fit status are logged.

4. **Logging and Results**:

   * Logging is written to `01_analysis_of_threshold.log` for traceability.
   * Results for each configuration are stored in `results_nu.jsonl` as individual JSON lines, capturing the configuration, F1-scores across folds, and fit diagnostics.

This file forms the quantitative backbone of the research, producing data required to analyze the sensitivity of OC-SVM performance to different kernel functions and `ν` values.

---

##### `02_plots_of_nu.py`

This script visualizes the experimental results generated in `results_nu.jsonl`. It reads the recorded F1-scores and produces comparative plots illustrating:

* The effect of `ν` on model performance across kernels and embedding types.
* Trends and variances to assist in selecting optimal configurations.


---

##### `03_embedding_analysis.py`
This script analyzes the internal structure and similarity of embeddings produced by three multilingual models: XLM-RoBERTa, LaBSE, and Distilled BERT. It computes intra-group variances (how dispersed the embedding dimensions are within each model) and inter-group distances (how far apart the average embeddings of different models are). Results are saved both as a textual report and as bar plots.

---

##### `04_PCA.py`
This script performs a PCA-based analysis of text embeddings to answer three research questions: (1) how many components are needed to retain various levels of variance across models (XLM-RoBERTa, LaBSE, Distilled BERT); (2) how well political texts and outliers separate in 2D and 3D PCA space; and (3) which components are most discriminative between the two classes. The script produces informative plots and logs summarizing variance thresholds, class separability, and component-wise interpretability.


## 2. Summary

This project investigates how different kernel functions affect the performance of the One-Class SVM (OC-SVM) algorithm when applied to text embeddings of political and non-political content. It explores the relationship between the `ν` parameter and the fraction of training errors, utilizing three multilingual embedding models (`xlm-roberta`, `labse`, `distill_bert`) and four kernel types (`linear`, `rbf`, `poly`, `sigmoid`). The study uses 10-fold evaluation and uses F1-score, supported by comprehensive logging, visualization, and embedding analyses, including PCA and inter-model comparisons. Scripts for data acquisition, preprocessing, and statistical evaluation are included to ensure full reproducibility of the results.
