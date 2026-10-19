# Steam Game Review Sentiment Analysis: Machine Learning vs. Transformers

**Course:** ADS-509 : Applied Large Language Models for Data Science  
**Program:** Master of Applied Data Science, University of San Diego  
**Project Status:** Completed  
**Author:** Sushma Kafle


## Project Title

This project is a part of the ADS-509 course in the Applied Data Science Program at the University of San Diego.

## Project Introduction / Objective

This project investigates how effectively different natural language processing approaches classify Steam game reviews as **positive or negative**. It compares a traditional machine learning baseline (TF-IDF + Logistic Regression), a fine-tuned transformer (DistilBERT) and a zero-shot generative language model (FLAN-T5 Small) on the same held-out test set.

The goal is to evaluate whether more advanced language models improve sentiment classification and whether a generative model can provide useful predictions without task-specific training. These findings can inform how gaming platforms, developers and analysts summarize large volumes of player feedback.

## Project Description

### Data source and scope

Reviews were collected using the **Steam Reviews API** for five games spanning different genres and player communities:

- Elden Ring
- Cyberpunk 2077
- No Man's Sky
- Cities: Skylines II
- Stardew Valley

The collection process used cursor-based pagination and sought a roughly balanced sample of positive and negative recommendations for each game. Reviews were combined, deduplicated using review identifiers, and cleaned. Six predominantly Chinese-language reviews were removed after the API's English-language filter returned some non-English text. The original review text was preserved alongside a normalized text column.

**Final dataset:** 4,869 reviews  **2,453 positive** and **2,416 negative**. Sentiment labels were derived from Steam's user recommendation indicator rather than manually annotated review text.

### Key data fields

| Field | Description |
| --- | --- |
| `review` | Original Steam review text |
| `clean_review` | Normalized text used for analysis and modeling |
| `voted_up` | Steam recommendation indicator used to derive sentiment |
| `recommendationid` | Review identifier used to detect duplicates |
| Game identifier/name | Identifies which of the five games the review describes |
| Review word count | Derived feature used in exploratory analysis |

Other Steam API metadata may be present in the raw and intermediate datasets.

### Research questions

1. How well does a traditional TF-IDF and Logistic Regression classifier predict review sentiment?
2. Does fine-tuning DistilBERT improve classification accuracy?
3. Can FLAN-T5 Small classify reviews with zero-shot instructions and no task-specific training?
4. What are the practical trade-offs among these approaches?

### Analysis and modeling

- **Collection and cleaning:** Steam API requests, cursor pagination, deduplication, language checks, text normalization, and CSV exports.
- **Exploratory data analysis:** Sentiment distribution by game, review lengths, box plots, and frequent-word comparisons.
- **Data splitting:** Stratified **70% training / 15% validation / 15% testing** split (3,408 / 730 / 731 reviews).
- **Model 1:** TF-IDF features with Logistic Regression.
- **Model 2:** Pretrained `distilbert-base-uncased` fine-tuned for two epochs.
- **Model 3:** Pretrained `google/flan-t5-small` used for zero-shot sentiment classification through written instructions, without fine-tuning.
- **Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrices; final comparisons use the same held-out test set.

### Final results

| Model | Test accuracy (731 reviews) |
| --- | ---: |
| TF-IDF + Logistic Regression | **87.96%** |
| Fine-tuned DistilBERT | **92.61%** |
| FLAN-T5 Small (zero-shot) | **87.96%** |

DistilBERT performed best, exceeding Logistic Regression by **4.65 percentage points**. FLAN-T5 matched the baseline without training on the project's labeled reviews.

### Challenges and limitations

- Training transformer models on a Mac encountered Apple MPS memory constraints; CPU inference was used where necessary.
- Attempts to fine-tune FLAN-T5 did not yield a reliable result, so the reported FLAN-T5 comparison is **zero-shot only**.
- Steam recommendation labels may not capture sarcasm, mixed opinions, or nuanced sentiment.
- Reviews from five games may not represent all Steam titles or gaming communities.
- Re-running live Steam API collection may retrieve different reviews, so exact results may vary unless the original collected dataset is reused.

## Methods Used

- Natural language processing and text mining
- Data collection and data cleaning
- Exploratory data analysis and data visualization
- Machine learning and predictive modeling
- Transformer fine-tuning
- Zero-shot generative AI classification
- Model evaluation and comparison

## Technologies

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Libraries:** pandas, requests, matplotlib, scikit-learn, PyTorch, Hugging Face Transformers, Hugging Face Datasets
- **Models:** Logistic Regression, DistilBERT, FLAN-T5 Small
- **Data source:** Steam Reviews API

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the placeholders with your actual GitHub repository details.

### 2. Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install jupyter pandas requests matplotlib scikit-learn torch transformers datasets
```

Package versions and PyTorch acceleration support may vary by operating system. An internet connection is required for Steam API collection and first-time downloads of pretrained Hugging Face models.

### 4. Run the notebook

```bash
jupyter notebook
```

Open the final project notebook (`Final_Project.ipynb`, or its actual filename in the repository) and run cells in order. The notebook collects and saves API data under `data/steam/`, performs cleaning and EDA, trains/evaluates the classifiers, and saves the fine-tuned DistilBERT model locally.

**Reproducibility note:** The notebook includes live API requests and model training. Running it from the beginning may take significant time and may not recreate the exact dataset or accuracy values shown above. To reproduce the reported analysis more closely, reuse the original cleaned dataset if it is provided with the repository and start from the relevant data-loading stage after adjusting notebook paths.

### 5. Edit or extend

Modify notebook cells to change game IDs, data collection targets, preprocessing, model hyperparameters, or evaluation plots. Keep training, validation, and test data separate when experimenting with models.

## Repository Files

- `Final_Project.ipynb` — data collection, EDA, model training, and evaluation notebook (rename the final notebook to this before pushing, if desired).
- `README.md` — project documentation.
- `data/steam/` — generated raw and cleaned data files, if included.
- Saved model folders and intermediate outputs — generated locally; large files need not be committed.

## Contributors

**Sushma Kafle** — individual project.

## License

No license has been selected yet. **All rights reserved** unless a license is explicitly added. If an open-source license is desired, add a `LICENSE` file and update this section.

## Acknowledgments

Thanks to the University of San Diego Applied Data Science program for the course framework; Valve/Steam for providing review data through its API; and the open-source Python, scikit-learn, PyTorch and Hugging Face communities. An AI assistant was used to support troubleshooting, clarify technical issues and organize the analysis; suggestions and model outputs were checked during the project.
