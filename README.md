<div align="center">

<img src="./ads509-banner.svg" alt="Steam Review Sentiment Analysis: Machine Learning vs. Transformers" width="100%" />

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**ADS 509 · Applied Large Language Models for Data Science**  
Master of Applied Data Science · University of San Diego  
**Author:** Sushma Kafle · **Status:** Completed

</div>

---

## 🎯 At a Glance

Can transformer-based language models better understand player sentiment than traditional text classifiers? This project evaluates **three NLP approaches on the same held-out test set**, using real Steam game reviews and a reproducible machine learning workflow.

| 📚 Reviews | 🎮 Games | 🧪 Test set | 🏆 Best model |
|:--:|:--:|:--:|:--:|
| **4,869** | **5** | **731 reviews** | **DistilBERT · 92.61%** |

## 📊 Model Performance

| Approach | Model | Held-out test accuracy |
|---|---|---:|
| Traditional ML | TF-IDF + Logistic Regression | **87.96%** |
| Fine-tuned transformer | **DistilBERT** | **92.61%** 🏆 |
| Zero-shot generative AI | FLAN-T5 Small | **87.96%** |

**Key takeaway:** Fine-tuned DistilBERT outperformed the traditional baseline by **4.65 percentage points**. Notably, zero-shot FLAN-T5 matched the baseline without training on the project's labeled reviews. These results highlight the trade-off between predictive performance and task-specific training effort.

## 🔎 Research Questions

1. How accurately can **TF-IDF + Logistic Regression** classify player sentiment?
2. Does fine-tuning **DistilBERT** improve performance?
3. Can **FLAN-T5 Small** classify reviews using zero-shot instructions alone?
4. What are the practical trade-offs among traditional ML, fine-tuned transformers, and generative models?

## 🎮 Data Collection & Preparation

Reviews were collected from the **Steam Reviews API** across five games and diverse player communities:

| Game | Game | Game | Game | Game |
|---|---|---|---|---|
| Elden Ring | Cyberpunk 2077 | No Man's Sky | Cities: Skylines II | Stardew Valley |

- Used **cursor-based pagination** and targeted approximately balanced positive and negative recommendations.
- Deduplicated records by `recommendationid`, checked language, and normalized review text while retaining originals.
- Removed **six predominantly Chinese-language reviews** returned despite the API's English-language filter.
- Derived labels from Steam's `voted_up` recommendation indicator, **not manual sentiment annotations**.

**Final class balance:** 2,453 positive reviews and 2,416 negative reviews.

### Key fields

| Field | Purpose |
|---|---|
| `review` | Original player review |
| `clean_review` | Normalized modeling text |
| `voted_up` | Recommendation-based sentiment label |
| `recommendationid` | Deduplication identifier |
| Game identifier | Game-level analysis |
| Review word count | Review-length EDA |

## 🧠 Methodology

```text
Steam Reviews API
       │
       ▼
Collection & deduplication
       │
       ▼
Text cleaning & exploratory analysis
       │
       ▼
Stratified split (70% train / 15% validation / 15% test)
       │
       ├── TF-IDF + Logistic Regression
       ├── Fine-tuned DistilBERT (2 epochs)
       └── Zero-shot FLAN-T5 Small
       │
       ▼
Accuracy · Precision · Recall · F1 · Confusion Matrices
```

**Data split:** 3,408 training / 730 validation / 731 testing reviews. All final reported accuracies are evaluated on the **same held-out test set**.

### Modeling approaches

- **Baseline:** TF-IDF vectorization with Logistic Regression provides an interpretable, computationally efficient reference.
- **Transformer:** `distilbert-base-uncased` is fine-tuned for two epochs on labeled training reviews.
- **Zero-shot LLM:** `google/flan-t5-small` uses natural-language instructions to classify reviews without task-specific fine-tuning.

## 💡 Findings & Interpretation

- **Fine-tuning matters:** DistilBERT delivered the strongest test-set accuracy.
- **Zero-shot is competitive:** FLAN-T5 Small matched the traditional baseline without training on project labels.
- **Model choice depends on constraints:** Training time, hardware resources, and inference costs should be weighed alongside accuracy.
- **Sentiment is nuanced:** Binary recommendation labels can miss sarcasm, mixed opinions, and context-dependent language.

## ⚠️ Challenges & Limitations

- Transformer experiments encountered **Apple MPS memory constraints**; CPU inference was used where needed.
- Attempts to fine-tune FLAN-T5 did not yield a reliable result; the reported FLAN-T5 experiment is **zero-shot only**.
- The dataset spans **five games** and may not generalize to every Steam community.
- Recollecting live API reviews may yield different records and results.

## 🛠️ Tools & Technologies

**Python · pandas · requests · matplotlib · scikit-learn · PyTorch · Hugging Face Transformers · Hugging Face Datasets · Jupyter Notebook**

## 🚀 Run the Project

```bash
git clone https://github.com/Sushma-Kafle/ADS_509_Final_Project.git
cd ADS_509_Final_Project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install jupyter pandas requests matplotlib scikit-learn torch transformers datasets
jupyter notebook
```

Open **`Final_Project.ipynb`** and run the cells in order. The notebook handles Steam API collection, data cleaning, EDA, model training, and evaluation.

> **Reproducibility note:** The repository contains the notebook and README, but not the generated dataset or trained model artifacts. Re-running live collection requires internet access and may produce a different sample. Transformer downloads and training can take time.

## 📁 Repository Structure

```text
ADS_509_Final_Project/
├── Final_Project.ipynb   # End-to-end analysis and modeling
├── README.md            # Project documentation
└── .gitignore
```

## 🔭 Future Directions

Explore more game genres, test robustness across individual games, evaluate additional transformer architectures, and examine errors in nuanced or sarcastic reviews. Consider deployment-oriented comparisons of inference latency and compute requirements.

## 🙌 Acknowledgments

Developed as an individual project for the **University of San Diego Master of Applied Data Science** program. Thanks to Valve/Steam for the review API and the open-source scikit-learn, PyTorch, and Hugging Face communities. AI assistance supported troubleshooting and documentation; technical outputs were reviewed during project development.

**License:** No license has been selected. All rights reserved unless a license is added.
