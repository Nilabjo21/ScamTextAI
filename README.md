# ScamText AI

An explainable binary text classifier for identifying spam and scam-like messages. ScamText AI uses genuine labeled public datasets, benchmarks TF-IDF baselines against modern NLP models, and provides an interactive Streamlit application for both single-message inspection and CSV batch analysis.

---

## Features

- **Binary Classification**: Flags messages as **spam/scam** or **legitimate**.
- **Dual Baseline Models**: Trains and benchmarks Logistic Regression and Multinomial Naive Bayes using TF-IDF unigram and bigram features.
- **Explainable Indicators**: Lexical heuristics (e.g., urgency markers, prize keywords, suspicious URLs, requests for credentials) are surfaced alongside model evidence to explain predictions transparently.
- **Interactive Streamlit Web Interface**:
  - **Detector**: Interactive interface to test individual messages or upload batch CSV files with instant visual risk indicators.
  - **Analytics**: Evaluates held-out test set performance, precision-recall curves, and class distributions.
  - **Train & Evaluate**: UI trigger to run pipeline training directly with optional transformer fine-tuning.
- **Optional Transformer Support**: Train and evaluate DistilBERT alongside TF-IDF baselines.

---

## Project Structure

```text
ScamText_AI_Project/
├── .streamlit/
│   └── config.toml             # Streamlit theme & UI styling
├── data/
│   └── README.md               # Dataset descriptions & citation info
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_loader.py          # Automatic dataset fetching & validation
│   ├── preprocessing.py       # Text cleaning & feature tokenization
│   ├── model_training.py       # Model training, calibration & evaluation
│   ├── prediction.py           # Real-time inference & heuristic analysis
│   └── pipeline.py             # End-to-end data & training pipeline
├── tests/
│   └── test_data_and_preprocessing.py # Unit tests for data & preprocessing
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Getting Started

### 1. Prerequisites

Python 3.10 or higher is recommended.

### 2. Installation

Clone this repository and install the dependencies:

```bash
git clone <your-repository-url>
cd ScamText_AI_Project
pip install -r requirements.txt
```

### 3. Train the Models

Run the end-to-end pipeline to download datasets, process features, and train the baseline classifiers:

```bash
python src/pipeline.py
```

*(Optional)* To also fine-tune DistilBERT:

```bash
python src/pipeline.py --with-distilbert
```

### 4. Run the Web Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the detector and review analytics.

---

## Dataset & Methodology

- **Primary Corpus**: Public [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). The pipeline automatically fetches this corpus on first run if not present locally.
- **Evaluation**: Stratified split (70% train, 10% validation, 20% test). Validation splits are used to tune optimal decision thresholds, while final metrics are calculated strictly on the held-out test split.
- **Custom CSVs**: You can drop custom binary-labeled CSV files into `data/` to incorporate additional samples into training.

---

## License

This project is licensed under the MIT License.
