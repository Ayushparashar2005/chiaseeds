# AI-Driven Phishing Email Detection Using NLP

This repository contains a complete, end-to-end B.Tech AIML semester project that detects phishing/spam emails using Natural Language Processing (NLP) and Machine Learning techniques.

## 🛡️ Project Overview
Phishing emails are one of the most common vectors for cyberattacks. This project demonstrates how text classification algorithms can analyze text structure, patterns, and vocabulary to accurately distinguish between **Spam (Phishing)** and **Ham (Legitimate)** emails.

Using the provided dataset `spam.csv`, the project achieves **98.45% accuracy** and **94.33% F1-score** with a Support Vector Machine (LinearSVC) model.

---

## 📁 Repository Structure
```text
├── spam.csv                 # Primary dataset
├── requirements.txt         # Project dependencies
├── app.py                   # Streamlit web application
├── README.md                # Project documentation
├── Project_Report.md        # Comprehensive 15-page project report
├── phishing_email_detection.ipynb  # Interactive Jupyter Notebook
├── models/
│   ├── best_model.pkl       # Serialized best-performing ML model
│   ├── tfidf_vectorizer.pkl # Serialized TF-IDF Vectorizer
│   └── experiments_summary.json # JSON containing metrics of all models
└── visualizations/          # Generated EDA and evaluation graphs
    ├── class_distribution.png
    ├── class_distribution_pie.png
    ├── email_length_distribution.png
    ├── word_count_distribution.png
    ├── top_20_words.png
    ├── spam_wordcloud.png
    ├── ham_wordcloud.png
    ├── model_accuracy_comparison.png
    ├── model_precision_comparison.png
    ├── model_recall_comparison.png
    ├── model_f1_comparison.png
    ├── confusion_matrices.png
    ├── rf_feature_importance.png
    └── roc_curves.png
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Datasets
The project requires the `stopwords`, `wordnet`, `punkt`, and `omw-1.4` corpora from NLTK. They are downloaded automatically on first run, but you can also download them manually using Python:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

---

## 🚀 Execution Guide

### 1. Running the Jupyter Notebook
Open the notebook in your Jupyter environment:
```bash
jupyter notebook phishing_email_detection.ipynb
```
Run all cells to step through the text preprocessing pipelines, EDA, model training, and performance comparisons.

### 2. Launching the Streamlit Web Application
To start the interactive web application:
```bash
streamlit run app.py
```
This will spin up a local development server and open the web interface in your default browser (usually at `http://localhost:8501`).

---

## 📊 Summary of Model Performance (TF-IDF Features)

| Machine Learning Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **98.45%** | **97.08%** | **91.72%** | **94.33%** |
| **Multinomial Naive Bayes** | 97.68% | 95.49% | 87.59% | 91.37% |
| **Logistic Regression** | 97.58% | 94.78% | 87.59% | 91.04% |
| **Random Forest** | 97.39% | 99.17% | 82.07% | 89.81% |
| **Decision Tree** | 94.68% | 80.82% | 81.38% | 81.10% |

*The best model was chosen based on its **F1-score**, which balances classification accuracy on both minority (Spam) and majority (Ham) classes.*

---

## 👤 Author
* **Submitted by:** Ayush
* **Title:** B.Tech Artificial Intelligence and Machine Learning Semester Project
