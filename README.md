# NewsGuard — Fake News Detection

A Machine Learning web application that classifies news text as **FAKE** or **REAL** using TF-IDF text vectorization and Logistic Regression.

## Live Demo

🚀 **Try the app:** https://siyam-fake-news-detector.streamlit.app/

## Project Overview

NewsGuard demonstrates how Natural Language Processing (NLP) and machine learning can be used to analyze news text and classify it into two categories: fake and real.

The application provides an interactive interface where users can enter news text and view the model's prediction and confidence score.

## Features

- News text classification
- TF-IDF feature extraction
- Logistic Regression classifier
- Prediction confidence display
- Interactive Streamlit interface
- Sample text testing

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Joblib
- Git and GitHub

## Model Evaluation

The saved model was evaluated on a held-out split of the available dataset.

| Metric | Result |
|---|---:|
| Test accuracy | 98.96% |
| Test samples | 7,821 |

**Note:** This result comes from the current evaluation script. Further validation is needed to rule out data leakage and confirm that the evaluation setup matches the original model training process.


## Project Structure

```text
Fake-News-Detection/
├── app.py
├── data/
├── docs/
│   └── screenshots/
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SiyamMukhtar/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal.

## Limitations

- Predictions depend on the training data and patterns learned by the model.
- Confidence scores are model outputs, not proof that a news claim is true or false.
- Important claims should be verified using reliable sources.
- Further testing on unseen, diverse news sources is needed to assess generalization.

## Author

**Siyam Mukhtar**

- GitHub: [SiyamMukhtar](https://github.com/SiyamMukhtar)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/siyam-mukhtar-619508401/)

---

If you find this project useful, feel free to explore the repository.
