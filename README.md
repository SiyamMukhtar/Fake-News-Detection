# Fake News Detection using Machine Learning

## Overview
This project uses Natural Language Processing (NLP) and Machine
Learning to classify news articles as FAKE or REAL.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Jupyter Notebook

## Project Workflow
1. Load the Fake and Real news datasets.
2. Combine both datasets and assign labels.
3. Remove blank news text and duplicate articles.
4. Split the dataset into training and testing sets.
5. Convert text into numerical features using TF-IDF.
6. Train a Logistic Regression classifier.
7. Evaluate the model using accuracy, precision, recall,
   F1-score, and a confusion matrix.

## Dataset
Fake and Real News Dataset from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Model Performance

- Algorithm: Logistic Regression
- Text Features: TF-IDF
- Original random-split test accuracy: 98.28%
- Title-grouped test accuracy: 98.03%
- Short-text-excluded experiment accuracy: 98.11%

The title-grouped model is saved in the `models/` directory. Its evaluation reduces overlap between training and testing articles that share the same title.

**Note:** These results are specific to this dataset and evaluation setup. High test accuracy does not guarantee reliable detection of real-world misinformation.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Open the Jupyter Notebook.
3. Run the notebook cells in order.

## Saved Model
The trained model and TF-IDF vectorizer are saved in the
models/ directory.

## Disclaimer
This model is an educational project. Its predictions should
not be treated as definitive verification of news.
