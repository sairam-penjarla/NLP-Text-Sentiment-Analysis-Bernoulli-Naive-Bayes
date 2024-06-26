# Sentiment Analysis - Twitter Data

This project explores sentiment analysis on a dataset of tweets. It performs the following tasks:

* **Data Preprocessing**
* **Feature Engineering**
* **Model Training**
* **Model Evaluation**
* **Saving Models and Vectorizer**

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sairam-penjarla/Sentiment-Analysis-on-Twitter-Data.git
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script:**
    ```bash
    `cd scripts`
    ```
    ```bash
    `python main.py`
    ```

## Required Libraries:

* pandas
* numpy
* matplotlib.pyplot
* seaborn 
* nltk
* sklearn

## Data Description:

The data used for this project is a subset of the Sentiment140 dataset containing 1.6 million tweets labeled as positive or negative.

## Data Preprocessing:

The preprocessing step involves cleaning the text data by:

* Converting text to lowercase
* Removing URLs and replacing them with "URL"
* Replacing emojis with a predefined dictionary mapping
* Replacing usernames with "USER"
* Removing non-alphanumeric characters
* Reducing consecutive letters to a maximum of two
* Removing short words (length less than 2)
* Lemmatization (converting words to their base form)

## Feature Engineering:

TF-IDF Vectorizer is used to transform the preprocessed text data into numerical features suitable for machine learning algorithms.

## Model Training:

Three machine learning models are trained to classify sentiment:

* Bernoulli Naive Bayes (BNB)
* Linear Support Vector Classification (SVC)
* Logistic Regression (LR)

## Model Evaluation:

The models are evaluated using classification report and confusion matrix on a held-out test set.

## Saving Models and Vectorizer:

The trained models (SVC, LR and BNB) and the TF-IDF Vectorizer are pickled for future use.

**Note:** The script is currently configured to load pre-trained models and vectorizer. Update the script to perform training and evaluation if needed.
