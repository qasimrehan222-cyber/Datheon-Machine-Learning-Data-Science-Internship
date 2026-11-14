# NLP Sentiment Classification on Customer Reviews (TF-IDF)

An end-to-end Natural Language Processing (NLP) text classification pipeline built in Python (Google Colab) that ingests raw customer/movie review records, cleans and normalizes text, extracts TF-IDF numerical features, and classifies review polarity using **Multinomial Naive Bayes** and **Logistic Regression**.

## Project Overview

* **Task:** Binary / Multi-class Sentiment Polarity Classification

* **Dataset:** `sentiment_reviews.csv`

* **Feature Extraction:** TF-IDF Vectorization (`max_features=5000`)

* **Models Evaluated:** Multinomial Naive Bayes (`MultinomialNB`) & Logistic Regression (`LogisticRegression`)

* **Deliverable:** Executable Jupyter Notebook (`.ipynb`) with accuracy benchmarks and an interactive `predict_sentiment()` test helper

## Pipeline Architecture

1. **Data Ingestion & Inspection**

   * Uploads `sentiment_reviews.csv` into Google Colab and automatically detects the review text column and target sentiment label column.

   * Removes missing (`NaN`) or empty entries.

2. **Text Preprocessing & Normalization (`NLTK` + `Regex`)**

   * Strips HTML tags (e.g., `<br />`), URLs, punctuation, and numeric characters.

   * Converts all text to lowercase.

   * Tokenizes sentences into individual words using `nltk.word_tokenize`.

   * Filters out English stopwords and reduces words to their base dictionary forms using `WordNetLemmatizer`.

3. **Feature Engineering (`TF-IDF`)**

   * Splits cleaned data into an 80% training set and 20% testing set.

   * Transforms text documents into a sparse numerical matrix using `TfidfVectorizer(max_features=5000)`.

4. **Model Training & Benchmarking**

   * Trains both **Multinomial Naive Bayes** and **Logistic Regression** on the TF-IDF training matrix.

   * Evaluates test performance using **Accuracy Score**, **Classification Reports** (Precision, Recall, F1-Score), and a **Confusion Matrix**.

   * Automatically selects the top-performing classifier for inference.

5. **Interactive Inference (`predict_sentiment()`)**

   * Applies the exact same preprocessing and TF-IDF transformation to unseen custom user reviews.

   * Outputs predicted sentiment polarity alongside a model confidence percentage.

## Tech Stack & Dependencies

* **Python 3.x** (Google Colab environment)

* **Data Manipulation:** `pandas`, `numpy`

* **NLP Preprocessing:** `nltk` (`punkt`, `stopwords`, `wordnet`), `re`

* **Machine Learning:** `scikit-learn` (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `train_test_split`, `metrics`)

* **Visualization:** `matplotlib`

## How to Run in Google Colab

1. Open a new notebook in [Google Colab](https://colab.research.google.com/).

2. Copy and run **Steps 1 through 6** sequentially from top to bottom.

3. When prompted in **Step 2**, upload your `sentiment_reviews.csv` dataset.

4. Review the accuracy benchmarks and classification reports generated in **Step 5**.

5. Test live custom phrases using the interactive prompt in **Step 6** (type `quit` when finished to cleanly terminate the loop).

6. Export your completed notebook via **File → Download → Download .ipynb**.

## Example Usage (`predict_sentiment`)

```
predict_sentiment("This product exceeded all my expectations! Super fast delivery and great build quality.")
# Output: Sentiment: positive (Confidence: 89.4%)

predict_sentiment("Terrible experience. It broke after two days and customer support refused to help.")
# Output: Sentiment: negative (Confidence: 92.1%)

```