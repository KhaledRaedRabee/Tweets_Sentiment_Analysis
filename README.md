# Sentiment Analysis on Tweets using Multinomial Naive Bayes

This project implements a sentiment analysis model that classifies tweets as either **positive** or **negative** using Natural Language Processing (NLP) techniques and the **Multinomial Naive Bayes** algorithm.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Modeling](#modeling)
- [Evaluation](#evaluation)


---

## 📝 Overview

This project focuses on classifying the sentiment of tweets using various NLP techniques. The dataset consists of tweets labeled as either **0 (negative)** or **1 (positive)**.

### Key Components:
- **Data Preprocessing**: Clean and tokenize text, remove stopwords, apply stemming.
- **Feature Extraction**: Transform text data using TF-IDF.
- **Modeling**: Train a Multinomial Naive Bayes classifier.
- **Evaluation**: Measure model accuracy and performance using classification metrics.

---

## 🧹 Data Preprocessing

Data preprocessing ensures that the input text is clean and standardized. The steps include:

- **Cleaning**: Removing special characters, URLs, and Twitter mentions.
- **Tokenization**: Splitting text into individual words (tokens).
- **Stopword Removal**: Eliminating common words like “the”, “is”, etc.
- **Stemming**: Reducing words to their root forms (e.g., “running” → “run”).

---

## 📊 Feature Extraction

The **TF-IDF (Term Frequency-Inverse Document Frequency)** method is used to convert textual data into numerical vectors. This method helps highlight important words in the dataset while down-weighting common terms.

---

## 🧠 Modeling

A **Multinomial Naive Bayes** classifier is used to predict sentiment. This algorithm is well-suited for text classification tasks that involve frequency-based features like TF-IDF.

---

## 📈 Evaluation

Model performance is assessed using the following metrics:

- **Accuracy**: Overall percentage of correct predictions.
- **Confusion Matrix**: Helps visualize the number of true/false positives and negatives.
- **Classification Report**: Includes precision, recall, and F1-score for each class.

---


