# Text_classifier

This project is a machine learning-based spam detection system built using Python. It utilizes Natural Language Processing (NLP) techniques and a **Random Forest Classifier** to classify SMS messages as either **"spam"** or **"ham"** (not spam).

---

## Overview

SMS spam detection is an important problem in communication systems. In this project:

- We preprocess text messages.
- Convert them into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency).
- Train a **Random Forest** model to classify new messages as spam or not.
- Evaluate the model using precision, recall, and F1-score.

---

## Libraries Used

- **pandas**: For data handling and manipulation.
- **nltk**: For natural language processing (e.g., stopword removal).
- **re** and **string**: For text cleaning using regular expressions.
- **scikit-learn (sklearn)**:
  - `TfidfVectorizer` for feature extraction from text.
  - `RandomForestClassifier` for building the prediction model.
  - `train_test_split` for splitting data.
  - `LabelEncoder` for encoding labels.
  - Evaluation metrics: `precision_score`, `recall_score`, and `f1_score`.

---

## Model Used

### Random Forest Classifier

An ensemble learning method that creates multiple decision trees and merges them to get more accurate and stable predictions.

**Advantages:**
- Handles both classification and regression tasks.
- Reduces overfitting.
- Works well with both categorical and numerical features.

---

## Steps Performed

1. **Load and Clean the Dataset**
   - Dataset: `spam.csv` (contains SMS messages labeled as spam or ham).
   - Dropped unnecessary columns and renamed remaining ones.

2. **Text Preprocessing**
   - Removed punctuation and stopwords.
   - Tokenized the messages.

3. **Feature Extraction**
   - Applied **TF-IDF Vectorization** to convert text to numerical features.

4. **Label Encoding**
   - Encoded labels ('ham' and 'spam') to numerical values.

5. **Model Training**
   - Split the data into training and testing sets (80/20).
   - Trained a **Random Forest** classifier.

6. **Evaluation**
   - Evaluated the model using precision, recall, and F1-score.

7. **Prediction**
   - Tested the model with a custom message like `Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.` to predict whether it’s spam or not.

---
# output:
Precision: 1.0 / Recall: 0.865 / F1-Score: 0.927
Prediction for Congratulations! You've won a $1000 Walmart gift card. Click here to claim now. : spam

## How to Run

1. Install required libraries:
   ```bash
   pip install pandas scikit-learn nltk
