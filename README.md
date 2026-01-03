# 🍽️ Aspect-Based Sentiment Analysis (ABSA) – Restaurant Reviews

This project implements an **Aspect-Based Sentiment Analysis (ABSA)** model using **Bidirectional LSTM** to classify restaurant reviews as **positive or negative** for specific aspects such as **FOOD, SERVICE, DELIVERY, and OVERALL experience**.

The goal is to help customers quickly understand **the sentiment of different aspects** in restaurant reviews, so they can make informed decisions.

---

## Features

- Preprocess restaurant reviews: clean text, remove punctuation, lowercase
- Focus on relevant aspects:
  - FOOD#QUALITY, FOOD#STYLE_OPTIONS, FOOD#PRICES
  - SERVICE#GENERAL
  - DELIVERY#GENERAL
  - OVERALL#GENERAL
- Handle class imbalance by upsampling
- Train a Bidirectional LSTM for sentiment classification
- Evaluate model with Accuracy, Precision, Recall, and F1-score

---

## Dataset

- **Training data:** `restaurants_train_single.csv`  
- **Test data:** `restaurants_test_single.csv`  
- Columns used:
  - `sentence`: Customer review text
  - `aspect_category`: Aspect of the review
  - `aspect_term`: Word/phrase related to the aspect
  - `polarity`: Sentiment (positive / negative / neutral)
- Preprocessing steps:
  - Removed neutral reviews
  - Filtered relevant aspects
  - Renamed `RESTAURANT#GENERAL` → `OVERALL#GENERAL`
  - Cleaned text and balanced positive/negative examples

**Dataset links:**  
- [Training CSV](https://raw.githubusercontent.com/Harjandar/absa-restaurant-sentiment/main/data/raw/restaurants_train_single.csv)  
- [Test CSV](https://raw.githubusercontent.com/Harjandar/absa-restaurant-sentiment/main/data/raw/restaurants_test_single.csv)

---

## Model Architecture

- **Embedding Layer:** Convert words to dense vectors (64 dimensions)  
- **Bidirectional LSTM:** Capture context from both directions  
- **Dropout (0.5):** Prevent overfitting  
- **Dense Layer with Sigmoid:** Binary sentiment output  

**Training parameters:**  
- Epochs: 5  
- Batch size: 32  
- Validation split: 20%

---

## Evaluation Metrics (Test Data)

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.7832 |
| Precision | 0.8889 |
| Recall    | 0.8082 |
| F1-score  | 0.8467 |

---

## How to Run

### Option 1: Google Colab
1. Open `ABSA_LSTM.ipynb` in Colab.
2. Ensure datasets are accessible via URLs or uploaded to Colab.
3. Run all cells sequentially:
   - Data preprocessing
   - Tokenization and padding
   - Model training
   - Evaluation

### Option 2: Local Environment
1. Clone the repository:
```bash
git clone <repo-url>
````

2. Install required packages:

```bash
pip install pandas numpy tensorflow scikit-learn
```

3. Open the notebook or script and run all cells.

---

## Future Improvements

* Use pre-trained embeddings (GloVe, FastText) for better performance
* Experiment with transformer-based models (BERT)
* Fine-tune LSTM/BiLSTM hyperparameters
* Extend to multi-class sentiment (positive / neutral / negative)
