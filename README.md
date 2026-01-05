# Aspect-Based Sentiment Analysis (ABSA) – LSTM Model

## Overview

This project performs **aspect-based sentiment analysis** on home meal / restaurant reviews using an **LSTM model**.
The model predicts **sentiment per aspect** like `FOOD`, `SERVICE`, `DELIVERY_OVERALL` instead of just overall sentiment.

**Example:**

> "Rice was fantastic but service was bad"

* FOOD: Positive
* SERVICE: Negative

---

## Dataset

* **Training:** [restaurants_train_single.csv](https://github.com/Harjandar/absa-restaurant-sentiment/blob/main/data/raw/restaurants_train_single.csv)
* **Test:** [restaurants_test_single.csv](https://github.com/Harjandar/absa-restaurant-sentiment/blob/main/data/raw/restaurants_test_single.csv)

Columns used:

* `sentence` → review text
* `aspect_category` → aspect type (mapped to simplified: FOOD / SERVICE / DELIVERY_OVERALL)
* `polarity` → sentiment label (positive / neutral / negative)

---

## Steps

1. **Data Preprocessing**

   * Filter only relevant aspects (FOOD, SERVICE, DELIVERY_OVERALL)
   * Map aspect categories to simplified names
   * Combine sentence + aspect for model input
   * Clean text (lowercase, remove punctuation)
   * Encode sentiment labels (`positive=2, neutral=1, negative=0`)

2. **Dataset Balancing**

   * Upsample `neutral` and `negative` to match `positive` count

3. **Tokenization**

   * Use Keras `Tokenizer` with `max_words=5000`
   * Pad sequences to `max_len=50`

4. **LSTM Model**

   * Embedding layer → Bidirectional LSTM → Dropout → Dense softmax
   * Loss: `sparse_categorical_crossentropy`
   * Optimizer: Adam

5. **Training**

   * 10 epochs, batch size 32, 10% validation split
   * Early stopping to prevent overfitting

6. **Evaluation (Test Data)**

```
✅ LSTM Test Performance
Accuracy: 0.7412
Precision: 0.6035
Recall: 0.5352
F1-score: 0.5470
```

7. **Aspect-wise Prediction**

```python
review = "rice was fantastic and tasty but service was bad"
aspects = ["FOOD", "SERVICE", "DELIVERY_OVERALL"]

result = predict_aspect_sentiment_detailed(review, aspects)
print(result)
```

**Output Example:**

```python
{
  'FOOD': {'sentiment': 'Positive'},
  'SERVICE': {'sentiment': 'Negative'},
  'DELIVERY_OVERALL': {'sentiment': 'Neutral'}
}
```

---

## Requirements

```text
Python >= 3.8
pandas
numpy
tensorflow >= 2.0
scikit-learn
```

---

## Notes

* Dataset **must be balanced** for better results.
* Can be extended with **pretrained embeddings** or **BERT** for improved accuracy.

