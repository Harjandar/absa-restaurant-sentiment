# Aspect-Based Sentiment Analysis (ABSA) using LSTM

This project implements an **Aspect-Based Sentiment Analysis (ABSA)** system for restaurant reviews using a **Bidirectional LSTM** model. The work follows a **step-by-step experimental methodology** to compare model performance before and after adding **Pakistani-style reviews**.

---

## 📌 Project Objective

The main objectives of this project are:

* To build an LSTM-based ABSA model using an existing (Western-style) dataset
* To evaluate model performance using standard metrics
* To analyze the impact of adding **Pakistani-style restaurant reviews**
* To highlight limitations of LSTM-based ABSA for multi-aspect sentences

---

## 📂 Dataset Description

### 1️⃣ Original Dataset (Western-style)

Source: Mam Nimra’s ABSA restaurant dataset

**Files used:**

* `restaurants_train_single.csv`
* `restaurants_test_single.csv`

**Columns:**

* `sentence` – review text
* `aspect_category` – fine-grained aspect label
* `polarity` – sentiment (`positive`, `negative`, `neutral`)

---

### 2️⃣ Pakistani-Style Reviews Dataset

**File:**

* `pakistani_reviews_150.csv`

**Purpose:**

* Introduce local linguistic patterns and expressions
* Study their effect on ABSA performance

**Structure:**

* Same columns as original dataset
* Only `positive` and `negative` labels are used

---

## 🔁 Aspect Mapping

Original aspect categories are mapped into four high-level aspects:

| Original Aspect Category | Mapped Aspect |
| ------------------------ | ------------- |
| FOOD#QUALITY             | FOOD          |
| FOOD#STYLE_OPTIONS       | FOOD          |
| FOOD#PRICES              | FOOD          |
| SERVICE#GENERAL          | SERVICE       |
| RESTAURANT#GENERAL       | OVERALL       |

This simplifies the ABSA task while preserving semantic meaning.

---

## 🧪 Experimental Methodology

### ✅ Step 1: Load Original Training Dataset

* Load Western-style training data
* Inspect shape and columns

---

### ✅ Step 2: Text Preprocessing

Each review undergoes the following preprocessing steps:

1. Convert text to lowercase
2. Remove punctuation and special characters
3. Remove stopwords (except negations: `not`, `no`, `never`)
4. Apply lemmatization using WordNet

---

### ✅ Step 3: Aspect Mapping

* Filter only required aspect categories
* Map them to FOOD, SERVICE, DELIVERY, OVERALL

---

### ✅ Step 4: Polarity Filtering

* Remove neutral reviews
* Keep only positive and negative samples

---

### ✅ Step 5: Model Input Construction

Each input is constructed as:

```
<sentence> [ASP] <aspect>
```

This conditions the model on the target aspect.

---

### ✅ Step 6: Train LSTM Model (Baseline)

**Model Architecture:**

* Embedding Layer
* Bidirectional LSTM (128 units)
* Dropout (0.3)
* Sigmoid output layer

**Loss Function:** Binary Cross-Entropy
**Optimizer:** Adam
**Metrics:** Accuracy

---

### ✅ Step 7: Evaluate Baseline Model

The model is evaluated on the original test dataset using:

* Accuracy
* Precision
* Recall
* F1-score

These results serve as **baseline performance**.

---

## 🇵🇰 Extension: Adding Pakistani-Style Reviews

### ✅ Step 8: Load Pakistani Reviews

* Verify dataset structure
* Apply same aspect mapping
* Apply same preprocessing steps

---

### ✅ Step 9: Merge Datasets

* Combine original training data with Pakistani reviews
* Shuffle data to avoid ordering bias

---

### ✅ Step 10: Re-balance Data (Optional)

* Balance positive and negative samples per aspect using upsampling

---

### ✅ Step 11: Retrain LSTM Model

* Refit tokenizer on combined dataset
* Train model from scratch

---

### ✅ Step 12: Re-evaluate Model

* Test retrained model on the **same original test set**
* Compare metrics with baseline results

This ensures a **fair comparison**.

---

## 🔍 Aspect-wise Sentiment Prediction

The system predicts sentiment only for **explicitly requested aspects**:

* No neutral class
* No rule-based detection
* No fake aspect inference

Example:

```python
predict_aspect_sentiment(
  "food was good but delivery was slow",
  ['FOOD', 'DELIVERY']
)
```

---

## ⚠️ Known Limitation

* The LSTM model processes the **entire sentence** for each aspect
* In multi-aspect sentences, sentiment words may influence other aspects
* This is a known limitation of non-attention-based sequence models

**This limitation is discussed transparently and motivates future work.**

---

## 🚀 Future Improvements

* Add attention mechanism
* Use transformer-based models (BERT, RoBERTa)
* Perform clause-level sentiment extraction

---

## ✅ Conclusion

This project successfully demonstrates:

* A complete ABSA pipeline using LSTM
* Honest evaluation and comparison
* Impact analysis of domain-specific (Pakistani) reviews
* Clear understanding of model limitations

The implementation is **academically sound** and suitable for undergraduate research.

---

📌 *End of README*
