<!--
Presentation-style README meant to be converted to PDF.
Recommended conversion:
- If you have pandoc: pandoc README.md -o README.pdf
- Otherwise: open this file in VSCode/Browser and use Print -> Save as PDF
-->

# 🎬 IMDB Sentiment Analysis — Project Presentation

---

## Slide 1 — Title

**IMDB Movie Review Sentiment Analysis**  
Classify reviews as: **Positive / Negative** using:

- **Naive Bayes (NB)** (TF‑IDF)
- **LSTM** (token vocab from training subset)
- **BERT** (fine‑tuning)

---

## Slide 2 — What you get

- A Streamlit app for inference
- One script to train **all models** end‑to‑end:
  - `train_all_models.py`
- Model artifacts stored in:
  - `models/`

---

## Slide 3 — Problem Statement

Given an IMDb movie review text, predict the sentiment label:

- **positive**
- **negative**

---

## Slide 4 — Dataset

**File:** `data/IMDB Dataset.csv`  
Each row contains:

- `review` (text)
- `sentiment` (string label)

During training we create:

- `clean_review` = preprocessed text
- `label` = `1` if sentiment == `positive`, else `0`

---

## Slide 5 — Preprocessing (Shared)

Implemented in `utils.py`:

- Lowercase
- Remove non-letters
- Tokenize
- Remove **NLTK stopwords**
- Apply **Porter stemming**

Output is a single cleaned string used by:

- Naive Bayes (TF‑IDF)
- LSTM (vocab + integer sequences)

---

## Slide 6 — Model 1: Naive Bayes (TF‑IDF)

**Training steps:**

1. Create TF‑IDF features from `clean_review`
2. Fit `MultinomialNB` classifier
3. Save:
   - `models/nb_model.pkl`
   - `models/tfidf_vectorizer.pkl`

**Inference logic:**

- `utils.predict_nb()` → returns `(sentiment, confidence_prob)`

---

## Slide 7 — Model 2: LSTM Sentiment Classifier

**Training strategy:**

- Train on a subset for speed (default: `lstm_samples=3000`)
- Build vocabulary from training subset:
  - `models/vocab.pkl`
- Convert reviews to padded integer sequences of fixed length:
  - `max_len=200`
- Train a small PyTorch LSTM:
  - Embedding → LSTM → Linear → Sigmoid

**Artifacts saved:**

- `models/lstm_model.pth`

**Inference:**

- `utils.predict_lstm()` → returns `(sentiment, prob)`

---

## Slide 8 — Model 3: BERT Fine‑Tuning

**Base model:** `bert-base-uncased`  
**Training strategy:**

- Sample subset for speed (default: `samples=2000`)
- Tokenize with:
  - padding to `max_length=128`
- Fine‑tune `BertForSequenceClassification(num_labels=2)` for 1 epoch by default

**Artifacts saved:**

- `models/bert_model/`
- `models/bert_tokenizer/`

**Inference:**

- `utils.predict_bert()` → returns `(sentiment, confidence_prob)`

---

## Slide 9 — Training Pipeline (Unified)

Use:

- `train_all_models.py`

What it does:

1. Train NB + LSTM (saves `models/nb_model.pkl`, `models/tfidf_vectorizer.pkl`, `models/vocab.pkl`, `models/lstm_model.pth`)
2. Train BERT (saves `models/bert_model/`, `models/bert_tokenizer/`)
3. Print completion messages

---

## Slide 10 — Streamlit App (Inference + Optional Retraining)

**Main app:** `app.py`

UI features:

- Input text box
- Model dropdown (auto-detects loaded artifacts)
- Predict button
- Metrics:
  - Sentiment
  - Confidence %
- Optional local training:
  - “Train LSTM (and NB) 🧠”
  - “Train BERT ⚡”
- After training, cached loader is cleared:
  - `load_models.clear()`
  - then `st.rerun()` updates the model list

---

## Slide 11 — Repository Structure

```text
.
app.py                 # Streamlit UI (inference + training buttons)
utils.py               # preprocessing + model loaders + predictors
main.py                # evaluation/plots (optional)
train_all_models.py   # train NB+LSTM and BERT end-to-end
data/                 # dataset file
models/               # trained model artifacts
requirements.txt
README.md
```

---

## Slide 12 — How to Run (Local)

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Ensure dataset exists

- Place `IMDB Dataset.csv` into `data/`

### 3) Train all models (recommended)

```bash
python train_all_models.py
```

### 4) Start the app

```bash
streamlit run app.py
```

---

## Slide 13 — How to Use (Inference)

1. Type a review in the text box
2. Choose model:
   - Naive Bayes / LSTM / BERT (only shown if artifacts exist)
3. Click **Predict 🎯**
4. Read:
   - **Sentiment**
   - **Confidence**

---

## Slide 14 — Quick Example

Input:
> “This movie was amazing! Best acting ever…”

Expected output (example):

- **Naive Bayes:** positive (~0.68)
- **LSTM:** positive (~0.52)
- **BERT:** positive (~0.98)

(Actual values vary slightly depending on training subset sizes.)

---

## Slide 15 — Results & Notes

- NB + TF‑IDF provides a fast baseline.
- LSTM adds sequence modeling using a learned embedding + recurrent layer.
- BERT usually performs best with strong contextual representations, at the cost of training time.

---

## Slide 16 — Common Issues / Troubleshooting

### No model available in dropdown

- Train using `python train_all_models.py`
- Or use training buttons in the Streamlit app

### BERT download is slow

- First run downloads model weights from Hugging Face

### NLTK stopwords error

- `utils.py` runs `nltk.download('stopwords', quiet=True)` automatically

---

## Slide 17 — License

Project licensed under **MIT** (see `LICENSE`).

---

## Slide 18 — End

✅ App + training pipeline + predictors are integrated and runnable locally.
