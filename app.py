import streamlit as st
import torch
import joblib
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset

from utils import (
    load_nb_model,
    predict_nb,
    load_lstm_model,
    predict_lstm,
    load_bert_model,
    predict_bert,
    load_vocab,
    preprocess_text,
    LSTMModel,
    build_vocab,
)

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")


# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource
def load_models():
    nb_model, nb_vectorizer = load_nb_model()
    vocab = load_vocab()
    lstm_model = load_lstm_model(len(vocab)) if vocab else None
    bert_model, bert_tokenizer = load_bert_model()
    return nb_model, nb_vectorizer, lstm_model, vocab, bert_model, bert_tokenizer


def predict(model_choice, text):
    nb_model, nb_vectorizer, lstm_model, vocab, bert_model, bert_tokenizer = load_models()

    if model_choice == "Naive Bayes":
        if nb_model is None:
            return None, 0.0, "🚀 NB model not found. Run training locally first."
        sentiment, prob = predict_nb(nb_model, nb_vectorizer, text)

    elif model_choice == "LSTM":
        if lstm_model is None or vocab is None:
            return None, 0.0, "🚀 LSTM model/vocab not found. Run training locally first."
        sentiment, prob = predict_lstm(lstm_model, vocab, text, max_len=200)

    elif model_choice == "BERT":
        if bert_model is None or bert_tokenizer is None:
            return None, 0.0, "🚀 BERT model not found. Run training locally first."
        sentiment, prob = predict_bert(bert_model, bert_tokenizer, text)

    else:
        return None, 0.0, "Select model."

    return sentiment, prob, None


# -----------------------------
# Training (run on button)
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, vocab: dict[str, int], max_len: int):
        self.texts = texts.tolist()
        self.labels = labels.values
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        tokens = [self.vocab.get(w, 0) for w in self.texts[idx].split()][: self.max_len]
        seq = tokens + [0] * (self.max_len - len(tokens))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


def train_lstm_and_nb(
    data_path: str = "data/IMDB Dataset.csv",
    nb_vectorizer_max_features: int = 5000,
    lstm_samples: int = 3000,
    max_len: int = 200,
    lstm_epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(data_path)
    df["clean_review"] = df["review"].apply(preprocess_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # -----------------
    # Naive Bayes
    # -----------------
    X_train, _, y_train, _ = train_test_split(
        df["clean_review"],
        df["label"],
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=nb_vectorizer_max_features)
    X_train_vec = vectorizer.fit_transform(X_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    joblib.dump(nb_model, "models/nb_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # -----------------
    # LSTM
    # -----------------
    df_small = df.sample(n=lstm_samples, random_state=seed) if len(df) > lstm_samples else df
    vocab = build_vocab(df_small)
    joblib.dump(vocab, "models/vocab.pkl")

    X_train, _, y_train, _ = train_test_split(
        df_small["clean_review"],
        df_small["label"],
        train_size=0.8,
        random_state=seed,
        stratify=df_small["label"],
    )

    train_ds = TextDataset(X_train, y_train, vocab=vocab, max_len=max_len)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(len(vocab)).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    model.train()
    for _ in range(lstm_epochs):
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "models/lstm_model.pth")


def train_bert(
    data_path: str = "data/IMDB Dataset.csv",
    samples: int = 2000,
    test_size: float = 0.2,
    max_length: int = 128,
    num_train_epochs: int = 1,
    batch_size: int = 4,
    seed: int = 42,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(data_path)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    df_sample = df.sample(n=samples, random_state=seed) if len(df) > samples else df
    train_df, test_df = train_test_split(df_sample, test_size=test_size, random_state=seed, stratify=df_sample["label"])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["review"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = HFDataset.from_pandas(train_df).map(tokenize_function, batched=True)
    test_dataset = HFDataset.from_pandas(test_df).map(tokenize_function, batched=True)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./temp",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    model.save_pretrained("models/bert_model")
    tokenizer.save_pretrained("models/bert_tokenizer")


# -----------------------------
# UI
# -----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown(
    "**Classify reviews** using available models (Naive Bayes, LSTM, BERT). "
    "*Train locally via the buttons below. Cloud needs files under `models/`. *"
)

# Model train controls
with st.expander("⚙️ Train models locally (updates `models/`)", expanded=False):
    st.caption("Training can take a while and may require downloading pretrained weights for BERT the first time.")

    train_col_1, train_col_2 = st.columns(2)

    with train_col_1:
        train_lstm_btn = st.button("Train LSTM (and NB) 🧠", type="primary")

    with train_col_2:
        train_bert_btn = st.button("Train BERT ⚡")

    data_exists = os.path.exists("data/IMDB Dataset.csv")

    if not data_exists:
        st.warning("Dataset not found at `data/IMDB Dataset.csv`. Add it or adjust paths in the code.")

    if train_lstm_btn:
        if not data_exists:
            st.error("Dataset not found; cannot train.")
        else:
            try:
                with st.spinner("Training Naive Bayes + LSTM..."):
                    train_lstm_and_nb()
                load_models.clear()
                st.success("✅ LSTM (and NB) trained and saved to `models/`.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    if train_bert_btn:
        if not data_exists:
            st.error("Dataset not found; cannot train.")
        else:
            try:
                with st.spinner("Training BERT..."):
                    train_bert()
                load_models.clear()
                st.success("✅ BERT trained and saved to `models/`.")
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

col1, col2 = st.columns([3, 1])
with col1:
    text_input = st.text_area(
        "Enter movie review:",
        height=150,
        placeholder="This movie was amazing! Best acting ever...",
    )

with col2:
    # Dynamic available models
    available_models = []

    nb_model, nb_vec = load_nb_model()
    vocab = load_vocab()
    lstm = load_lstm_model(len(vocab)) if vocab else None
    bert_model, bert_tok = load_bert_model()

    if nb_vec:
        available_models.append("Naive Bayes")
    if lstm:
        available_models.append("LSTM")
    if bert_tok:
        available_models.append("BERT")

    if not available_models:
        available_models = ["No models loaded"]

    model_choice = st.selectbox("Model:", available_models)

if st.button("Predict 🎯", type="primary"):
    if text_input:
        sentiment, confidence, error = predict(model_choice, text_input)
        if error:
            st.error(error)
        else:
            col_left, col_right = st.columns(2)
            if sentiment.lower() == "positive":
                color = "normal"
            else:
                color = "inverse"

            with col_left:
                st.metric("🎭 Sentiment", sentiment.title())
            with col_right:
                st.metric("📊 Confidence", f"{confidence:.1%}")

            emoji = "✅" if sentiment.lower() == "positive" else "❌"
            st.markdown(f"**{emoji} Prediction: {sentiment.title()}** with **{confidence:.1%}** confidence!")

            if confidence < 0.6:
                st.info("💡 Low confidence - try a more detailed review.")
    else:
        st.warning("Enter a review!")

st.markdown("---")
st.markdown(
    "*Run `streamlit run app.py`. Train first: use the buttons above. "
    "[GitHub](https://github.com/starkindustriestony575-alt/IMDB-Sentiment-Analysis)*"
)
