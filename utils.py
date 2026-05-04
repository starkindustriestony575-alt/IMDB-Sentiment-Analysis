import re
import os
import nltk
import joblib
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pathlib import Path
from nltk.stem import PorterStemmer

BASE_DIR = Path(__file__).parent.resolve()
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass

download_nltk_data()

try:
    stop_words = set(nltk.corpus.stopwords.words("english"))
except:
    stop_words = set()

stemmer = PorterStemmer()

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text).lower())
    tokens = [stemmer.stem(word) for word in text.split() 
              if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)


def load_nb_model():
    nb_path = os.path.join(MODELS_DIR, "nb_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(nb_path) and os.path.exists(tfidf_path):
        return joblib.load(nb_path), joblib.load(tfidf_path)
    return None, None


def predict_nb(model, vectorizer, text):
    clean_text = preprocess_text(text)
    vec = vectorizer.transform([clean_text])
    prob = model.predict_proba(vec)[0]
    label = model.predict(vec)[0]
    return 'positive' if label == 1 else 'negative', prob[1]


def load_vocab():
    vocab_path = os.path.join(MODELS_DIR, "vocab.pkl")
    if os.path.exists(vocab_path):
        try:
            return joblib.load(vocab_path)
        except:
            pass
    return None


def load_lstm_model(vocab_size):
    try:
        vocab = load_vocab()
        if vocab:
            vocab_size = len(vocab)
        model = LSTMModel(vocab_size)
        lstm_path = os.path.join(MODELS_DIR, "lstm_model.pth")
        if os.path.exists(lstm_path):
            model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
        model.eval()
        return model
    except:
        return None


class LSTMModel(torch.nn.Module):
    def __init__(self, vsize):
        super().__init__()
        self.embedding = torch.nn.Embedding(vsize + 1, 64)
        self.lstm = torch.nn.LSTM(64, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h.squeeze(0))
        return self.sigmoid(out)


def predict_lstm(model, vocab, text, max_len=200):
    clean_text = preprocess_text(text)
    tokens = [vocab.get(word, 0) for word in clean_text.split()]
    seq = torch.tensor([tokens[:max_len] + [0]*(max_len - len(tokens))])
    with torch.no_grad():
        prob = model(seq).squeeze().item()
    return ('positive' if prob > 0.5 else 'negative', prob)


def load_bert_model():
    try:
        model_path = os.path.join(MODELS_DIR, "bert_model")
        tokenizer_path = os.path.join(MODELS_DIR, "bert_tokenizer")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    except:
        return None, None


def predict_bert(model, tokenizer, text, device='cpu'):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
    return 'positive' if label == 1 else 'negative', probs[0][1].item()


def build_vocab(df):
    all_words = " ".join(df["clean_review"]).split()
    vocab = {word: i + 1 for i, word in enumerate(set(all_words))}
    return vocab