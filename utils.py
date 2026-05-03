import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text).lower())
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def load_nb_model():
    """Load NB model and TF-IDF vectorizer."""
    if os.path.exists('models/nb_model.pkl') and os.path.exists('models/tfidf_vectorizer.pkl'):
        model = joblib.load('models/nb_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    print("NB models not found.")
    return None, None

def predict_nb(model, vectorizer, text):
    """Predict with NB."""
    clean_text = preprocess_text(text)
    vec = vectorizer.transform([clean_text])
    prob = model.predict_proba(vec)[0]
    label = model.predict(vec)[0]
    return 'positive' if label == 1 else 'negative', prob[1]

def load_vocab():
    """Load vocab or return None."""
    try:
        if os.path.exists('models/vocab.pkl'):
            return joblib.load('models/vocab.pkl')
    except FileNotFoundError:
        pass
    return None

def load_lstm_model(vocab_size):
    """Load LSTM model - keep arg for app.py compat, but prefer saved vocab."""
    try:
        vocab = load_vocab()
        if vocab:
            vocab_size = len(vocab)
        model = LSTMModel(vocab_size)
        if os.path.exists('models/lstm_model.pth'):
            checkpoint = torch.load('models/lstm_model.pth', map_location='cpu')
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except (FileNotFoundError, Exception):
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
    """Predict with LSTM."""
    clean_text = preprocess_text(text)
    tokens = [vocab.get(word, 0) for word in clean_text.split()]
    seq = torch.tensor([tokens[:max_len] + [0]*(max_len - len(tokens))])
    with torch.no_grad():
        prob = model(seq).squeeze().item()
    label = 'positive' if prob > 0.5 else 'negative'
    return label, prob

def load_bert_model():
    """Load BERT model and tokenizer."""
    try:
        model = BertForSequenceClassification.from_pretrained('models/bert_model')
        tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
        return model, tokenizer
    except (FileNotFoundError, Exception):
        print("BERT models not found.")
        return None, None

def predict_bert(model, tokenizer, text, device='cpu'):
    """Predict with BERT."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
    return 'positive' if label == 1 else 'negative', probs[0][1].item()

def build_vocab(df):
    """Build vocab from df."""
    all_words = ' '.join(df['clean_review']).split()
    vocab = {word: i+1 for i, word in enumerate(set(all_words))}
