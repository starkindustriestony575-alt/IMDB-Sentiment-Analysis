#!/usr/bin/env python3
"""
Train all models quick - fixed.
"""
import os
os.makedirs('models', exist_ok=True)
import pandas as pd
import numpy as np
from utils import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import torch
import joblib
from torch.utils.data import Dataset, DataLoader

def load_and_preprocess():
    df = pd.read_csv('data/IMDB Dataset.csv')
    df['clean_review'] = df['review'].apply(preprocess_text)
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    return df[['clean_review', 'label']]

df = load_and_preprocess()

# Train NB
print("Training Naive Bayes...")
X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
joblib.dump(nb_model, 'models/nb_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("1. NB ready.")

# LSTM fixed
print("2. LSTM...")
df_small = df.sample(n=3000, random_state=42)
vocab = {w: i+1 for i, w in enumerate(set(' '.join(df_small['clean_review']).split()))}
joblib.dump(vocab, 'models/vocab.pkl')

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts.tolist()
        self.labels = labels.values
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = [self.vocab.get(w, 0) for w in self.texts[idx].split()][:200]
        seq = tokens + [0] * (200 - len(tokens))
        return torch.tensor(seq), torch.tensor(self.labels[idx])

X_train, _, y_train, _ = train_test_split(df_small['clean_review'], df_small['label'], train_size=0.8, random_state=42)
train_ds = TextDataset(X_train, y_train, vocab)
loader = DataLoader(train_ds, batch_size=32, shuffle=True)

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

model = LSTMModel(len(vocab))
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCELoss()

model.train()
for epoch in range(2):
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
print("LSTM saved.")

torch.save(model.state_dict(), 'models/lstm_model.pth')

print("✅ Training complete! All models ready.")
print("")
print("Next steps:")
print("  python train_BERT.py     # BERT fine-tuning")
print("  python main.py          # Evaluate + plots")
print("  streamlit run streamlit_app.py  # Demo UI")
