#!/usr/bin/env python3
"""
Train all models quick - fixed.
"""
import os
os.makedirs('models', exist_ok=True)
import pandas as pd
import numpy as np
from src.preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
import torch
import joblib
from torch.utils.data import Dataset, DataLoader

df = load_and_preprocess()

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

print("3. BERT ready (run train_bert_quick.py).")

print("Train complete! Run python main.py or app.py")
