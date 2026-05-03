import pandas as pd
from src.preprocess import load_and_preprocess
from utils import load_nb_model, load_lstm_model, load_bert_model, predict_nb, predict_bert, build_vocab
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import joblib
import os

print("Loading dataset...")
df = load_and_preprocess()

test_df = df.tail(1000).reset_index(drop=True)
X_test = test_df['clean_review']
y_test = test_df['label'].values

# NB
if os.path.exists('models/nb_model.pkl'):
    nb_model, vectorizer = load_nb_model()
    nb_preds = [predict_nb(nb_model, vectorizer, text)[0] for text in X_test]
    nb_preds_num = [1 if p == 'positive' else 0 for p in nb_preds]
    nb_acc = accuracy_score(y_test, nb_preds_num)
    print(f"NB Accuracy: {nb_acc:.4f}")
else:
    nb_acc = 0

# LSTM
if os.path.exists('models/lstm_model.pth') and os.path.exists('models/vocab.pkl'):
    vocab_size = len(joblib.load('models/vocab.pkl'))
    lstm_model = load_lstm_model(vocab_size)
    device = torch.device('cpu')
    lstm_model.to(device)
    lstm_model.eval()
    vocab = joblib.load('models/vocab.pkl')
    lstm_preds_num = []
    for text in X_test:
        tokens = [vocab.get(word, 0) for word in text.split()]
        seq = torch.tensor([tokens[:200] + [0]*(200 - len(tokens))]).to(device)
        with torch.no_grad():
            prob = lstm_model(seq).squeeze().item()
            pred = 1 if prob > 0.5 else 0
        lstm_preds_num.append(pred)
    lstm_acc = accuracy_score(y_test, lstm_preds_num)
    print(f"LSTM Accuracy: {lstm_acc:.4f}")
else:
    lstm_acc = 0

# BERT
if os.path.exists('models/bert_model'):
    bert_model, bert_tokenizer = load_bert_model()
    bert_preds_num = []
    for text in X_test:
        _, prob = predict_bert(bert_model, bert_tokenizer, text)
        pred = 1 if prob > 0.5 else 0
        bert_preds_num.append(pred)
    bert_acc = accuracy_score(y_test, bert_preds_num)
    print(f"BERT Accuracy: {bert_acc:.4f}")
else:
    bert_acc = 0

# Plot
models = ['Naive Bayes', 'LSTM', 'BERT']
accs = [nb_acc, lstm_acc, bert_acc]
plt.figure(figsize=(10,6))
sns.barplot(x=models, y=accs)
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.ylim(0,1)
for i, acc in enumerate(accs):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
plt.tight_layout()
plt.show()

if nb_acc > 0:
    cm = confusion_matrix(y_test, nb_preds_num)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()

print("Done! Demo: python app.py")
