import os
import joblib
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset

from utils import preprocess_text, load_vocab, build_vocab, LSTMModel


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


def train_nb_and_lstm(
    data_path: str = "data/IMDB Dataset.csv",
    nb_vectorizer_max_features: int = 5000,
    lstm_samples: int = 3000,
    max_len: int = 200,
    lstm_epochs: int = 2,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(data_path)
    df["clean_review"] = df["review"].apply(preprocess_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # Train Naive Bayes on full dataset (clean_review)
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

    # Train LSTM on a subset
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
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(data_path)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    df_sample = df.sample(n=samples, random_state=seed) if len(df) > samples else df
    train_df, test_df = train_test_split(
        df_sample,
        test_size=test_size,
        random_state=seed,
        stratify=df_sample["label"],
    )

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
        logging_steps=10,
        evaluation_strategy="epoch" if num_train_epochs >= 1 else "no",
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


def main() -> None:
    print("Training Naive Bayes + LSTM...")
    train_nb_and_lstm()
    print("✅ Naive Bayes + LSTM ready.")

    print("Training BERT...")
    train_bert()
    print("✅ BERT ready.")

    print("All models trained. Start app: streamlit run app.py")


if __name__ == "__main__":
    main()
