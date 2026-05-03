import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

os.makedirs('models', exist_ok=True)

print("Quick BERT (2k samples)...")

df = pd.read_csv('data/IMDB Dataset.csv')
df['label'] = (df['sentiment'] == 'positive').astype(int)
df_sample = df.sample(n=2000, random_state=42)
train_df, test_df = train_test_split(df_sample, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['review'], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./temp',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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

model.save_pretrained('models/bert_model')
tokenizer.save_pretrained('models/bert_tokenizer')

print("BERT ready! streamlit run streamlit_app.py")
