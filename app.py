import gradio as gr
from utils import load_nb_model, predict_nb, predict_bert, build_vocab, load_lstm_model, predict_lstm, load_bert_model
from src.preprocess import load_and_preprocess, preprocess_text
import torch
import joblib
import pandas as pd
import os

# Load models (use saved vocab for consistency)
if os.path.exists('models/vocab.pkl'):
    vocab = joblib.load('models/vocab.pkl')
    vocab_size = len(vocab)
else:
    df = load_and_preprocess()
    vocab, vocab_size = build_vocab(df)

nb_model, nb_vectorizer = load_nb_model() if os.path.exists('models/nb_model.pkl') else (None, None)
bert_model, bert_tokenizer = load_bert_model() if os.path.exists('models/bert_model') else (None, None)
lstm_model = load_lstm_model(vocab_size) if os.path.exists('models/lstm_model.pth') else None

def predict_sentiment(text, model_choice):
    clean_text = preprocess_text(text)
    if model_choice == "Naive Bayes":
        if nb_model is None:
            return "Model not loaded. Run python train_LSTM.py first."
        sentiment, prob = predict_nb(nb_model, nb_vectorizer, text)
        return f"Sentiment: {sentiment}\nConfidence: {prob:.2%}"
    elif model_choice == "BERT":
        if bert_model is None:
            return "Model not loaded. Run python train_LSTM.py first."
        sentiment, prob = predict_bert(bert_model, bert_tokenizer, text)
        return f"Sentiment: {sentiment}\nConfidence: {prob:.2%}"
    elif model_choice == "LSTM":
        if lstm_model is None:
            return "Model not loaded. Run python train_LSTM.py first."
        device = torch.device('cpu')
        lstm_model.to(device)
        lstm_model.eval()
        tokens = [vocab.get(word, 0) for word in clean_text.split()]
        seq = torch.tensor([tokens[:200] + [0]*(200 - len(tokens))]).to(device)
        with torch.no_grad():
            prob = lstm_model(seq).squeeze().item()
            sentiment = 'positive' if prob > 0.5 else 'negative'
        return f"Sentiment: {sentiment}\nConfidence: {prob:.2%}"
    return "Select a model."

with gr.Blocks(title="IMDB Sentiment Analysis Demo") as demo:
    gr.Markdown("# 🎬 IMDB Movie Review Sentiment Analysis")
    gr.Markdown("Classify reviews as positive/negative using NB, LSTM, or BERT.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Movie Review", placeholder="Enter a movie review...", lines=5)
            model_dropdown = gr.Dropdown(choices=["Naive Bayes", "LSTM", "BERT"], label="Select Model", value="Naive Bayes")
            predict_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Prediction", lines=5)
    
    predict_btn.click(predict_sentiment, inputs=[text_input, model_dropdown], outputs=output)
    
gr.Markdown("### Usage: python train_LSTM.py → python main.py → python app.py")

if __name__ == "__main__":
    demo.launch(share=True)

