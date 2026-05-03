import streamlit as st
import torch
import joblib
from utils import load_nb_model, predict_nb, load_lstm_model, predict_lstm, load_bert_model, predict_bert, load_vocab
import os

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")

# Load models (cache)
@st.cache_resource
def load_models():
    nb_model, nb_vectorizer = load_nb_model()
    vocab = load_vocab()
    lstm_model = load_lstm_model(len(vocab)) if vocab else None
    bert_model, bert_tokenizer = load_bert_model()
    return nb_model, nb_vectorizer, lstm_model, vocab, bert_model, bert_tokenizer

def predict(model_choice, text):
    nb_model, nb_vectorizer, lstm_model, vocab, bert_model, bert_tokenizer = load_models()
    clean_text = preprocess_text(text)

    if model_choice == "Naive Bayes":
        if nb_model is None:
            return None, 0.0, "NB model not found. Run train_LSTM.py."
        sentiment, prob = predict_nb(nb_model, nb_vectorizer, text)
    elif model_choice == "LSTM":
        if lstm_model is None or vocab is None:
            return None, 0.0, "LSTM model/vocab not found. Run train_LSTM.py."
        sentiment, prob = predict_lstm(lstm_model, vocab, text, max_len=200)
    elif model_choice == "BERT":
        if bert_model is None:
            return None, 0.0, "BERT model not found. Run train_BERT.py."
        sentiment, prob = predict_bert(bert_model, bert_tokenizer, text)
    else:
        return None, 0.0, "Select model."

    return sentiment, prob, None

# UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Classify reviews using **Naive Bayes**, **LSTM**, or **BERT**. Models in `models/`.")

col1, col2 = st.columns([3,1])
with col1:
    text_input = st.text_area("Enter movie review:", height=150, placeholder="This movie was amazing! Best acting ever...")
with col2:
    model_choice = st.selectbox("Model:", ["Naive Bayes", "LSTM", "BERT"])

if st.button("Predict 🎯", type="primary"):
    if text_input:
        sentiment, confidence, error = predict(model_choice, text_input)
        if error:
            st.error(error)
        else:
            # Clean metrics display
            col_left, col_right = st.columns(2)
            color = "normal" if sentiment.lower() == 'positive' else "inverse"
            with col_left:
                st.metric("🎭 Sentiment", sentiment.title())
            with col_right:
                st.metric("📊 Confidence", f"{confidence:.1%}")
            
            # Summary with emoji
            emoji = "✅" if sentiment.lower() == 'positive' else "❌"
            st.markdown(f"**{emoji} Prediction: {sentiment.title()}** with **{confidence:.1%}** confidence!")
            
            if confidence < 0.6:
                st.info("💡 Low confidence - try a more detailed review.")
    else:
        st.warning("Enter a review!")

st.markdown("---")
st.markdown("*Run `streamlit run streamlit_app.py`. Train first: `train_LSTM.py`, `train_BERT.py`. [GitHub](https://github.com/starkindustriestony575-alt/IMDB-Sentiment-Analysis)*")
