import streamlit as st
import torch
import joblib
import os

from utils import (
    load_nb_model, predict_nb,
    load_lstm_model, predict_lstm,
    load_bert_model, predict_bert,
    load_vocab, preprocess_text,
    LSTMModel, build_vocab, MODELS_DIR
)

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="wide")


@st.cache_resource
def load_models():
    nb_model, nb_vectorizer = load_nb_model()
    vocab = load_vocab()
    lstm_model = load_lstm_model(len(vocab) if vocab else 10000)
    bert_model, bert_tokenizer = load_bert_model()
    return nb_model, nb_vectorizer, lstm_model, vocab, bert_model, bert_tokenizer


def predict(model_choice, text):
    nb_model, nb_vectorizer, lstm_model, vocab, bert_model, bert_tokenizer = load_models()

    if model_choice == "Naive Bayes":
        if nb_model is None:
            return None, 0.0, "Naive Bayes model not found. Please train it first."
        sentiment, prob = predict_nb(nb_model, nb_vectorizer, text)

    elif model_choice == "LSTM":
        if lstm_model is None or vocab is None:
            return None, 0.0, "LSTM model not found. Please train it first."
        sentiment, prob = predict_lstm(lstm_model, vocab, text)

    elif model_choice == "BERT":
        if bert_model is None:
            return None, 0.0, "BERT model not found. Training BERT on free tier is slow."
        sentiment, prob = predict_bert(bert_model, bert_tokenizer, text)

    return sentiment, prob, None


def train_lstm_and_nb():
    # Same as before but we keep it light
    from train_all_models import train_nb_and_lstm
    train_nb_and_lstm(lstm_samples=2000, lstm_epochs=2)
    st.success("✅ Naive Bayes & LSTM trained successfully!")


def train_bert_light():
    from train_all_models import train_bert
    with st.spinner("Training BERT (this may take 10-20 minutes on free tier)..."):
        train_bert(samples=600, num_train_epochs=1, batch_size=8)
    st.success("✅ BERT trained!")


st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("**Fast & Light models recommended for Streamlit Cloud**")

with st.expander("⚙️ Train Models (Recommended: NB + LSTM first)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Naive Bayes + LSTM 🧠", type="primary"):
            try:
                train_lstm_and_nb()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if st.button("Train BERT (Slow)", type="secondary"):
            try:
                train_bert_light()
                st.rerun()
            except Exception as e:
                st.error(f"BERT training failed: {e}")

available_models = []
nb_m, nb_v = load_nb_model()
if nb_m: available_models.append("Naive Bayes")
if load_lstm_model(10000): available_models.append("LSTM")
if load_bert_model()[0]: available_models.append("BERT")

if not available_models:
    available_models = ["Naive Bayes"]  # fallback

model_choice = st.selectbox("Select Model:", available_models)

text_input = st.text_area("Enter your movie review:", height=150, 
                         placeholder="This movie was fantastic! Great acting and story...")

if st.button("Predict 🎯", type="primary"):
    if text_input.strip():
        with st.spinner("Predicting..."):
            sentiment, confidence, error = predict(model_choice, text_input)
        
        if error:
            st.error(error)
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                st.metric("Sentiment", sentiment.title())
            with col_right:
                st.metric("Confidence", f"{confidence:.1%}")
            
            emoji = "✅" if sentiment == "positive" else "❌"
            st.success(f"{emoji} **{sentiment.title()}** with **{confidence:.1%}** confidence")
    else:
        st.warning("Please enter a review!")

st.caption("💡 Tip: Train **Naive Bayes + LSTM** first — they are fast and work well on free tier.")