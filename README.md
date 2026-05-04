# 🎬 IMDB Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Classify IMDb movie reviews as positive or negative using three different models:

- Naive Bayes with TF-IDF features
- LSTM neural network
- BERT (fine-tuned bert-base-uncased)

## Live Demo

GitHub - (<https://github.com/starkindustriestony575-alt/IMDB-Sentiment-Analysis>)🎯

Streamlit Cloud - (<https://imdb-sentiment-analysis-deexkp6eos2hnwmxz7npxh.streamlit.app/>) 🎯

## Features

- Streamlit web application for interactive sentiment prediction
- Unified training script to train all models
- Preprocessed model artifacts stored in `models/` directory
- Shared text preprocessing pipeline (lowercase, cleaning, stopword removal, stemming)

## Models

1. **Naive Bayes (TF-IDF)**: Fast baseline model using TF-IDF vectorization
2. **LSTM**: Recurrent neural network for sequence modeling
3. **BERT**: Transformer-based model for contextual understanding (usually highest accuracy)

## Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the dataset file `IMDB Dataset.csv` in the `data/` directory

## Usage

### Train all models (recommended first step)

```bash
python train_LSTM.py   # NB + LSTM → models/
python train_BERT.py   # BERT → models/
```

### Launch the Streamlit app

```bash
streamlit run app.py
```
   Local URL: http://localhost:8501

In the web interface:

1. Enter a movie review in the text box
2. Select a model from the dropdown (available models appear after training)
3. Click "Predict" to see sentiment and confidence score

### Optional: Train individual models via the Streamlit app

Use the training buttons in the UI to train LSTM/NB or BERT models locally.

## Repository Structure

```text
.
├── app.py                 # Streamlit UI (inference + training controls)
├── utils.py               # Preprocessing, model loading, prediction functions
├── main.py                # Evaluation and plotting utilities (optional)
├── train_all_models.py    # Script to train all models end-to-end
├── data/                  # Contains IMDB Dataset.csv
├── models/                # Saved model artifacts (created during training)
├── requirements.txt       # Python dependencies
├── README.md
└── LICENSE
```

## How It Works

**Preprocessing** (shared across models):

- Text converted to lowercase
- Non-letter characters removed
- Tokenized and cleaned (stopwords removed, Porter stemming applied)

**Model-specific processing**:

- **Naive Bayes**: TF-IDF vectorization of cleaned text
- **LSTM**: Vocabulary built from training data, text converted to padded sequences
- **BERT**: Hugging Face tokenizer with padding/truncation to max length

## Model Performance Visualization

The following figures illustrate key aspects of model performance:

![Model Comparison](models/Figure_1.png)
*Figure 1: Model performance comparison*

![Training Metrics](models/Figure_2.png)
*Figure 2: Training metrics and loss curves*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Notes

- First-time BERT usage downloads model weights from Hugging Face (may be slow initially)
- NLTK stopwords are downloaded automatically on first run
- Training uses subsets by default for faster execution (adjust parameters in training scripts)
