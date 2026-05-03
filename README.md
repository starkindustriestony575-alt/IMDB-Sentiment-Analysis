# 🎬 IMDB Sentiment Analysis

[![GitHub stars](https://img.shields.io/github/stars/starkindustriestony575-alt/IMDB-Sentiment-Analysis?style=social)](https://github.com/starkindustriestony575-alt/IMDB-Sentiment-Analysis)
[![GitHub issues](https://img.shields.io/github/issues/starkindustriestony575-alt/IMDB-Sentiment-Analysis)](https://github.com/starkindustriestony575-alt/IMDB-Sentiment-Analysis/issues)
[![GitHub license](https://img.shields.io/github/license/starkindustriestony575-alt/IMDB-Sentiment-Analysis)](https://github.com/starkindustriestony575-alt/IMDB-Sentiment-Analysis/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Author**: [Tony Stark](https://github.com/starkindustriestony575-alt)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%96-HuggingFace-purple.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF3C37?style=for-the-badge&logo=streamlit)](https://streamlit.io/)

## ✨ Features

- **3 Advanced Models**: Naive Bayes (85%+), LSTM (87%+), BERT (91%+)
- **Preprocessing**: NLTK stemming, stopwords, regex cleaning
- **Interactive UI**: Streamlit dashboard with **clean metrics display** 🎭📊 (Sentiment, Confidence %, emojis, low-confidence warnings)
- **Evaluation**: Accuracy, F1, confusion matrices, plots
- **Production**: Dockerized, HF Spaces ready
- **Dataset**: 50K IMDB movie reviews

## 📊 Performance

| Model | Accuracy | F1 Score | Train Time |
|-------|----------|----------|------------|
| Naive Bayes | **85.2%** | 0.85 | <1 min |
| LSTM | **87.3%** | 0.87 | 10-15 min |
| **BERT** | **91.2%** | 0.91 | ~5 min |

## 🏗️ Structure

```
.
├── streamlit_app.py     # 🎨 Streamlit UI (updated: clean output with st.metric)
├── main.py             # 📈 Eval & plots
├── utils.py            # 🤖 Model predict/load
├── train_LSTM.py       # NB + LSTM training
├── train_BERT.py       # BERT fine-tune
├── src/preprocess.py   # 🧹 Text cleaning
├── data/IMDB Dataset.csv
├── models/             # 💾 All trained models
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quickstart

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Download data

Place [IMDB Dataset.csv](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) in `data/`

### 3. Train

```bash
python train_LSTM.py  # NB + LSTM
python train_BERT.py  # BERT
```

### 4. Evaluate

```bash
python main.py  # Prints metrics + saves plots/
```

### 5. **Demo** (Now with clean output!)

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) 🚀

**Example:** "This movie sucked!" → Red **NEGATIVE** metric, ❌ summary, confidence %.

## 🐋 Docker

```bash
docker build -t imdb-sentiment .
docker run -p 8501:8501 imdb-sentiment streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## ☁️ Deploy

- Render/Railway: Add `Procfile: web: streamlit run streamlit_app.py --server.port $PORT`
- Vercel (static + API)

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| No models | `train_LSTM.py` / `train_BERT.py` |
| NLTK data | Auto-downloads |
| CUDA error | Falls back to CPU |
| Large files | `.gitignore` excludes |
| Messy output | Fixed in latest streamlit_app.py! |

## 🤝 Contributing

1. Fork repo
2. Create branch
3. PR to `main`

See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

[MIT](LICENSE) © Tony Stark

⭐ **Star if useful!** 🎥
