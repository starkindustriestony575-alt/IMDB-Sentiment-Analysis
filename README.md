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
- **3 Advanced Models**: Naive Bayes, LSTM, BERT
- **Clean Streamlit UI**: Metrics display (Sentiment/Confidence), emojis, cloud-safe
- **Preprocessing**: Stemming, stopwords in utils.py
- **Evaluation**: main.py accuracy/plots
- **Production Ready**: Docker, Streamlit Cloud deploy

## 📊 Performance (example)
| Model | Accuracy |
|-------|----------|
| Naive Bayes | ~85% |
| LSTM | ~87% |
| BERT | ~91% |

## 🏗️ Structure
```
.
├── streamlit_app.py     # 🎨 Main Streamlit UI (clean output)
├── utils.py             # 🤖 All models/preprocessing
├── train_LSTM.py        # Train NB+LSTM
├── train_BERT.py        # Train BERT
├── main.py              # Evaluate + plots
├── data/                # IMDB Dataset.csv
├── models/              # Trained models (*.pkl/.pth)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quickstart
### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Data
Download [IMDB Dataset.csv](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) to `data/`

### 3. Train
```bash
python train_LSTM.py   # NB + LSTM → models/
python train_BERT.py   # BERT → models/
```

### 4. Demo
```bash
streamlit run streamlit_app.py
```
http://localhost:8501 🎯

### 5. Evaluate
```bash
python main.py  # Metrics + plots
```

## ☁️ Deploy Streamlit Cloud
1. Push to GitHub
2. Streamlit Cloud → New app → Select repo/branch
3. Main file: `streamlit_app.py`
4. Works without models (shows instructions)

## 🐋 Docker
```bash
docker build -t imdb-sentiment .
docker run -p 8501:8501 -v $(pwd)/models:/app/models imdb-sentiment streamlit run streamlit_app.py
```

## 🔧 Troubleshooting
| Issue | Solution |
|-------|----------|
| No models | Run train scripts, check models/ |
| Cloud error | Models not needed (app handles gracefully) |
| NLTK | Auto-downloads |
| CUDA | CPU fallback |

## 📄 License
[MIT](LICENSE)

⭐ **Star if useful!** Deploy live: [Streamlit template](https://streamlit.io/cloud)
