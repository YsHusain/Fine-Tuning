# 📈 Financial Twitter Sentiment Analyzer
**Fine-Tuning DistilBERT for Financial News Sentiment Classification**
---

## 📌 Project Overview

This project fine-tunes a pre-trained transformer model to classify sentiment in financial Twitter posts and news headlines.

Financial social media content contains valuable market signals, but sentiment analysis in this domain is challenging due to:
- Domain-specific terminology (upgrades, downgrades, price targets)
- Stock ticker symbols ($AAPL, $TSLA)
- Subtle sentiment indicators in analyst language

The goal of this project is to build a sentiment classifier that accurately identifies:
- 🟢 **Bullish** — Positive market sentiment
- 🔴 **Bearish** — Negative market sentiment  
- 🟡 **Neutral** — Factual or mixed signals

We apply fine-tuning using the **Hugging Face Trainer API** on a pre-trained DistilBERT model.

---

## 🧠 Key Objectives

- Adapt a general-purpose language model to the financial domain
- Compare baseline vs fine-tuned model performance
- Conduct comprehensive hyperparameter optimization
- Perform error analysis to identify model limitations
- Deploy an interactive inference interface using Gradio

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Dataset** | Twitter Financial News Sentiment |
| **Source** | [Hugging Face Datasets](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) |
| **Task** | Multi-class sentiment classification |
| **License** | MIT |

### Dataset Structure

| Split | Samples |
|-------|---------|
| Train | ~8,400 |
| Validation | ~2,400 |
| Test | ~1,500 |

### Label Distribution

| Label | Description | Examples |
|-------|-------------|----------|
| **Bearish (0)** | Negative sentiment | Downgrades, price cuts, sell ratings |
| **Bullish (1)** | Positive sentiment | Upgrades, strong earnings, buy ratings |
| **Neutral (2)** | Neutral/factual | Market updates, mixed signals |

---

## 🏗️ Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | `distilbert-base-uncased` |
| **Parameters** | 66M |
| **Task Type** | Sequence Classification (3 classes) |
| **Fine-Tuning** | Full fine-tuning with Hugging Face Trainer API |

### Why DistilBERT?

- ✅ 40% smaller than BERT, 60% faster training
- ✅ Retains 97% of BERT's language understanding
- ✅ Efficient for iterative hyperparameter experiments
- ✅ Works well on short text (tweets)
- ✅ Clear improvement baseline (general → financial domain)

### Alternative Models Considered

| Model | Reason Not Selected |
|-------|---------------------|
| `bert-base-uncased` | Larger, slower, marginal improvement |
| `ProsusAI/finbert` | Already financial-tuned, less room to show improvement |
| `roberta-base` | Better but requires more compute |

---

## ⚙️ Training Strategy

We evaluated three hyperparameter configurations to optimize performance:

| Config | Learning Rate | Batch Size | Epochs | Weight Decay |
|--------|---------------|------------|--------|--------------|
| **Config 1** (Conservative) | 2e-5 | 16 | 3 | 0.01 |
| **Config 2** (Aggressive) | 5e-5 | 32 | 4 | 0.01 |
| **Config 3** (Small Batch) | 1e-5 | 8 | 5 | 0.05 |

### Training Features

- ✅ Early stopping (patience=2)
- ✅ Model checkpointing (save best model)
- ✅ Comprehensive logging
- ✅ Stratified train/test split

---

## 🚀 Inference & Deployment

The project includes:

- ✅ A reusable `FinancialSentimentAnalyzer` Python class
- ✅ Single and batch prediction methods
- ✅ An interactive **Gradio web interface** for live demonstrations

### Example Usage

```python
from inference import FinancialSentimentAnalyzer

analyzer = FinancialSentimentAnalyzer("./financial_sentiment_model/best_model")

result = analyzer.predict("$AAPL upgraded to Buy at Goldman Sachs")
print(result)
# {'sentiment': 'Bullish', 'confidence': 0.94, 'emoji': '🟢'}
```

## 🧪 Tools & Libraries

| Library | Purpose |
|---------|---------|
| **Hugging Face Transformers** | Model & Trainer API |
| **Hugging Face Datasets** | Dataset loading |
| **PyTorch** | Deep learning framework |
| **scikit-learn** | Evaluation metrics |
| **Gradio** | Interactive web interface |
| **Matplotlib / Seaborn** | Visualizations |
| **Pandas / NumPy** | Data processing |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Training (Optional)

```bash
python src/train.py
```

### 5. Launch Gradio Demo

```bash
python app.py
```

---

## 📋 Requirements

```txt
transformers>=4.36.0
datasets>=2.16.0
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
gradio>=4.0.0
accelerate>=0.25.0
```

---

## 📌 Conclusion

This project demonstrates how targeted fine-tuning of transformer models can significantly improve sentiment classification in the financial domain. By combining:

- A pre-trained language model (DistilBERT)
- The Hugging Face Trainer API
- Comprehensive hyperparameter optimization
- Rigorous evaluation and error analysis

We achieve **strong performance gains** while maintaining training efficiency.

The approach highlights a practical pathway for deploying language models in financial analysis workflows, with careful consideration of model limitations and deployment usability.

---

## 📚 References

1. Sanh et al., [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
2. Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
3. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
4. [Twitter Financial News Sentiment Dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
5. [Hugging Face Trainer API Guide](https://huggingface.co/docs/transformers/training)
6. [Gradio Documentation](https://gradio.app/docs/)

