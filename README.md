# 🤖 Climate Sentiment Classification & Twitter NER

![Typing Header](https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&pause=1000&color=36D7B7&center=true&vCenter=true&width=850&lines=🌡️+Classify+ESG+Disclosure+Sentiment;🐦+Extract+Entities+From+Tweet+Text)

<p align="center">
  <img src="https://img.shields.io/badge/NLP-Climate%20Sentiment-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NER-Twitter-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Transformer-BERT/DistilBERT-yellow?style=for-the-badge"/>
</p>

---

## 🧠 Project Overview
This project integrates two natural language processing applications:

- **Classifying climate-related financial disclosures** into sentiment categories: Risk, Opportunity, or Neutral.
- **Performing Named Entity Recognition (NER)** on noisy Twitter text using transformer-based architectures.

> 💡 The goal is to enable better automated interpretation of environmental finance reports and real-world event mentions from unstructured, informal text sources.

---

## 🌍 Climate Sentiment Classification

### 🔍 Objective
Automatically identify whether a financial or ESG-related statement expresses Risk, Opportunity, or Neutral sentiment regarding climate impact.

### 📚 Methods and Models
| Model       | Accuracy |
|-------------|----------|
| Naive Bayes | 0.79     |
| FNN         | 0.47     |
| BERT        | 0.66     |

- Naive Bayes: Tuned with `ngram_range=(2,2)` and `min_df=2`, outperforming contextual models on sparse data.
- Feedforward Neural Network: Utilized GloVe embeddings.
- BERT: Fine-tuned via HuggingFace; struggled with mixed sentiment but captured nuanced phrases.

### 📈 Feature Engineering
- Stopword removal, TF-IDF vectorization, bigram modeling
- BERT settings: max_seq_len=128, batch_size=16, epochs=4

### ✅ Final Result & Significance
Naive Bayes achieved **0.79 accuracy**, proving that traditional models can outperform transformers in sparse, domain-specific ESG datasets.

---

## 📊 BERTopic Topic Modeling

### ⚙️ Pipeline
- Embedding: MiniLM
- Dimensionality reduction: UMAP
- Clustering: HDBSCAN
- Labeling: c-TF-IDF

### 📌 Results
- 🧱 Risk: climate, disaster, liability
- 🌞 Opportunity: renewable, energy, investment
- 🚗 Tech: carbon, EVs, charging

### ✅ Final Result & Significance
BERTopic generated clearer and more actionable themes than LDA, indicating the strength of semantic clustering for ESG topics.

---

## 🐦 Twitter Named Entity Recognition (NER)

### 🔍 Objective
Extract named entities from informal tweets using a transformer-based model.

### 🤖 Model
- Fine-tuned `distilbert-base-uncased`
- Dataset: Broad Twitter Corpus (BTC)

### 📊 Metrics
| Entity       | Precision | Recall | F1    |
|--------------|-----------|--------|-------|
| Person       | 0.73      | 0.68   | 0.70  |
| Location     | 0.69      | 0.63   | 0.66  |
| Corporation  | 0.65      | 0.58   | 0.61  |
| Product      | 0.50      | 0.56   | 0.53  |

### ✅ Final Result & Significance
Achieved **0.60 macro F1 score**, demonstrating strong generalization of DistilBERT to messy, real-world text.

---

## 🧰 Technologies Used
- Python 3.10+
- `scikit-learn`, `transformers`, `HuggingFace`, `pandas`, `nltk`, `matplotlib`, `seaborn`
- `BERTopic`, `MiniLM`, `UMAP`, `HDBSCAN`, `c-TFIDF`, `sentence-transformers`
- `seqeval` for NER evaluation

---

## 🚀 Getting Started

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
