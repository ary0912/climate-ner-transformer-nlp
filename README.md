# 🤖 AI Project — Climate Sentiment Classification & Twitter NER

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

All components are developed independently using transformer models and topic modeling techniques.

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
- Feedforward Neural Network: Utilized GloVe embeddings; underperformed due to lack of deep contextual signal.
- BERT: Fine-tuned with HuggingFace transformers; struggled with mixed sentiment tone but captured nuanced context.

### 📈 Feature Engineering
- Stopword filtering using sklearn  
- TF-IDF vectorization (with and without bigrams)  
- Max sequence length tuning for BERT: 128  
- Batch size: 16, epochs: 4  

---

## 📊 BERTopic Topic Modeling (Extension)

### ⚙️ Pipeline
- Sentence embedding: MiniLM via `sentence-transformers`
- Dimensionality reduction: UMAP (10 neighbors)
- Clustering: HDBSCAN
- Topic representation: c-TF-IDF

### 📌 Results
Generated interpretable clusters:
- 🧱 Risk: climate, disaster, impact, liability  
- 🌞 Opportunity: energy, transition, investment  
- 🚗 Green Tech: EV, carbon, hydrogen  

BERTopic achieved better topic coherence than LDA, particularly on short ESG documents.

---

## 🐦 Twitter Named Entity Recognition (NER)

### 🔍 Goal
Identify named entities (Person, Location, Corporation, Product, etc.) in informal and noisy Twitter data.

### 🤖 Model
- Pretrained model: `distilbert-base-uncased`
- Fine-tuned using HuggingFace Trainer
- Dataset: Broad Twitter Corpus (BTC)

### 🔧 Token Alignment
- WordPiece tokenization handled via token-level label alignment
- Subword masking applied to ignore non-initial subwords during training

### 📊 Metrics
| Entity Type     | Precision | Recall | F1    |
|------------------|-----------|--------|-------|
| Person           | 0.73      | 0.68   | 0.70  |
| Location         | 0.69      | 0.63   | 0.66  |
| Corporation      | 0.65      | 0.58   | 0.61  |
| Product          | 0.50      | 0.56   | 0.53  |

Final macro-averaged F1 score: **0.60**

---

## 🧰 Technologies Used
- Python 3.10+
- `scikit-learn`, `transformers`, `pandas`, `nltk`, `matplotlib`, `seaborn`
- `BERTopic`, `MiniLM`, `UMAP`, `HDBSCAN`, `sentence-transformers`
- `seqeval`, `HuggingFace Trainer`, `c-TFIDF`

---

## 🚀 Getting Started

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```
<p align="center"> <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=2000&color=F97316&center=true&vCenter=true&width=900&lines=From+climate+disclosures+to+tweet+NER...;Transformers+decoded+semantic+insight+across+domains."/> </p>
