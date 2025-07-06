# 📊 AI-Powered Financial Sentiment Analysis using FinBERT and XGBoost

This project implements a sentiment analysis system specifically designed for financial text such as news headlines and articles. It combines **FinBERT** — a BERT-based model pre-trained on financial data — with **XGBoost** to improve classification across **Positive**, **Neutral**, and **Negative** categories. The system addresses class imbalance and enhances generalization using **data augmentation** and **SMOTE-based oversampling**.

---

## 🔍 Problem Statement

Extracting sentiment from financial text is challenging due to subtle language and **class imbalance** — especially underrepresented classes like **Neutral**. The goal was to build a robust system that could provide accurate and balanced sentiment classification for financial applications.

---

## 🧪 Experimental Approach

### ✅ Baseline Model
- Fine-tuned FinBERT on the original imbalanced dataset
- High precision for Positive class, but lower recall for Neutral and Negative

### 🔁 Data Balancing & Augmentation
To improve generalization and fairness:
- Applied **SMOTE** to oversample minority classes
- Used **NLP-based augmentation**:
  - Synonym replacement
  - Back-translation (English → French → English)

These steps increased the diversity of input text and reduced the risk of overfitting.

---

## ⚙️ Final Training Setup
- FinBERT fine-tuned for 5 epochs  
- Optimizer: **AdamW** (learning rate = 3e-6)  
- Loss Function: **Class-weighted cross-entropy**  
- **XGBoost** trained on FinBERT logits for enhanced predictions

---

## 📈 Results Overview
- **Accuracy**: 95.62% on test set  
- Improved recall and F1-scores for Neutral and Negative classes  
- Evaluated using:
  - Confusion matrix  
  - Classification report  
  - Precision-recall curves

---

## 🚀 Deployment

Deployed using **Gradio** for real-time sentiment prediction.  
🔹 Users can input financial text and instantly receive sentiment output via an intuitive interface (see uploaded screenshots).

---

## 🧰 Technologies Used

- **Languages**: Python  
- **Tools**: Google Colab, GitHub  
- **Libraries/Models**:  
  - FinBERT (via Hugging Face Transformers)  
  - XGBoost  
  - Scikit-learn  
  - SMOTE (imbalanced-learn)  
  - Pandas, NumPy, Matplotlib 
  - Gradio (for UI)

---

## 📂 Dataset

- Source: [Kaggle Financial Markets Dataset – Prices & News](https://www.kaggle.com/datasets/znevzz/the-news-dataset)
- The dataset used for this project is a labeled CSV (`labeled.csv`) downloaded from the above source.
- Labels used: **Positive**, **Neutral**, **Negative**

---

## 📁 File Overview

- `Financial_Sentiment_Analysis.ipynb` – Complete notebook with training, evaluation, and Gradio UI  
- `gradio_screenshots/` – Folder containing screenshots of the Gradio interface
- `README.md` – Project documentation

---

## 👩‍💻 Author

**Udisha Panwar**  
B.Tech – Electronics and Communication Engineering with Artificial Intelligence  
[GitHub Profile](https://github.com/UdishaPanwar)

---
