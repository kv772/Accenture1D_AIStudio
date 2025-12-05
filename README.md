# Fake News Detection & Classification

**Accenture AI Studio Challenge Project - Fall 2025**

Investigated how well machine learning models can identify fake news articles compared to human review, applying Python, NLP, and deep learning methods within Break Through Tech AI's AI Studio accelerator program.

## Problem Statement

Trust in digital media and content moderation are critical challenges in today's information ecosystem. Social media platforms, publishers, and advertisers face financial and reputational risks when their services propagate false information. Manual review of news articles is infeasible at scale, with moderators vulnerable to fatigue and inconsistent judgment. With the exponential growth of online content, there is a growing need for automated tools that can support content moderators and improve detection consistency. This project explores whether AI — specifically deep learning and NLP models — can classify fake and real news articles with meaningful accuracy. Understanding AI's strengths and weaknesses in this task has real impact on platform safety, brand trust, and public discourse.

## Key Results

1. Successfully trained 4 different models: LSTM, Baseline Neural Network, Logistic Regression, and BERT (Bidirectional Encoder Representations from Transformers)
2. Processed and explored the Kaggle Fake and Real News dataset containing labeled articles with title, text, subject, and date fields
3. Identified and mitigated major data leakage issues:
   * Fake news articles contained 12x more exclamation marks and question marks than real news
   * URL tokens (`<URL>`) appeared predominantly in fake news, creating artificial classification signals
   * Subject column (political topics) provided overly strong prediction signals
4. Analyzed potential biases in the dataset:
   * Political news dominance reducing generalizability to other news domains
   * Source bias from original article publishers
   * Class imbalance requiring specialized training techniques
5. Implemented comprehensive text preprocessing: URL removal, punctuation analysis, stop word handling (retained for BERT), and tokenization
6. Achieved clinically viable performance metrics across multiple model architectures
7. Demonstrated models' viability as screening/triage tools to assist human content moderators in detecting misleading content

## Methodologies

To accomplish this, we utilized Python to load, preprocess, and analyze text data from the Fake and Real News dataset. We implemented comprehensive data preprocessing including text cleaning (HTML tags, special characters), normalization, and feature engineering. We built multiple model architectures: a traditional Logistic Regression baseline, an LSTM (Long Short-Term Memory) neural network for sequential text analysis, a simple feedforward neural network baseline, and a fine-tuned BERT transformer model specifically designed for natural language understanding. To address the data leakage issues, we conducted extensive exploratory data analysis using n-grams and punctuation pattern analysis, then systematically removed problematic features (URLs, subject column, excessive punctuation). The BERT model was trained using pre-trained `bert-base-uncased` weights with 2 epochs, batch size of 16, and learning rate of 2e-5. We evaluated performance using classification metrics including accuracy, precision, recall, F1-score, and confusion matrices. Finally, we compared models with and without suspected leakage features to validate our data cleaning decisions and ensure models learned genuine content patterns rather than dataset artifacts.

## Data Sources

* **Kaggle Fake and Real News Dataset**: Two CSV files containing fake news and real news articles
* Dataset includes fields: title, text, subject, date
* Related documentation and research on fake news detection methodologies

## Technologies Used

* Python
* TensorFlow / Keras
* PyTorch
* Transformers (Hugging Face)
* scikit-learn
* pandas
* NumPy
* matplotlib
* seaborn
* BERT (`bert-base-uncased`)
* LSTM Networks
* Google Colab (GPU)
* Jupyter Notebook

## Usage

### Loading Saved Models
```python
import joblib
from tensorflow.keras.models import load_model
from transformers import BertForSequenceClassification, BertTokenizer

# Load Keras models
lstm_model = load_model('lstm_model.keras')
baseline_model = load_model('baseline_model.keras')

# Load Logistic Regression
log_reg = joblib.load('logistic_regression_model.pkl')

# Load BERT model
bert_model = BertForSequenceClassification.from_pretrained('bert_fake_news_model')
tokenizer = BertTokenizer.from_pretrained('bert_fake_news_model')
```

### Making Predictions
```python
import torch

# Example with BERT
text = "Your news article here..."
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
outputs = bert_model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
print("Fake" if prediction == 1 else "Real")
```

## Installation
```bash
pip install tensorflow torch transformers scikit-learn pandas numpy matplotlib seaborn joblib
```

## Authors

This project was completed in collaboration with:

* Lin Zhang
* Kashvi Vijay
* Nancy Huang
* Adriena Jiang
* Ousman Baldeh
* Sanskriti Khadka
* Harshika Agrawal

**Coach:** Jenna Hunte  
**Challenge Advisor:** Abdul (Accenture)

---

**Acknowledgments:** Thanks to Accenture and Break Through Tech AI for this opportunity and guidance throughout the project.
