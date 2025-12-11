# Fake News Detection & Classification 

#### Accenture AI Studio Challenge Project
Investigated how well machine learning models can identify fake news articles compared to human review, applying Python, NLP, and deep learning methods within Break Through Tech AI's AI Studio accelerator program.


## Team Members

| Name | GitHub | Contribution |
|------|--------|--------------|
| Lin Zhang | [@lin-zhang88](https://github.com/lin-zhang88) | Exploratory Data Analysis, BERT, Feature Engineering |
| Kashvi Vijay | [@kv772](https://github.com/kv772) | Logistic Regression, Feature Engineering |
| Nancy Huang | [@naanci](https://github.com/naanci) | Feature Engineering, BERT, Exploratory Data Analysis |
| Adriena Jiang | [@adrienajiang](https://github.com/adrienajiang) | Exploratory Data Analysis, Visualizations, Feature Engineering |
| Ousman Bah | [@Ousmanbah10](https://github.com/Ousmanbah10) | Feature Engineering, Exploratory Data Analysis, CNN |
| Sanskriti Khadka | [@Sanskritik7](https://github.com/sanskritik7) | Exploratory Data Analysis, CNN, Feature Engineering |
| Harshika Agrawal | [@HarshikaAgr](https://github.com/HarshikaAgr) | Exploratory Data Analysis, Logistic Regression |


## Project Highlights

* Developed a machine learning model using Logistic Regression, BERT and Neural Networks to identify fake news articles compared to human review.
* Achieved 74% accuracy with Logistic Regression, 94% accuracy with BERT, 93.5% accuracy with average embedding model and 96% with CNN model.
* Technologies Used: Python, TensorFlow, Keras, PyTorch, Transformers, scikit-learn, pandas, NumPy, matplotlib, seaborn, BERT, LSTM Networks, Google Colab and Jupyter Notebook.


## Setup and Installation
```bash
pip install tensorflow torch transformers scikit-learn pandas numpy matplotlib seaborn joblib
```

### Usage

#### Load Saved Models
```python
import joblib
from tensorflow.keras.models import load_model
from transformers import BertForSequenceClassification, BertTokenizer

# Load Keras Models
lstm_model = load_model('lstm_model.keras')
baseline_model = load_model('baseline_model.keras')

# Load Logistic Regression
log_reg = joblib.load('logistic_regression_model.pkl')

# Load BERT model
bert_model = BertForSequenceClassification.from_pretrained('bert_fake_news_model')
tokenizer = BertTokenizer.from_pretrained('bert_fake_news_model')
```

#### Making Predictions
```python
import torch

# Example with BERT
text = "Your news article here..."
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
outputs = bert_model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
print("Fake" if prediction == 1 else "Real")
```

---

## Project Overview

* Trust in digital media and content moderation are critical challenges in today's information ecosystem. Social media platforms, publishers, and advertisers face financial and reputational risk when their services propogate false information. Manual review of news articles is infeasible at scale. With the exponential growth of online content, there is a growing need for automated tools that can support content moderators and improve detection consistency.
* This project with Accenture aims to utilize deep learning techniques and NLP models to accurately classify real and fake news. Understanding the projects strength and weaknesses align with Accentures responsible AI initiatives, strengthen digital trust offerings for clients and automate content vertifcation/risk detection.

## Data Exploration

* Used datasets from [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data), which includes two CSV files: one containing real news articles and one containing fake news articles.
* The true news file included ~21,000 unique entries whie the fake news file included ~18,000 unique entries.
* Each dataset contains fields such as title, text, subject and date providing multiple features for analysis.

#### Preprocessing & Cleaning
* Loaded and processed data using Python to prepare it for modeling.
* Applied text preprocessing steps including cleaning, nromalization and basic feature engineering.
* Conducted extensive EDA to identify potential data leakages or fields that could unintentionally reveal class labels.
* Removed features that were problematic or not useful for modeling such as URLs, subject labels, and stop words, to prevent models from relying on non-semantic shortcuts.

## Model Developement
### 1. Logistic Regression
  * Logistic Regression was selected because it is lightweight, interpretable, and a strong baseline for text classification.
  * Paired with TF-IDF, it effectively captures key linguistic and stylistic cues that differentiate real and fake news.
  * Used HalvingGridSearchCV to tune hyperparameters; the best configuration was C = 1 with an L2 penalty.
  * Training included 5-fold cross validation to ensure consistent performance across splits.
  * Performance: 74% accuracy, True F1-score: 97%, Fake F1-score: 96%.
### 2. BERT
  * BERT was chosen because it understands contextual meaning by reading text bidirectionally, allowing it to detect tone, writing style, and subtle misleading cues.
  * Kept stop words since BERT performs better on full sentence structure, which also reduced overfitting.
  * Performance: 94% accuracy, F1-score: 96%.
### 3. Neural Networks
  * Neural Networks allowed us to capture more complex linguistic patterns through deep learning architectures built with TensorFlow/Keras.
  * 
  #### Model A: Global Average Pooling
  * Uses word embeddings and averages them to learn the overall meaning of the article.
  * Serves as a simple and fast baseline deep learning model.
  * Performance: 93.5% accuracy.

  #### Model B: 1D CNN
  * Uses embeddings combined with a convolutional layer to learn phrase-level patterns (n-grams).
  * Better at capturing tone and structural signals within the text.
  * Performance: 96% accuracy.

## Code Highlights

### `Accenture_1D_Model.ipynb`

This notebook contains the full workflow/pipeline for building and evaluation our models for fake news detection.


## Results and Key Findings

#### Model Performance
We successfully trained and evaluated four different models for fake news classification:
- Logistic Regression
- BERT (Bidirectional Encoder Representations from Transformers)
- Neural Networks
Each model achieved strong performance, demonstrating that both traditional ML and deep learning architectures can effectively support misinformation detection tasks.

#### Dataset Insights
During exploration of the Kaggle Fake News datasets, we identified several key findings:
- Fake news articles contained significantly more punctuation, including ~12× more exclamation and question marks than real news.
- URL tokens (`<URL>`) appeared far more often in fake articles, creating an unintended shortcut for classification.
- The subject column (political categories) acted as a strong predictor, introducing data leakage risks.

#### Bias Analysis
We examined dataset-level biases that could affect model generalizability:
- Heavy political-topic dominance, limiting performance on non-political news.
- Publisher/source bias, since original news outlets influence writing style.
- Class imbalance, requiring careful preprocessing and model tuning.

#### Preprocessing Summary
To reduce leakage and prepare the text for modeling, we implemented:
- Removal of URLs  
- Punctuation review and filtering  
- Stop-word removal (kept for BERT models)  
- Tokenization and text normalization  

#### Overall Findings
- All models achieved viable performance metrics, confirming that machine learning can serve as an effective first layer of misinformation detection.
- The models show strong potential as screening or triage tools to assist human content moderators by flagging potentially misleading content for further review.

## Discusson and Reflection

Throughout this project, our team found different modeling approaches excelled for different reasons. Traditional machine learning models like Logistic Regression performed suprisingly well, especially when paired with TF-IDF, because they captured strong stylistic signals in the text. Deep learning models, such as neural networks and BERT, performed well they captured phrase level patterns effectively and leverage contextual understanding to handle subtle word differences. Our deep learning models are still experiencing overfitting indicating the importance of careful data exploration before model development.

## Next Steps

- Although our models achieved high accuracy, this may indicate remaining sources of data leakage. Our next step is to perform deeper cleaning and feature analysis to identify and remove any remaining unintended signals.
- Continue refining BERT and neural network models to reduce overfitting and improve their generalization to unseen data.
- Re-train all models under stricter preprocessing conditions with a target accuracy of **70–75%**, which likely reflects the dataset’s true difficulty once leakage is fully mitigated.

## Acknowledgments 

Special thanks to Accenture and Break Through Tech AI for making this project possible. We also express our deep appreciation to our coach, Jenna Hunte, and our challenge advisor, Abdul Wahab, for their expert guidance and mentorship throughout the project.
