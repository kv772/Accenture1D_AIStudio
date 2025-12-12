# Fake News Detection Backend

Flask API backend serving the Logistic Regression model for fake news detection.

## Requirements

- Python 3.8 or higher
- pip or pip3

## Installation

Install the required packages:

```bash
pip3 install Flask flask-cors joblib scikit-learn numpy scipy
```

Or using the requirements file:

```bash
pip3 install --user -r requirements.txt
```

## Running the Server

Start the Flask backend:

```bash
python3 app.py
```

The server will start on **http://localhost:5001**

You should see:
```
Loading Logistic Regression model...
Model loaded successfully: LogisticRegression(...)
Starting Flask server on http://localhost:5001
```

## API Endpoints

### Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict
```
POST /predict
Content-Type: application/json

{
  "text": "Article text to analyze..."
}
```

Response:
```json
{
  "prediction": "Fake",
  "confidence": 85.5,
  "probabilities": {
    "real": 14.5,
    "fake": 85.5
  },
  "model": "Logistic Regression"
}
```

## Model Details

- **Model Type**: Logistic Regression (scikit-learn)
- **Input Features**: 3000 TF-IDF features
- **Training**: Trained on 40,000+ news articles
- **Output**: Binary classification (Fake/Real) with confidence scores

## Architecture

The backend:
1. Loads the pre-trained logistic regression model (`logistic_regression_model.joblib`)
2. Preprocesses incoming text (lowercase, remove URLs, special characters)
3. Vectorizes text using TF-IDF (3000 features, bigrams, English stop words)
4. Pads feature vector to match model's expected input shape
5. Returns prediction with confidence scores

## Notes

- The TF-IDF vectorizer is initialized at runtime (not saved with the model)
- This is a workaround - ideally the vectorizer should be saved during training
- CORS is enabled for frontend integration
- Debug mode is enabled for development

## Troubleshooting

**Port already in use:**
```bash
lsof -ti:5001 | xargs kill -9
```

**Missing packages:**
```bash
pip3 install --user Flask flask-cors joblib scikit-learn numpy scipy
```

**macOS externally-managed-environment error:**
```bash
pip3 install --user -r requirements.txt
```
