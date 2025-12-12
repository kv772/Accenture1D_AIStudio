from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)

# Load the pipeline (contains both vectorizer and logistic regression model)
pipeline = joblib.load('../Models/logistic_regression_model.joblib')

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether news article is fake or real"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        article_text = data['text']

        if not article_text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        # Preprocess the text
        cleaned_text = preprocess_text(article_text)

        # Make prediction using the pipeline
        # Pipeline automatically handles vectorization with the correct fitted vectorizer
        prediction = pipeline.predict([cleaned_text])[0]
        probabilities = pipeline.predict_proba([cleaned_text])[0]

        # Get confidence score (probability of predicted class)
        confidence = float(max(probabilities) * 100)

        # Determine if fake or real (1 = Fake, 0 = Real)
        is_fake = bool(prediction == 1)

        result = {
            'prediction': 'Fake' if is_fake else 'Real',
            'confidence': round(confidence, 2),
            'probabilities': {
                'real': round(float(probabilities[0]) * 100, 2),
                'fake': round(float(probabilities[1]) * 100, 2)
            },
            'model': 'Logistic Regression'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading Logistic Regression pipeline (vectorizer + model)...")
    print(f"Pipeline loaded successfully: {pipeline}")
    print("Starting Flask server on http://localhost:5001")
    app.run(debug=True, port=5001)
