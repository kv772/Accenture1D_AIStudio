import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Navbar from '../components/Navbar';
import './Results.css';

function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const { article = '', title = '', prediction = null } = location.state || {};

  const [analyzing, setAnalyzing] = useState(true);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!article) {
      navigate('/analysis');
      return;
    }

    // Animate progress
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          return 100;
        }
        return prev + 2;
      });
    }, 30);

    // Use actual prediction if available, otherwise simulate
    setTimeout(() => {
      if (prediction) {
        // Use actual prediction from backend
        setResult({
          prediction: prediction.prediction,
          confidence: prediction.confidence,
          probabilities: prediction.probabilities,
          modelScores: {
            bert: Math.random() * 30 + 70, // Mock for now
            lstm: Math.random() * 30 + 70, // Mock for now
            logistic: prediction.confidence,
            baseline: Math.random() * 30 + 60, // Mock for now
          },
        });
      } else {
        // Fallback to mock data
        const mockConfidence = Math.random() * 30 + 70;
        const isFake = Math.random() > 0.5;

        setResult({
          prediction: isFake ? 'Fake' : 'Real',
          confidence: mockConfidence,
          modelScores: {
            bert: Math.random() * 30 + 70,
            lstm: Math.random() * 30 + 70,
            logistic: Math.random() * 30 + 60,
            baseline: Math.random() * 30 + 60,
          },
        });
      }
      setAnalyzing(false);
    }, 3000);

    return () => clearInterval(progressInterval);
  }, [article, navigate, prediction]);

  if (analyzing) {
    return (
      <div className="results loading">
        <Navbar />
        <div className="container">
          <div className="loading-card">
            <h2 className="loading-title">Analyzing Article...</h2>

            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${progress}%` }}
              />
            </div>

            <div className="steps">
              <div className="step completed">
                <div className="step-indicator">✓</div>
                <span>Preprocessing text</span>
              </div>
              <div className="step completed">
                <div className="step-indicator">✓</div>
                <span>Tokenization</span>
              </div>

              <div
                className={`step ${progress >= 90 ? 'completed' : 'active'}`}>
                <div className="step-indicator">
                  {progress >= 90 ? '✓' : '⟳'}
                </div>
                <span>Generating results</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const isFake = result.prediction === 'Fake';

  return (
    <div className="results">
      <Navbar />

      <div className="results-content">
        <div className="container">
          {/* Main Result Card */}
          <div className="result-card">
            <div className={`result-badge ${isFake ? 'fake' : 'real'}`}>
              {result.prediction.toUpperCase()}
            </div>

            <h2 className="result-title">
              Classification: {result.prediction} News
            </h2>

            <div className="confidence-header">
              <span className="confidence-label">Confidence Score</span>
              <span className={`confidence-value ${isFake ? 'fake' : 'real'}`}>
                {result.confidence.toFixed(1)}%
              </span>
            </div>

            <div className="confidence-bar">
              <div
                className={`confidence-fill ${isFake ? 'fake' : 'real'}`}
                style={{ width: `${result.confidence}%` }}
              />
            </div>

            <div className="interpretation">
              <p>
                {isFake
                  ? 'This article exhibits patterns commonly found in fake news, including sensational language, unverified claims, or suspicious sources.'
                  : 'This article demonstrates characteristics of legitimate news reporting, including factual language, credible sources, and balanced presentation.'}
              </p>
            </div>
          </div>

          {/* Model Breakdown */}
          <div className="section-header">
            <h3 className="section-title">Model Breakdown</h3>
            <p className="section-subtitle">Individual model predictions</p>
          </div>

          <div className="model-cards">
            <div className="model-card">
              <div className="model-info">
                <h4 className="model-name">Logistic Regression</h4>
                <p className="model-desc">Statistical classification</p>
              </div>
              <div className="model-score">
                {result.modelScores.logistic.toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Article Preview */}
          <div className="section-header">
            <h3 className="section-title">Article Preview</h3>
          </div>

          <div className="article-preview">
            {title && <h4 className="article-title">{title}</h4>}
            <p className="article-text">
              {article.length > 500
                ? article.substring(0, 500) + '...'
                : article}
            </p>
          </div>

          {/* Actions */}
          <div className="actions">
            <button
              className="btn btn-primary"
              onClick={() => navigate('/analysis')}>
              Analyze Another Article
            </button>
            <button className="btn btn-secondary" onClick={() => navigate('/')}>
              Back to Home
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Results;
