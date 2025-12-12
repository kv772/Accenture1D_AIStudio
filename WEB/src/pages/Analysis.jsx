import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import './Analysis.css';

function Analysis() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('single');
  const [articleTitle, setArticleTitle] = useState('');
  const [articleText, setArticleText] = useState('');
  const [selectedModel, setSelectedModel] = useState('ensemble');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleLoadDemo = () => {
    setArticleTitle('Breaking: Major Scientific Discovery Announced');
    setArticleText(
      'Scientists at leading research institutions have announced a groundbreaking discovery in renewable energy technology. The new solar panel design demonstrates 45% efficiency, a significant improvement over current commercial panels averaging 20% efficiency. The research team, led by Dr. Sarah Chen, published their findings in Nature Energy today. "This represents a major step forward in our transition to sustainable energy," Dr. Chen stated during the press conference. The technology uses a novel multi-junction architecture combining perovskite and silicon layers, enabling broader spectrum light absorption.',
    );
  };

  const handleAnalysis = async () => {
    if (!articleText.trim()) {
      alert('Please enter article text to analyze');
      return;
    }

    setIsAnalyzing(true);

    try {
      // Call the backend API
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: articleText,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction from server');
      }

      const result = await response.json();

      setIsAnalyzing(false);
      navigate('/results', {
        state: {
          article: articleText,
          title: articleTitle,
          model: selectedModel,
          prediction: result,
        },
      });
    } catch (error) {
      setIsAnalyzing(false);
      alert(
        `Error: ${error.message}. Make sure the backend server is running on http://localhost:5001`,
      );
    }
  };

  return (
    <div className="analysis">
      <Navbar />

      <div className="analysis-content">
        <div className="container">
          {/* Model Selection */}
          <div className="model-selection">
            <div className="model-options">
              <button
                className={`model-option ${
                  selectedModel === 'logistic' ? 'active' : ''
                }`}
                onClick={() => setSelectedModel('logistic')}>
                <div className="model-option-title">
                  LOGISTIC REGRESSION MODEL
                </div>
              </button>
            </div>
          </div>

          {/* Input Fields */}
          <div className="input-section">
            <label className="label">Article Title (Optional)</label>
            <input
              type="text"
              className="input"
              placeholder="Enter article title..."
              value={articleTitle}
              onChange={e => setArticleTitle(e.target.value)}
            />

            <label className="label">Article Text *</label>
            <textarea
              className="textarea"
              placeholder="Paste or type the article content here..."
              value={articleText}
              onChange={e => setArticleText(e.target.value)}
              rows={12}
            />

            <p className="help-text">
              Paste the full article text for analysis. Our AI models will
              analyze the content, writing style, and linguistic patterns.
            </p>
          </div>

          {/* Info Card */}
          <div className="info-card">
            <h3 className="info-title">How It Works</h3>
            <ul className="info-list">
              <li>Text preprocessing and tokenization</li>

              <li>Model predicting</li>
            </ul>
          </div>

          {/* Analyze Button */}
          <button
            className={`analyze-button ${isAnalyzing ? 'analyzing' : ''}`}
            onClick={handleAnalysis}
            disabled={isAnalyzing}>
            {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Analysis;
