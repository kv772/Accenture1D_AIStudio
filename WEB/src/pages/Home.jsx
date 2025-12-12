import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import './Home.css';
import LinPhoto from '../Images/LIN.jpeg';
import AdrienaPhoto from '../Images/Adriena.jpeg';
import OusmanPhoto from '../Images/Ousman.jpeg';
import SanskritiPhoto from '../Images/Sanskriti.jpeg';
import KashviPhoto from '../Images/Kashvi.jpeg';
import NancyPhoto from '../Images/Nancy.jpeg';
import HarshikaPhoto from '../Images/harshika.jpeg';
import JennaPhoto from '../Images/Jenna.jpeg';
import AbdulPhoto from '../Images/Abdul.jpeg';

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home">
      <Navbar />

      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <div className="badge">AI-POWERED DETECTION</div>

          <h1 className="hero-title">
            Fake News
            <br />
            Detection
          </h1>

          <p className="hero-subtitle">
            Advanced machine learning models to identify fake news articles with
            high accuracy. Built with BERT, LSTM, and NLP technology.
          </p>

          <div className="cta-buttons">
            <button
              className="btn btn-primary"
              onClick={() => navigate('/analysis')}>
              Launch Analysis Tool
            </button>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats">
        <div className="container">
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value secondary">4</div>
              <div className="stat-label">ML Models</div>
            </div>
            <div className="stat-card">
              <div className="stat-value accent">40k+</div>
              <div className="stat-label">Articles Trained</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="container">
          <h2 className="section-title">Why Fake News Detection Matters</h2>

          <div className="feature-cards">
            <div className="feature-card">
              <div className="feature-icon">🛡️</div>
              <h3 className="feature-title">Protect Brand Trust</h3>
              <p className="feature-text">
                Safeguard your platform and users from misinformation that
                damages reputation and credibility.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3 className="feature-title">Real-time Analysis</h3>
              <p className="feature-text">
                Instant detection using state-of-the-art BERT and LSTM neural
                networks trained on thousands of articles.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3 className="feature-title">High Precision</h3>
              <p className="feature-text">
                Comprehensive NLP preprocessing and deep learning models ensure
                reliable classification results.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Models Section */}
      <section id="models" className="models-section">
        <div className="container">
          <h2 className="section-title">Our AI Models</h2>
          <p className="section-subtitle">
            State-of-the-art machine learning models working together
          </p>

          <div className="models-grid">
            <div className="model-box">
              <div className="model-icon">🧠</div>
              <h3 className="model-box-title">BERT Transformer</h3>
              <p className="model-box-desc">
                Bidirectional Encoder Representations from Transformers.
                Analyzes context and semantic meaning with deep learning.
              </p>
              <div className="model-stats">
                <div className="model-stat">
                  <span className="stat-label">Type</span>
                  <span className="stat-value">Transformer</span>
                </div>
              </div>
            </div>

            <div className="model-box">
              <div className="model-icon">⚡</div>
              <h3 className="model-box-title">LSTM Network</h3>
              <p className="model-box-desc">
                Long Short-Term Memory neural network. Captures sequential
                patterns and temporal dependencies in text.
              </p>
              <div className="model-stats">
                <div className="model-stat">
                  <span className="stat-label">Type</span>
                  <span className="stat-value">RNN</span>
                </div>
              </div>
            </div>

            <div className="model-box">
              <div className="model-icon">📊</div>
              <h3 className="model-box-title">Logistic Regression</h3>
              <p className="model-box-desc">
                Statistical classification model. Provides fast, interpretable
                predictions based on text features.
              </p>
              <div className="model-stats">
                <div className="model-stat">
                  <span className="stat-label">Type</span>
                  <span className="stat-value">Statistical</span>
                </div>
              </div>
            </div>

            <div className="model-box">
              <div className="model-icon">🎯</div>
              <h3 className="model-box-title">Ensemble</h3>
              <p className="model-box-desc">
                Combines all models for superior accuracy. Leverages strengths
                of each approach for reliable predictions.
              </p>
              <div className="model-stats">
                <div className="model-stat">
                  <span className="stat-label">Type</span>
                  <span className="stat-value">Combined</span>
                </div>
              </div>
            </div>
          </div>

          <div className="tech-stack-box">
            <h3 className="tech-stack-title">Built With</h3>
            <div className="tech-badges">
              <span className="tech-badge">BERT</span>
              <span className="tech-badge">LSTM</span>
              <span className="tech-badge">TensorFlow</span>
              <span className="tech-badge">PyTorch</span>
              <span className="tech-badge">scikit-learn</span>
              <span className="tech-badge">Transformers</span>
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section id="team" className="team">
        <div className="container">
          <h2 className="section-title">Meet Our Team</h2>

          <div className="team-grid-photos">
            <div className="team-member">
              <a href="https://www.linkedin.com/in/lin-zhang-850a8a248/" target="_blank" rel="noopener noreferrer">
                <img src={LinPhoto} alt="Lin Zhang" className="member-photo" />
              </a>
              <h3 className="member-name">Lin Zhang</h3>
              <p className="member-school">Lipscomb University</p>
            </div>

            <div className="team-member">
              <a href="https://www.linkedin.com/in/adrienajiang/" target="_blank" rel="noopener noreferrer">
                <img src={AdrienaPhoto} alt="Adriena Jiang" className="member-photo" />
              </a>
              <h3 className="member-name">Adriena Jiang</h3>
              <p className="member-school">Stony Brook University</p>
            </div>

            <div className="team-member">
              <a href="https://www.linkedin.com/in/ousman-bah/" target="_blank" rel="noopener noreferrer">
                <img src={OusmanPhoto} alt="Ousman Bah" className="member-photo" />
              </a>
              <h3 className="member-name">Ousman Bah</h3>
              <p className="member-school">Florida International University</p>
            </div>

            <div className="team-member">
              <a href="https://www.linkedin.com/in/sanskritikhadka/" target="_blank" rel="noopener noreferrer">
                <img src={SanskritiPhoto} alt="Sanskriti Khadka" className="member-photo" />
              </a>
              <h3 className="member-name">Sanskriti Khadka</h3>
              <p className="member-school">City College of New York</p>
            </div>
          </div>

          <div className="team-grid-photos">
            <div className="team-member">
              <a href="https://www.linkedin.com/in/kashvi-vijay-12b7bb271/" target="_blank" rel="noopener noreferrer">
                <img src={KashviPhoto} alt="Kashvi Vijay" className="member-photo" />
              </a>
              <h3 className="member-name">Kashvi Vijay</h3>
              <p className="member-school">UT Austin</p>
            </div>

            <div className="team-member">
              <a href="https://www.linkedin.com/in/nancyyy-huang/" target="_blank" rel="noopener noreferrer">
                <img src={NancyPhoto} alt="Nancy Huang" className="member-photo" />
              </a>
              <h3 className="member-name">Nancy Huang</h3>
              <p className="member-school">Stony Brook University</p>
            </div>

            <div className="team-member">
              <a href="https://www.linkedin.com/in/agrawal-harshika/" target="_blank" rel="noopener noreferrer">
                <img src={HarshikaPhoto} alt="Harshika Agrawal" className="member-photo" />
              </a>
              <h3 className="member-name">Harshika Agrawal</h3>
              <p className="member-school">New Jersey Institute of Technology</p>
            </div>
          </div>

          <div className="mentors-section">
            <div className="mentor-box">
              <div className="mentor-label">Coach</div>
              <a href="https://www.linkedin.com/in/jennahunte/" target="_blank" rel="noopener noreferrer">
                <img src={JennaPhoto} alt="Jenna Hunte" className="mentor-photo" />
              </a>
              <div className="mentor-name">Jenna Hunte</div>
            </div>
            <div className="mentor-box">
              <div className="mentor-label">Challenge Advisor</div>
              <a href="https://www.linkedin.com/in/abdulwahabbb65/" target="_blank" rel="noopener noreferrer">
                <img src={AbdulPhoto} alt="Abdul" className="mentor-photo" />
              </a>
              <div className="mentor-name">Abdul (Accenture)</div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p>Accenture AI Studio Challenge - Fall 2025</p>
        <p>Break Through Tech AI</p>
      </footer>
    </div>
  );
}

export default Home;
