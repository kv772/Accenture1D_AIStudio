import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import './Navbar.css';

function Navbar() {
  const location = useLocation();
  const { isDark, toggleTheme } = useTheme();

  const scrollToSection = sectionId => {
    if (location.pathname !== '/') {
      window.location.href = `/#${sectionId}`;
    } else {
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <span className="logo-icon">🔍</span>
          <span className="logo-text">Accenture 1D Fake News Detector</span>
        </Link>

        <div className="navbar-menu">
          <Link to="/" className="navbar-link">
            Home
          </Link>
          <button
            onClick={() => scrollToSection('models')}
            className="navbar-link">
            Models
          </button>
          <button
            onClick={() => scrollToSection('team')}
            className="navbar-link">
            Our Team
          </button>
          <Link to="/analysis" className="navbar-link navbar-link-cta">
            Analyze
          </Link>
        </div>

        <button
          className="theme-toggle-nav"
          onClick={toggleTheme}
          aria-label="Toggle theme">
          {isDark ? '☀️' : '🌙'}
        </button>
      </div>
    </nav>
  );
}

export default Navbar;
