# Fake News Detector - React Web App

A beautiful, modern React web application for detecting fake news articles using AI and machine learning. Built for the **Accenture 1D AI Studio Challenge** with a sleek UI inspired by NeuroScan.

## ✨ Features

- **Modern UI/UX**: Clean, professional interface with smooth animations
- **Dark/Light Mode**: Automatic theme detection with manual toggle
- **Real-time Analysis**: Analyze news articles with AI-powered detection
- **Multiple Models**: Displays results from BERT, LSTM, Logistic Regression, and Baseline models
- **Interactive Results**: Detailed breakdown of predictions and confidence scores
- **Demo Mode**: Try the app with pre-loaded sample articles
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites

- Node.js 18 or higher
- npm or yarn

### Installation

1. **Install dependencies**
   ```bash
   cd WEB
   npm install
   ```

2. **Start development server**
   ```bash
   npm run dev
   ```

3. **Open in browser**
   ```
   http://localhost:3000
   ```

## 📁 Project Structure

```
WEB/
├── index.html              # HTML entry point
├── package.json            # Dependencies
├── vite.config.js          # Vite configuration
├── src/
│   ├── main.jsx            # App entry point
│   ├── App.jsx             # Main app with routing
│   ├── App.css             # Global app styles
│   ├── index.css           # Base styles
│   ├── context/
│   │   └── ThemeContext.jsx # Theme management
│   └── pages/
│       ├── Home.jsx        # Landing page
│       ├── Home.css
│       ├── Analysis.jsx    # Article input
│       ├── Analysis.css
│       ├── Results.jsx     # Results display
│       └── Results.css
```

## 🎨 Design System

### Color Palette

#### Light Theme
- Primary: `#2563EB` (Blue)
- Secondary: `#10B981` (Green)
- Accent: `#8B5CF6` (Purple)
- Background: `#FFFFFF`
- Surface: `#F8F9FA`

#### Dark Theme
- Primary: `#3B82F6` (Light Blue)
- Secondary: `#10B981` (Green)
- Accent: `#A78BFA` (Light Purple)
- Background: `#0F172A` (Dark Blue)
- Surface: `#1E293B` (Dark Gray)

### Typography
- Font Family: System fonts (-apple-system, Roboto, etc.)
- Headings: 700-800 weight (bold/extra bold)
- Body: 400-600 weight (regular/semi-bold)

## 📱 Pages

### 1. Home Page (`/`)
- Hero section with project branding
- Statistics display (95%+ accuracy, 4 models, 40k+ articles)
- Feature highlights
- Technology badges (BERT, LSTM, TensorFlow, PyTorch)
- Theme toggle

### 2. Analysis Page (`/analysis`)
- Single/Paste text tabs
- Demo article loader
- Article title input (optional)
- Article text input (required)
- "How It Works" information
- Run Analysis button

### 3. Results Page (`/results`)
**Loading State:**
- Animated progress bar
- Step-by-step indicators

**Results State:**
- Classification result (Real/Fake)
- Confidence score with visual bar
- Interpretation text
- Individual model breakdowns
- Article preview
- Action buttons

## 🔧 Tech Stack

- **React 18.2**: UI library
- **Vite 5.0**: Build tool and dev server
- **React Router 6**: Client-side routing
- **CSS Variables**: Dynamic theming
- **LocalStorage**: Theme persistence

## 🌐 Integrating Your ML Models

Currently, the app uses mock predictions. To integrate your actual ML models:

### Option 1: Backend API (Recommended)

1. **Create an API endpoint** for your models
2. **Update the analysis function** in `src/pages/Analysis.jsx`:

```javascript
const handleAnalysis = async () => {
  if (!articleText.trim()) {
    alert('Please enter article text to analyze')
    return
  }

  setIsAnalyzing(true)

  try {
    const response = await fetch('https://your-api.com/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: articleText,
        title: articleTitle,
      }),
    })

    const data = await response.json()

    navigate('/results', {
      state: {
        article: articleText,
        title: articleTitle,
        prediction: data.prediction,
        confidence: data.confidence,
        modelScores: data.modelScores
      }
    })
  } catch (error) {
    console.error('Analysis failed:', error)
    alert('Failed to analyze article')
  } finally {
    setIsAnalyzing(false)
  }
}
```

3. **Update Results.jsx** to use real data instead of mock data

### Expected API Response Format

```json
{
  "prediction": "Real" | "Fake",
  "confidence": 87.3,
  "modelScores": {
    "bert": 89.2,
    "lstm": 85.7,
    "logistic": 82.1,
    "baseline": 80.5
  }
}
```

### Option 2: Serverless Functions

Deploy your models as serverless functions on:
- Vercel Functions
- Netlify Functions
- AWS Lambda
- Google Cloud Functions

### Option 3: Static Demo

Keep the mock data for demonstration purposes and add a disclaimer.

## 🎯 Customization

### Change Colors

Edit the CSS variables in `src/pages/Home.css`:

```css
[data-theme="light"] {
  --color-primary: #YOUR_COLOR;
  /* ... other colors */
}
```

### Update Statistics

Edit `src/pages/Home.jsx`:

```javascript
<div className="stat-value primary">YOUR_VALUE</div>
<div className="stat-label">YOUR_LABEL</div>
```

### Modify Content

All text content is in the JSX files - simply edit the strings in:
- `src/pages/Home.jsx`
- `src/pages/Analysis.jsx`
- `src/pages/Results.jsx`

## 📦 Build for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## 🚀 Deployment

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

### Deploy to Netlify

```bash
npm install -g netlify-cli
netlify deploy --prod
```

### Deploy to GitHub Pages

1. Update `vite.config.js`:
   ```javascript
   export default defineConfig({
     base: '/your-repo-name/',
     // ... other config
   })
   ```

2. Build and deploy:
   ```bash
   npm run build
   gh-pages -d dist
   ```

## 🧪 Development

### Run Development Server
```bash
npm run dev
```

### Lint Code
```bash
npm run lint
```

### Format Code
```bash
npm run format
```

## 🎓 Team & Credits

### Development Team
- Lin Zhang
- Kashvi Patel
- Nancy Huang
- Adriena Jiang
- Ousman Baldeh
- Sanskriti Sharma
- Harshika Patel

### Mentorship
- **Coach**: Jenna Hunte
- **Challenge Advisor**: Abdul (Accenture)

### Program
- Accenture AI Studio Challenge
- Break Through Tech AI
- Fall 2025

## 📚 Additional Resources

- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [React Router Documentation](https://reactrouter.com/)

## 🔗 Related Files

- Dataset: `../Fake News Detection Datasets/`
- Models: BERT, LSTM, Logistic Regression, Baseline
- Notebook: `../Accenture_1D_Model.ipynb`
- Main README: `../README.md`

## 📝 License

This project is part of the Accenture AI Studio Challenge - Fall 2025.

---

**Built with ❤️ using React and Vite**
