# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies

```bash
cd WEB
npm install
```

This will install:
- React 18.2
- Vite 5.0
- React Router 6

### 2. Start Development Server

```bash
npm run dev
```

The app will automatically open in your browser at `http://localhost:3000`

### 3. Explore the App

**Home Page** - Landing page with:
- Project overview
- Statistics (95%+ accuracy, 4 models, 40k+ articles)
- Feature highlights
- Theme toggle (light/dark mode)

**Try It Out:**
1. Click "Launch Analysis Tool"
2. Click "Load Demo Article" (or paste your own)
3. Click "Run Analysis"
4. View results with confidence scores and model breakdown

## 📁 Project Structure

```
WEB/
├── src/
│   ├── pages/           # Home, Analysis, Results
│   ├── context/         # Theme management
│   └── main.jsx         # App entry
├── index.html          # HTML template
└── package.json        # Dependencies
```

## 🎨 Features

✅ Dark/Light mode with auto-detection
✅ Responsive design (mobile/tablet/desktop)
✅ Animated loading states
✅ Demo mode with sample articles
✅ Beautiful UI inspired by NeuroScan

## 🔧 Build for Production

```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## 🌐 Deploy

### Vercel (Recommended)
```bash
npm install -g vercel
vercel
```

### Netlify
```bash
npm install -g netlify-cli
netlify deploy --prod
```

## 🔗 Integrate Your ML Models

Update `src/pages/Analysis.jsx` line ~30:

```javascript
const response = await fetch('YOUR_API_ENDPOINT', {
  method: 'POST',
  body: JSON.stringify({ text: articleText })
})
```

See [README.md](README.md) for detailed integration guide.

## 📚 Learn More

- **Full Documentation**: [README.md](README.md)
- **React**: https://react.dev
- **Vite**: https://vitejs.dev
- **React Router**: https://reactrouter.com

---

**Need help?** Check the main [README.md](README.md) or the project's dataset files.
