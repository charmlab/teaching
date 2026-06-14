# ECE 457B / 657 — Teaching Repository

**University of Waterloo · Dr. Amir-Hossein Karimi**

This repository powers two things:

## 1. AI Learning Map — [teaching.amirhkarimi.com](https://teaching.amirhkarimi.com)

An interactive skill-tree website for students, executives, and the general public to
navigate AI/ML topics. Built as a static site (HTML/CSS/JS, no build step).

**Key files:**
- `index.html` — skill tree map
- `module.html` — per-module page (Listen / Play / Build / Quiz)
- `data.js` — all module content, layout, audience logic
- `styles.css` — shared styles
- `modules/{id}/play.html` — interactive playground per module (drop-in)
- `modules/{id}/assets/` — lecture PDFs (gitignored for raw sources)

**To run locally:**
```bash
python3 -m http.server 8787
# open http://localhost:8787
```

## 2. Assignment Starter Files (ECE 457B / 657)

Starter notebooks for course assignments. Each folder contains partially implemented
code with `TODO` sections.

```
assignments/
  assignment-2/   jax.ipynb
  assignment-3/   MLE.ipynb, Gaussian_Process_Regression.ipynb, RNN_Music_Generation.ipynb
  assignment-4/   AE.ipynb, bandit_Q.ipynb, Kaggle_starter_code.ipynb
```

**Dependencies:** Python, `jax`, `numpy`, `torch` (see individual notebooks).

---

**Instructor:** Prof. Amir-Hossein Karimi  
**Teaching Assistant:** Arezoo Alipanah  
**Course:** ECE 457B / 657 — Foundations of Computational Intelligence, Spring 2025
