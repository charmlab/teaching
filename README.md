# The AI Learning Map

**[teaching.amirhkarimi.com](https://teaching.amirhkarimi.com)**

An interactive skill-tree for navigating artificial intelligence — built for three audiences: the curious public, university students, and executives leading AI-driven organizations.

---

## Vision

Most AI education is siloed. Courses for students assume you're aiming for a PhD. Executive briefings skip the substance. Public-facing content rarely goes deep enough to be useful.

This learning map tries something different: a single, shared knowledge graph where each audience follows their own path through the same underlying ideas. A concept like "decision trees" looks different depending on whether you're building intuition, implementing it for an assignment, or deciding whether to deploy it in your organization — and the map reflects that.

**Three audiences, one map:**

- **General public** — build genuine intuition about how AI works, without the math overhead. Follow the public path to understand what's real and what's hype.
- **ECE 657 students** — a structured guide through the full curriculum, with interactive playgrounds, lecture notes, and assignments for each module.
- **Executives & CTOs** — a focused path through the concepts that matter most for strategy, governance, and leading technical teams.

---

## About the Instructor

Amir H. Karimi is an Assistant Professor of Machine Learning at the University of Waterloo, where he holds the O'Donovan Chair in Trustworthy AI and directs the [CHARM Lab](https://charm-lab.com) (Collaborative Human-AI Reasoning Machines). His career spans top academic institutions (Toronto, Waterloo, Stanford, ETH Zürich, Max Planck), leading industry labs (Google Brain, DeepMind, Meta AI), and major tech firms (Meta, BlackBerry).

He received his PhD from ETH Zürich and the Max Planck Institute for Intelligent Systems, earning the **ETH Zurich Medal** for outstanding doctoral performance and the **Google PhD Fellowship in AI for Social Good** (one of 17 awarded globally). He holds the **Alumni Gold Medal** — the University of Waterloo's top master's award.

As an educator, Amir has taught **over 40,000 learners** across in-person university courses and free online content on YouTube, Instagram, Substack, and Medium. In his very first teaching term at Waterloo, he received the **Igor Ivkovic Teaching Excellence Award** — a competitive, student-nominated honour recognizing exceptional teaching across the Faculty of Engineering.

---

## So what will you learn?

The map covers the arc of modern AI — from foundational ideas to the systems shaping the world today:

- **How machines learn from data** — the principles behind every model you've heard of
- **Classical ML** — nearest neighbors, decision trees, ensembles, and why they still matter
- **Neural networks & deep learning** — from perceptrons to the architectures powering GPT
- **Computer vision & language models** — how machines see, read, and generate
- **Trustworthy AI** — fairness, robustness, explainability, and what it means to deploy AI responsibly
- **AI governance & agents** — the policy landscape and where autonomous systems are headed

Each module has four layers: **Listen** (lecture), **Play** (interactive demo), **Build** (code), and **Quiz** (self-check). You can go as shallow or as deep as your goals require.

---

## Running locally

```bash
python3 -m http.server 8787
# open http://localhost:8787
```

No build step. Pure HTML/CSS/JS.

---

## Key files

| File | Purpose |
|------|---------|
| `index.html` | Interactive skill-tree map |
| `module.html` | Per-module page (Listen / Play / Build / Quiz) |
| `data.js` | All module content, layout, and audience logic |
| `styles.css` | Shared styles |
| `modules/{id}/play.html` | Interactive playground per module |
| `modules/{id}/assets/` | Lecture PDFs (public-safe only; raw sources gitignored) |

**Security note:** Raw lecture PDFs live in `modules/*/sources/` and `modules/*/out/` — both gitignored. Only processed assets in `modules/*/assets/` are tracked.
