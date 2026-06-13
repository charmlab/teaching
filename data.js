/* ============================================================
   Shared data for The AI Learning Map (teaching.amirhkarimi.com)
   ============================================================ */

/* Google Analytics 4 — same property as amirhkarimi.com.
   For a separate teaching property, create a new GA4 data stream
   and swap the ID here. */
const GA_ID = "G-PLH5YY7PQS";
window.dataLayer = window.dataLayer || [];
function gtag(){ dataLayer.push(arguments); }
(function () {
  if (!GA_ID || GA_ID.indexOf("XXXX") !== -1) return;
  function load() {
    if (window.__gaLoaded) return;
    window.__gaLoaded = true;
    var s = document.createElement("script");
    s.async = true;
    s.src = "https://www.googletagmanager.com/gtag/js?id=" + GA_ID;
    document.head.appendChild(s);
    gtag("js", new Date());
    gtag("config", GA_ID);
  }
  if ("requestIdleCallback" in window) {
    window.addEventListener("load", function () { requestIdleCallback(load, { timeout: 3000 }); }, { once: true });
  } else {
    window.addEventListener("load", load, { once: true });
  }
})();
function track(name, params) {
  try { gtag("event", name, Object.assign({ transport_type: "beacon" }, params || {})); } catch (e) {}
}

const ING_META = {
  listen: { icon: "\u{1F3A7}", name: "Listen", tag: "for concepts",
    desc: "Watch or listen to the lecture — slides, video, or podcast." },
  play:   { icon: "\u{1F3AE}", name: "Play", tag: "for intuition",
    desc: "Interactive course notes — read, then poke at the algorithms and watch them respond." },
  build:  { icon: "\u{1F528}", name: "Build", tag: "for mastery",
    desc: "Implement it yourself in a guided notebook, with feedback." },
};

const AUD_LABEL = { student: "Student", executive: "Executive" };

const MODULES = [
  // ---- Foundations ----
  { id:"f-la", tier:"foundations", title:"Linear Algebra", sub:"vectors → SVD", ref:true,
    aud:["student","executive"], prereqs:[],
    blurb:"Vectors, matrices, eigendecomposition, SVD and low-rank approximation — the language every model is written in.",
    resources:[
      { title:"3Blue1Brown — Essence of Linear Algebra", type:"video", url:"https://www.3blue1brown.com/topics/linear-algebra" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },
  { id:"f-calc", tier:"foundations", title:"Calculus", sub:"gradients & chain rule", ref:true,
    aud:["student","executive"], prereqs:[],
    blurb:"Multivariable differentiation, gradients, and the chain rule — everything backprop quietly assumes you know.",
    resources:[
      { title:"3Blue1Brown — Essence of Calculus", type:"video", url:"https://www.3blue1brown.com/topics/calculus" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },
  { id:"f-prob", tier:"foundations", title:"Probability & Statistics", sub:"uncertainty & inference", ref:true,
    aud:["student","executive"], prereqs:[],
    blurb:"Distributions, expectation, estimation, and statistical inference — why ML claims anything at all about unseen data.",
    resources:[
      { title:"Khan Academy — Statistics & Probability", type:"website", url:"https://www.khanacademy.org/math/statistics-probability" },
      { title:"Seeing Theory — visual introduction to probability", type:"website", url:"https://seeing-theory.brown.edu" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },
  { id:"f-opt", tier:"foundations", title:"Optimization", sub:"gradient descent & friends", ref:true,
    aud:["student","executive"], prereqs:[],
    blurb:"Loss landscapes, convexity, gradient descent, and constrained optimization — how learning actually happens.",
    resources:[
      { title:"Khan Academy — Multivariable Calculus (optimization units)", type:"website", url:"https://www.khanacademy.org/math/multivariable-calculus" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },
  { id:"f-prog", tier:"foundations", title:"Programming", sub:"Python, NumPy, Git", ref:true,
    aud:["student","executive"], prereqs:[],
    blurb:"Python, NumPy/pandas, data munging, visualization, Git and the terminal — the workshop where everything gets built.",
    resources:[
      { title:"Programming with Mosh — Python for Beginners", type:"video", url:"https://www.youtube.com/watch?v=_uQrJ0TkZlc" },
      { title:"Harvard CS50P — Introduction to Programming with Python", type:"website", url:"https://cs50.harvard.edu/python/" },
      { title:"Automate the Boring Stuff with Python", type:"book", url:"https://automatetheboringstuff.com" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  // ---- Core ----
  { id:"c-ml", tier:"core", title:"Classical ML", sub:"KNN → ensembles → SVMs",
    aud:["student","executive"], prereqs:["f-la","f-calc","f-prob","f-opt","f-prog"],
    blurb:"Supervised learning end-to-end: KNN, decision trees, ensembles, linear regression, linear classification, logistic regression and SVMs. The running thread: linear models and linear separability — and where they break.",
    resources:[
      { title:"StatQuest — Machine Learning playlists", type:"video", url:"https://statquest.org" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },
  { id:"c-nn", tier:"core", title:"Neural Networks", sub:"beyond linear separability",
    aud:["student","executive"], prereqs:["c-ml"],
    blurb:"Linear models as neural nets, activations and nonlinearity, backprop intuition, loss functions, gradient descent. The answer to “what if the data isn’t linearly separable?”",
    resources:[
      { title:"3Blue1Brown — Neural Networks", type:"video", url:"https://www.3blue1brown.com/topics/neural-networks" },
      { title:"Karpathy — Neural Networks: Zero to Hero", type:"video", url:"https://karpathy.ai/zero-to-hero.html" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  // ---- Applied ----
  { id:"a-vision", tier:"applied", title:"Computer Vision", sub:"CNNs & transfer learning",
    aud:["student","executive"], prereqs:["c-nn"],
    blurb:"CNNs and feature hierarchies, convolution / pooling / normalization, transfer learning, data augmentation, evaluation.",
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },
  { id:"a-nlp", tier:"applied", title:"Language Models", sub:"embeddings → transformers",
    aud:["student","executive"], prereqs:["c-nn"],
    blurb:"Sequence models, embeddings, attention, transformers, pretraining and fine-tuning, decoding. From n-grams to LLMs.",
    resources:[
      { title:"Jay Alammar — The Illustrated Transformer", type:"website", url:"https://jalammar.github.io/illustrated-transformer/" },
      { title:"3Blue1Brown — Neural Networks (attention & transformers chapters)", type:"video", url:"https://www.3blue1brown.com/topics/neural-networks" },
      { title:"Karpathy — Neural Networks: Zero to Hero (GPT lectures)", type:"video", url:"https://karpathy.ai/zero-to-hero.html" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },
  { id:"a-agents", tier:"applied", title:"AI Agents", sub:"reasoning, tools, memory",
    aud:["executive"], prereqs:["a-nlp"],
    blurb:"Agents and reasoning, planning and memory, tool / function calling, workflows and orchestration, multi-agent concepts.",
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },
  { id:"a-systems", tier:"applied", title:"AI Systems & MLOps", sub:"models → production",
    aud:["executive"], prereqs:["a-agents","a-vision"],
    blurb:"From models to systems: retrieval and RAG, APIs and services, end-to-end evaluation, reliability and observability.",
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  // ---- Advanced ----
  { id:"x-prob", tier:"advanced", title:"Probabilistic & Causal AI", sub:"Bayes, uncertainty, causality",
    aud:["student"], prereqs:["c-ml"],
    blurb:"Probabilistic learning, Bayesian modeling, uncertainty quantification, and causal reasoning — growing into a full module.",
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },
  { id:"x-unsup", tier:"advanced", title:"Unsupervised Learning", sub:"PCA, clustering, embeddings",
    aud:["student"], prereqs:["c-ml"],
    blurb:"Dimensionality reduction, clustering, and representation learning — finding structure without labels.",
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },
  { id:"x-rl", tier:"advanced", title:"Reinforcement Learning", sub:"primer → module",
    aud:["student"], prereqs:["c-nn"],
    blurb:"From a brief primer to a full module: MDPs, value functions, policy gradients.",
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },
  { id:"x-rai", tier:"advanced", title:"Responsible AI", sub:"governance, risk, compliance",
    aud:["executive"], prereqs:["a-systems"],
    blurb:"Lead AI responsibly: governance, risk and compliance, responsible-AI principles, privacy, security and policy, industry cases.",
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },
];

const byId = Object.fromEntries(MODULES.map(m => [m.id, m]));

/* One shared layout for all tracks.
   x = horizontal position (% of canvas width, card center)
   y = vertical level (0 = bottom row, larger = higher up)  */
const LAYOUT = {
  "f-la":     { x: 10, y: 0 },
  "f-calc":   { x: 30, y: 0 },
  "f-prob":   { x: 50, y: 0 },
  "f-opt":    { x: 70, y: 0 },
  "f-prog":   { x: 90, y: 0 },
  "c-ml":     { x: 30, y: 1.15 },
  "c-nn":     { x: 52, y: 1.95 },   // slightly above Classical ML
  "x-unsup":  { x: 10, y: 2.5 },
  "a-vision": { x: 30, y: 2.9 },
  "a-nlp":    { x: 62, y: 2.9 },
  "x-prob":   { x: 12, y: 3.5 },
  "a-agents": { x: 82, y: 3.65 },   // slightly above Language Models
  "x-rl":     { x: 36, y: 3.95 },
  "a-systems":{ x: 62, y: 4.4 },
  "x-rai":    { x: 82, y: 5.1 },
};

const EDGES = (() => {
  const out = [];
  for (const m of MODULES) {
    if (!(m.id in LAYOUT)) continue;
    for (const p of m.prereqs) if (p in LAYOUT) out.push([p, m.id]);
  }
  return out;
})();

const PATHS = {
  student: {
    hint: "The full technical path. Foundations are assumed; brush up as needed.",
    steps: [
      { id:"c-ml",     where:"Weeks 1–4" },
      { id:"c-nn",     where:"Weeks 5–6" },
      { id:"a-vision", where:"2 lectures" },
      { id:"a-nlp",    where:"NLP + transformers" },
      { id:"x-prob",   where:"Advanced lectures" },
      { id:"x-unsup",  where:"Advanced lectures" },
      { id:"x-rl",     where:"Primer" },
    ],
  },
  executive: {
    hint: "The 6-week executive program — 3-hour sessions: interactive lectures + hands-on notebooks. Build something every week.",
    steps: [
      { id:"c-ml",     where:"Week 1" },
      { id:"c-nn",     where:"Week 2" },
      { id:"a-vision", where:"Week 3" },
      { id:"a-nlp",    where:"Week 4" },
      { id:"a-agents", where:"Week 5" },
      { id:"a-systems",where:"Week 6" },
      { id:"x-rai",    where:"Bonus", bonus:true },
    ],
  },
};

function getAudience() {
  const a = localStorage.getItem("alm-audience");
  if (a === "practitioner") return "executive"; // migrate old value
  return (a === "student" || a === "executive") ? a : "student";
}
function setAudience(a) { localStorage.setItem("alm-audience", a); }

/* BFS shortest path between two modules (undirected prereq graph) */
function shortestPath(from, to) {
  if (from === to) return [from];
  const adj = {};
  for (const [a, b] of EDGES) {
    (adj[a] = adj[a] || []).push(b);
    (adj[b] = adj[b] || []).push(a);
  }
  const prev = { [from]: null };
  const queue = [from];
  while (queue.length) {
    const cur = queue.shift();
    for (const nxt of adj[cur] || []) {
      if (nxt in prev) continue;
      prev[nxt] = cur;
      if (nxt === to) {
        const path = [to];
        let p = cur;
        while (p !== null) { path.push(p); p = prev[p]; }
        return path.reverse();
      }
      queue.push(nxt);
    }
  }
  return [from, to];
}
