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

/* ---------- i18n ---------- */
function getLang() {
  return localStorage.getItem("alm-lang") === "fa" ? "fa" : "en";
}
function setLang(l) { localStorage.setItem("alm-lang", l); }

const STRINGS = {
  en: {
    find_level: "Find your level · تعیین سطح",
    lang_btn: "فا",
    audience: { student: "Student", executive: "Executive" },
    views: { map: "Skill tree", path: "Your path" },
    map_title: "The AI Learning Map",
    map_subtitle: "One coherent path from mathematical foundations to production AI systems — at the right depth for who you are. Each module, three ways in:",
    ing: {
      listen: { name: "Listen", tag: "for concepts",
        desc: "Watch or listen to the lecture — slides, video, or podcast." },
      play:   { name: "Play",   tag: "for intuition",
        desc: "Interactive course notes — read, then poke at the algorithms and watch them respond." },
      build:  { name: "Build",  tag: "for mastery",
        desc: "Implement it yourself in a guided notebook, with feedback." },
    },
    path_hints: {
      student:   "The full technical path. Foundations are assumed; brush up as needed.",
      executive: "The 6-week executive program — 3-hour sessions: interactive lectures + hands-on notebooks. Build something every week.",
    },
    ref_note: "Reference module — no original content here. Learn this from the excellent resources below.",
    back_to_map: "← back to the map",
    minimap_label: "you are here — click a node to travel",
    resources_title: "\u{1F4DA} Learn more",
    resources_sub: "Hand-picked videos, websites, and readings from other excellent teachers.",
    resources_empty: "Curated resources for this module are on their way.",
    ref_resources_title: "\u{1F4DA} Learn it from the best",
    ref_resources_sub: "No original content here for now — these excellent resources cover everything this map needs from",
    listen_choose: "Choose PDF source",
    listen_gen_pdf: "Generated PDF",
    listen_gen_pdf_sub: "Compiled from module content",
    listen_own_pdf: "Upload your PDF",
    listen_own_pdf_sub: "Use your own lecture notes",
    listen_gen_unavail: "Not built yet — run `make listen MODULE=",
    listen_open_pdf: "Open PDF",
    listen_view_pdf: "Viewing:",
    footer: "Sequence may evolve based on feedback and emerging technology.",
  },
  fa: {
    find_level: "تعیین سطح",
    lang_btn: "EN",
    audience: { student: "دانشجو", executive: "مدیر اجرایی" },
    views: { map: "درخت مهارت", path: "مسیر شما" },
    map_title: "نقشه یادگیری هوش مصنوعی",
    map_subtitle: "یک مسیر منسجم از مبانی ریاضی تا سیستم‌های هوش مصنوعی — با عمق مناسب. هر ماژول، سه روش:",
    ing: {
      listen: { name: "گوش ده",  tag: "برای مفاهیم",
        desc: "جلسات درس را تماشا کنید یا گوش دهید." },
      play:   { name: "بازی",          tag: "برای شهود",
        desc: "یادداشت‌های تعاملی — بخوانید، سپس با الگوریتم‌ها بازی کنید." },
      build:  { name: "بساز",          tag: "برای تسلط",
        desc: "خودتان پیاده‌سازی کنید در یک نوت‌بوک راهنما." },
    },
    path_hints: {
      student:   "مسیر کامل فنی. مبانی فرض شده‌اند؛ در صورت لزوم مرور کنید.",
      executive: "برنامه شش هفته‌ای — جلسات سه ساعته. هر هفته یک پروژه بسازید.",
    },
    ref_note: "ماژول مرجع — محتوای اصلی ندارد. از منابع عالی زیر یاد بگیرید.",
    back_to_map: "→ بازگشت به نقشه",
    minimap_label: "شما اینجایید — روی یک گره کلیک کنید",
    resources_title: "\u{1F4DA} بیشتر بیاموزید",
    resources_sub: "ویدیوها، وبسایت‌ها و منابع انتخابی.",
    resources_empty: "منابع بهزودی برای این ماژول در راه است.",
    ref_resources_title: "\u{1F4DA} از بهترین‌ها یاد بگیرید",
    ref_resources_sub: "این منابع عالی هرآنچه نیاز است را پوشش می‌دهند:",
    listen_choose: "منبع PDF را انتخاب کنید",
    listen_gen_pdf: "فایل PDF تولیدی",
    listen_gen_pdf_sub: "ساخته شده از محتوای ماژول",
    listen_own_pdf: "آپلود PDF خودتان",
    listen_own_pdf_sub: "از جزوات درسی خود استفاده کنید",
    listen_gen_unavail: "هنوز ساخته نشده — اجرا کنید: make listen MODULE=",
    listen_open_pdf: "باز کردن PDF",
    listen_view_pdf: "در حال نمایش:",
    footer: "ترتیب براساس بازخورد تغییر میکند.",
  }
};

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
  { id:"f-la", tier:"foundations",
    title:"Linear Algebra", sub:"vectors → SVD",
    title_fa:"جبر خطی", sub_fa:"بردارها تا SVD",
    ref:true, aud:["student","executive"], prereqs:[],
    blurb:"Vectors, matrices, eigendecomposition, SVD and low-rank approximation — the language every model is written in.",
    resources:[
      { title:"3Blue1Brown — Essence of Linear Algebra", type:"video", url:"https://www.3blue1brown.com/topics/linear-algebra" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-calc", tier:"foundations",
    title:"Calculus", sub:"gradients & chain rule",
    title_fa:"حساب دیفرانسیل", sub_fa:"گرادیان و قانون زنجیره",
    ref:true, aud:["student","executive"], prereqs:[],
    blurb:"Multivariable differentiation, gradients, and the chain rule — everything backprop quietly assumes you know.",
    resources:[
      { title:"3Blue1Brown — Essence of Calculus", type:"video", url:"https://www.3blue1brown.com/topics/calculus" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-prob", tier:"foundations",
    title:"Probability & Statistics", sub:"uncertainty & inference",
    title_fa:"احتمال و آمار", sub_fa:"عدم قطعیت و استنتاج",
    ref:true, aud:["student","executive"], prereqs:[],
    blurb:"Distributions, expectation, estimation, and statistical inference — why ML claims anything at all about unseen data.",
    resources:[
      { title:"Khan Academy — Statistics & Probability", type:"website", url:"https://www.khanacademy.org/math/statistics-probability" },
      { title:"Seeing Theory — visual introduction to probability", type:"website", url:"https://seeing-theory.brown.edu" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-opt", tier:"foundations",
    title:"Optimization", sub:"gradient descent & friends",
    title_fa:"بهینه‌سازی", sub_fa:"گرادیان کاهشی",
    ref:true, aud:["student","executive"], prereqs:[],
    blurb:"Loss landscapes, convexity, gradient descent, and constrained optimization — how learning actually happens.",
    resources:[
      { title:"Khan Academy — Multivariable Calculus (optimization units)", type:"website", url:"https://www.khanacademy.org/math/multivariable-calculus" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-prog", tier:"foundations",
    title:"Programming", sub:"Python, NumPy, Git",
    title_fa:"برنامه‌نویسی", sub_fa:"پایتون و NumPy",
    ref:true, aud:["student","executive"], prereqs:[],
    blurb:"Python, NumPy/pandas, data munging, visualization, Git and the terminal — the workshop where everything gets built.",
    resources:[
      { title:"Programming with Mosh — Python for Beginners", type:"video", url:"https://www.youtube.com/watch?v=_uQrJ0TkZlc" },
      { title:"Harvard CS50P — Introduction to Programming with Python", type:"website", url:"https://cs50.harvard.edu/python/" },
      { title:"Automate the Boring Stuff with Python", type:"book", url:"https://automatetheboringstuff.com" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  // ---- Core ----
  { id:"c-intro", tier:"core", mini:true,
    title:"Intro to ML", sub:"the what & why",
    title_fa:"مقدمه یادگیری ماشین", sub_fa:"چیستی و چرایی",
    aud:["student","executive"], prereqs:["f-la","f-calc","f-prob","f-opt","f-prog"],
    blurb:"What machine learning actually is, where it sits in the broader AI landscape, when to use it (and when not to), and the common thread across all the methods ahead.",
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"c-ml", tier:"core",
    title:"Classical ML", sub:"KNN → ensembles → SVMs",
    title_fa:"یادگیری ماشین کلاسیک", sub_fa:"KNN تا SVM",
    aud:["student","executive"], prereqs:["c-intro"],
    blurb:"Supervised learning end-to-end: KNN, decision trees, ensembles, linear regression, linear classification, logistic regression and SVMs. The running thread: linear models and linear separability — and where they break.",
    resources:[
      { title:"StatQuest — Machine Learning playlists", type:"video", url:"https://statquest.org" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"c-nn", tier:"core",
    title:"Neural Networks", sub:"beyond linear separability",
    title_fa:"شبکه‌های عصبی", sub_fa:"فراتر از جداسازی خطی",
    aud:["student","executive"], prereqs:["c-ml"],
    blurb:"Linear models as neural nets, activations and nonlinearity, backprop intuition, loss functions, gradient descent. The answer to “what if the data isn’t linearly separable?”",
    resources:[
      { title:"3Blue1Brown — Neural Networks", type:"video", url:"https://www.3blue1brown.com/topics/neural-networks" },
      { title:"Karpathy — Neural Networks: Zero to Hero", type:"video", url:"https://karpathy.ai/zero-to-hero.html" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  // ---- Applied ----
  { id:"a-vision", tier:"applied",
    title:"Computer Vision", sub:"CNNs & transfer learning",
    title_fa:"بینایی ماشین", sub_fa:"CNN و انتقال یادگیری",
    aud:["student","executive"], prereqs:["c-nn"],
    blurb:"CNNs and feature hierarchies, convolution / pooling / normalization, transfer learning, data augmentation, evaluation.",
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"a-nlp", tier:"applied",
    title:"Language Models", sub:"embeddings → transformers",
    title_fa:"مدل‌های زبانی", sub_fa:"تعبیه‌ها تا ترانسفورمر",
    aud:["student","executive"], prereqs:["c-nn"],
    blurb:"Sequence models, embeddings, attention, transformers, pretraining and fine-tuning, decoding. From n-grams to LLMs.",
    resources:[
      { title:"Jay Alammar — The Illustrated Transformer", type:"website", url:"https://jalammar.github.io/illustrated-transformer/" },
      { title:"3Blue1Brown — Neural Networks (attention & transformers chapters)", type:"video", url:"https://www.3blue1brown.com/topics/neural-networks" },
      { title:"Karpathy — Neural Networks: Zero to Hero (GPT lectures)", type:"video", url:"https://karpathy.ai/zero-to-hero.html" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"a-agents", tier:"applied",
    title:"AI Agents", sub:"reasoning, tools, memory",
    title_fa:"عامل‌های هوش مصنوعی", sub_fa:"استدلال، ابزار، حافظه",
    aud:["executive"], prereqs:["a-nlp"],
    blurb:"Agents and reasoning, planning and memory, tool / function calling, workflows and orchestration, multi-agent concepts.",
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"a-systems", tier:"applied",
    title:"AI Systems & MLOps", sub:"models → production",
    title_fa:"سیستم‌های هوش مصنوعی", sub_fa:"مدل‌ها تا تولید",
    aud:["executive"], prereqs:["a-agents","a-vision"],
    blurb:"From models to systems: retrieval and RAG, APIs and services, end-to-end evaluation, reliability and observability.",
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  // ---- Advanced ----
  { id:"x-prob", tier:"advanced",
    title:"Probabilistic & Causal AI", sub:"Bayes, uncertainty, causality",
    title_fa:"هوش مصنوعی احتمالی", sub_fa:"بیز، عدم قطعیت، علیت",
    aud:["student"], prereqs:["c-ml"],
    blurb:"Probabilistic learning, Bayesian modeling, uncertainty quantification, and causal reasoning — growing into a full module.",
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"x-unsup", tier:"advanced",
    title:"Unsupervised Learning", sub:"PCA, clustering, embeddings",
    title_fa:"یادگیری بدون نظارت", sub_fa:"PCA، خوشه‌بندی",
    aud:["student"], prereqs:["c-ml"],
    blurb:"Dimensionality reduction, clustering, and representation learning — finding structure without labels.",
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"x-rl", tier:"advanced",
    title:"Reinforcement Learning", sub:"primer → module",
    title_fa:"یادگیری تقویتی", sub_fa:"مقدمه",
    aud:["student"], prereqs:["c-nn"],
    blurb:"From a brief primer to a full module: MDPs, value functions, policy gradients.",
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"x-rai", tier:"advanced",
    title:"Responsible AI", sub:"governance, risk, compliance",
    title_fa:"هوش مصنوعی مسئولانه", sub_fa:"حاکمیت و ریسک",
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
  "c-intro":  { x: 50, y: 0.62 },  // mini gateway between foundations and core
  "c-ml":     { x: 30, y: 1.25 },
  "c-nn":     { x: 52, y: 2.1 },
  "x-unsup":  { x: 10, y: 2.65 },
  "a-vision": { x: 30, y: 3.05 },
  "a-nlp":    { x: 62, y: 3.05 },
  "x-prob":   { x: 12, y: 3.65 },
  "a-agents": { x: 82, y: 3.8 },
  "x-rl":     { x: 36, y: 4.1 },
  "a-systems":{ x: 62, y: 4.55 },
  "x-rai":    { x: 82, y: 5.25 },
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
      { id:"c-intro",  where:"Week 1 intro" },
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
      { id:"c-intro",   where:"Session 0" },
      { id:"c-ml",      where:"Week 1" },
      { id:"c-nn",      where:"Week 2" },
      { id:"a-vision",  where:"Week 3" },
      { id:"a-nlp",     where:"Week 4" },
      { id:"a-agents",  where:"Week 5" },
      { id:"a-systems", where:"Week 6" },
      { id:"x-rai",     where:"Bonus", bonus:true },
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
