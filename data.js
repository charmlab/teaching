/* ============================================================
   Shared data for The AI Learning Map (teaching.amirhkarimi.com)
   ============================================================ */

/* Google Analytics 4 */
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

/* ---------- Audience ---------- */
const AUD_LABEL   = { student: "Student",   executive: "Executive",        public: "Public"         };
const AUD_TAGLINE = { student: "build AI",  executive: "lead with AI",     public: "use AI"         };
const AUD_TAGLINE_FA = { student: "هوش مصنوعی بسازید", executive: "با هوش مصنوعی رهبری کنید", public: "از هوش مصنوعی استفاده کنید" };

/* ---------- Tier display names (3-tier: foundations / concepts / applications) ---------- */
const TIER_LABEL = {
  foundations: "Foundations",
  concepts:    "Concepts",
  applied:     "Applications",
};
const TIER_LABEL_FA = {
  foundations: "مبانی",
  concepts:    "مفاهیم",
  applied:     "کاربردها",
};

/* ---------- i18n string tables ---------- */
const STRINGS = {
  en: {
    find_level: "Find your level · تعیین سطح",
    lang_btn: "فا",
    audience: { student: "Student", executive: "Executive", public: "Public" },
    tagline:  { student: "build AI", executive: "lead with AI", public: "use AI" },
    tier_label: { foundations:"Foundations", concepts:"Concepts", applied:"Applications" },
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
      public:    "Start here. No technical background required — just curiosity about AI.",
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
    listen_open_pdf: "Open PDF",
    listen_view_pdf: "Viewing:",
    footer: "Sequence may evolve based on feedback and emerging technology.",
  },
  fa: {
    find_level: "تعیین سطح",
    lang_btn: "EN",
    audience: { student: "دانشجو", executive: "مدیر اجرایی", public: "عموم" },
    tagline:  { student: "هوش مصنوعی بسازید", executive: "با هوش مصنوعی رهبری کنید", public: "از هوش مصنوعی استفاده کنید" },
    tier_label: { foundations:"مبانی", concepts:"مفاهیم", applied:"کاربردها" },
    views: { map: "درخت مهارت", path: "مسیر شما" },
    map_title: "نقشه یادگیری هوش مصنوعی",
    map_subtitle: "یک مسیر منسجم از مبانی ریاضی تا سیستم‌های هوش مصنوعی — با عمق مناسب برای شما. هر ماژول، سه روش:",
    ing: {
      listen: { name: "گوش بده",  tag: "برای مفاهیم",
        desc: "جلسات درس را تماشا کنید یا گوش دهید." },
      play:   { name: "بازی کن",    tag: "برای شهود",
        desc: "یادداشت‌های تعاملی — بخوانید، سپس با الگوریتم‌ها بازی کنید." },
      build:  { name: "بساز",    tag: "برای تسلط",
        desc: "خودتان پیاده‌سازی کنید در یک نوت‌بوک راهنما." },
    },
    path_hints: {
      student:   "مسیر کامل فنی. مبانی فرض شده‌اند؛ در صورت لزوم مرور کنید.",
      executive: "برنامه شش هفته‌ای — جلسات سه ساعته. هر هفته یک پروژه بسازید.",
      public:    "از اینجا شروع کنید. پیش‌زمینه فنی لازم نیست — فقط کنجکاوی درباره AI.",
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
    listen_open_pdf: "باز کردن PDF",
    listen_view_pdf: "در حال نمایش:",
    footer: "ترتیب براساس بازخورد تغییر می‌کند.",
  }
};

const ING_META = {
  listen: { icon: "\u{1F3A7}", name: "Listen", tag: "for concepts",
    desc: "Watch or listen to the lecture — slides, video, or podcast." },
  play:   { icon: "\u{1F3AE}", name: "Play",   tag: "for intuition",
    desc: "Interactive course notes — read, then poke at the algorithms and watch them respond." },
  build:  { icon: "\u{1F528}", name: "Build",  tag: "for mastery",
    desc: "Implement it yourself in a guided notebook, with feedback." },
};

const MODULES = [
  // ---- Foundations ----
  { id:"f-la", tier:"foundations", short:"LA",
    title:"Linear Algebra", sub:"vectors → SVD",
    title_fa:"جبر خطی", sub_fa:"بردارها تا SVD",
    blurb:"Vectors, matrices, eigendecomposition, SVD and low-rank approximation — the language every model is written in.",
    blurb_fa:"بردارها، ماتریس‌ها، تجزیه مقدار ویژه، SVD — زبانی که هر مدل در آن نوشته می‌شود.",
    ref:true, aud:["student","executive"], prereqs:[],
    resources:[
      { title:"3Blue1Brown — Essence of Linear Algebra", type:"video", url:"https://www.3blue1brown.com/topics/linear-algebra" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-calc", tier:"foundations", short:"∂x",
    title:"Calculus", sub:"gradients & chain rule",
    title_fa:"حساب دیفرانسیل", sub_fa:"گرادیان و قانون زنجیره",
    blurb:"Multivariable differentiation, gradients, and the chain rule — everything backprop quietly assumes you know.",
    blurb_fa:"مشتق چندمتغیره، گرادیان‌ها و قانون زنجیره — هر آنچه پس‌انتشار پنهانی فرض می‌کند می‌دانید.",
    ref:true, aud:["student","executive"], prereqs:[],
    resources:[
      { title:"3Blue1Brown — Essence of Calculus", type:"video", url:"https://www.3blue1brown.com/topics/calculus" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-prob", tier:"foundations", short:"P&S",
    title:"Probability & Statistics", sub:"uncertainty & inference",
    title_fa:"احتمال و آمار", sub_fa:"عدم قطعیت و استنتاج",
    blurb:"Distributions, expectation, estimation, and statistical inference — why ML claims anything at all about unseen data.",
    blurb_fa:"توزیع‌ها، امید ریاضی، تخمین و استنتاج آماری — چرا یادگیری ماشین درباره داده‌های ندیده ادعایی می‌کند.",
    ref:true, aud:["student","executive"], prereqs:[],
    resources:[
      { title:"Khan Academy — Statistics & Probability", type:"website", url:"https://www.khanacademy.org/math/statistics-probability" },
      { title:"Seeing Theory — visual introduction to probability", type:"website", url:"https://seeing-theory.brown.edu" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-opt", tier:"foundations", short:"Opt",
    title:"Optimization", sub:"gradient descent & friends",
    title_fa:"بهینه‌سازی", sub_fa:"گرادیان کاهشی",
    blurb:"Loss landscapes, convexity, gradient descent, and constrained optimization — how learning actually happens.",
    blurb_fa:"فضای تابع خسارت، تحدب، گرادیان کاهشی و بهینه‌سازی مقیدشده — یادگیری در واقع چگونه اتفاق می‌افتد.",
    ref:true, aud:["student","executive"], prereqs:[],
    resources:[
      { title:"Khan Academy — Multivariable Calculus (optimization units)", type:"website", url:"https://www.khanacademy.org/math/multivariable-calculus" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"f-prog", tier:"foundations", short:"Py",
    title:"Programming", sub:"Python, NumPy, Git",
    title_fa:"برنامه‌نویسی", sub_fa:"پایتون و NumPy",
    blurb:"Python, NumPy/pandas, data munging, visualization, Git and the terminal — the workshop where everything gets built.",
    blurb_fa:"پایتون، NumPy/pandas، پردازش داده، تصویرسازی، Git — کارگاهی که همه چیز در آن ساخته می‌شود.",
    ref:true, aud:["student","executive"], prereqs:[],
    resources:[
      { title:"Programming with Mosh — Python for Beginners", type:"video", url:"https://www.youtube.com/watch?v=_uQrJ0TkZlc" },
      { title:"Harvard CS50P — Introduction to Programming with Python", type:"website", url:"https://cs50.harvard.edu/python/" },
      { title:"Automate the Boring Stuff with Python", type:"book", url:"https://automatetheboringstuff.com" },
    ],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  // ---- Junction node (renders as dot, not a card) ----
  { id:"junc-f2c", junction:true, tier:"concepts", short:"·",
    title:"", sub:"", title_fa:"", sub_fa:"", blurb:"", blurb_fa:"",
    aud:["student"], prereqs:["f-la","f-calc","f-prob","f-opt","f-prog"],
    resources:[], ing:{} },

  // ---- Core / Concepts ----
  { id:"c-intro", tier:"foundations", short:"Lit",
    title:"AI Literacy", sub:"what AI is & its history",
    title_fa:"مبانی هوش مصنوعی", sub_fa:"هوش مصنوعی چیست و تاریخچه آن",
    blurb:"What AI and machine learning actually are, where they sit in the broader landscape, when to use them (and when not to), and the common thread across all the methods ahead.",
    blurb_fa:"هوش مصنوعی و یادگیری ماشین واقعاً چه هستند، کجا در چشم‌انداز گسترده‌تر جای می‌گیرند، چه زمانی باید از آن‌ها استفاده کرد، و وجه مشترک تمام روش‌های پیش رو.",
    aud:["public"], prereqs:[],
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"c-flu", tier:"foundations", short:"Flu",
    title:"AI Fluency", sub:"prompts, tools & workflows",
    title_fa:"مهارت هوش مصنوعی", sub_fa:"پرامپت‌نویسی، ابزارها و گردش‌های کاری",
    blurb:"How to use AI tools effectively: prompting strategies, working with LLMs, integrating AI into real workflows, and evaluating outputs critically.",
    blurb_fa:"چگونه ابزارهای هوش مصنوعی را به طور موثر استفاده کنیم: استراتژی‌های پرامپت، کار با LLM‌ها، ادغام هوش مصنوعی در گردش‌های کاری واقعی و ارزیابی نتایج.",
    aud:["public"], prereqs:[],
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"c-ml", tier:"concepts", short:"ML",
    title:"Supervised Learning", sub:"nearest neighbors, decision trees, regression, classification",
    title_fa:"یادگیری با نظارت", sub_fa:"نزدیک‌ترین همسایه، درخت‌های تصمیم، رگرسیون، طبقه‌بندی",
    blurb:"KNN, decision trees, ensembles, linear regression, logistic regression and SVMs — supervised learning end-to-end. The running thread: linear models and linear separability, and where they break.",
    blurb_fa:"KNN، درخت‌های تصمیم، مجموعه‌ها، رگرسیون خطی، رگرسیون لجستیک و SVM. موضوع مشترک: مدل‌های خطی و جداسازی خطی — و جایی که شکست می‌خورند.",
    aud:["student","executive"], prereqs:["junc-f2c"],
    resources:[
      { title:"StatQuest — Machine Learning playlists", type:"video", url:"https://statquest.org" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"c-nn", tier:"concepts", short:"DL",
    title:"Deep Learning", sub:"neural nets, backprop, depth & scale",
    title_fa:"یادگیری عمیق", sub_fa:"شبکه‌های عصبی، پس‌انتشار، عمق و مقیاس",
    blurb:"Linear models as neural nets, activations and nonlinearity, backprop intuition, loss functions, gradient descent — then deeper: depth, regularization, and the transformer architecture.",
    blurb_fa:"مدل‌های خطی به عنوان شبکه‌های عصبی، فعال‌سازی‌ها و غیرخطی بودن، شهود پس‌انتشار، توابع خسارت، گرادیان کاهشی — و عمیق‌تر: عمق، منظم‌سازی، و معماری ترانسفورمر.",
    aud:["student","executive"], prereqs:["c-ml"],
    resources:[
      { title:"3Blue1Brown — Neural Networks", type:"video", url:"https://www.3blue1brown.com/topics/neural-networks" },
      { title:"Karpathy — Neural Networks: Zero to Hero", type:"video", url:"https://karpathy.ai/zero-to-hero.html" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  // ---- Applied / Applications ----
  { id:"a-vision", tier:"applied", short:"CV",
    title:"Computer Vision", sub:"CNNs & transfer learning",
    title_fa:"بینایی ماشین", sub_fa:"CNN و انتقال یادگیری",
    blurb:"CNNs and feature hierarchies, convolution / pooling / normalization, transfer learning, data augmentation, evaluation.",
    blurb_fa:"CNN‌ها و سلسله مراتب ویژگی، کانولوشن / پولینگ / نرمال‌سازی، انتقال یادگیری، بهبود داده، ارزیابی.",
    aud:["student","executive"], prereqs:["c-nn"],
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"a-nlp", tier:"applied", short:"LM",
    title:"Language Models", sub:"embeddings → transformers",
    title_fa:"مدل‌های زبانی", sub_fa:"تعبیه‌ها تا ترانسفورمر",
    blurb:"Sequence models, embeddings, attention, transformers, pretraining and fine-tuning, decoding. From n-grams to LLMs.",
    blurb_fa:"مدل‌های دنباله، تعبیه‌ها، توجه، ترانسفورمرها، پیش‌آموزش و تنظیم دقیق، رمزگشایی. از n-gram تا LLM.",
    aud:["student","executive"], prereqs:["c-nn"],
    resources:[
      { title:"Jay Alammar — The Illustrated Transformer", type:"website", url:"https://jalammar.github.io/illustrated-transformer/" },
      { title:"3Blue1Brown — Neural Networks (attention & transformers chapters)", type:"video", url:"https://www.3blue1brown.com/topics/neural-networks" },
      { title:"Karpathy — Neural Networks: Zero to Hero (GPT lectures)", type:"video", url:"https://karpathy.ai/zero-to-hero.html" },
    ],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"a-agents", tier:"applied", short:"Ag",
    title:"AI Agents", sub:"reasoning, tools, memory",
    title_fa:"عامل‌های هوش مصنوعی", sub_fa:"استدلال، ابزار، حافظه",
    blurb:"Agents and reasoning, planning and memory, tool / function calling, workflows and orchestration, multi-agent concepts.",
    blurb_fa:"عامل‌ها و استدلال، برنامه‌ریزی و حافظه، فراخوانی ابزار/تابع، گردش‌های کاری و هماهنگی، مفاهیم چند عاملی.",
    aud:["executive"], prereqs:["a-nlp"],
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  { id:"a-systems", tier:"applied", short:"Sys",
    title:"AI Systems & MLOps", sub:"models → production",
    title_fa:"سیستم‌های هوش مصنوعی", sub_fa:"مدل‌ها تا تولید",
    blurb:"From models to systems: retrieval and RAG, APIs and services, end-to-end evaluation, reliability and observability.",
    blurb_fa:"از مدل‌ها به سیستم‌ها: بازیابی و RAG، API‌ها و سرویس‌ها، ارزیابی انتها به انتها، قابلیت اطمینان و مشاهده‌پذیری.",
    aud:["executive"], prereqs:["a-agents","a-vision"],
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },

  // ---- Advanced / Specializations ----
  { id:"x-prob", tier:"concepts", short:"Pr",
    title:"Probabilistic & Causal AI", sub:"Bayes, uncertainty & causality",
    title_fa:"هوش مصنوعی احتمالی", sub_fa:"بیز، عدم قطعیت و علیت",
    blurb:"Probabilistic learning, Bayesian modeling, uncertainty quantification, and causal reasoning.",
    blurb_fa:"یادگیری احتمالی، مدل‌سازی بیزی، کمی‌سازی عدم قطعیت و استدلال علّی.",
    aud:["student"], prereqs:["c-ml"],
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"x-unsup", tier:"concepts", short:"Un",
    title:"Unsupervised Learning", sub:"PCA, clustering, Gaussian mixtures",
    title_fa:"یادگیری بدون نظارت", sub_fa:"PCA، خوشه‌بندی، مخلوط گاوسی",
    blurb:"Dimensionality reduction, clustering, and representation learning — finding structure without labels.",
    blurb_fa:"کاهش بعد، خوشه‌بندی و یادگیری بازنمایی — یافتن ساختار بدون برچسب.",
    aud:["student"], prereqs:["junc-f2c"],
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"x-rl", tier:"concepts", short:"RL",
    title:"Reinforcement Learning", sub:"bandits, MDPs, value functions, policy gradient",
    title_fa:"یادگیری تقویتی", sub_fa:"باندیت‌ها، MDP، توابع ارزش، گرادیان سیاست",
    blurb:"MDPs, value functions, Q-learning, and policy gradients — learning by interacting with an environment.",
    blurb_fa:"MDP، توابع ارزش، Q-learning و گرادیان سیاست — یادگیری از طریق تعامل با محیط.",
    aud:["student"], prereqs:["junc-f2c"],
    resources:[],
    ing:{listen:"planned", play:"planned", build:"planned"} },

  { id:"x-rai", tier:"concepts", short:"RAI",
    title:"Responsible AI", sub:"fairness, robustness, security & explainability",
    title_fa:"هوش مصنوعی مسئولانه", sub_fa:"انصاف، استحکام، امنیت و تبیین‌پذیری",
    blurb:"Case studies in responsible AI: fairness and bias, robustness under distribution shift, security against adversarial attacks, explainability, privacy, and accountability. What does it mean to build AI that is safe and trustworthy?",
    blurb_fa:"مطالعات موردی در هوش مصنوعی مسئولانه: انصاف و تعصب، استحکام در برابر تغییر توزیع، امنیت در برابر حملات، تبیین‌پذیری، حریم خصوصی و پاسخگویی. هوش مصنوعی ایمن و قابل اعتماد یعنی چه؟",
    aud:["student","executive","public"], prereqs:[],
    resources:[],
    ing:{listen:"wip", play:"planned", build:"planned"} },
];

const byId = Object.fromEntries(MODULES.map(m => [m.id, m]));

/* One shared layout — x = % of canvas width (card center), y = level (0=bottom) */
/* Layout: bottom=foundations, middle=concepts blob, top=applications
   Row y-values:
     0.0  — foundations row 1: AI Literacy | AI Fluency | Responsible AI (bottom entry row)
     0.9  — foundations row 2: 5 math foundations (reference, students/execs)
     1.75 — concepts row 1: Unsupervised | Supervised | RL
     2.65 — concepts row 2: Deep Learning | Probabilistic AI
     3.65 — applications row 1: Computer Vision | Language Models
     4.45 — applications row 2: AI Systems & MLOps | AI Agents
*/
const LAYOUT = {
  // ── Foundations row 1 (entry points, very bottom) ───────────
  "c-intro":   { x: 18, y: 0    },  // AI Literacy
  "c-flu":     { x: 50, y: 0    },  // AI Fluency
  "x-rai":     { x: 82, y: 0    },  // Responsible AI
  // ── Junction dot (between math row and concepts row 1) ──────
  "junc-f2c":  { x: 50, y: 1.325 },
  // ── Foundations row 2 (math references) ─────────────────────
  "f-la":      { x: 10, y: 0.9  },
  "f-calc":    { x: 30, y: 0.9  },
  "f-prob":    { x: 50, y: 0.9  },
  "f-opt":     { x: 70, y: 0.9  },
  "f-prog":    { x: 90, y: 0.9  },
  // ── Concepts row 1 ───────────────────────────────────────────
  "x-unsup":   { x: 20, y: 1.75 },
  "c-ml":      { x: 50, y: 1.75 },
  "x-rl":      { x: 80, y: 1.75 },
  // ── Concepts row 2 (Deep Learning left, Prob AI right) ──────
  "c-nn":      { x: 30, y: 2.65 },
  "x-prob":    { x: 70, y: 2.65 },
  // ── Applications ─────────────────────────────────────────────
  "a-vision":  { x: 25, y: 3.65 },
  "a-nlp":     { x: 75, y: 3.65 },
  "a-agents":  { x: 72, y: 4.45 },
  "a-systems": { x: 28, y: 4.45 },
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
      { id:"x-rai",    where:"" },
    ],
  },
  executive: {
    hint: "The 6-week executive program — 3-hour sessions: interactive lectures + hands-on notebooks. Build something every week.",
    steps: [
      { id:"c-ml",      where:"" },
      { id:"c-nn",      where:"" },
      { id:"a-vision",  where:"" },
      { id:"a-nlp",     where:"" },
      { id:"a-agents",  where:"" },
      { id:"a-systems", where:"" },
      { id:"x-rai",     where:"", bonus:true },
    ],
  },
  public: {
    hint: "Start here. No technical background required — just curiosity about AI.",
    steps: [
      { id:"c-intro", where:"" },
      { id:"c-flu",   where:"" },
      { id:"x-rai",   where:"", bonus:true },
    ],
  },
};

function getAudience() {
  const a = localStorage.getItem("alm-audience");
  if (a === "practitioner") return "executive"; // migrate old value
  return (a === "student" || a === "executive" || a === "public") ? a : "student";
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
