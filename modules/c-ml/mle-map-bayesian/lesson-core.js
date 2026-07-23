/* ══════════════════════════════════════════════════════════════
   lesson-core.js — shared framework for the MLE / MAP / Bayesian
   lesson. Canvas/plot helpers, the estimation core (likelihoods,
   Beta updates, the three estimators: real computation, nothing
   staged), quiz engine, and the Python highlighter.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  const LR = (window.LR = window.LR || {});
  LR.figs = {};

  LR.reducedMotion =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ── palette (kept in sync with lesson.css) ─────────────── */
  LR.C = {
    text: "#111111",
    muted: "#555555",
    faint: "#868e96",
    grid: "#ececec",
    axis: "#333333",
    orange: "#e8590c",
    orangeSoft: "#ffd8bd",
    green: "#2f9e44",
    purple: "#7048e8",
    red: "#e03131",
    amber: "#f0a202",
    yellow: "#e6b800",
  };

  /* ── deterministic RNG ──────────────────────────────────── */
  LR.rng = function (seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  LR.gauss = function (rand) {
    // Box–Muller
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  /* ── formatting ─────────────────────────────────────────── */
  LR.fmt = function (n, d) {
    if (!isFinite(n)) return n > 0 ? "∞" : "-∞";
    if (d === undefined) d = 2;
    return Number(n.toFixed(d)).toString();
  };
  LR.fmtF = function (n, d) {
    if (!isFinite(n)) return n > 0 ? "∞" : "-∞";
    return n.toFixed(d === undefined ? 2 : d);
  };

  /* ── canvas with HiDPI scaling ──────────────────────────── */
  LR.canvas = function (parent, W, H, opts) {
    opts = opts || {};
    const cv = document.createElement("canvas");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = W * dpr;
    cv.height = H * dpr;
    cv.className = "fig-canvas" + (opts.cls ? " " + opts.cls : "");
    cv.style.maxWidth = W + "px";
    cv.style.margin = "0 auto";
    if (opts.aria) { cv.setAttribute("role", "img"); cv.setAttribute("aria-label", opts.aria); }
    parent.appendChild(cv);
    const ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);
    return { cv, ctx, W, H };
  };

  /* map pointer event → logical canvas coords */
  LR.evtXY = function (cv, W, H, e) {
    const r = cv.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * W,
      y: ((e.clientY - r.top) / r.height) * H,
    };
  };

  /* generic pointer-drag wiring */
  LR.drag = function (cv, W, H, handlers) {
    let active = false;
    cv.addEventListener("pointerdown", function (e) {
      const p = LR.evtXY(cv, W, H, e);
      if (handlers.hit && !handlers.hit(p)) return;
      active = true;
      cv.setPointerCapture(e.pointerId);
      cv.classList.add("grabbing");
      if (handlers.down) handlers.down(p);
      e.preventDefault();
    });
    cv.addEventListener("pointermove", function (e) {
      const p = LR.evtXY(cv, W, H, e);
      if (active) { if (handlers.move) handlers.move(p); }
      else if (handlers.hover) handlers.hover(p);
    });
    const end = function (e) {
      if (!active) return;
      active = false;
      cv.classList.remove("grabbing");
      if (handlers.up) handlers.up(LR.evtXY(cv, W, H, e));
    };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
    cv.classList.add("grab");
  };

  /* ── plot scaffold: axes, ticks, gridlines, labels ──────── */
  LR.plot = function (ctx, o) {
    // o: {x0,y0,w,h, xmin,xmax,ymin,ymax, xlabel,ylabel, xticks,yticks, title}
    const sx = (x) => o.x0 + ((x - o.xmin) / (o.xmax - o.xmin)) * o.w;
    const sy = (y) => o.y0 + o.h - ((y - o.ymin) / (o.ymax - o.ymin)) * o.h;
    const inv = (px, py) => ({
      x: o.xmin + ((px - o.x0) / o.w) * (o.xmax - o.xmin),
      y: o.ymin + ((o.y0 + o.h - py) / o.h) * (o.ymax - o.ymin),
    });

    ctx.save();
    ctx.font = "11px Inter, -apple-system, sans-serif";

    // gridlines + ticks
    const xt = o.xticks || LR.ticks(o.xmin, o.xmax, 6);
    const yt = o.yticks || LR.ticks(o.ymin, o.ymax, 5);
    ctx.strokeStyle = LR.C.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = LR.C.faint;
    xt.forEach((t) => {
      const px = sx(t);
      if (px < o.x0 - 0.5 || px > o.x0 + o.w + 0.5) return;
      ctx.beginPath(); ctx.moveTo(px, o.y0); ctx.lineTo(px, o.y0 + o.h); ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(LR.fmt(t, 2), px, o.y0 + o.h + 15);
    });
    yt.forEach((t) => {
      const py = sy(t);
      if (py < o.y0 - 0.5 || py > o.y0 + o.h + 0.5) return;
      ctx.beginPath(); ctx.moveTo(o.x0, py); ctx.lineTo(o.x0 + o.w, py); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(LR.fmt(t, 2), o.x0 - 7, py + 3.5);
    });

    // frame
    ctx.strokeStyle = LR.C.axis;
    ctx.lineWidth = 1.4;
    ctx.strokeRect(o.x0, o.y0, o.w, o.h);

    // labels
    ctx.fillStyle = LR.C.text;
    ctx.font = "600 12px Inter, -apple-system, sans-serif";
    if (o.xlabel) {
      ctx.textAlign = "center";
      ctx.fillText(o.xlabel, o.x0 + o.w / 2, o.y0 + o.h + 32);
    }
    if (o.ylabel) {
      ctx.save();
      ctx.translate(o.x0 - 38, o.y0 + o.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(o.ylabel, 0, 0);
      ctx.restore();
    }
    if (o.title) {
      ctx.textAlign = "left";
      ctx.font = "700 12.5px Inter, -apple-system, sans-serif";
      ctx.fillText(o.title, o.x0, o.y0 - 8);
    }
    ctx.restore();
    return { sx, sy, inv };
  };

  LR.ticks = function (min, max, n) {
    const span = max - min;
    const step0 = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    let step = mag;
    for (const m of [1, 2, 2.5, 5, 10]) {
      if (step0 <= m * mag) { step = m * mag; break; }
    }
    const out = [];
    for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) out.push(t);
    return out;
  };

  LR.dot = function (ctx, x, y, r, fill, stroke) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.4; ctx.stroke(); }
  };

  LR.arrow = function (ctx, x1, y1, x2, y2, color, width) {
    const a = Math.atan2(y2 - y1, x2 - x1);
    const hl = Math.max(7, (width || 2) * 3.2);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width || 2;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - hl * Math.cos(a - 0.42), y2 - hl * Math.sin(a - 0.42));
    ctx.lineTo(x2 - hl * Math.cos(a + 0.42), y2 - hl * Math.sin(a + 0.42));
    ctx.closePath(); ctx.fill();
  };

  /* ══════════════════════════════════════════════════════════
     The estimation core — shared by every figure in this lesson.
     Likelihood, log-likelihood, Beta machinery, and the three
     estimators (MLE, MAP, posterior predictive). All real math.
     ══════════════════════════════════════════════════════════ */
  const EST = (LR.est = {
    // log Γ(x), Lanczos approximation (accurate to ~1e-13 for x > 0)
    lgamma: function (x) {
      const g = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
      ];
      let y = x, tmp = x + 5.5;
      tmp -= (x + 0.5) * Math.log(tmp);
      let ser = 1.000000000190015;
      for (let j = 0; j < 6; j++) ser += g[j] / ++y;
      return -tmp + Math.log((2.5066282746310005 * ser) / x);
    },
    // log B(a,b) = log Γ(a) + log Γ(b) − log Γ(a+b)
    logBeta: function (a, b) {
      return EST.lgamma(a) + EST.lgamma(b) - EST.lgamma(a + b);
    },
    // log C(n,k)
    lchoose: function (n, k) {
      return EST.lgamma(n + 1) - EST.lgamma(k + 1) - EST.lgamma(n - k + 1);
    },

    /* Bernoulli / Binomial */
    // ℓ(θ) = N_H log θ + N_T log(1−θ)  (−∞ at clashing edges)
    logLik: function (theta, nh, nt) {
      if (theta <= 0) return nh > 0 ? -Infinity : nt * Math.log(1 - theta);
      if (theta >= 1) return nt > 0 ? -Infinity : nh * Math.log(theta);
      return nh * Math.log(theta) + nt * Math.log(1 - theta);
    },
    // Binomial pmf p(k heads in n | θ)
    binomPmf: function (n, k, theta) {
      if (theta <= 0) return k === 0 ? 1 : 0;
      if (theta >= 1) return k === n ? 1 : 0;
      return Math.exp(EST.lchoose(n, k) + k * Math.log(theta) + (n - k) * Math.log(1 - theta));
    },

    /* the three estimators for coin data (N_H, N_T) with Beta(a,b) */
    mle: function (nh, nt) {
      return nh + nt === 0 ? NaN : nh / (nh + nt);
    },
    mapEst: function (nh, nt, a, b) {
      const den = nh + nt + a + b - 2;
      return den <= 0 ? NaN : (nh + a - 1) / den;
    },
    postPred: function (nh, nt, a, b) {
      return (nh + a) / (nh + nt + a + b);
    },

    /* Beta density and its shape summaries */
    betaPdf: function (x, a, b) {
      if (x <= 0 || x >= 1) return 0;
      return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - EST.logBeta(a, b));
    },
    betaMean: function (a, b) { return a / (a + b); },
    betaMode: function (a, b) {
      return a > 1 && b > 1 ? (a - 1) / (a + b - 2) : NaN;
    },

    /* fully Bayesian predictive for a whole future dataset (h, t)
       given history (nh, nt) and prior Beta(a, b):
       C(h+t, h) · B(a+nh+h, b+nt+t) / B(a+nh, b+nt)   (source formula) */
    predDataset: function (nh, nt, a, b, h, t) {
      return Math.exp(
        EST.lchoose(h + t, h) +
        EST.logBeta(a + nh + h, b + nt + t) -
        EST.logBeta(a + nh, b + nt)
      );
    },

    /* Gaussian */
    gaussPdf: function (x, mu, sigma) {
      const z = (x - mu) / sigma;
      return Math.exp(-0.5 * z * z) / (Math.sqrt(2 * Math.PI) * sigma);
    },
    gaussLogLik: function (xs, mu, sigma) {
      let s = 0;
      for (const x of xs) s += Math.log(EST.gaussPdf(x, mu, sigma));
      return s;
    },
    gaussMLE: function (xs) {
      const n = xs.length;
      const mu = xs.reduce((a, b) => a + b, 0) / n;
      const v = xs.reduce((s, x) => s + (x - mu) * (x - mu), 0) / n;
      return { mu, sigma: Math.sqrt(v) };
    },

    /* ── shared datasets ──────────────────────────────────── */
    // the running coin example from the slides: 100 tosses, 55 heads
    COIN: { NH: 55, NT: 45 },

    /* Waterloo May daily-high temperatures (°C).
       Flagged addition: the source names the example but gives no
       numbers; these ten values are chosen by us so the MLE comes
       out clean (μ̂ = 15, σ̂ = √6 ≈ 2.449). Everything computed. */
    TEMPS: [11, 14, 17, 13, 18, 16, 12, 15, 19, 15],

    /* the coins-in-a-pouch table.
       Flagged addition: the source specifies the setup, D_new = (9,1),
       and the winners (MLE → coin 3, MAP and Bayesian → coin 2) but
       not the table entries; these counts and frequencies are chosen
       by us so the source's winner flip genuinely happens. Verified:
       MLE plug-in argmax = coin 3, MAP plug-in argmax = coin 2,
       Bayesian predictive argmax = coin 2, all computed live. */
    POUCH: [
      { name: "coin 1", nh: 50, nt: 50, pi: 0.35 },
      { name: "coin 2", nh: 80, nt: 20, pi: 0.20 },
      { name: "coin 3", nh: 9,  nt: 1,  pi: 0.15 },
      { name: "coin 4", nh: 30, nt: 70, pi: 0.30 },
    ],
    POUCH_NEW: { h: 9, t: 1 },
  });

  /* ── figure chrome helpers ──────────────────────────────── */
  LR.el = function (tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  };
  LR.header = function (mount, title, sub) {
    mount.appendChild(LR.el("div", "fig-title", title));
    if (sub) mount.appendChild(LR.el("div", "fig-sub", sub));
  };
  LR.controls = function (mount) {
    const c = LR.el("div", "fig-controls");
    mount.appendChild(c);
    return c;
  };
  LR.slider = function (parent, label, min, max, step, value, oninput, fmt) {
    const lab = LR.el("label");
    lab.appendChild(document.createTextNode(label));
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = min; inp.max = max; inp.step = step; inp.value = value;
    inp.setAttribute("aria-label", label);
    const val = LR.el("span", "val", (fmt || LR.fmt)(value));
    inp.addEventListener("input", function () {
      const v = parseFloat(inp.value);
      val.textContent = (fmt || LR.fmt)(v);
      oninput(v);
    });
    lab.appendChild(inp); lab.appendChild(val);
    parent.appendChild(lab);
    return {
      input: inp,
      set: function (v) { inp.value = v; val.textContent = (fmt || LR.fmt)(v); },
    };
  };
  LR.button = function (parent, text, onclick, cls) {
    const b = LR.el("button", "btn" + (cls ? " " + cls : ""), text);
    b.type = "button";
    b.addEventListener("click", onclick);
    parent.appendChild(b);
    return b;
  };
  LR.readout = function (mount, items) {
    // items: [{k, label}], returns {set(k, text)}
    const box = LR.el("div", "fig-readout");
    const map = {};
    items.forEach(function (it) {
      const ro = LR.el("span", "ro");
      ro.appendChild(LR.el("span", "ro-label", it.label));
      const v = LR.el("span", "ro-val", "–");
      ro.appendChild(v);
      box.appendChild(ro);
      map[it.k] = v;
    });
    mount.appendChild(box);
    return { set: function (k, text, color) { map[k].textContent = text; if (color) map[k].style.color = color; } };
  };
  LR.msg = function (mount) {
    const m = LR.el("div", "fig-msg");
    mount.appendChild(m);
    return {
      show: function (text, kind) { m.textContent = text; m.className = "fig-msg show " + kind; },
      hide: function () { m.className = "fig-msg"; },
    };
  };

  /* number input (for counts like N_H, N_T) */
  LR.numInput = function (parent, label, value, min, max, oninput) {
    const lab = LR.el("label");
    lab.appendChild(document.createTextNode(label));
    const inp = document.createElement("input");
    inp.type = "number";
    inp.min = min; inp.max = max; inp.step = 1; inp.value = value;
    inp.className = "num-input";
    inp.setAttribute("aria-label", label);
    inp.addEventListener("input", function () {
      let v = parseInt(inp.value, 10);
      if (isNaN(v)) return;
      v = Math.max(min, Math.min(max, v));
      oninput(v);
    });
    lab.appendChild(inp);
    parent.appendChild(lab);
    return {
      input: inp,
      set: function (v) { inp.value = v; },
    };
  };

  /* ── Python syntax highlighter (for pre.code blocks) ────── */
  LR.highlight = function (src) {
    // escape
    let s = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const out = [];
    let i = 0;
    while (i < s.length) {
      const rest = s.slice(i);
      let m;
      if ((m = rest.match(/^#[^\n]*/))) {
        out.push('<span class="tok-com">' + m[0] + "</span>"); i += m[0].length;
      } else if ((m = rest.match(/^("([^"\\]|\\.)*"|'([^'\\]|\\.)*')/))) {
        out.push('<span class="tok-str">' + m[0] + "</span>"); i += m[0].length;
      } else if ((m = rest.match(/^\b\d+(\.\d+)?(e-?\d+)?\b/))) {
        out.push('<span class="tok-num">' + m[0] + "</span>"); i += m[0].length;
      } else if ((m = rest.match(/^\b(np|len|range|print|def|for|in|return|if|elif|else|while|import|as|from|not|and|or|True|False|None)\b/))) {
        const cls = /^(np)$/.test(m[0]) ? "tok-fn" : "tok-kw";
        out.push('<span class="' + cls + '">' + m[0] + "</span>"); i += m[0].length;
      } else if ((m = rest.match(/^\.\w+/))) {
        out.push('.<span class="tok-fn">' + m[0].slice(1) + "</span>"); i += m[0].length;
      } else if ((m = rest.match(/^(@|\+=|-=|\*|\+|-|\/|=|&lt;|&gt;)/))) {
        out.push('<span class="tok-op">' + m[0] + "</span>"); i += m[0].length;
      } else {
        out.push(s[i]); i += 1;
      }
    }
    return out.join("");
  };

  function highlightAll() {
    document.querySelectorAll("pre.code code").forEach(function (el) {
      el.innerHTML = LR.highlight(el.textContent);
    });
  }

  /* ── quiz engine ────────────────────────────────────────── */
  function wireQuizzes() {
    document.querySelectorAll(".quiz").forEach(function (q) {
      const correct = q.getAttribute("data-correct");
      const buttons = q.querySelectorAll(".quiz-opts button");
      buttons.forEach(function (btn) {
        btn.addEventListener("click", function () {
          const k = btn.getAttribute("data-k");
          // hide all feedback
          q.querySelectorAll(".quiz-fb").forEach((f) => f.classList.remove("show", "right", "wrong"));
          buttons.forEach((b) => b.classList.remove("picked-wrong"));
          const fb = q.querySelector('.quiz-fb[data-k="' + k + '"]');
          if (k === correct) {
            btn.classList.add("picked-right");
            buttons.forEach((b) => (b.disabled = true));
            if (fb) fb.classList.add("show", "right");
          } else {
            btn.classList.add("picked-wrong");
            if (fb) fb.classList.add("show", "wrong");
          }
        });
      });
    });
  }

  /* ── lazy figure instantiation ──────────────────────────── */
  function initFigures() {
    const mounts = document.querySelectorAll(".fig-mount[data-fig]");
    const build = function (mount) {
      const name = mount.getAttribute("data-fig");
      if (mount.__built) return;
      mount.__built = true;
      const fn = LR.figs[name];
      if (fn) {
        try { fn(mount); }
        catch (err) {
          mount.appendChild(LR.el("div", "fig-msg show bad", "Figure failed to load: " + err.message));
          console.error("[fig:" + name + "]", err);
        }
      }
    };
    const eager = /[?&]eager/.test(location.search); // build everything up front (testing / print)
    if (!eager && "IntersectionObserver" in window) {
      const io = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (en) {
            if (en.isIntersecting) { build(en.target); io.unobserve(en.target); }
          });
        },
        { rootMargin: "600px 0px" }
      );
      mounts.forEach((m) => io.observe(m));
    } else {
      mounts.forEach(build);
    }
  }

  /* ── boot ───────────────────────────────────────────────── */
  document.addEventListener("DOMContentLoaded", function () {
    highlightAll();
    wireQuizzes();
    // defer one tick so figures-a/b (also deferred) have registered
    setTimeout(initFigures, 0);
  });
})();
