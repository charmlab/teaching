/* ══════════════════════════════════════════════════════════════
   lesson-core.js — shared framework for the Linear Regression lesson
   Canvas/plot helpers, math (real computation, nothing staged),
   quiz engine, Python highlighter, and the code stepper (Fig 9.1).
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

  /* ── math: real fits, real losses ───────────────────────── */

  // mean-squared loss J = 1/(2N) Σ (w0 + w1 x - t)^2
  LR.loss1d = function (pts, w0, w1) {
    let s = 0;
    for (const p of pts) { const r = w0 + w1 * p.x - p.t; s += r * r; }
    return s / (2 * pts.length);
  };

  // closed-form 1D least squares (centered form)
  LR.fit1d = function (pts) {
    const n = pts.length;
    let xb = 0, tb = 0;
    for (const p of pts) { xb += p.x; tb += p.t; }
    xb /= n; tb /= n;
    let num = 0, den = 0;
    for (const p of pts) { num += (p.x - xb) * (p.t - tb); den += (p.x - xb) * (p.x - xb); }
    const w1 = num / den;
    return { w0: tb - w1 * xb, w1 };
  };

  // solve A x = b by Gaussian elimination with partial pivoting
  LR.solve = function (A, b) {
    const n = A.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      const tmp = M[c]; M[c] = M[piv]; M[piv] = tmp;
      if (Math.abs(M[c][c]) < 1e-12) continue;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((row, i) => (Math.abs(row[i]) < 1e-12 ? 0 : row[n] / row[i]));
  };

  // ridge polynomial fit: w = (Φ'Φ + λI)^{-1} Φ' t   (λ=0 → plain LS)
  LR.polyfit = function (xs, ts, M, lambda) {
    lambda = lambda || 0;
    const d = M + 1, n = xs.length;
    const A = Array.from({ length: d }, () => new Array(d).fill(0));
    const b = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const phi = new Array(d);
      let v = 1;
      for (let j = 0; j < d; j++) { phi[j] = v; v *= xs[i]; }
      for (let j = 0; j < d; j++) {
        b[j] += phi[j] * ts[i];
        for (let k = 0; k < d; k++) A[j][k] += phi[j] * phi[k];
      }
    }
    for (let j = 0; j < d; j++) A[j][j] += lambda;
    return LR.solve(A, b);
  };

  LR.polyval = function (w, x) {
    let v = 0;
    for (let j = w.length - 1; j >= 0; j--) v = v * x + w[j];
    return v;
  };

  LR.polyLoss = function (w, xs, ts) {
    let s = 0;
    for (let i = 0; i < xs.length; i++) { const r = LR.polyval(w, xs[i]) - ts[i]; s += r * r; }
    return s / (2 * xs.length);
  };

  /* ── shared datasets ────────────────────────────────────── */

  // house-style scatter around 0.5 + 1.5x, x∈[0.3, 4.7]  (used by several figures)
  LR.houseData = function (n, seed, noise) {
    const rand = LR.rng(seed || 42);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = 0.3 + (4.4 * i) / (n - 1) + (rand() - 0.5) * 0.35;
      const t = 0.5 + 1.5 * x + LR.gauss(rand) * (noise === undefined ? 0.55 : noise);
      pts.push({ x, t });
    }
    return pts;
  };

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

  /* ── Python syntax highlighter (for pre.code blocks) ────── */
  const PY_KW = /\b(def|for|in|range|return|if|elif|else|while|import|as|from|len|not|and|or|True|False|None|print)\b/g;
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

  /* ══════════════════════════════════════════════════════════
     Fig 9.1 — the code stepper: gradient descent, line by line
     ══════════════════════════════════════════════════════════ */
  LR.figs.stepper = function (mount) {
    LR.header(
      mount,
      "Gradient descent, executed line by line",
      "The Python on the left really runs. Step through it and watch each line act on the data on the right."
    );

    // ---- the program (displayed) and its real semantics -----
    const CODE = [
      "w = np.zeros(2)          # [w0, w1]",
      "alpha = 0.3",
      "for epoch in range(30):",
      "    y = X @ w            # predict",
      "    r = y - t            # residuals",
      "    grad = X.T @ r / N   # gradient",
      "    w -= alpha * grad    # step",
    ];
    const NARRATE = [
      "Initialize the weights at the origin: w₀ = 0, w₁ = 0. The model starts as a flat line at height zero.",
      "Choose the learning rate α = 0.3, a hyperparameter. We picked it by hand; too big diverges, too small crawls.",
      "Begin the next pass over the data. One loop iteration = one full-batch gradient step.",
      "Predict: multiply the design matrix by w. Each open circle on the line is one prediction y⁽ⁱ⁾ = w₀ + w₁x⁽ⁱ⁾.",
      "Subtract the targets to get residuals, drawn as red vertical gaps. Their signs say which way each point pushes.",
      "Assemble the gradient: average of residual-weighted inputs. It points uphill in weight space.",
      "Step against the gradient. Watch the line jump toward the data, and J(w) drop in the readout.",
    ];

    // dataset (real, deterministic)
    const pts = LR.houseData(8, 7, 0.5);
    const N = pts.length;

    // program state
    let st;
    function resetState() {
      st = { pc: -1, w: [0, 0], alpha: 0.3, epoch: -1, y: null, r: null, grad: null, prevW: null, J: LR.loss1d(pts, 0, 0) };
    }
    resetState();

    // ---- layout ----
    const grid = LR.el("div", "stepper-grid");
    mount.appendChild(grid);

    const left = LR.el("div");
    const codeBox = LR.el("div", "stepper-code");
    codeBox.setAttribute("aria-label", "Python gradient descent code");
    const lineEls = CODE.map(function (line) {
      const d = LR.el("div", "stepper-line");
      d.innerHTML = LR.highlight(line);
      codeBox.appendChild(d);
      return d;
    });
    left.appendChild(codeBox);

    const varsBox = LR.el("div", "stepper-vars", "");
    left.appendChild(varsBox);
    const narrate = LR.el("div", "stepper-narrate", "Press <b>Step</b> to execute the first line.");
    left.appendChild(narrate);
    grid.appendChild(left);

    const right = LR.el("div");
    const { cv, ctx, W, H } = LR.canvas(right, 430, 330, {
      aria: "Scatter plot showing the regression line updating as the code runs",
    });
    grid.appendChild(right);

    const ro = LR.readout(mount, [
      { k: "epoch", label: "epoch" },
      { k: "w", label: "w = [w₀, w₁]" },
      { k: "J", label: "loss J(w)" },
    ]);

    const bar = LR.controls(mount);
    const stepBtn = LR.button(bar, "Step ▸", step, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, LR.reducedMotion ? 900 : 420);
    });
    LR.button(bar, "Reset ⟲", function () {
      stopPlay(); resetState(); render(); updateSide();
      narrate.innerHTML = "Press <b>Step</b> to execute the first line.";
    });
    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    // ---- semantics of each line ----
    function exec(pc) {
      switch (pc) {
        case 0: st.w = [0, 0]; st.epoch = -1; st.y = null; st.r = null; st.grad = null; break;
        case 1: st.alpha = 0.3; break;
        case 2: st.epoch += 1; st.y = null; st.r = null; st.grad = null; break;
        case 3: st.y = pts.map((p) => st.w[0] + st.w[1] * p.x); break;
        case 4: st.r = st.y.map((y, i) => y - pts[i].t); break;
        case 5: {
          let g0 = 0, g1 = 0;
          for (let i = 0; i < N; i++) { g0 += st.r[i]; g1 += st.r[i] * pts[i].x; }
          st.grad = [g0 / N, g1 / N];
          break;
        }
        case 6:
          st.prevW = st.w.slice();
          st.w = [st.w[0] - st.alpha * st.grad[0], st.w[1] - st.alpha * st.grad[1]];
          break;
      }
      st.J = LR.loss1d(pts, st.w[0], st.w[1]);
    }

    function nextPC(pc) {
      if (pc < 6) return pc + 1;
      return st.epoch >= 29 ? -2 : 2; // -2 = done
    }

    function step() {
      if (st.pc === -2) { stopPlay(); return; }
      st.pc = st.pc === -1 ? 0 : nextPC(st.pc);
      if (st.pc === -2) {
        stopPlay();
        narrate.innerHTML = "<b>Done.</b> 30 epochs. Compare w with the closed-form optimum in the readout: they agree to two decimals.";
        render(); updateSide();
        return;
      }
      exec(st.pc);
      narrate.textContent = NARRATE[st.pc];
      render(); updateSide();
    }

    function updateSide() {
      lineEls.forEach((el, i) => el.classList.toggle("active", i === st.pc));
      const rows = [];
      rows.push('<span class="var-name">w</span> = [' + LR.fmtF(st.w[0], 3) + ", " + LR.fmtF(st.w[1], 3) + "]");
      rows.push('<span class="var-name">alpha</span> = ' + st.alpha);
      rows.push('<span class="var-name">epoch</span> = ' + (st.epoch < 0 ? "–" : st.epoch));
      rows.push('<span class="var-name">y</span> = ' + (st.y ? "[" + st.y.slice(0, 3).map((v) => LR.fmtF(v, 2)).join(", ") + ", …]" : "–"));
      rows.push('<span class="var-name">r</span> = ' + (st.r ? "[" + st.r.slice(0, 3).map((v) => LR.fmtF(v, 2)).join(", ") + ", …]" : "–"));
      rows.push('<span class="var-name">grad</span> = ' + (st.grad ? "[" + LR.fmtF(st.grad[0], 3) + ", " + LR.fmtF(st.grad[1], 3) + "]" : "–"));
      varsBox.innerHTML = rows.join("<br>");

      const opt = LR.fit1d(pts);
      ro.set("epoch", st.epoch < 0 ? "–" : String(st.epoch));
      ro.set("w", "[" + LR.fmtF(st.w[0], 3) + ", " + LR.fmtF(st.w[1], 3) + "]  (optimum: [" + LR.fmtF(opt.w0, 3) + ", " + LR.fmtF(opt.w1, 3) + "])");
      ro.set("J", LR.fmtF(st.J, 4), LR.C.orange);
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const P = { x0: 52, y0: 18, w: W - 70, h: H - 60, xmin: 0, xmax: 5, ymin: -1, ymax: 9, xlabel: "x (input feature)", ylabel: "t (target)" };
      const { sx, sy } = LR.plot(ctx, P);

      // ghost of previous line during a step
      if (st.pc === 6 && st.prevW) {
        ctx.strokeStyle = LR.C.orangeSoft;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(sx(0), sy(st.prevW[0]));
        ctx.lineTo(sx(5), sy(st.prevW[0] + st.prevW[1] * 5));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // residuals
      if (st.r && st.pc >= 4) {
        ctx.strokeStyle = LR.C.red;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([3, 3]);
        pts.forEach(function (p, i) {
          ctx.beginPath();
          ctx.moveTo(sx(p.x), sy(p.t));
          ctx.lineTo(sx(p.x), sy(st.y[i]));
          ctx.stroke();
        });
        ctx.setLineDash([]);
      }

      // current model line
      ctx.strokeStyle = LR.C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(sx(0), sy(st.w[0]));
      ctx.lineTo(sx(5), sy(st.w[0] + st.w[1] * 5));
      ctx.stroke();

      // gradient arrow annotation during grad step (in a small inset, weight space direction)
      if (st.pc === 5 && st.grad) {
        ctx.font = "600 11.5px Inter, sans-serif";
        ctx.fillStyle = LR.C.purple;
        ctx.textAlign = "left";
        ctx.fillText("∇J = [" + LR.fmtF(st.grad[0], 2) + ", " + LR.fmtF(st.grad[1], 2) + "] → step opposite", P.x0 + 8, P.y0 + 16);
      }

      // data points
      pts.forEach(function (p) { LR.dot(ctx, sx(p.x), sy(p.t), 4.5, LR.C.text); });

      // predictions
      if (st.y && st.pc >= 3) {
        pts.forEach(function (p, i) {
          LR.dot(ctx, sx(p.x), sy(st.y[i]), 4, "#fff", LR.C.orange);
        });
      }
    }

    render(); updateSide();
  };

  /* ── boot ───────────────────────────────────────────────── */
  document.addEventListener("DOMContentLoaded", function () {
    highlightAll();
    wireQuizzes();
    // defer one tick so figures-a/b (also deferred) have registered
    setTimeout(initFigures, 0);
  });
})();

/* == code-fold == */
(function () {
  "use strict";
  function fold(el, lang) {
    var d = document.createElement("details");
    d.className = "code-fold";
    var s = document.createElement("summary");
    var t = document.createElement("span");
    t.textContent = "The code";
    s.appendChild(t);
    if (lang) {
      var c = document.createElement("span");
      c.className = "cf-lang";
      c.textContent = lang;
      s.appendChild(c);
    }
    s.setAttribute("aria-label", "Show the " + (lang || "code") + " for this section");
    el.parentNode.insertBefore(d, el);
    d.appendChild(s);
    d.appendChild(el);
  }
  function run() {
    document.querySelectorAll(".code-compare").forEach(function (g) {
      if (g.closest("details") || g.closest(".fig-mount")) return;
      var p = g.querySelector("pre.code");
      fold(g, p ? p.getAttribute("data-lang") : null);
    });
    document.querySelectorAll("pre.code").forEach(function (p) {
      if (p.closest("details") || p.closest(".code-compare") || p.closest(".fig-mount")) return;
      fold(p, p.getAttribute("data-lang"));
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", run);
  else run();
})();
