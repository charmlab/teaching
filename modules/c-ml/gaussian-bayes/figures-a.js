/* ══════════════════════════════════════════════════════════════
   figures-a.js — Gaussian Bayes lesson, sections 0–3
   Fig 0.1 hookgallery · Fig 1.1 genvdisc · Fig 2.1 bayes1d
   (signature) · Fig 3.1 fitmle
   Also defines the shared Gaussian-Bayes core (LR.gb) used by
   figures-b.js: 1-D and 2-D Gaussian pdfs, MLE fits, pooled and
   diagonal covariances, eigen-ellipses, posterior computation,
   decision-region rendering, and the shared datasets.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared Gaussian-Bayes core (also used by figures-b.js) ── */
  const GB = (LR.gb = {
    /* 1-D Gaussian density */
    g1: function (x, mu, s2) {
      return Math.exp(-((x - mu) * (x - mu)) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2);
    },
    /* 1-D MLE fit: sample mean, sample variance (divide by N) */
    fit1d: function (xs) {
      const n = xs.length;
      let mu = 0;
      for (const x of xs) mu += x;
      mu /= n;
      let s2 = 0;
      for (const x of xs) s2 += (x - mu) * (x - mu);
      s2 /= n;
      return { mu: mu, s2: s2 };
    },
    /* i.i.d. Gaussian log-likelihood of a 1-D sample */
    logL1d: function (xs, mu, s2) {
      let ll = 0;
      for (const x of xs) {
        ll += -0.5 * Math.log(2 * Math.PI * s2) - ((x - mu) * (x - mu)) / (2 * s2);
      }
      return ll;
    },

    /* 2×2 matrix helpers */
    det2: function (S) { return S[0][0] * S[1][1] - S[0][1] * S[1][0]; },
    inv2: function (S) {
      const d = GB.det2(S);
      return [[S[1][1] / d, -S[0][1] / d], [-S[1][0] / d, S[0][0] / d]];
    },
    /* multivariate Gaussian density in 2-D */
    mvn2: function (x, y, mu, S) {
      return Math.exp(GB.logMvn2(x, y, mu, S));
    },
    logMvn2: function (x, y, mu, S) {
      const inv = GB.inv2(S);
      const dx = x - mu[0], dy = y - mu[1];
      const q = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
      return -Math.log(2 * Math.PI) - 0.5 * Math.log(GB.det2(S)) - 0.5 * q;
    },
    /* 2-D MLE fit: mean vector + covariance matrix (divide by N) */
    fit2d: function (pts) {
      const n = pts.length;
      let mx = 0, my = 0;
      for (const p of pts) { mx += p.x; my += p.y; }
      mx /= n; my /= n;
      let sxx = 0, sxy = 0, syy = 0;
      for (const p of pts) {
        const dx = p.x - mx, dy = p.y - my;
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
      }
      return { mu: [mx, my], S: [[sxx / n, sxy / n], [sxy / n, syy / n]], n: n };
    },
    /* pooled (shared) covariance, weighted by class sizes */
    pool: function (fits) {
      let n = 0, sxx = 0, sxy = 0, syy = 0;
      fits.forEach(function (f) {
        n += f.n;
        sxx += f.n * f.S[0][0]; sxy += f.n * f.S[0][1]; syy += f.n * f.S[1][1];
      });
      return [[sxx / n, sxy / n], [sxy / n, syy / n]];
    },
    diagOf: function (S) { return [[S[0][0], 0], [0, S[1][1]]]; },

    /* eigen-decomposition of a symmetric 2×2 (for contour ellipses) */
    eig2: function (S) {
      const a = S[0][0], b = S[0][1], c = S[1][1];
      const tr2 = (a + c) / 2;
      const disc = Math.sqrt(((a - c) / 2) * ((a - c) / 2) + b * b);
      return { l1: tr2 + disc, l2: Math.max(tr2 - disc, 1e-9), theta: 0.5 * Math.atan2(2 * b, a - c) };
    },
    /* draw the r-sigma contour ellipse of N(mu, S) through a plot map */
    ellipse: function (ctx, map, mu, S, r, color, width, dash) {
      const e = GB.eig2(S);
      const ct = Math.cos(e.theta), st = Math.sin(e.theta);
      const a = r * Math.sqrt(e.l1), b = r * Math.sqrt(e.l2);
      ctx.strokeStyle = color;
      ctx.lineWidth = width || 1.6;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      const N = 72;
      for (let i = 0; i <= N; i++) {
        const t = (i / N) * 2 * Math.PI;
        const u = a * Math.cos(t), v = b * Math.sin(t);
        const x = mu[0] + u * ct - v * st;
        const y = mu[1] + u * st + v * ct;
        if (i === 0) ctx.moveTo(map.sx(x), map.sy(y));
        else ctx.lineTo(map.sx(x), map.sy(y));
      }
      ctx.stroke();
      ctx.setLineDash([]);
    },
    meanCross: function (ctx, map, mu, color) {
      const px = map.sx(mu[0]), py = map.sy(mu[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(px - 6, py); ctx.lineTo(px + 6, py);
      ctx.moveTo(px, py - 6); ctx.lineTo(px, py + 6);
      ctx.stroke();
    },

    /* sample n points from N(mu, S) via Cholesky (deterministic rand) */
    cluster: function (rand, n, mu, S) {
      const l11 = Math.sqrt(S[0][0]);
      const l21 = S[0][1] / l11;
      const l22 = Math.sqrt(Math.max(S[1][1] - l21 * l21, 1e-9));
      const pts = [];
      for (let i = 0; i < n; i++) {
        const z1 = LR.gauss(rand), z2 = LR.gauss(rand);
        pts.push({ x: mu[0] + l11 * z1, y: mu[1] + l21 * z1 + l22 * z2 });
      }
      return pts;
    },

    /* posteriors of a query under k classes [{mu,S,prior}] */
    posteriors: function (x, y, classes) {
      const logs = classes.map(function (c) {
        return Math.log(c.prior) + GB.logMvn2(x, y, c.mu, c.S);
      });
      const m = Math.max.apply(null, logs);
      const e = logs.map((v) => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return { post: e.map((v) => v / s), logs: logs };
    },

    /* class colours + faint region tints */
    COLS: [C.green, C.orange, C.purple],
    TINTS: ["rgba(47,158,68,0.10)", "rgba(232,89,12,0.10)", "rgba(112,72,232,0.12)"],

    /* decision regions + boundary, computed on a grid over the plot area */
    regions: function (ctx, P, map, classes, opts) {
      opts = opts || {};
      const step = opts.step || 3;
      const cols = Math.ceil(P.w / step), rows = Math.ceil(P.h / step);
      const lab = new Int8Array(cols * rows);
      const pre = classes.map(function (c) {
        const inv = GB.inv2(c.S);
        return { inv: inv, half: 0.5 * Math.log(GB.det2(c.S)), lp: Math.log(c.prior), mu: c.mu };
      });
      for (let r = 0; r < rows; r++) {
        for (let cc = 0; cc < cols; cc++) {
          const px = P.x0 + cc * step + step / 2;
          const py = P.y0 + r * step + step / 2;
          const pt = map.inv(px, py);
          let best = -Infinity, bi = 0;
          for (let k = 0; k < pre.length; k++) {
            const g = pre[k];
            const dx = pt.x - g.mu[0], dy = pt.y - g.mu[1];
            const q = dx * (g.inv[0][0] * dx + g.inv[0][1] * dy) + dy * (g.inv[1][0] * dx + g.inv[1][1] * dy);
            const s = g.lp - g.half - 0.5 * q;
            if (s > best) { best = s; bi = k; }
          }
          lab[r * cols + cc] = bi;
        }
      }
      const tints = opts.tints || GB.TINTS;
      for (let r = 0; r < rows; r++) {
        for (let cc = 0; cc < cols; cc++) {
          ctx.fillStyle = tints[lab[r * cols + cc]];
          ctx.fillRect(P.x0 + cc * step, P.y0 + r * step, step, step);
        }
      }
      // boundary: mark grid edges where the winning class changes
      ctx.fillStyle = opts.boundary || "#333";
      for (let r = 0; r < rows; r++) {
        for (let cc = 0; cc < cols; cc++) {
          const me = lab[r * cols + cc];
          if (cc + 1 < cols && lab[r * cols + cc + 1] !== me) {
            ctx.fillRect(P.x0 + (cc + 1) * step - 1, P.y0 + r * step, 2, step);
          }
          if (r + 1 < rows && lab[(r + 1) * cols + cc] !== me) {
            ctx.fillRect(P.x0 + cc * step, P.y0 + (r + 1) * step - 1, step, 2);
          }
        }
      }
    },

    /* apply a covariance mode to per-class MLE fits.
       "full": each class keeps its own Σ_k (QDA)
       "shared": one pooled Σ for every class (LDA)
       "diag": pooled Σ with off-diagonals zeroed (Gaussian Naive Bayes,
               shared-diagonal form, so the linear boundary is exact) */
    withMode: function (fits, mode) {
      const total = fits.reduce((a, f) => a + f.n, 0);
      let shared = null;
      if (mode !== "full") {
        shared = GB.pool(fits);
        if (mode === "diag") shared = GB.diagOf(shared);
      }
      return fits.map(function (f) {
        return { mu: f.mu, S: mode === "full" ? f.S : shared, prior: f.n / total };
      });
    },
    /* Gaussian parameter count for k classes in d dims under a mode */
    paramCount: function (k, d, mode) {
      const cov = d + (d * d - d) / 2; // variances + covariances (symmetric)
      if (mode === "full") return k * (d + cov);
      if (mode === "shared") return k * d + cov;
      return k * d + d; // shared diagonal
    },

    /* ── shared datasets (deterministic) ─────────────────────
       Flagged addition: these synthetic datasets (spam features,
       three-class demo, morph demo) are generated by us for
       pedagogy; the source specifies the models, not the data. */
    spamData: function () {
      const r0 = LR.rng(6571);
      const ham = GB.cluster(r0, 40, [2.6, 2.4], [[1.1, 0.5], [0.5, 0.9]]);
      const r1 = LR.rng(6572);
      const spam = GB.cluster(r1, 25, [6.6, 6.1], [[2.4, -1.0], [-1.0, 1.5]]);
      const clip = (p) => ({ x: Math.min(9.7, Math.max(0.3, p.x)), y: Math.min(9.7, Math.max(0.3, p.y)) });
      return [ham.map(clip), spam.map(clip)];
    },
    triData: function () {
      const r0 = LR.rng(1711);
      const r1 = LR.rng(1712);
      const r2 = LR.rng(1713);
      const clip = (p) => ({ x: Math.min(9.7, Math.max(0.3, p.x)), y: Math.min(9.7, Math.max(0.3, p.y)) });
      return [
        GB.cluster(r0, 30, [2.9, 7.0], [[1.3, 0.6], [0.6, 0.9]]).map(clip),
        GB.cluster(r1, 24, [7.1, 6.9], [[0.8, -0.45], [-0.45, 1.2]]).map(clip),
        GB.cluster(r2, 36, [5.0, 2.7], [[2.1, 0.2], [0.2, 0.65]]).map(clip),
      ];
    },
    morphData: function () {
      const r0 = LR.rng(7401);
      const r1 = LR.rng(7402);
      const clip = (p) => ({ x: Math.min(9.7, Math.max(0.3, p.x)), y: Math.min(9.7, Math.max(0.3, p.y)) });
      return [
        GB.cluster(r0, 42, [3.4, 3.5], [[2.5, 1.25], [1.25, 1.5]]).map(clip),
        GB.cluster(r1, 26, [6.9, 6.0], [[0.5, -0.15], [-0.15, 0.42]]).map(clip),
      ];
    },
  });

  /* draw a scatter of points */
  function drawPts(ctx, map, pts, color) {
    pts.forEach(function (p) {
      LR.dot(ctx, map.sx(p.x), map.sy(p.y), 3.4, color);
    });
  }

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — the destination gallery (hook)
     ══════════════════════════════════════════════════════════ */
  LR.figs.hookgallery = function (mount) {
    LR.header(
      mount,
      "One dataset, three covariance assumptions, three boundaries",
      "Real mail (green) and spam (orange), each fitted with a Gaussian by MLE. Switch the covariance assumption and the implied boundary changes shape. Drag the ✉ query (or use arrow keys) to classify a new email."
    );

    const data = GB.spamData();
    const fits = data.map(GB.fit2d);
    let mode = "full";
    let q = { x: 4.9, y: 4.6 };

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const MODES = [
      { key: "full", label: "Per-class Σ (QDA)" },
      { key: "shared", label: "Shared Σ (LDA)" },
      { key: "diag", label: "Shared diagonal (Naive Bayes)" },
    ];
    const btns = MODES.map(function (m) {
      return LR.button(grp, m.label, function () { mode = m.key; render(); }, mode === m.key ? "on" : "");
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 430, {
      aria: "Two-class spam scatter with fitted Gaussian contours and the live decision boundary; a draggable query point is classified under the chosen covariance assumption",
    });
    cv.tabIndex = 0;

    const ro = LR.readout(mount, [
      { k: "mode", label: "regime" },
      { k: "post", label: "p(spam | x)" },
      { k: "verdict", label: "verdict" },
    ]);

    const P = { x0: 56, y0: 16, w: W - 76, h: H - 66, xmin: 0, xmax: 10, ymin: 0, ymax: 10, xlabel: "capitalized words (%)", ylabel: "links per email" };
    let map = null;

    function classes() { return GB.withMode(fits, mode); }

    function render() {
      btns.forEach((b, i) => b.classList.toggle("on", MODES[i].key === mode));
      ctx.clearRect(0, 0, W, H);
      map = LR.plot(ctx, P);
      const cls = classes();
      GB.regions(ctx, P, map, cls, { step: 3 });
      drawPts(ctx, map, data[0], GB.COLS[0]);
      drawPts(ctx, map, data[1], GB.COLS[1]);
      cls.forEach(function (c, k) {
        GB.ellipse(ctx, map, c.mu, c.S, 1, GB.COLS[k], 2);
        GB.ellipse(ctx, map, c.mu, c.S, 2, GB.COLS[k], 1.2, [4, 4]);
        GB.meanCross(ctx, map, c.mu, GB.COLS[k]);
      });
      // query marker (envelope)
      const px = map.sx(q.x), py = map.sy(q.y);
      ctx.fillStyle = C.purple;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.rect(px - 11, py - 8, 22, 16);
      ctx.fill(); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - 11, py - 8); ctx.lineTo(px, py + 2); ctx.lineTo(px + 11, py - 8);
      ctx.stroke();

      const r = GB.posteriors(q.x, q.y, cls);
      const spamP = r.post[1];
      const name = MODES.find((m) => m.key === mode).label;
      ro.set("mode", name);
      ro.set("post", LR.fmtF(spamP, 3) + "  (real mail: " + LR.fmtF(r.post[0], 3) + ")");
      ro.set("verdict", spamP > 0.5 ? "spam" : "real mail", spamP > 0.5 ? C.orange : C.green);
      cv.setAttribute("aria-label", "Query at " + LR.fmtF(q.x, 1) + ", " + LR.fmtF(q.y, 1) +
        "; posterior probability of spam " + LR.fmtF(spamP, 2) + " under " + name);
    }

    LR.drag(cv, W, H, {
      hit: function () { return true; },
      down: move, move: move,
    });
    function move(p) {
      if (!map) return;
      const pt = map.inv(p.x, p.y);
      q.x = Math.min(P.xmax, Math.max(P.xmin, pt.x));
      q.y = Math.min(P.ymax, Math.max(P.ymin, pt.y));
      render();
    }
    cv.addEventListener("keydown", function (e) {
      const s = e.shiftKey ? 1 : 0.25;
      if (e.key === "ArrowLeft") q.x = Math.max(P.xmin, q.x - s);
      else if (e.key === "ArrowRight") q.x = Math.min(P.xmax, q.x + s);
      else if (e.key === "ArrowUp") q.y = Math.min(P.ymax, q.y + s);
      else if (e.key === "ArrowDown") q.y = Math.max(P.ymin, q.y - s);
      else return;
      e.preventDefault();
      render();
    });

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — discriminative vs generative, side by side
     ══════════════════════════════════════════════════════════ */
  LR.figs.genvdisc = function (mount) {
    LR.header(
      mount,
      "Two philosophies, one dataset",
      "Left: learn a boundary directly, model nothing else. Right: model each class's distribution plus a prior; a boundary emerges where the posteriors tie. Toggle to spotlight each side."
    );

    const data = GB.spamData();
    const fits = data.map(GB.fit2d);
    const total = fits[0].n + fits[1].n;
    const priors = [fits[0].n / total, fits[1].n / total];
    // pooled covariance gives the best linear boundary for this data,
    // the kind of line logistic regression or an SVM would learn directly
    const Sp = GB.pool(fits);
    const inv = GB.inv2(Sp);
    const dm = [fits[1].mu[0] - fits[0].mu[0], fits[1].mu[1] - fits[0].mu[1]];
    const w = [inv[0][0] * dm[0] + inv[0][1] * dm[1], inv[1][0] * dm[0] + inv[1][1] * dm[1]];
    const mid = [(fits[1].mu[0] + fits[0].mu[0]) / 2, (fits[1].mu[1] + fits[0].mu[1]) / 2];
    const b = -(w[0] * mid[0] + w[1] * mid[1]) + Math.log(priors[1] / priors[0]);

    let view = 0; // 0 = discriminative, 1 = generative

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bd = LR.button(grp, "Discriminative", function () { view = 0; render(); }, "on");
    const bg = LR.button(grp, "Generative", function () { view = 1; render(); });

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left);
    grid.appendChild(right);
    left.appendChild(LR.el("div", "pane-label", "discriminative: a boundary, nothing else"));
    right.appendChild(LR.el("div", "pane-label", "generative: a Gaussian + prior per class"));
    const cvL = LR.canvas(left, 330, 300, { aria: "Discriminative view: two-class scatter with a single learned linear boundary" });
    const cvR = LR.canvas(right, 330, 300, { aria: "Generative view: the same scatter with a fitted Gaussian contour per class and the implied boundary" });
    const learnL = LR.el("div", "stepper-vars");
    const learnR = LR.el("div", "stepper-vars");
    left.appendChild(learnL);
    right.appendChild(learnR);
    const msg = LR.msg(mount);

    const P = { x0: 40, y0: 12, w: 330 - 54, h: 300 - 58, xmin: 0, xmax: 10, ymin: 0, ymax: 10, xlabel: "capitalized words (%)", ylabel: "links" };

    function lineEndpoints() {
      // w·x + b = 0 across the plot box
      const pts = [];
      const xs = [P.xmin, P.xmax];
      if (Math.abs(w[1]) > 1e-9) {
        xs.forEach(function (x) {
          const y = -(w[0] * x + b) / w[1];
          if (y >= P.ymin - 3 && y <= P.ymax + 3) pts.push([x, y]);
        });
      }
      [P.ymin, P.ymax].forEach(function (y) {
        if (Math.abs(w[0]) > 1e-9) {
          const x = -(w[1] * y + b) / w[0];
          if (x >= P.xmin && x <= P.xmax) pts.push([x, y]);
        }
      });
      return pts.slice(0, 2);
    }

    function drawPane(cvo, generative) {
      const ctx = cvo.ctx;
      ctx.clearRect(0, 0, cvo.W, cvo.H);
      const map = LR.plot(ctx, P);
      drawPts(ctx, map, data[0], GB.COLS[0]);
      drawPts(ctx, map, data[1], GB.COLS[1]);
      const ends = lineEndpoints();
      if (generative) {
        [0, 1].forEach(function (k) {
          const cc = { mu: fits[k].mu, S: fits[k].S };
          GB.ellipse(ctx, map, cc.mu, cc.S, 1, GB.COLS[k], 2);
          GB.ellipse(ctx, map, cc.mu, cc.S, 2, GB.COLS[k], 1.1, [4, 4]);
          GB.meanCross(ctx, map, cc.mu, GB.COLS[k]);
        });
        if (ends.length === 2) {
          ctx.strokeStyle = "#333";
          ctx.lineWidth = 1.8;
          ctx.setLineDash([6, 5]);
          ctx.beginPath();
          ctx.moveTo(map.sx(ends[0][0]), map.sy(ends[0][1]));
          ctx.lineTo(map.sx(ends[1][0]), map.sy(ends[1][1]));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      } else if (ends.length === 2) {
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(map.sx(ends[0][0]), map.sy(ends[0][1]));
        ctx.lineTo(map.sx(ends[1][0]), map.sy(ends[1][1]));
        ctx.stroke();
      }
    }

    function fmtS(S) {
      return "[[" + LR.fmtF(S[0][0], 2) + ", " + LR.fmtF(S[0][1], 2) + "], [" +
        LR.fmtF(S[1][0], 2) + ", " + LR.fmtF(S[1][1], 2) + "]]";
    }

    function render() {
      bd.classList.toggle("on", view === 0);
      bg.classList.toggle("on", view === 1);
      drawPane(cvL, false);
      drawPane(cvR, true);
      cvL.cv.style.opacity = view === 0 ? 1 : 0.35;
      cvR.cv.style.opacity = view === 1 ? 1 : 0.35;
      learnL.innerHTML =
        '<span class="var-name">learned</span>: one boundary, w·x + b = 0<br>' +
        '<span class="var-name">w</span> = [' + LR.fmtF(w[0], 2) + ", " + LR.fmtF(w[1], 2) + "], " +
        '<span class="var-name">b</span> = ' + LR.fmtF(b, 2) + "<br>" +
        "no model of p(x | y): it cannot say what spam looks like";
      learnR.innerHTML =
        '<span class="var-name">learned</span>: p(x | y) and p(y) per class<br>' +
        '<span class="var-name">μ̂₀</span> = [' + LR.fmtF(fits[0].mu[0], 2) + ", " + LR.fmtF(fits[0].mu[1], 2) + "], " +
        '<span class="var-name">Σ̂₀</span> = ' + fmtS(fits[0].S) + "<br>" +
        '<span class="var-name">μ̂₁</span> = [' + LR.fmtF(fits[1].mu[0], 2) + ", " + LR.fmtF(fits[1].mu[1], 2) + "], " +
        '<span class="var-name">Σ̂₁</span> = ' + fmtS(fits[1].S) + "<br>" +
        '<span class="var-name">priors</span> = [' + LR.fmtF(priors[0], 2) + ", " + LR.fmtF(priors[1], 2) + "] (from counts)";
      if (view === 0) {
        msg.show("The discriminative side commits to 3 numbers: a weight per feature and an offset. It separates, and that is all it can do.", "info");
      } else {
        msg.show("The generative side learns 11 numbers: two means, two covariances, and the priors, each fitted in closed form. The dashed boundary was never optimized; it is where the two posteriors tie.", "info");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — the 1-D Bayes classifier (signature figure)
     ══════════════════════════════════════════════════════════ */
  LR.figs.bayes1d = function (mount) {
    LR.header(
      mount,
      "The Bayes classifier on a number line",
      "Curves are prior-weighted class-likelihoods p(x | C) p(C). Drag the query, move the means and variances, slide the prior; every likelihood, posterior, and boundary is recomputed live."
    );

    const st = { mu0: 50, s0: 4, mu1: 42, s1: 6, p1: 0.3, x: 48 };
    const X0 = 20, X1 = 72;

    const { cv, ctx, W, H } = LR.canvas(mount, 660, 330, {
      aria: "Two class-conditional Gaussians weighted by their priors, with a draggable query point, shaded decision regions, and boundary lines where the posteriors cross",
    });
    cv.tabIndex = 0;

    // posterior bars (HTML)
    const barBox = LR.el("div", "attn-weightbar");
    mount.appendChild(barBox);
    const mkBar = function (label, color) {
      const item = LR.el("div", "wb-item");
      item.appendChild(LR.el("div", "wb-lab", label));
      const track = LR.el("div", "wb-track");
      const fill = LR.el("div", "wb-fill");
      fill.style.background = color;
      track.appendChild(fill);
      item.appendChild(track);
      const val = LR.el("div", "wb-val", "–");
      item.appendChild(val);
      barBox.appendChild(item);
      return { fill: fill, val: val };
    };
    const bar0 = mkBar("p(C=0 | x)", C.green);
    const bar1 = mkBar("p(C=1 | x)", C.orange);

    const ro = LR.readout(mount, [
      { k: "lik", label: "likelihoods at x" },
      { k: "verdict", label: "verdict" },
      { k: "bound", label: "decision point(s)" },
    ]);

    const ctl = LR.controls(mount);
    const sMu0 = LR.slider(ctl, "μ₀", 30, 65, 0.5, st.mu0, function (v) { st.mu0 = v; render(); });
    const sS0 = LR.slider(ctl, "σ₀", 1, 12, 0.25, st.s0, function (v) { st.s0 = v; render(); });
    const sMu1 = LR.slider(ctl, "μ₁", 30, 65, 0.5, st.mu1, function (v) { st.mu1 = v; render(); });
    const sS1 = LR.slider(ctl, "σ₁", 1, 12, 0.25, st.s1, function (v) { st.s1 = v; render(); });
    const sP1 = LR.slider(ctl, "p(C=1)", 0.05, 0.95, 0.05, st.p1, function (v) { st.p1 = v; render(); });
    LR.button(ctl, "Diabetes preset (x = 48)", function () {
      st.mu0 = 50; st.s0 = 4; st.mu1 = 42; st.s1 = 6; st.p1 = 0.3; st.x = 48;
      sMu0.set(50); sS0.set(4); sMu1.set(42); sS1.set(6); sP1.set(0.3);
      render();
    }, "primary");

    /* boundary: p1 N(x;μ1,σ1²) = p0 N(x;μ0,σ0²)  →  a x² + b x + c = 0 */
    function boundaryRoots() {
      const p0 = 1 - st.p1, p1 = st.p1;
      const v0 = st.s0 * st.s0, v1 = st.s1 * st.s1;
      const a = 1 / (2 * v0) - 1 / (2 * v1);
      const b = st.mu1 / v1 - st.mu0 / v0;
      const c = (st.mu0 * st.mu0) / (2 * v0) - (st.mu1 * st.mu1) / (2 * v1) +
        Math.log(st.s0 / st.s1) + Math.log(p1 / p0);
      if (Math.abs(a) < 1e-10) {
        return Math.abs(b) < 1e-12 ? [] : [-c / b];
      }
      const disc = b * b - 4 * a * c;
      if (disc < 0) return [];
      const r = Math.sqrt(disc);
      return [(-b - r) / (2 * a), (-b + r) / (2 * a)].sort((u, v) => u - v);
    }

    function score(k, x) {
      return k === 0
        ? (1 - st.p1) * GB.g1(x, st.mu0, st.s0 * st.s0)
        : st.p1 * GB.g1(x, st.mu1, st.s1 * st.s1);
    }

    let map = null;
    const P = { x0: 56, y0: 16, w: W - 76, h: H - 66, xmin: X0, xmax: X1, ymin: 0, ymax: 0.09, xlabel: "x (white-blood-cell count)", ylabel: "p(x | C) · p(C)" };

    function render() {
      // dynamic y scale
      let peak = 0;
      for (let x = X0; x <= X1; x += 0.25) peak = Math.max(peak, score(0, x), score(1, x));
      P.ymax = peak * 1.15;

      ctx.clearRect(0, 0, W, H);
      map = LR.plot(ctx, P);

      // decision-region band along the top of the plot
      const roots = boundaryRoots().filter((r) => r > X0 && r < X1);
      for (let px = P.x0; px < P.x0 + P.w; px += 2) {
        const x = map.inv(px, 0).x;
        const win = score(1, x) > score(0, x) ? 1 : 0;
        ctx.fillStyle = win === 1 ? "rgba(232,89,12,0.14)" : "rgba(47,158,68,0.12)";
        ctx.fillRect(px, P.y0, 2, 8);
      }

      // curves
      [0, 1].forEach(function (k) {
        ctx.strokeStyle = k === 0 ? C.green : C.orange;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        let first = true;
        for (let x = X0; x <= X1; x += 0.2) {
          const px = map.sx(x), py = map.sy(score(k, x));
          if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      });

      // boundary lines
      roots.forEach(function (r) {
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1.6;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(map.sx(r), P.y0);
        ctx.lineTo(map.sx(r), P.y0 + P.h);
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // query
      const l0 = GB.g1(st.x, st.mu0, st.s0 * st.s0);
      const l1 = GB.g1(st.x, st.mu1, st.s1 * st.s1);
      const w0 = (1 - st.p1) * l0, w1 = st.p1 * l1;
      const post1 = w1 / (w0 + w1);
      const qpx = map.sx(st.x);
      ctx.strokeStyle = C.purple;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(qpx, P.y0); ctx.lineTo(qpx, P.y0 + P.h);
      ctx.stroke();
      LR.dot(ctx, qpx, map.sy(w0), 4.5, "#fff", C.green);
      LR.dot(ctx, qpx, map.sy(w1), 4.5, "#fff", C.orange);
      ctx.fillStyle = C.purple;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("x = " + LR.fmtF(st.x, 1), qpx, P.y0 + P.h + 30);

      // class labels on curves
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.green;
      ctx.textAlign = "center";
      ctx.fillText("C = 0 (healthy)", map.sx(st.mu0), map.sy(score(0, st.mu0)) - 10);
      ctx.fillStyle = C.orange;
      ctx.fillText("C = 1 (diabetic)", map.sx(st.mu1), map.sy(score(1, st.mu1)) - 10);

      bar0.fill.style.width = Math.round((1 - post1) * 100) + "%";
      bar0.val.textContent = LR.fmtF(1 - post1, 3);
      bar1.fill.style.width = Math.round(post1 * 100) + "%";
      bar1.val.textContent = LR.fmtF(post1, 3);

      ro.set("lik", "p(x|C=0) = " + LR.fmtF(l0, 4) + ", p(x|C=1) = " + LR.fmtF(l1, 4));
      ro.set("verdict", post1 > 0.5 ? "C = 1 (diabetic)" : "C = 0 (healthy)", post1 > 0.5 ? C.orange : C.green);
      ro.set("bound", roots.length ? roots.map((r) => "x = " + LR.fmtF(r, 2)).join(", ") : "none in view");
      cv.setAttribute("aria-label", "Query x " + LR.fmtF(st.x, 1) + "; posterior of class 1 " + LR.fmtF(post1, 3) +
        "; verdict " + (post1 > 0.5 ? "class 1" : "class 0"));
    }

    LR.drag(cv, W, H, {
      hit: function () { return true; },
      down: move, move: move,
    });
    function move(p) {
      if (!map) return;
      st.x = Math.min(X1, Math.max(X0, map.inv(p.x, 0).x));
      render();
    }
    cv.addEventListener("keydown", function (e) {
      const s = e.shiftKey ? 2 : 0.5;
      if (e.key === "ArrowLeft") st.x = Math.max(X0, st.x - s);
      else if (e.key === "ArrowRight") st.x = Math.min(X1, st.x + s);
      else return;
      e.preventDefault();
      render();
    });

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — fit a Gaussian by MLE, literally
     ══════════════════════════════════════════════════════════ */
  LR.figs.fitmle = function (mount) {
    LR.header(
      mount,
      "Fit each class by MLE, made literal",
      "One class's points. Guess a Gaussian with the sliders and watch the log-likelihood; then press Fit by MLE and the closed-form μ̂, σ̂ take over."
    );

    let seed = 21;
    let pts = sample(seed);
    let fit = GB.fit1d(pts);
    const st = { mu: 46, s: 8, fitted: false };

    function sample(sd) {
      const rand = LR.rng(sd);
      const n = 14;
      const out = [];
      for (let i = 0; i < n; i++) out.push(42 + 6 * LR.gauss(rand));
      return out;
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 300, {
      aria: "A one-dimensional class sample with a candidate Gaussian curve and the maximum-likelihood fit",
    });

    const ro = LR.readout(mount, [
      { k: "guess", label: "your curve" },
      { k: "mle", label: "MLE fit" },
      { k: "ll", label: "log-likelihood (yours vs MLE)" },
    ]);

    const ctl = LR.controls(mount);
    const sMu = LR.slider(ctl, "μ (guess)", 25, 65, 0.25, st.mu, function (v) { st.mu = v; st.fitted = false; render(); });
    const sS = LR.slider(ctl, "σ (guess)", 1.5, 14, 0.25, st.s, function (v) { st.s = v; st.fitted = false; render(); });
    LR.button(ctl, "Fit by MLE", function () {
      st.mu = fit.mu; st.s = Math.sqrt(fit.s2); st.fitted = true;
      sMu.set(st.mu); sS.set(st.s);
      render();
    }, "primary");
    LR.button(ctl, "Resample ⟲", function () {
      seed += 1;
      pts = sample(seed);
      fit = GB.fit1d(pts);
      st.fitted = false;
      render();
    });
    const msg = LR.msg(mount);

    const P = { x0: 56, y0: 16, w: W - 76, h: H - 66, xmin: 20, xmax: 70, ymin: 0, ymax: 0.1, xlabel: "x (one class's feature values)", ylabel: "density" };

    function render() {
      const guessS2 = st.s * st.s;
      P.ymax = Math.max(GB.g1(fit.mu, fit.mu, fit.s2), GB.g1(st.mu, st.mu, guessS2)) * 1.2;
      ctx.clearRect(0, 0, W, H);
      const map = LR.plot(ctx, P);

      // rug of points
      pts.forEach(function (x) {
        ctx.strokeStyle = C.text;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(map.sx(x), P.y0 + P.h - 14);
        ctx.lineTo(map.sx(x), P.y0 + P.h);
        ctx.stroke();
      });

      // MLE curve (always visible, faint until fitted)
      ctx.strokeStyle = st.fitted ? C.orange : LR.C.faint;
      ctx.lineWidth = st.fitted ? 2.6 : 1.4;
      if (!st.fitted) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      let first = true;
      for (let x = P.xmin; x <= P.xmax; x += 0.25) {
        const px = map.sx(x), py = map.sy(GB.g1(x, fit.mu, fit.s2));
        if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // guess curve
      if (!st.fitted) {
        ctx.strokeStyle = C.purple;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        first = true;
        for (let x = P.xmin; x <= P.xmax; x += 0.25) {
          const px = map.sx(x), py = map.sy(GB.g1(x, st.mu, guessS2));
          if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }

      // mean markers
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(map.sx(fit.mu), P.y0);
      ctx.lineTo(map.sx(fit.mu), P.y0 + P.h);
      ctx.stroke();
      ctx.setLineDash([]);

      const llGuess = GB.logL1d(pts, st.mu, guessS2);
      const llMLE = GB.logL1d(pts, fit.mu, fit.s2);
      ro.set("guess", "μ = " + LR.fmtF(st.mu, 2) + ", σ = " + LR.fmtF(st.s, 2), C.purple);
      ro.set("mle", "μ̂ = " + LR.fmtF(fit.mu, 2) + ", σ̂ = " + LR.fmtF(Math.sqrt(fit.s2), 2) + "  (σ̂² = " + LR.fmtF(fit.s2, 2) + ")", C.orange);
      ro.set("ll", LR.fmtF(llGuess, 2) + " vs " + LR.fmtF(llMLE, 2));
      if (st.fitted) {
        msg.show("Fitted: the curve is the sample mean and sample variance of the " + pts.length + " points, computed in closed form. No hand-tuned (μ, σ) scores a higher log-likelihood.", "good");
      } else if (llMLE - llGuess < 0.5) {
        msg.show("Close. Your curve is within " + LR.fmtF(llMLE - llGuess, 2) + " log-likelihood of the optimum; press Fit by MLE to land exactly on it.", "info");
      } else {
        msg.show("Your curve trails the MLE by " + LR.fmtF(llMLE - llGuess, 1) + " in log-likelihood. Slide μ toward the data's centre of mass and match σ to its spread, or let the closed form do it.", "info");
      }
    }
    render();
  };
})();
