/* ══════════════════════════════════════════════════════════════
   figures-a.js — Logistic Regression lesson figures, sections 0–3
   Fig 0.1 hook · Fig 1.1 tries · Fig 1.2 losscompare
   Fig 2.1 shaper · Fig 3.1 probmap · Fig 3.2 exam
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared logistic core (also used by figures-b.js) ─────
     One implementation of the sigmoid, cross-entropy, the
     (y − t)x gradient, softmax, and gradient-descent fitting,
     shared by every figure in the lesson.                     */
  const LG = (LR.logit = {
    sig: function (z) { return 1 / (1 + Math.exp(-z)); },
    dot: function (w, f) {
      let s = 0;
      for (let i = 0; i < w.length; i++) s += w[i] * f[i];
      return s;
    },
    // per-example cross-entropy, clamped only to keep display finite
    ce: function (y, t) {
      const e = 1e-9;
      return -t * Math.log(Math.max(y, e)) - (1 - t) * Math.log(Math.max(1 - y, e));
    },
    // pts: [{f: [1, x1, ...], t: 0|1}]
    lossCE: function (w, pts, lam) {
      let s = 0;
      for (const p of pts) s += LG.ce(LG.sig(LG.dot(w, p.f)), p.t);
      s /= pts.length;
      if (lam) for (let j = 1; j < w.length; j++) s += lam * w[j] * w[j];
      return s;
    },
    lossSQ: function (w, pts) {
      let s = 0;
      for (const p of pts) {
        const y = LG.sig(LG.dot(w, p.f));
        s += 0.5 * (y - p.t) * (y - p.t);
      }
      return s / pts.length;
    },
    // gradient; loss "ce" → (y−t)x  (the cancellation),
    //           loss "sq" → (y−t)·y(1−y)·x  (the saturating one)
    grad: function (w, pts, lam, loss) {
      const g = new Array(w.length).fill(0);
      for (const p of pts) {
        const y = LG.sig(LG.dot(w, p.f));
        let r = y - p.t;
        if (loss === "sq") r *= y * (1 - y);
        for (let j = 0; j < w.length; j++) g[j] += r * p.f[j];
      }
      for (let j = 0; j < w.length; j++) g[j] /= pts.length;
      if (lam) for (let j = 1; j < w.length; j++) g[j] += 2 * lam * w[j];
      return g;
    },
    // full-batch gradient descent
    fit: function (pts, opts) {
      opts = opts || {};
      const d = pts[0].f.length;
      let w = (opts.w0 || new Array(d).fill(0)).slice();
      const lr = opts.lr === undefined ? 0.5 : opts.lr;
      const iters = opts.iters || 2000;
      const lam = opts.lam || 0;
      const loss = opts.loss || "ce";
      for (let k = 0; k < iters; k++) {
        const g = LG.grad(w, pts, lam, loss);
        for (let j = 0; j < d; j++) w[j] -= lr * g[j];
      }
      return w;
    },
    softmax: function (zs) {
      const m = Math.max.apply(null, zs);
      const e = zs.map((z) => Math.exp(z - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    },
  });

  const CLS_COLOR = { 0: C.green, 1: C.purple };

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — the hook: a verdict is not a confidence
     ════════════════════════════════════════════════════════════ */
  LR.figs.hook = function (mount) {
    LR.header(
      mount,
      "How sure is it?",
      "A trained hard-threshold classifier on one feature. Drag the white query anywhere, then toggle the probabilistic view."
    );

    // 1-D two-class data (deterministic, cleanly separated for the hook)
    const pts = [];
    [0.8, 1.4, 1.9, 2.5, 3.1, 3.7].forEach((x) => pts.push({ x, t: 0, f: [1, x] }));
    [5.3, 5.9, 6.4, 7.0, 7.6, 8.3].forEach((x) => pts.push({ x, t: 1, f: [1, x] }));
    const thr = 4.5; // hard-threshold boundary (any b in the gap gives zero training error)
    // sigmoid fit for the probabilistic view; small L2 keeps the ramp readable
    // on separable data instead of racing to a step
    const wFit = LG.fit(pts, { lr: 0.5, iters: 3000, lam: 0.02 });

    let probView = false;
    let qx = 4.9;

    const bar = LR.controls(mount);
    const tog = LR.button(bar, "Probabilistic view: off", function () {
      probView = !probView;
      tog.textContent = "Probabilistic view: " + (probView ? "on" : "off");
      tog.classList.toggle("on", probView);
      draw();
    }, "primary");
    const qs = LR.slider(bar, "query x", 0.2, 9.3, 0.05, qx, function (v) { qx = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 340, {
      aria: "One-dimensional classifier: a draggable query gets a hard verdict from a threshold, or a graded probability from a sigmoid",
    });
    const ro = LR.readout(mount, [
      { k: "verdict", label: "verdict" },
      { k: "conf", label: "how sure?" },
    ]);

    const P = { x0: 58, y0: 22, w: W - 82, h: H - 76, xmin: 0, xmax: 9.5, ymin: -0.18, ymax: 1.18, xlabel: "x (the one feature)", ylabel: "output y", yticks: [0, 0.5, 1] };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => SC && Math.abs(p.x - SC.sx(qx)) < 26,
      move: function (p) {
        qx = Math.max(0.2, Math.min(9.3, SC.inv(p.x, p.y).x));
        qs.set(qx);
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      if (!probView) {
        // hard step
        ctx.strokeStyle = C.text; ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(sx(P.xmin), sy(0)); ctx.lineTo(sx(thr), sy(0));
        ctx.moveTo(sx(thr), sy(1)); ctx.lineTo(sx(P.xmax), sy(1));
        ctx.stroke();
        ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(sx(thr), sy(0)); ctx.lineTo(sx(thr), sy(1)); ctx.stroke();
        ctx.setLineDash([]);
      } else {
        // fitted sigmoid
        ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
        ctx.beginPath();
        for (let i = 0; i <= 240; i++) {
          const x = P.xmin + ((P.xmax - P.xmin) * i) / 240;
          const y = LG.sig(wFit[0] + wFit[1] * x);
          i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
        }
        ctx.stroke();
        // 0.5 line
        ctx.setLineDash([4, 4]); ctx.strokeStyle = C.faint; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(sx(P.xmin), sy(0.5)); ctx.lineTo(sx(P.xmax), sy(0.5)); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore();

      // data
      pts.forEach((p) => LR.dot(ctx, sx(p.x), sy(p.t), 5.5, CLS_COLOR[p.t], "#fff"));

      // query
      const z = wFit[0] + wFit[1] * qx;
      const yq = probView ? LG.sig(z) : (qx >= thr ? 1 : 0);
      ctx.setLineDash([3, 3]); ctx.strokeStyle = C.orange; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(qx), sy(P.ymin)); ctx.lineTo(sx(qx), sy(yq)); ctx.stroke();
      ctx.setLineDash([]);
      LR.dot(ctx, sx(qx), sy(yq), 9, "#fff", C.orange);
      ctx.font = "800 12px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "center";
      ctx.fillText("?", sx(qx), sy(yq) + 4);

      // legend
      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = CLS_COLOR[0]; ctx.fillText("● class 0", P.x0 + 10, P.y0 + 16);
      ctx.fillStyle = CLS_COLOR[1]; ctx.fillText("● class 1", P.x0 + 10, P.y0 + 33);

      if (!probView) {
        const cls = qx >= thr ? 1 : 0;
        ro.set("verdict", "class " + cls, CLS_COLOR[cls]);
        ro.set("conf", "it cannot say: the output is the same 1 mm or 1 mile from the boundary", C.red);
      } else {
        const p = LG.sig(z);
        const cls = p >= 0.5 ? 1 : 0;
        ro.set("verdict", "class " + cls + " (p ≥ 0.5)", CLS_COLOR[cls]);
        ro.set("conf", "p(class 1 | x) = " + LR.fmtF(p, 3) + "  (" + LR.fmtF(p * 100, 1) + "%)", C.orange);
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — the four tries (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.tries = function (mount) {
    LR.header(
      mount,
      "Four tries, one dataset",
      "Each try is refit to the data live. Left: model output and boundary. Right: the loss terrain an optimizer would have to walk. Drag the ringed outlier."
    );

    // 1-D two-class data + one draggable class-1 outlier
    const base = [];
    [-3.4, -2.8, -2.3, -1.9, -1.4, -1.0, -0.6].forEach((x) => base.push({ x, t: 0 }));
    [0.4, 0.7, 1.1, 1.5, 1.9].forEach((x) => base.push({ x, t: 1 }));
    const outlier = { x: 6.0, t: 1, isOutlier: true };
    const pts = base.concat([outlier]);
    const aug = () => pts.map((p) => ({ f: [1, p.x], t: p.t, src: p }));

    let tri = 1;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = {};
    [[1, "Try #1 · threshold"], [2, "Try #2 · least squares"], [3, "Try #3 · σ + squared"], [4, "Try #4 · σ + cross-entropy"]].forEach(function (m) {
      btns[m[0]] = LR.button(group, m[1], function () {
        tri = m[0];
        for (const k in btns) btns[k].classList.toggle("on", +k === tri);
        draw();
      }, "small" + (m[0] === 1 ? " on" : ""));
    });
    const oSlider = LR.slider(bar, "outlier x", -4.5, 11.5, 0.1, outlier.x, function (v) {
      outlier.x = v; draw();
    }, (v) => LR.fmtF(v, 1));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Four attempts at a trainable classifier on one dataset, with the model output on the left and the loss terrain on the right; the outlier point is draggable",
    });
    const ro = LR.readout(mount, [
      { k: "loss", label: "training loss J" },
      { k: "b", label: "boundary" },
      { k: "miss", label: "misclassified" },
      { k: "og", label: "outlier's gradient pull" },
    ]);
    const msg = LR.msg(mount);

    const PL = { x0: 56, y0: 26, w: 470, h: H - 86, xmin: -5, xmax: 12, ymin: -0.6, ymax: 1.6, xlabel: "x", ylabel: "model output", yticks: [-0.5, 0, 0.5, 1, 1.5], title: "data + model" };
    const PR = { x0: 596, y0: 26, w: W - 596 - 18, h: H - 86, xmin: -5, xmax: 12, ymin: 0, ymax: 1, xlabel: "boundary position b", ylabel: "loss J(b)", title: "loss vs boundary" };
    let SCL = null;

    LR.drag(cv, W, H, {
      hit: (p) => SCL && (p.x - SCL.sx(outlier.x)) ** 2 + (p.y - SCL.sy(outlier.t)) ** 2 < 900,
      move: function (p) {
        outlier.x = Math.max(-4.5, Math.min(11.5, SCL.inv(p.x, p.y).x));
        oSlider.set(outlier.x);
        draw();
      },
    });

    // ---- per-try fits (all recomputed on every drag) ----
    function fitTry1() {
      // best hard threshold by direct 0-1 search (class 1 to the right)
      const cands = [];
      const xs = pts.map((p) => p.x).sort((a, b) => a - b);
      cands.push(xs[0] - 0.5);
      for (let i = 0; i + 1 < xs.length; i++) cands.push((xs[i] + xs[i + 1]) / 2);
      cands.push(xs[xs.length - 1] + 0.5);
      let best = cands[0], bestE = Infinity;
      cands.forEach(function (b) {
        let e = 0;
        pts.forEach((p) => { if ((p.x >= b ? 1 : 0) !== p.t) e++; });
        if (e < bestE) { bestE = e; best = b; }
      });
      return { b: best, err: bestE / pts.length };
    }
    function l01(b) {
      let e = 0;
      pts.forEach((p) => { if ((p.x >= b ? 1 : 0) !== p.t) e++; });
      return e / pts.length;
    }

    function draw() {
      const A = aug();
      ctx.clearRect(0, 0, W, H);
      SCL = LR.plot(ctx, PL);
      const { sx, sy } = SCL;

      // model + boundary + loss-vs-b curve, per try
      let boundary, model, J, gradTxt, missN = 0, outG = 0, lossOfB, steep = null;

      if (tri === 1) {
        const f1 = fitTry1();
        boundary = f1.b;
        model = (x) => (x >= boundary ? 1 : 0);
        J = f1.err;
        gradTxt = "0 (flat almost everywhere)";
        lossOfB = l01;
        outG = 0;
      } else if (tri === 2) {
        const ls = LR.fit1d(pts.map((p) => ({ x: p.x, t: p.t })));
        model = (x) => ls.w0 + ls.w1 * x;
        // decide by thresholding the score at the midpoint of the target values
        // (targets are 0/1 here; with ±1 targets this is the source's threshold at 0)
        boundary = (0.5 - ls.w0) / ls.w1;
        J = pts.reduce((s, p) => s + 0.5 * (model(p.x) - p.t) ** 2, 0) / pts.length;
        lossOfB = function (b) {
          let s = 0;
          pts.forEach((p) => { const z = ls.w1 * (p.x - b) + 0.5; s += 0.5 * (z - p.t) ** 2; });
          return s / pts.length;
        };
        outG = Math.hypot(model(outlier.x) - outlier.t, (model(outlier.x) - outlier.t) * outlier.x);
      } else {
        const loss = tri === 3 ? "sq" : "ce";
        const w = tri === 3
          ? LG.fit(A, { loss: "sq", lr: 1.2, iters: 4000, w0: [0, 0.4] })
          : LG.fit(A, { loss: "ce", lr: 0.5, iters: 2500, lam: 0.01 });
        model = (x) => LG.sig(w[0] + w[1] * x);
        boundary = -w[0] / w[1];
        steep = w[1];
        J = tri === 3 ? LG.lossSQ(w, A) : LG.lossCE(w, A, 0);
        lossOfB = function (b) {
          let s = 0;
          pts.forEach(function (p) {
            const y = LG.sig(steep * (p.x - b));
            s += tri === 3 ? 0.5 * (y - p.t) ** 2 : LG.ce(y, p.t);
          });
          return s / pts.length;
        };
        // outlier's per-example gradient magnitude (norm over [1, x] features)
        const yo = model(outlier.x);
        let r = yo - outlier.t;
        if (tri === 3) r *= yo * (1 - yo);
        outG = Math.abs(r) * Math.hypot(1, outlier.x);
      }
      missN = pts.reduce((s, p) => s + (((tri === 2 ? model(p.x) >= 0.5 : tri === 1 ? p.x >= boundary : model(p.x) >= 0.5) ? 1 : 0) !== p.t ? 1 : 0), 0);

      // ---- left panel ----
      ctx.save();
      ctx.beginPath(); ctx.rect(PL.x0, PL.y0, PL.w, PL.h); ctx.clip();
      // boundary
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.orange; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(sx(boundary), sy(PL.ymin)); ctx.lineTo(sx(boundary), sy(PL.ymax)); ctx.stroke();
      ctx.setLineDash([]);
      // model curve
      ctx.strokeStyle = tri === 4 ? C.orange : tri === 3 ? C.purple : tri === 2 ? C.red : C.text;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      if (tri === 1) {
        ctx.moveTo(sx(PL.xmin), sy(0)); ctx.lineTo(sx(boundary), sy(0));
        ctx.moveTo(sx(boundary), sy(1)); ctx.lineTo(sx(PL.xmax), sy(1));
      } else {
        for (let i = 0; i <= 260; i++) {
          const x = PL.xmin + ((PL.xmax - PL.xmin) * i) / 260;
          const y = model(x);
          i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
        }
      }
      ctx.stroke();
      ctx.restore();

      // points (misclassified get a red ring)
      pts.forEach(function (p) {
        const pred = tri === 1 ? (p.x >= boundary ? 1 : 0) : (model(p.x) >= 0.5 ? 1 : 0);
        const wrong = pred !== p.t;
        LR.dot(ctx, sx(p.x), sy(p.t), p.isOutlier ? 8 : 5.5, CLS_COLOR[p.t], wrong ? C.red : p.isOutlier ? C.orange : "#fff");
      });
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "center";
      ctx.fillText("drag me", sx(outlier.x), sy(outlier.t) - 13);

      // ---- right panel: loss vs boundary position ----
      const bs = [], ls = [];
      let lmax = 0;
      for (let i = 0; i <= 140; i++) {
        const b = PR.xmin + ((PR.xmax - PR.xmin) * i) / 140;
        const v = lossOfB(b);
        bs.push(b); ls.push(v);
        if (isFinite(v) && v > lmax) lmax = v;
      }
      PR.ymax = Math.max(0.5, lmax * 1.08);
      const SCR = LR.plot(ctx, PR);
      ctx.save();
      ctx.beginPath(); ctx.rect(PR.x0, PR.y0, PR.w, PR.h); ctx.clip();
      ctx.strokeStyle = C.purple; ctx.lineWidth = 2.2;
      ctx.beginPath();
      bs.forEach((b, i) => (i === 0 ? ctx.moveTo(SCR.sx(b), SCR.sy(ls[i])) : ctx.lineTo(SCR.sx(b), SCR.sy(ls[i]))));
      ctx.stroke();
      // current boundary marker
      LR.dot(ctx, SCR.sx(boundary), SCR.sy(lossOfB(boundary)), 5.5, C.orange, "#fff");
      ctx.restore();

      // ---- readouts + narration ----
      ro.set("loss", LR.fmtF(J, 3), C.orange);
      ro.set("b", "x = " + LR.fmtF(boundary, 2));
      ro.set("miss", missN + " of " + pts.length, missN ? C.red : C.green);
      ro.set("og", tri === 1 ? "0 (no gradient exists)" : "‖(∂L/∂w)‖ = " + LR.fmtF(outG, 3),
        tri === 1 ? C.red : outG < 0.05 ? C.red : C.green);

      const far = outlier.x > 7, wrongSide = outlier.x < -0.5;
      if (tri === 1) {
        msg.show("Try #1: the loss terrain on the right is a staircase. Zero slope on every flat step, so gradient descent cannot even tell which direction improves. Training is impossible by smooth optimization.", "bad");
      } else if (tri === 2) {
        msg.show(far
          ? (missN > 0
            ? "Try #2: the outlier is far on its own (correct!) side, but squared loss punishes its score for being “too correct”, so the fit has tilted far enough to give away " + missN + " good point" + (missN > 1 ? "s" : "") + " near the boundary. Count the red rings."
            : "Try #2: watch the boundary readout climb as you drag. The outlier is on the correct side, yet the fit keeps tilting toward it. Keep going right until a good point gets sacrificed.")
          : "Try #2: differentiable at last, and reasonable while no point is extreme. Now drag the outlier to the right and watch the boundary chase it.", far ? "bad" : "info");
      } else if (tri === 3) {
        msg.show(wrongSide
          ? "Try #3: the outlier is now a confident mistake, and its gradient pull (readout) is nearly zero: the y(1−y) factor mutes exactly the point that most needs fixing. Note the flat plateaus in the loss terrain."
          : "Try #3: bounded output fixes the “too correct” bug, and the far outlier no longer tilts anything. But check the loss terrain: plateaus far from the optimum. Drag the outlier to the far LEFT to make it a confident mistake.", wrongSide ? "bad" : "info");
      } else {
        msg.show(wrongSide
          ? "Try #4: same confident mistake, but cross-entropy keeps a strong gradient on it (readout) and the terrain stays sloped everywhere. The mislabelled-looking point still shouts, as it should."
          : "Try #4: bounded output AND a loss with slope everywhere. Drag the outlier far right (boundary holds) and far left (the gradient stays alive).", wrongSide ? "good" : "good");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.2 — the four losses for t = 1, as functions of z
     ════════════════════════════════════════════════════════════ */
  LR.figs.losscompare = function (mount) {
    LR.header(
      mount,
      "Four losses, one mistake",
      "One example with t = 1. Each curve is that example's loss as a function of the score z. Slide the probe, toggle curves."
    );

    const CURVES = [
      { k: "01", name: "zero/one", color: C.text, on: true, f: (z) => (z >= 0 ? 0 : 1), d: () => 0 },
      { k: "sq", name: "squared on z", color: C.red, on: true, f: (z) => 0.5 * (z - 1) ** 2, d: (z) => z - 1 },
      { k: "ss", name: "σ + squared", color: C.purple, on: true, f: (z) => 0.5 * (LG.sig(z) - 1) ** 2, d: (z) => { const y = LG.sig(z); return (y - 1) * y * (1 - y); } },
      { k: "ce", name: "σ + cross-entropy", color: C.orange, on: true, f: (z) => -Math.log(Math.max(LG.sig(z), 1e-12)), d: (z) => LG.sig(z) - 1 },
    ];

    let zq = -3;

    const bar = LR.controls(mount);
    CURVES.forEach(function (c) {
      const b = LR.button(bar, c.name, function () {
        c.on = !c.on;
        b.classList.toggle("on", c.on);
        draw();
      }, "small on");
      b.style.borderColor = c.color;
      b.style.color = c.color;
      b.classList.remove("on"); b.classList.add("on"); // keep .on styling minimal; colours carry identity
    });
    LR.slider(bar, "probe z", -5, 5, 0.05, zq, function (v) { zq = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 410, {
      aria: "The zero-one, squared, sigmoid-plus-squared, and cross-entropy losses for a positive example, plotted against the score",
    });
    const ro = LR.readout(mount, [
      { k: "v", label: "losses at z" },
      { k: "g", label: "slopes dL/dz at z" },
    ]);

    const P = { x0: 58, y0: 24, w: W - 84, h: H - 84, xmin: -5, xmax: 5, ymin: 0, ymax: 5, xlabel: "score z (t = 1, so left = wrong, right = correct)", ylabel: "loss L(z)" };

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const { sx, sy } = LR.plot(ctx, P);
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      CURVES.forEach(function (c) {
        if (!c.on) return;
        ctx.strokeStyle = c.color; ctx.lineWidth = 2.3;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i <= 400; i++) {
          const z = P.xmin + ((P.xmax - P.xmin) * i) / 400;
          const v = c.f(z);
          if (!isFinite(v)) { started = false; continue; }
          if (c.k === "01" && Math.abs(z) < 0.013) { started = false; continue; } // show the jump
          if (!started) { ctx.moveTo(sx(z), sy(Math.min(v, P.ymax))); started = true; }
          else ctx.lineTo(sx(z), sy(Math.min(v, P.ymax)));
        }
        ctx.stroke();
      });

      // probe
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C.faint; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(zq), sy(0)); ctx.lineTo(sx(zq), sy(P.ymax)); ctx.stroke();
      ctx.setLineDash([]);
      CURVES.forEach(function (c) {
        if (!c.on) return;
        const v = c.f(zq);
        if (v <= P.ymax) LR.dot(ctx, sx(zq), sy(v), 5, c.color, "#fff");
      });
      ctx.restore();

      // legend
      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      let ly = P.y0 + 16;
      CURVES.forEach(function (c) {
        ctx.fillStyle = c.on ? c.color : "#ccc";
        ctx.fillText("— " + c.name, P.x0 + P.w - 165, ly);
        ly += 17;
      });

      ro.set("v", CURVES.filter((c) => c.on).map((c) => c.name.split(" ")[0] + ": " + LR.fmtF(c.f(zq), 3)).join("   "));
      ro.set("g", CURVES.filter((c) => c.on).map((c) => c.name.split(" ")[0] + ": " + LR.fmtF(c.d(zq), 3)).join("   "), C.orange);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — the sigmoid shaper
     ════════════════════════════════════════════════════════════ */
  LR.figs.shaper = function (mount) {
    LR.header(
      mount,
      "Shape the sigmoid",
      "y = σ(w₀ + w₁x). The bias shifts the curve, the weight sharpens it, and at large w₁ it becomes Try #1's hard step."
    );

    let w0 = 0, w1 = 1, anim = null;

    const bar = LR.controls(mount);
    const s0 = LR.slider(bar, "w₀ (shift)", -8, 8, 0.1, w0, function (v) { w0 = v; draw(); }, (v) => LR.fmtF(v, 1));
    const s1 = LR.slider(bar, "w₁ (steepness)", -40, 40, 0.2, w1, function (v) { w1 = v; draw(); }, (v) => LR.fmtF(v, 1));
    LR.button(bar, "Step-function limit ▸", function () {
      if (anim) cancelAnimationFrame(anim);
      if (LR.reducedMotion) { w1 = 40; s1.set(w1); draw(); return; }
      const tick = function () {
        w1 = Math.min(40, w1 < 1 ? 1 : w1 * 1.045);
        s1.set(w1); draw();
        if (w1 < 40) anim = requestAnimationFrame(tick);
      };
      tick();
    }, "primary");
    LR.button(bar, "Reset ⟲", function () {
      if (anim) cancelAnimationFrame(anim);
      w0 = 0; w1 = 1; s0.set(w0); s1.set(w1); draw();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 380, {
      aria: "Sigmoid curve with sliders for the bias, which shifts it, and the weight, which sharpens it toward a step function",
    });
    const ro = LR.readout(mount, [
      { k: "cross", label: "y = 0.5 crossing (boundary)" },
      { k: "slope", label: "slope at the crossing (w₁/4)" },
      { k: "gsig", label: "avg gradient factor y(1−y) on [−6,6]" },
    ]);

    const P = { x0: 58, y0: 22, w: W - 84, h: H - 80, xmin: -6, xmax: 6, ymin: -0.08, ymax: 1.08, xlabel: "x", ylabel: "y = σ(w₀ + w₁x)", yticks: [0, 0.25, 0.5, 0.75, 1] };

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const { sx, sy } = LR.plot(ctx, P);

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // y = 0.5 guide
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C.faint; ctx.lineWidth = 1.1;
      ctx.beginPath(); ctx.moveTo(sx(P.xmin), sy(0.5)); ctx.lineTo(sx(P.xmax), sy(0.5)); ctx.stroke();
      ctx.setLineDash([]);

      // curve
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.8;
      ctx.beginPath();
      for (let i = 0; i <= 480; i++) {
        const x = P.xmin + ((P.xmax - P.xmin) * i) / 480;
        const y = LG.sig(w0 + w1 * x);
        i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
      }
      ctx.stroke();

      // crossing marker
      if (Math.abs(w1) > 1e-9) {
        const xc = -w0 / w1;
        if (xc >= P.xmin && xc <= P.xmax) {
          ctx.setLineDash([3, 3]); ctx.strokeStyle = C.green; ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(sx(xc), sy(P.ymin)); ctx.lineTo(sx(xc), sy(0.5)); ctx.stroke();
          ctx.setLineDash([]);
          LR.dot(ctx, sx(xc), sy(0.5), 6, C.green, "#fff");
        }
      }
      ctx.restore();

      // annotations
      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.muted;
      ctx.fillText("y → 1 as z → +∞", P.x0 + P.w - 140, sy(0.965));
      ctx.fillText("y → 0 as z → −∞", P.x0 + 10, sy(0.03));

      // real computed readouts
      let g = 0;
      const NS = 240;
      for (let i = 0; i <= NS; i++) {
        const x = -6 + (12 * i) / NS;
        const y = LG.sig(w0 + w1 * x);
        g += y * (1 - y);
      }
      g /= NS + 1;
      const xc = Math.abs(w1) > 1e-9 ? -w0 / w1 : null;
      ro.set("cross", xc === null ? "undefined (w₁ = 0, flat model)" : "x = " + LR.fmtF(xc, 3), C.green);
      ro.set("slope", LR.fmtF(w1 / 4, 2));
      ro.set("gsig", LR.fmtF(g, 4) + (g < 0.02 ? "  ← nearly a step: gradients dead" : ""), g < 0.02 ? C.red : undefined);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — probability heatmap with a draggable boundary
     ════════════════════════════════════════════════════════════ */
  LR.figs.probmap = function (mount) {
    LR.header(
      mount,
      "A line with probabilities attached",
      "The shading is σ(w₀ + w₁x₁ + w₂x₂), darker purple = higher p(t = 1). Drag the square to move the boundary, the circle to rotate it, the white point to query."
    );

    // deterministic 2-D dataset (shared style with the trainer in figures-b.js)
    const rand = LR.rng(11);
    const pts = [];
    for (let i = 0; i < 14; i++) pts.push({ x: 2.1 + LR.gauss(rand) * 0.75, y: 2.3 + LR.gauss(rand) * 0.75, t: 0 });
    for (let i = 0; i < 14; i++) pts.push({ x: 4.4 + LR.gauss(rand) * 0.75, y: 4.2 + LR.gauss(rand) * 0.75, t: 1 });
    const aug = pts.map((p) => ({ f: [1, p.x, p.y], t: p.t }));

    // boundary parametrized by a point c on the line, normal angle th, steepness s
    let c = { x: 3.4, y: 3.2 }, th = Math.PI / 4, s = 1.2;
    let q = { x: 4.6, y: 2.4 };

    function weights() {
      const w1 = s * Math.cos(th), w2 = s * Math.sin(th);
      return [-(w1 * c.x + w2 * c.y), w1, w2];
    }

    const bar = LR.controls(mount);
    LR.slider(bar, "steepness ‖w‖", 0.3, 6, 0.05, s, function (v) { s = v; heatDirty = true; draw(); }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Fit with cross-entropy ▸", function () {
      const w = LG.fit(aug, { lr: 0.5, iters: 3000, lam: 0.02 });
      s = Math.hypot(w[1], w[2]);
      th = Math.atan2(w[2], w[1]);
      // put c at the boundary point nearest the plot centre
      const pc = { x: 3.25, y: 3.25 };
      const d = (w[0] + w[1] * pc.x + w[2] * pc.y) / (s * s);
      c = { x: pc.x - d * w[1], y: pc.y - d * w[2] };
      heatDirty = true; draw();
    }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Two-class scatter with a probability heatmap, a draggable linear decision boundary at p equals one half, and a movable query point with its probability",
    });
    const ro = LR.readout(mount, [
      { k: "w", label: "w = [w₀, w₁, w₂]" },
      { k: "z", label: "logit at query" },
      { k: "p", label: "p(t = 1 | query)" },
    ]);

    const P = { x0: 58, y0: 20, w: W - 84, h: H - 76, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null, heatDirty = true;

    // cached heatmap
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const offCtx = off.getContext("2d");
    function renderHeat() {
      const w = weights();
      offCtx.clearRect(0, 0, W, H);
      const BLK = 6;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const p = LG.sig(w[0] + w[1] * wx + w[2] * wy);
          // single-hue ramp: white → purple with p
          offCtx.fillStyle = "rgba(112,72,232," + (p * 0.42).toFixed(3) + ")";
          offCtx.fillRect(px, py, BLK, BLK);
        }
      }
      heatDirty = false;
    }

    function handles() {
      // square translate handle at c, round rotate handle along the line direction
      const dir = { x: -Math.sin(th), y: Math.cos(th) };
      return { sq: c, rot: { x: c.x + 1.5 * dir.x, y: c.y + 1.5 * dir.y } };
    }

    LR.drag(cv, W, H, {
      hit: function (p) {
        const h = handles();
        if ((p.x - SC.sx(q.x)) ** 2 + (p.y - SC.sy(q.y)) ** 2 < 750) { dragging = "q"; return true; }
        if ((p.x - SC.sx(h.rot.x)) ** 2 + (p.y - SC.sy(h.rot.y)) ** 2 < 750) { dragging = "rot"; return true; }
        if ((p.x - SC.sx(h.sq.x)) ** 2 + (p.y - SC.sy(h.sq.y)) ** 2 < 750) { dragging = "sq"; return true; }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        d.x = Math.max(0.15, Math.min(6.35, d.x));
        d.y = Math.max(0.15, Math.min(6.35, d.y));
        if (dragging === "q") { q = d; draw(); return; }
        if (dragging === "sq") { c = d; }
        else if (dragging === "rot") { th = Math.atan2(d.y - c.y, d.x - c.x) - Math.PI / 2; }
        heatDirty = true;
        draw();
      },
      up: function () { dragging = null; },
    });

    function draw() {
      if (heatDirty) renderHeat();
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(off, 0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;
      const w = weights();

      // p = 0.5 contour: the line through c with direction (−sinθ, cosθ)
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      const dir = { x: -Math.sin(th), y: Math.cos(th) };
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(sx(c.x - 20 * dir.x), sy(c.y - 20 * dir.y));
      ctx.lineTo(sx(c.x + 20 * dir.x), sy(c.y + 20 * dir.y));
      ctx.stroke();
      ctx.restore();

      // data
      pts.forEach((p) => LR.dot(ctx, sx(p.x), sy(p.y), 5.5, CLS_COLOR[p.t], "#fff"));

      // handles
      const h = handles();
      ctx.fillStyle = C.orange;
      ctx.fillRect(sx(h.sq.x) - 7, sy(h.sq.y) - 7, 14, 14);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.strokeRect(sx(h.sq.x) - 7, sy(h.sq.y) - 7, 14, 14);
      LR.dot(ctx, sx(h.rot.x), sy(h.rot.y), 8, "#fff", C.orange);

      // query
      const zq = w[0] + w[1] * q.x + w[2] * q.y;
      const pq = LG.sig(zq);
      LR.dot(ctx, sx(q.x), sy(q.y), 9, "#fff", C.text);
      ctx.font = "800 11px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "center";
      ctx.fillText("?", sx(q.x), sy(q.y) + 4);
      ctx.font = "700 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.text;
      ctx.fillText("p = " + LR.fmtF(pq, 2), sx(q.x) + 12, sy(q.y) - 8);

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = CLS_COLOR[0]; ctx.fillText("● class 0", P.x0 + 10, P.y0 + 16);
      ctx.fillStyle = CLS_COLOR[1]; ctx.fillText("● class 1", P.x0 + 10, P.y0 + 33);
      ctx.fillStyle = C.orange; ctx.fillText("— p = 0.5 boundary", P.x0 + 10, P.y0 + 50);

      ro.set("w", "[" + LR.fmtF(w[0], 2) + ", " + LR.fmtF(w[1], 2) + ", " + LR.fmtF(w[2], 2) + "]");
      ro.set("z", LR.fmtF(zq, 3));
      ro.set("p", LR.fmtF(pq, 3) + "  (" + LR.fmtF(pq * 100, 1) + "%)", C.purple);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.2 — the exam-pass running example
     ════════════════════════════════════════════════════════════ */
  LR.figs.exam = function (mount) {
    LR.header(
      mount,
      "Will you pass?",
      "Twenty students, hours studied vs pass/fail, and the logistic curve fit to them by gradient descent. Read your own odds off the curve."
    );

    // hours-studied vs pass/fail. The slides use this scenario as the running
    // example; these specific 20 rows are the classic textbook version of it
    // (flagged: the concrete numbers are not printed in lecture-06-source-content.md).
    const hours = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5];
    const pass  = [0,   0,    0,   0,    0,   0,    1,    0,   1,    0,   1,    0,   1,    0,   1,   1,    1,   1,    1,   1];

    // fit by gradient descent on standardized hours (fast, well-conditioned),
    // then map the weights back to raw hours: same model, exact same curve
    const mu = hours.reduce((a, b) => a + b, 0) / hours.length;
    const sd = Math.sqrt(hours.reduce((a, b) => a + (b - mu) ** 2, 0) / hours.length);
    const augS = hours.map((h, i) => ({ f: [1, (h - mu) / sd], t: pass[i] }));
    const ws = LG.fit(augS, { lr: 1.0, iters: 20000 });
    const w1 = ws[1] / sd;
    const w0 = ws[0] - (ws[1] * mu) / sd;

    let hq = 2.0, tau = 0.5;

    const bar = LR.controls(mount);
    LR.slider(bar, "hours studied", 0, 6, 0.05, hq, function (v) { hq = v; draw(); }, (v) => LR.fmtF(v, 2) + " h");
    LR.slider(bar, "threshold τ", 0.05, 0.95, 0.01, tau, function (v) { tau = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Pass or fail versus hours studied with a fitted logistic curve, a movable query, and an adjustable decision threshold",
    });
    const ro = LR.readout(mount, [
      { k: "w", label: "fitted w₀, w₁" },
      { k: "p", label: "p(pass | hours)" },
      { k: "dec", label: "decision at τ" },
      { k: "target", label: "hours where p = τ" },
    ]);

    const P = { x0: 58, y0: 22, w: W - 84, h: H - 80, xmin: 0, xmax: 6, ymin: -0.08, ymax: 1.08, xlabel: "hours studied", ylabel: "p(pass)", yticks: [0, 0.25, 0.5, 0.75, 1] };

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const { sx, sy } = LR.plot(ctx, P);

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // threshold line
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C.red; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(sx(P.xmin), sy(tau)); ctx.lineTo(sx(P.xmax), sy(tau)); ctx.stroke();
      ctx.setLineDash([]);

      // fitted curve
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.8;
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const h = P.xmin + ((P.xmax - P.xmin) * i) / 300;
        const y = LG.sig(w0 + w1 * h);
        i === 0 ? ctx.moveTo(sx(h), sy(y)) : ctx.lineTo(sx(h), sy(y));
      }
      ctx.stroke();

      // query line
      const pq = LG.sig(w0 + w1 * hq);
      ctx.setLineDash([3, 3]); ctx.strokeStyle = C.text; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(hq), sy(P.ymin)); ctx.lineTo(sx(hq), sy(pq)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // data
      hours.forEach((h, i) => LR.dot(ctx, sx(h), sy(pass[i]), 5, CLS_COLOR[pass[i]], "#fff"));

      // query marker
      LR.dot(ctx, sx(hq), sy(pq), 8, "#fff", C.orange);

      // labels
      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = CLS_COLOR[1]; ctx.fillText("● passed (t = 1)", P.x0 + 10, P.y0 + 16);
      ctx.fillStyle = CLS_COLOR[0]; ctx.fillText("● failed (t = 0)", P.x0 + 10, P.y0 + 33);
      ctx.fillStyle = C.red; ctx.fillText("τ = " + LR.fmtF(tau, 2), P.x0 + P.w - 70, sy(tau) - 7);

      const pDec = pq >= tau;
      const hStar = (Math.log(tau / (1 - tau)) - w0) / w1;
      ro.set("w", LR.fmtF(w0, 2) + ", " + LR.fmtF(w1, 2));
      ro.set("p", LR.fmtF(pq, 3) + "  (" + LR.fmtF(pq * 100, 1) + "%)", C.orange);
      ro.set("dec", pDec ? "predict PASS" : "predict FAIL", pDec ? C.green : C.red);
      ro.set("target", LR.fmtF(hStar, 2) + " h");
    }
    draw();
  };
})();
