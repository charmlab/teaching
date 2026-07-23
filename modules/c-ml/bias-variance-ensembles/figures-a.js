/* ══════════════════════════════════════════════════════════════
   figures-a.js — Bias-Variance & Ensembles lesson, sections 0–3
   Fig 0.1 dartboard · Fig 1.1 resample · Fig 2.1 decomp
   Fig 3.1 vote
   All numbers computed live; nothing staged. Datasets are
   constructed for the figures (the lecture source describes the
   concepts, not specific datasets) — flagged here once.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared ensemble core (also used by figures-b.js) ───── */
  const E = (LR.ens = {
    // P(k or more of N wrong), each wrong independently w.p. eps.
    // Iterative binomial pmf: p_{k+1} = p_k · (N-k)/(k+1) · eps/(1-eps)
    binomTail: function (N, eps, kmin) {
      if (eps <= 0) return 0;
      if (eps >= 1) return 1;
      let p = Math.pow(1 - eps, N); // pmf at k = 0
      let tail = kmin <= 0 ? p : 0;
      for (let k = 0; k < N; k++) {
        p = (p * (N - k) * eps) / ((k + 1) * (1 - eps));
        if (k + 1 >= kmin) tail += p;
      }
      return Math.min(1, tail);
    },
    // majority of N (odd) simultaneously wrong
    majorityWrong: function (N, eps) {
      return E.binomTail(N, eps, Math.floor(N / 2) + 1);
    },
    // bootstrap resample of indices [0, n) with replacement
    bootstrap: function (n, rand) {
      const idx = new Array(n);
      for (let i = 0; i < n; i++) idx[i] = Math.floor(rand() * n);
      return idx;
    },
    mean: function (a) {
      let s = 0;
      for (const v of a) s += v;
      return s / a.length;
    },
    variance: function (a) {
      const m = E.mean(a);
      let s = 0;
      for (const v of a) s += (v - m) * (v - m);
      return s / a.length;
    },
  });

  /* ── shared 1-D regression world for Figs 1.1 and 2.1 ─────
     True function y*(x) = sin(2πx) on [0,1], label noise σ = 0.25.
     Each "dataset drawn from the population" is n = 20 pairs.  */
  const BV = (LR.bv = {
    f: (x) => Math.sin(2 * Math.PI * x),
    sigma: 0.25,
    n: 20,
    makeData: function (seed) {
      const rand = LR.rng(seed);
      const xs = [], ts = [];
      for (let i = 0; i < BV.n; i++) {
        const x = rand();
        xs.push(x);
        ts.push(BV.f(x) + LR.gauss(rand) * BV.sigma);
      }
      return { xs, ts };
    },
  });

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — the dartboard: two axes of error
     ════════════════════════════════════════════════════════════ */
  LR.figs.dartboard = function (mount) {
    LR.header(
      mount,
      "Four players, two kinds of error",
      "Each dart is one trained model's prediction; the bullseye is the truth. Throw a round and read the two numbers under each board: they are computed from the actual darts."
    );

    // board configs: aim offset (bias) and hand shake (variance), in board units
    const BOARDS = [
      { name: "low bias · low variance", ox: 0, oy: 0, sd: 7, tag: "the champion" },
      { name: "low bias · high variance", ox: 0, oy: 0, sd: 26, tag: "shaky hands" },
      { name: "high bias · low variance", ox: 36, oy: -26, sd: 7, tag: "systematic aim error" },
      { name: "high bias · high variance", ox: 36, oy: -26, sd: 26, tag: "both problems at once" },
    ];
    const NDARTS = 12;

    let throwCount = 0;
    let darts = BOARDS.map(() => []); // [{x,y} offsets from centre]
    let shown = 0; // darts revealed so far (animation)
    let anim = null;

    const bar = LR.controls(mount);
    LR.button(bar, "Throw a round 🎯", doThrow, "primary");
    LR.button(bar, "Reset ⟲", function () {
      if (anim) cancelAnimationFrame(anim);
      darts = BOARDS.map(() => []);
      shown = 0;
      draw();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 500, {
      aria: "Four dartboards showing the combinations of low and high bias with low and high variance; a button throws a round of darts on each",
    });

    function doThrow() {
      if (anim) cancelAnimationFrame(anim);
      throwCount += 1;
      const rand = LR.rng(300 + throwCount * 17); // deterministic per round
      darts = BOARDS.map(function (b) {
        const ds = [];
        for (let i = 0; i < NDARTS; i++) {
          ds.push({ x: b.ox + LR.gauss(rand) * b.sd, y: b.oy + LR.gauss(rand) * b.sd });
        }
        return ds;
      });
      if (LR.reducedMotion) { shown = NDARTS; draw(); return; }
      shown = 0;
      let last = 0;
      const tick = function (t) {
        if (t - last > 70) { shown = Math.min(NDARTS, shown + 1); last = t; draw(); }
        if (shown < NDARTS) anim = requestAnimationFrame(tick);
      };
      anim = requestAnimationFrame(tick);
    }

    function boardStats(ds) {
      // bias = distance of the dart centroid from the bullseye
      // spread = RMS distance of darts from their own centroid
      if (!ds.length) return null;
      const mx = E.mean(ds.map((d) => d.x)), my = E.mean(ds.map((d) => d.y));
      let s = 0;
      ds.forEach((d) => { s += (d.x - mx) ** 2 + (d.y - my) ** 2; });
      return { bias: Math.hypot(mx, my), spread: Math.sqrt(s / ds.length), mx, my };
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const R = 82;
      const CENTRES = [
        { x: 215, y: 130 }, { x: 605, y: 130 },
        { x: 215, y: 372 }, { x: 605, y: 372 },
      ];
      BOARDS.forEach(function (b, bi) {
        const c = CENTRES[bi];
        // rings
        [R, R * 0.66, R * 0.33].forEach(function (r, ri) {
          ctx.beginPath();
          ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
          ctx.fillStyle = ri === 0 ? "#fafafa" : ri === 1 ? "#f1f1f1" : "#e8e8e8";
          ctx.fill();
          ctx.strokeStyle = "#cfcfcf";
          ctx.lineWidth = 1.2;
          ctx.stroke();
        });
        // bullseye = the true target
        LR.dot(ctx, c.x, c.y, 5, C.green, "#ffffff");

        // darts thrown so far
        const ds = darts[bi].slice(0, shown);
        ds.forEach(function (d) {
          LR.dot(ctx, c.x + d.x, c.y + d.y, 4, C.orange, "#ffffff");
        });

        const st = boardStats(ds);
        if (st && ds.length > 1) {
          // centroid: the "average prediction"
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = C.purple;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(c.x, c.y);
          ctx.lineTo(c.x + st.mx, c.y + st.my);
          ctx.stroke();
          ctx.setLineDash([]);
          LR.dot(ctx, c.x + st.mx, c.y + st.my, 5, C.purple, "#ffffff");
        }

        // labels
        ctx.font = "700 13px Inter, sans-serif";
        ctx.fillStyle = C.text;
        ctx.textAlign = "center";
        ctx.fillText(b.name, c.x, c.y - R - 14);
        ctx.font = "500 11.5px Inter, sans-serif";
        ctx.fillStyle = C.faint;
        ctx.fillText(b.tag, c.x, c.y - R - 0);
        if (st && ds.length > 1) {
          ctx.font = "600 12px 'JetBrains Mono', Menlo, monospace";
          ctx.fillStyle = C.purple;
          ctx.fillText("bias " + LR.fmtF(st.bias, 1), c.x - 52, c.y + R + 20);
          ctx.fillStyle = C.orange;
          ctx.fillText("spread " + LR.fmtF(st.spread, 1), c.x + 52, c.y + R + 20);
        } else {
          ctx.font = "500 12px Inter, sans-serif";
          ctx.fillStyle = C.faint;
          ctx.fillText("press Throw", c.x, c.y + R + 20);
        }
      });

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.green;
      ctx.fillText("● truth", 16, 18);
      ctx.fillStyle = C.orange;
      ctx.fillText("● one model's prediction", 76, 18);
      ctx.fillStyle = C.purple;
      ctx.fillText("● average prediction", 260, 18);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — resample-and-refit (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.resample = function (mount) {
    LR.header(
      mount,
      "Same population, different datasets, different models",
      "Every press of Resample draws a fresh 20-point dataset from the same generator and refits a polynomial. Old fits stay as ghosts. Watch the spread of predictions at the marked query point x̂."
    );

    const XQ = 0.7; // fixed query input
    let deg = 3;
    let seedCounter = 0;
    let fits = []; // stored coefficient vectors, oldest first
    let cur = null; // {xs, ts, w}

    const bar = LR.controls(mount);
    LR.slider(bar, "model complexity (poly degree M)", 0, 9, 1, deg, function (v) {
      deg = Math.round(v);
      resetHistory();
      refit();
      draw();
      msg.show("Degree changed to " + deg + ": history cleared, one fresh fit drawn. Resample a dozen times to see the new spread.", "info");
    }, (v) => String(Math.round(v)));
    LR.button(bar, "Resample & refit ▸", function () { refit(); draw(); msg.hide(); }, "primary");
    LR.button(bar, "Resample ×10 ▸▸", function () {
      for (let i = 0; i < 10; i++) refit();
      draw();
      msg.hide();
    });
    LR.button(bar, "Reset ⟲", function () { resetHistory(); refit(); draw(); msg.hide(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 440, {
      aria: "Noisy one-dimensional datasets resampled from a fixed generator, each refit with a polynomial whose predictions at a fixed query point form a spreading cloud",
    });
    const ro = LR.readout(mount, [
      { k: "n", label: "models trained" },
      { k: "mean", label: "mean ŷ at x̂" },
      { k: "bias", label: "bias at x̂ (mean − y*)" },
      { k: "sd", label: "spread at x̂ (std of ŷ)" },
    ]);
    const msg = LR.msg(mount);

    function resetHistory() { fits = []; seedCounter = 0; }
    function refit() {
      seedCounter += 1;
      const d = BV.makeData(900 + seedCounter * 31);
      const w = LR.polyfit(d.xs, d.ts, deg, 0);
      cur = { xs: d.xs, ts: d.ts, w };
      fits.push(w);
      if (fits.length > 40) fits.shift();
      updateRO();
    }

    function updateRO() {
      const preds = fits.map((w) => LR.polyval(w, XQ));
      const m = E.mean(preds);
      const sd = Math.sqrt(E.variance(preds));
      const truth = BV.f(XQ);
      ro.set("n", String(fits.length));
      ro.set("mean", LR.fmtF(m, 3) + "  (y* = " + LR.fmtF(truth, 3) + ")");
      ro.set("bias", LR.fmtF(m - truth, 3), Math.abs(m - truth) > 0.15 ? C.red : C.green);
      ro.set("sd", fits.length > 1 ? LR.fmtF(sd, 3) : "need ≥ 2 fits", sd > 0.3 ? C.red : C.green);
    }

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 1, ymin: -2.2, ymax: 2.2, xlabel: "x", ylabel: "t" };

    function drawPoly(w, color, width, dash) {
      const { sx, sy } = SC;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= 240; i++) {
        const x = i / 240;
        let y = LR.polyval(w, x);
        y = Math.max(P.ymin - 0.5, Math.min(P.ymax + 0.5, y)); // clip exploding fits
        if (!started) { ctx.moveTo(sx(x), sy(y)); started = true; }
        else ctx.lineTo(sx(x), sy(y));
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    let SC = null;
    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath();
      ctx.rect(P.x0, P.y0, P.w, P.h);
      ctx.clip();

      // ghost fits (up to the last 15, excluding the current one)
      const ghosts = fits.slice(Math.max(0, fits.length - 16), fits.length - 1);
      ghosts.forEach(function (w) { drawPoly(w, "rgba(26,26,26,0.16)", 1.4); });

      // true function
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i <= 240; i++) {
        const x = i / 240;
        i === 0 ? ctx.moveTo(sx(x), sy(BV.f(x))) : ctx.lineTo(sx(x), sy(BV.f(x)));
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // current fit + its data
      if (cur) {
        drawPoly(cur.w, C.orange, 2.6);
        for (let i = 0; i < cur.xs.length; i++) {
          LR.dot(ctx, sx(cur.xs[i]), sy(Math.max(P.ymin, Math.min(P.ymax, cur.ts[i]))), 3.6, C.text);
        }
      }

      // query line and prediction cloud
      ctx.strokeStyle = C.purple;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(XQ), sy(P.ymin));
      ctx.lineTo(sx(XQ), sy(P.ymax));
      ctx.stroke();
      ctx.setLineDash([]);

      const preds = fits.map((w) => Math.max(P.ymin, Math.min(P.ymax, LR.polyval(w, XQ))));
      preds.forEach(function (y, i) {
        const isCur = i === preds.length - 1;
        LR.dot(ctx, sx(XQ), sy(y), isCur ? 5 : 3.4, isCur ? C.orange : "rgba(150,150,150,0.5)");
      });
      if (preds.length > 1) {
        const m = E.mean(fits.map((w) => LR.polyval(w, XQ)));
        LR.dot(ctx, sx(XQ), sy(Math.max(P.ymin, Math.min(P.ymax, m))), 6.5, C.purple, "#ffffff");
      }
      LR.dot(ctx, sx(XQ), sy(BV.f(XQ)), 6.5, C.green, "#ffffff");

      ctx.restore();

      // legend
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.green; ctx.fillText("- - true function y*(x), ● its value at x̂", P.x0 + 8, P.y0 + 16);
      ctx.fillStyle = C.orange; ctx.fillText("— current fit (ghosts: previous fits)", P.x0 + 8, P.y0 + 32);
      ctx.fillStyle = C.purple; ctx.fillText("● prediction cloud at x̂, ● its mean", P.x0 + 8, P.y0 + 48);
      ctx.fillStyle = C.text;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("x̂ = " + XQ, SC.sx(XQ), P.y0 + 14);
    }

    refit();
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — the decomposition, computed from many refits
     ════════════════════════════════════════════════════════════ */
  LR.figs.decomp = function (mount) {
    LR.header(
      mount,
      "Bias², variance, and the sweet spot",
      "For every degree, 60 datasets were resampled and refit, and the curves below are the real averages. Click the degree you think minimizes total test error; the total curve stays hidden until you commit."
    );

    // ---- the simulation: shared datasets across degrees ----
    // (degrees stop at 6: beyond that, unregularized fits on 20 random
    // points are so wild that even the Monte Carlo bias² estimate drowns
    // in noise — the variance story is already unmistakable by 6)
    const DEGS = [0, 1, 2, 3, 4, 5, 6];
    const R = 60; // resampled datasets
    const QX = []; // query grid, away from the edges where high-degree fits explode
    for (let i = 0; i <= 20; i++) QX.push(0.1 + (0.8 * i) / 20);

    const datasets = [];
    for (let r = 0; r < R; r++) datasets.push(BV.makeData(4000 + r * 13));

    const bias2 = [], vari = [], total = [];
    const bayes = BV.sigma * BV.sigma; // Var(t|x) = σ² = 0.0625
    DEGS.forEach(function (deg) {
      // preds[r][q]
      const preds = datasets.map((d) => {
        const w = LR.polyfit(d.xs, d.ts, deg, 0);
        return QX.map((x) => LR.polyval(w, x));
      });
      let b2 = 0, vv = 0;
      QX.forEach(function (x, qi) {
        const col = preds.map((row) => row[qi]);
        const m = E.mean(col);
        b2 += (m - BV.f(x)) ** 2;
        vv += E.variance(col);
      });
      b2 /= QX.length;
      vv /= QX.length;
      bias2.push(b2);
      vari.push(vv);
      total.push(b2 + vv + bayes);
    });
    const bestIdx = total.indexOf(Math.min.apply(null, total));

    let picked = null, revealed = false;

    const bar = LR.controls(mount);
    LR.button(bar, "Reset ⟲", function () { picked = null; revealed = false; msg.hide(); draw(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Bias squared, variance, and Bayes error versus polynomial degree, computed from sixty resampled fits; click a degree to reveal the total error curve",
    });
    const msg = LR.msg(mount);

    const ymax = Math.min(0.9, Math.max.apply(null, total) * 1.08);
    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: -0.4, xmax: 6.4, ymin: 0, ymax: ymax, xlabel: "model complexity (polynomial degree M)", ylabel: "error at test time", xticks: DEGS };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => SC && p.x > P.x0 && p.x < P.x0 + P.w && p.y > P.y0 && p.y < P.y0 + P.h,
      down: function (p) {
        picked = Math.max(0, Math.min(DEGS.length - 1, Math.round(SC.inv(p.x, p.y).x)));
        revealed = true;
        if (picked === bestIdx) {
          msg.show("✓ Degree " + DEGS[picked] + " is the computed minimum: total error " + LR.fmtF(total[picked], 3) + " = bias² " + LR.fmtF(bias2[picked], 3) + " + variance " + LR.fmtF(vari[picked], 3) + " + Bayes " + LR.fmtF(bayes, 4) + ".", "good");
        } else {
          msg.show("You picked degree " + DEGS[picked] + " (total " + LR.fmtF(total[picked], 3) + "). The computed minimum is degree " + DEGS[bestIdx] + " at " + LR.fmtF(total[bestIdx], 3) + ". Left of it bias² dominates; right of it variance takes over.", "info");
        }
        draw();
      },
      move: function () {},
    });

    function curve(arr, color, dash) {
      const { sx, sy } = SC;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.4;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      arr.forEach(function (v, i) {
        const y = Math.min(P.ymax, v);
        i === 0 ? ctx.moveTo(sx(i), sy(y)) : ctx.lineTo(sx(i), sy(y));
      });
      ctx.stroke();
      ctx.setLineDash([]);
      arr.forEach((v, i) => LR.dot(ctx, SC.sx(i), SC.sy(Math.min(P.ymax, v)), 3.2, color));
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // Bayes floor
      ctx.strokeStyle = C.faint;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(-0.4), sy(bayes));
      ctx.lineTo(sx(6.4), sy(bayes));
      ctx.stroke();
      ctx.setLineDash([]);

      curve(bias2, C.green);
      curve(vari, C.purple);
      if (revealed) {
        curve(total, C.orange);
        LR.dot(ctx, sx(bestIdx), sy(total[bestIdx]), 7, "#ffffff", C.orange);
        ctx.font = "600 11.5px Inter, sans-serif";
        ctx.fillStyle = C.orange;
        ctx.textAlign = "left";
        ctx.fillText("sweet spot: M = " + DEGS[bestIdx], sx(bestIdx) + 10, sy(total[bestIdx]) - 8);
      }
      if (picked !== null) {
        ctx.strokeStyle = C.text;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(sx(picked), sy(0));
        ctx.lineTo(sx(picked), sy(P.ymax));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = "600 11.5px Inter, sans-serif";
        ctx.fillStyle = C.text;
        ctx.textAlign = "left";
        ctx.fillText("your pick", sx(picked) + 6, P.y0 + 14);
      }

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.green; ctx.fillText("— bias²  (underfitting's term)", P.x0 + P.w - 232, P.y0 + 16);
      ctx.fillStyle = C.purple; ctx.fillText("— variance  (overfitting's term)", P.x0 + P.w - 232, P.y0 + 32);
      ctx.fillStyle = C.faint; ctx.fillText("· · Bayes error σ² = " + LR.fmtF(bayes, 4), P.x0 + P.w - 232, P.y0 + 48);
      ctx.fillStyle = revealed ? C.orange : "#bbbbbb";
      ctx.fillText(revealed ? "— total (revealed)" : "— total (hidden: click a degree)", P.x0 + P.w - 232, P.y0 + 64);

      if (!revealed) {
        ctx.font = "700 13px Inter, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText("click the degree you would ship", P.x0 + P.w / 2, P.y0 + P.h / 2 + 30);
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the majority-vote calculator
     ════════════════════════════════════════════════════════════ */
  LR.figs.vote = function (mount) {
    LR.header(
      mount,
      "How wrong can a majority be?",
      "N classifiers, each wrong with probability ε. If their errors are independent, the chance that a majority is wrong at once is a binomial tail, and it collapses as N grows. Correlation puts a floor under it."
    );

    let eps = 0.3, N = 21, rho = 0;
    const NS = [];
    for (let n = 1; n <= 101; n += 2) NS.push(n);

    const bar = LR.controls(mount);
    LR.slider(bar, "base error ε", 0.05, 0.45, 0.01, eps, function (v) { eps = v; draw(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar, "voters N (odd)", 1, 101, 2, N, function (v) { N = Math.round(v); draw(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "error correlation ρ", 0, 0.9, 0.05, rho, function (v) { rho = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 420, {
      aria: "Probability that a majority of N classifiers is simultaneously wrong versus N, for independent and correlated errors",
    });
    const ro = LR.readout(mount, [
      { k: "base", label: "single model error" },
      { k: "ind", label: "ensemble error (independent)" },
      { k: "cor", label: "ensemble error (ρ correlated)" },
    ]);
    const msg = LR.msg(mount);

    // Correlated-errors model (added beyond the source, flagged): with
    // probability ρ all N voters copy one shared classifier (wrong w.p. ε),
    // with probability 1−ρ they vote independently. This makes the
    // "independence caveat" quantitative in the simplest honest way.
    function corWrong(n, e, r) {
      return r * e + (1 - r) * E.majorityWrong(n, e);
    }

    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: 1, xmax: 101, ymin: 0, ymax: 0.5, xlabel: "N (number of voters)", ylabel: "P(majority wrong)" };

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // baseline: one model alone
      ctx.strokeStyle = C.faint;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(1), sy(eps));
      ctx.lineTo(sx(101), sy(eps));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.textAlign = "left";
      ctx.fillText("one model: ε = " + LR.fmtF(eps, 2), sx(72), sy(eps) - 6);

      // independent curve
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      NS.forEach(function (n, i) {
        const v = E.majorityWrong(n, eps);
        i === 0 ? ctx.moveTo(sx(n), sy(v)) : ctx.lineTo(sx(n), sy(v));
      });
      ctx.stroke();

      // correlated curve
      if (rho > 0) {
        ctx.strokeStyle = C.purple;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        NS.forEach(function (n, i) {
          const v = corWrong(n, eps, rho);
          i === 0 ? ctx.moveTo(sx(n), sy(v)) : ctx.lineTo(sx(n), sy(v));
        });
        ctx.stroke();
        // floor
        ctx.strokeStyle = "rgba(150,150,150,0.4)";
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(sx(1), sy(rho * eps));
        ctx.lineTo(sx(101), sy(rho * eps));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.purple;
        ctx.fillText("floor: ρ·ε = " + LR.fmtF(rho * eps, 3), sx(72), sy(rho * eps) - 6);
      }

      // markers at current N
      const pInd = E.majorityWrong(N, eps);
      const pCor = corWrong(N, eps, rho);
      LR.dot(ctx, sx(N), sy(pInd), 6, "#ffffff", C.orange);
      if (rho > 0) LR.dot(ctx, sx(N), sy(pCor), 6, "#ffffff", C.purple);

      // legend
      ctx.fillStyle = C.orange; ctx.fillText("— independent errors", P.x0 + P.w - 180, P.y0 + 16);
      ctx.fillStyle = rho > 0 ? C.purple : "#bbbbbb";
      ctx.fillText("— correlated (ρ = " + LR.fmtF(rho, 2) + ")", P.x0 + P.w - 180, P.y0 + 32);

      ro.set("base", LR.fmtF(eps, 3));
      ro.set("ind", LR.fmtF(pInd, 4), pInd < eps ? C.green : C.red);
      ro.set("cor", rho > 0 ? LR.fmtF(pCor, 4) : "ρ = 0 (same as independent)", rho > 0 ? C.purple : undefined);

      if (Math.abs(eps - 0.3) < 1e-9 && N === 21 && rho === 0) {
        msg.show("This is the lecture's example: ε = 0.30, N = 21, P(11 or more wrong) = " + LR.fmtF(pInd, 4) + " ≈ 0.026, an order of magnitude below the base rate of 0.3.", "good");
      } else if (eps >= 0.5 - 1e-9) {
        msg.show("At ε ≥ 0.5 voting cannot help: the tail no longer shrinks. Every voter must be better than chance.", "bad");
      } else {
        msg.hide();
      }
    }
    draw();
  };
})();
