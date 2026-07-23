/* ══════════════════════════════════════════════════════════════
   figures-a.js — MLE / MAP / Bayesian lesson, sections 0–2
   Fig 0.1 hook3 · Fig 1.1 problik (signature) · Fig 2.1 maximizer
   Fig 2.2 gaussfit
   Uses the shared estimation core LR.est from lesson-core.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const EST = LR.est;

  /* helper: draw a density curve y = f(theta) on a plot scaffold */
  function curve(ctx, sx, sy, f, xmin, xmax, color, width, dash) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2.4;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    const STEPS = 240;
    let started = false;
    for (let i = 0; i <= STEPS; i++) {
      const x = xmin + ((xmax - xmin) * i) / STEPS;
      const y = f(x);
      if (!isFinite(y)) { started = false; continue; }
      const px = sx(x), py = sy(y);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    if (dash) ctx.setLineDash([]);
  }

  /* helper: vertical marker line with a label */
  function marker(ctx, sx, sy, x, ytop, color, label, offset) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(sx(x), sy(0));
    ctx.lineTo(sx(x), sy(ytop));
    ctx.stroke();
    ctx.setLineDash([]);
    if (label) {
      ctx.fillStyle = color;
      ctx.font = "700 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, sx(x) + (offset || 0), sy(ytop) - 6);
    }
  }

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — the destination: one posterior, three readings
     ══════════════════════════════════════════════════════════ */
  LR.figs.hook3 = function (mount) {
    LR.header(
      mount,
      "Three answers to one question",
      "The posterior belief over θ after the observed flips, under a Beta(2,2) prior. The three estimators are three ways to read the same curve."
    );

    // start at the lecture's data; more flips come from a hidden true coin
    let nh = EST.COIN.NH, nt = EST.COIN.NT;
    const A = 2, B = 2;
    const TRUE_THETA = 0.55; // hidden coin behind the "flip more" button
    const rand = LR.rng(657);

    const bar = LR.controls(mount);
    LR.button(bar, "Flip 100 more ▸", function () {
      for (let i = 0; i < 100; i++) {
        if (rand() < TRUE_THETA) nh += 1; else nt += 1;
      }
      render();
    }, "primary");
    LR.button(bar, "Reset to 55 / 45 ⟲", function () {
      nh = EST.COIN.NH; nt = EST.COIN.NT;
      render();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 320, {
      aria: "Posterior density over theta with MLE, MAP, and posterior predictive mean marked; flipping more coins narrows the curve",
    });

    const ro = LR.readout(mount, [
      { k: "data", label: "data (N_H, N_T)" },
      { k: "mle", label: "θ̂ MLE" },
      { k: "map", label: "θ̂ MAP" },
      { k: "pred", label: "predictive mean" },
      { k: "sd", label: "posterior spread (sd)" },
    ]);

    function render() {
      const pa = nh + A, pb = nt + B;
      const mle = EST.mle(nh, nt);
      const map = EST.mapEst(nh, nt, A, B);
      const mean = EST.postPred(nh, nt, A, B);
      const sd = Math.sqrt((pa * pb) / ((pa + pb) * (pa + pb) * (pa + pb + 1)));

      // y range from the posterior peak
      let ymax = 0;
      for (let i = 1; i < 400; i++) {
        const v = EST.betaPdf(i / 400, pa, pb);
        if (v > ymax) ymax = v;
      }
      ymax *= 1.15;

      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 58, y0: 20, w: W - 84, h: H - 66,
        xmin: 0, xmax: 1, ymin: 0, ymax: ymax,
        xlabel: "θ (probability of heads)", ylabel: "posterior density p(θ | D)",
      };
      const { sx, sy } = LR.plot(ctx, P);

      // filled posterior
      ctx.beginPath();
      ctx.moveTo(sx(0), sy(0));
      for (let i = 0; i <= 240; i++) {
        const x = i / 240;
        ctx.lineTo(sx(x), sy(Math.min(EST.betaPdf(x, pa, pb), ymax)));
      }
      ctx.lineTo(sx(1), sy(0));
      ctx.closePath();
      ctx.fillStyle = "rgba(26,26,26, 0.10)";
      ctx.fill();
      curve(ctx, sx, sy, (x) => Math.min(EST.betaPdf(x, pa, pb), ymax), 0, 1, C.orange, 2.6);

      // the three estimator markers (offset labels so they never collide)
      const peak = EST.betaPdf(map, pa, pb);
      marker(ctx, sx, sy, mle, peak * 0.92, C.green, "MLE " + LR.fmtF(mle, 3), -46);
      marker(ctx, sx, sy, map, peak, C.orange, "MAP " + LR.fmtF(map, 3), 0);
      marker(ctx, sx, sy, mean, peak * 0.80, C.purple, "mean " + LR.fmtF(mean, 3), 50);

      ctx.fillStyle = C.muted;
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("posterior: Beta(" + pa + ", " + pb + ")", P.x0 + 8, P.y0 + 14);

      ro.set("data", "(" + nh + ", " + nt + ")  N = " + (nh + nt));
      ro.set("mle", LR.fmtF(mle, 4), C.green);
      ro.set("map", LR.fmtF(map, 4), C.orange);
      ro.set("pred", LR.fmtF(mean, 4), C.purple);
      ro.set("sd", LR.fmtF(sd, 4) + (sd < 0.03 ? "  (tight)" : ""));
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — probability vs likelihood (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.problik = function (mount) {
    LR.header(
      mount,
      "One Binomial formula, two readings",
      "Probability: θ fixed, distribution over data. Likelihood: data fixed, function over θ. Toggle between the two readings of p(D | θ) = C(n,k) θᵏ(1−θ)ⁿ⁻ᵏ."
    );

    let view = "prob"; // 'prob' | 'lik'
    let theta = 0.5;   // fixed parameter in probability view
    let nh = 55, nt = 45; // fixed data in likelihood view

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bProb = LR.button(grp, "probability: fix θ, vary data", function () { view = "prob"; render(); }, "on");
    const bLik = LR.button(grp, "likelihood: fix data, vary θ", function () { view = "lik"; render(); });

    const ctlProb = LR.el("div", "fig-controls");
    mount.appendChild(ctlProb);
    const thetaSlider = LR.slider(ctlProb, "θ (fixed)", 0.05, 0.95, 0.01, theta, function (v) {
      theta = v; render();
    }, (v) => LR.fmtF(v, 2));

    const ctlLik = LR.el("div", "fig-controls");
    mount.appendChild(ctlLik);
    const nhIn = LR.numInput(ctlLik, "N_H (fixed) ", nh, 0, 500, function (v) { nh = v; render(); });
    const ntIn = LR.numInput(ctlLik, "N_T (fixed) ", nt, 0, 500, function (v) { nt = v; render(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Toggleable chart: bar chart of binomial probabilities over outcomes, or likelihood curve over theta",
    });

    const ro = LR.readout(mount, [
      { k: "eq", label: "reading" },
      { k: "sum", label: "total" },
      { k: "star", label: "highlight" },
    ]);

    function renderProb() {
      const n = 100;
      const probs = [];
      let sum = 0, ymax = 0;
      for (let k = 0; k <= n; k++) {
        const p = EST.binomPmf(n, k, theta);
        probs.push(p);
        sum += p;
        if (p > ymax) ymax = p;
      }
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 58, y0: 20, w: W - 84, h: H - 66,
        xmin: 0, xmax: n, ymin: 0, ymax: ymax * 1.15,
        xlabel: "k (number of heads in n = 100 flips)", ylabel: "p(k heads | θ)",
        xticks: [0, 20, 40, 55, 60, 80, 100],
      };
      const { sx, sy } = LR.plot(ctx, P);
      const bw = Math.max(1.5, (P.w / (n + 1)) * 0.85);
      for (let k = 0; k <= n; k++) {
        const is55 = k === 55;
        ctx.fillStyle = is55 ? C.purple : C.orange;
        const px = sx(k) - bw / 2;
        const py = sy(probs[k]);
        ctx.fillRect(px, py, bw, sy(0) - py);
      }
      // annotate k = 55
      ctx.fillStyle = C.purple;
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.textAlign = "center";
      const y55 = sy(probs[55]);
      ctx.fillText("k = 55", sx(55), Math.max(P.y0 + 12, y55 - 8));

      ro.set("eq", "p(D | θ = " + LR.fmtF(theta, 2) + "), a distribution over outcomes");
      ro.set("sum", "Σₖ p(k) = " + LR.fmtF(sum, 4) + "  (sums to 1)", C.green);
      ro.set("star", "p(55 heads | θ = " + LR.fmtF(theta, 2) + ") = " + probs[55].toExponential(3), C.purple);
    }

    function renderLik() {
      const n = nh + nt;
      const f = (th) => EST.binomPmf(n, nh, th);
      let ymax = 0, area = 0;
      const STEPS = 400;
      for (let i = 0; i <= STEPS; i++) {
        const v = f(i / STEPS);
        if (v > ymax) ymax = v;
        area += v / (STEPS + 1);
      }
      ymax = ymax * 1.15 || 1;
      const mle = EST.mle(nh, nt);

      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 58, y0: 20, w: W - 84, h: H - 66,
        xmin: 0, xmax: 1, ymin: 0, ymax: ymax,
        xlabel: "θ (candidate parameter)", ylabel: "L(θ; D) = p(D | θ)",
      };
      const { sx, sy } = LR.plot(ctx, P);
      curve(ctx, sx, sy, f, 0, 1, C.orange, 2.6);
      if (isFinite(mle)) {
        marker(ctx, sx, sy, mle, f(mle), C.green, "peak at " + LR.fmtF(mle, 3), 0);
      }
      ctx.fillStyle = C.muted;
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("data frozen at (N_H, N_T) = (" + nh + ", " + nt + ")", P.x0 + 8, P.y0 + 14);

      ro.set("eq", "L(θ; D), a score over candidate θ");
      ro.set("sum", "∫ L dθ ≈ " + LR.fmtF(area, 4) + "  (not 1: not a distribution over θ)", C.red);
      ro.set("star", "L(" + LR.fmtF(mle, 2) + "; D) = " + (isFinite(mle) ? f(mle).toExponential(3) : "–"), C.green);
    }

    function render() {
      bProb.classList.toggle("on", view === "prob");
      bLik.classList.toggle("on", view === "lik");
      ctlProb.style.display = view === "prob" ? "" : "none";
      ctlLik.style.display = view === "lik" ? "" : "none";
      if (view === "prob") renderProb();
      else renderLik();
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — the likelihood maximizer: draggable θ
     ══════════════════════════════════════════════════════════ */
  LR.figs.maximizer = function (mount) {
    LR.header(
      mount,
      "Find the peak by hand, then trust the formula",
      "Drag the orange handle along ℓ(θ) = N_H log θ + N_T log(1−θ). The green marker is the analytic maximizer N_H / (N_H + N_T)."
    );

    let nh = 55, nt = 45;
    let theta = 0.30; // start away from the peak so there is something to find

    const bar = LR.controls(mount);
    const nhIn = LR.numInput(bar, "N_H ", nh, 0, 1000, function (v) { nh = v; clampTheta(); render(); });
    const ntIn = LR.numInput(bar, "N_T ", nt, 0, 1000, function (v) { nt = v; clampTheta(); render(); });
    LR.button(bar, "Snap to θ̂ ▸", function () {
      const m = EST.mle(nh, nt);
      if (isFinite(m)) theta = Math.min(0.99, Math.max(0.01, m));
      render();
    }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Log-likelihood curve over theta with a draggable handle; use arrow keys to move theta",
    });
    cv.tabIndex = 0;
    cv.addEventListener("keydown", function (e) {
      const step = e.shiftKey ? 0.05 : 0.01;
      if (e.key === "ArrowLeft") { theta = Math.max(0.01, theta - step); render(); e.preventDefault(); }
      if (e.key === "ArrowRight") { theta = Math.min(0.99, theta + step); render(); e.preventDefault(); }
    });

    const ro = LR.readout(mount, [
      { k: "theta", label: "your θ" },
      { k: "ell", label: "ℓ(θ)" },
      { k: "mle", label: "θ̂ = N_H/(N_H+N_T)" },
      { k: "gap", label: "gap to peak" },
    ]);
    const msg = LR.msg(mount);

    function clampTheta() { theta = Math.min(0.99, Math.max(0.01, theta)); }

    // plot geometry kept in closure for dragging
    let geom = null;

    function yRange() {
      // show the curve from its peak down a sensible depth
      const m = EST.mle(nh, nt);
      const peak = isFinite(m) ? EST.logLik(Math.min(0.99, Math.max(0.01, m)), nh, nt) : 0;
      const lo = Math.min(EST.logLik(0.02, nh, nt), EST.logLik(0.98, nh, nt));
      const depth = Math.max(6, peak - lo);
      return { ymin: peak - Math.min(depth, Math.max(10, (nh + nt) * 0.25)), ymax: peak + Math.max(1.5, (peak - lo) * 0.06) };
    }

    function render() {
      const { ymin, ymax } = yRange();
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 64, y0: 20, w: W - 92, h: H - 66,
        xmin: 0, xmax: 1, ymin: ymin, ymax: ymax,
        xlabel: "θ", ylabel: "log-likelihood ℓ(θ)",
      };
      const { sx, sy } = LR.plot(ctx, P);
      geom = { sx, sy, P };

      const f = (th) => Math.max(ymin, EST.logLik(th, nh, nt));
      curve(ctx, sx, sy, f, 0.005, 0.995, C.orange, 2.6);

      const m = EST.mle(nh, nt);
      if (isFinite(m) && m > 0 && m < 1) {
        // green analytic peak
        ctx.setLineDash([4, 3]);
        ctx.strokeStyle = C.green;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(sx(m), sy(ymin));
        ctx.lineTo(sx(m), sy(EST.logLik(m, nh, nt)));
        ctx.stroke();
        ctx.setLineDash([]);
        LR.dot(ctx, sx(m), sy(EST.logLik(m, nh, nt)), 5, C.green, "#ffffff");
        ctx.fillStyle = C.green;
        ctx.font = "700 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("θ̂ = " + LR.fmtF(m, 3), sx(m), sy(EST.logLik(m, nh, nt)) - 12);
      }

      // draggable handle
      const ly = EST.logLik(theta, nh, nt);
      LR.dot(ctx, sx(theta), sy(Math.max(ymin, ly)), 8, C.orange, "#ffffff");
      ctx.fillStyle = C.orange;
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("θ = " + LR.fmtF(theta, 2), sx(theta), sy(Math.max(ymin, ly)) + 24);

      const gap = isFinite(m) ? EST.logLik(m, nh, nt) - ly : NaN;
      ro.set("theta", LR.fmtF(theta, 3), C.orange);
      ro.set("ell", LR.fmtF(ly, 2));
      ro.set("mle", isFinite(m) ? LR.fmtF(m, 4) + "  (ℓ = " + LR.fmtF(EST.logLik(m, nh, nt), 2) + ")" : "undefined (no data)", C.green);
      ro.set("gap", isFinite(gap) ? LR.fmtF(gap, 3) : "–");

      if (isFinite(gap) && gap < 0.005) {
        msg.show("You are at the maximum: the derivative N_H/θ − N_T/(1−θ) crosses zero here, and no other θ explains this data better.", "good");
      } else if (isFinite(gap) && gap < 0.15) {
        msg.show("Warm. The curve is flat near its peak: nearby θ values explain the data almost as well, which is what posterior width will quantify later.", "info");
      } else {
        msg.hide();
      }
    }

    function setThetaFromX(px) {
      if (!geom) return;
      theta = Math.min(0.99, Math.max(0.01, (px - geom.P.x0) / geom.P.w));
      render();
    }
    LR.drag(cv, W, H, {
      hit: function (p) { return geom && p.x > geom.P.x0 - 20 && p.x < geom.P.x0 + geom.P.w + 20; },
      down: function (p) { setThetaFromX(p.x); },
      move: function (p) { setThetaFromX(p.x); },
    });

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.2 — Gaussian fit by hand vs the MLE
     ══════════════════════════════════════════════════════════ */
  LR.figs.gaussfit = function (mount) {
    LR.header(
      mount,
      "Fit the bell to the temperatures",
      "Drag on the plot: horizontal moves μ, vertical changes σ. The dashed green bell is the MLE (μ̂ = 15, σ̂ = √6 ≈ 2.449); try to beat its log-likelihood."
    );

    const xs = EST.TEMPS;
    const fit = EST.gaussMLE(xs);
    let mu = 12.0, sigma = 4.0; // deliberately off so there is a game

    const bar = LR.controls(mount);
    const muS = LR.slider(bar, "μ", 8, 22, 0.1, mu, function (v) { mu = v; render(); }, (v) => LR.fmtF(v, 1));
    const sgS = LR.slider(bar, "σ", 0.5, 8, 0.1, sigma, function (v) { sigma = v; render(); }, (v) => LR.fmtF(v, 1));
    LR.button(bar, "Snap to MLE ▸", function () {
      mu = fit.mu; sigma = fit.sigma;
      muS.set(mu); sgS.set(LR.fmt(sigma, 1));
      render();
    }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Scatter of ten temperature values with a draggable Gaussian curve and the MLE fit shown dashed",
    });

    const ro = LR.readout(mount, [
      { k: "yours", label: "your (μ, σ)" },
      { k: "ll", label: "ℓ(μ, σ)" },
      { k: "best", label: "MLE (μ̂, σ̂)" },
      { k: "bestll", label: "ℓ at MLE" },
    ]);
    const msg = LR.msg(mount);

    const P = {
      x0: 58, y0: 20, w: 820 - 84, h: 330 - 66,
      xmin: 6, xmax: 24, ymin: 0, ymax: 0.28,
      xlabel: "x (daily high, °C)", ylabel: "density",
    };

    function render() {
      ctx.clearRect(0, 0, W, H);
      const { sx, sy } = LR.plot(ctx, P);

      // MLE bell (dashed green)
      curve(ctx, sx, sy, (x) => EST.gaussPdf(x, fit.mu, fit.sigma), P.xmin, P.xmax, C.green, 2, [6, 4]);
      // your bell
      curve(ctx, sx, sy, (x) => Math.min(EST.gaussPdf(x, mu, sigma), P.ymax), P.xmin, P.xmax, C.orange, 2.6);

      // data points on the axis, with per-point density stems
      ctx.strokeStyle = "rgba(26,26,26,0.35)";
      ctx.lineWidth = 1.2;
      xs.forEach(function (x) {
        ctx.beginPath();
        ctx.moveTo(sx(x), sy(0));
        ctx.lineTo(sx(x), sy(Math.min(EST.gaussPdf(x, mu, sigma), P.ymax)));
        ctx.stroke();
      });
      xs.forEach(function (x) { LR.dot(ctx, sx(x), sy(0), 4.5, C.text); });

      // mu marker
      ctx.fillStyle = C.orange;
      ctx.font = "700 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("μ = " + LR.fmtF(mu, 1), sx(mu), sy(Math.min(EST.gaussPdf(mu, mu, sigma), P.ymax)) - 8);

      const ll = EST.gaussLogLik(xs, mu, sigma);
      const bestll = EST.gaussLogLik(xs, fit.mu, fit.sigma);
      ro.set("yours", "(" + LR.fmtF(mu, 2) + ", " + LR.fmtF(sigma, 2) + ")", C.orange);
      ro.set("ll", LR.fmtF(ll, 3), C.orange);
      ro.set("best", "(" + LR.fmtF(fit.mu, 2) + ", " + LR.fmtF(fit.sigma, 3) + ")", C.green);
      ro.set("bestll", LR.fmtF(bestll, 3), C.green);

      const gap = bestll - ll;
      if (gap < 0.005) {
        msg.show("This is the maximum likelihood fit: sample mean and sample standard deviation. The stems show each point's density under the bell; their log-sum is what you maximized.", "good");
      } else if (gap < 0.5) {
        msg.show("Close: within " + LR.fmtF(gap, 2) + " log-likelihood of the optimum. The surface is flat near the top, steep far away.", "info");
      } else {
        msg.hide();
      }
    }

    LR.drag(cv, W, H, {
      move: function (p) {
        mu = Math.min(P.xmax, Math.max(P.xmin, P.xmin + ((p.x - P.x0) / P.w) * (P.xmax - P.xmin)));
        // vertical position maps to sigma: top = narrow, bottom = wide
        const frac = Math.min(1, Math.max(0, (p.y - P.y0) / P.h));
        sigma = 0.5 + frac * 7.5;
        muS.set(LR.fmt(mu, 1));
        sgS.set(LR.fmt(sigma, 1));
        render();
      },
    });

    render();
  };
})();
