/* ══════════════════════════════════════════════════════════════
   figures-b.js — Gaussian Bayes lesson, sections 4–7
   Fig 4.1 covshape (signature) · Fig 5.1 gda2d (signature)
   Fig 5.2 gdacode · Fig 6.1 nbparams · Fig 7.1 boundary1d
   Fig 7.2 morph
   Uses the shared Gaussian-Bayes core LR.gb from figures-a.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const GB = LR.gb;

  /* single-hue ramp: white → orange (no rainbow) */
  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255 + (232 - 255) * t);
    const g = Math.round(255 + (89 - 255) * t);
    const b = Math.round(255 + (12 - 255) * t);
    return [r, g, b];
  }

  function drawPts(ctx, map, pts, color) {
    pts.forEach(function (p) {
      LR.dot(ctx, map.sx(p.x), map.sy(p.y), 3.4, color);
    });
  }

  /* match native selects to the lesson's button chrome */
  function styleSel(sel) {
    sel.style.font = "600 13px Inter, -apple-system, sans-serif";
    sel.style.padding = "6px 10px";
    sel.style.border = "1.5px solid #ccc";
    sel.style.borderRadius = "8px";
    sel.style.background = "#fff";
    sel.style.color = "#111";
  }

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — the covariance-shape explorer (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.covshape = function (mount) {
    LR.header(
      mount,
      "How Σ shapes a 2-D Gaussian",
      "The density (single-hue ramp) and its 1σ/2σ contour ellipses, recomputed from the Σ you dial in. Diagonal entries stretch, the off-diagonal tilts, and its sign picks the tilt direction."
    );

    const st = { v1: 1.0, v2: 1.0, rho: 0.0 };

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    LR.button(grp, "Isotropic σ²I", function () { setTo(1.0, 1.0, 0); }, "");
    LR.button(grp, "Axis-aligned", function () { setTo(2.5, 0.6, 0); }, "");
    LR.button(grp, "Correlated +", function () { setTo(1.6, 1.6, 0.7); }, "");
    LR.button(grp, "Correlated −", function () { setTo(1.6, 1.6, -0.7); }, "");

    const { cv, ctx, W, H } = LR.canvas(mount, 620, 410, {
      aria: "Density heatmap of a two-dimensional Gaussian with contour ellipses, controlled by two variance sliders and a correlation slider",
    });

    const ro = LR.readout(mount, [
      { k: "S", label: "Σ" },
      { k: "det", label: "|Σ|" },
      { k: "shape", label: "contours" },
    ]);

    const ctl = LR.controls(mount);
    const s1 = LR.slider(ctl, "σ₁² (variance of x₁)", 0.2, 3, 0.05, st.v1, function (v) { st.v1 = v; render(); });
    const s2 = LR.slider(ctl, "σ₂² (variance of x₂)", 0.2, 3, 0.05, st.v2, function (v) { st.v2 = v; render(); });
    const s3 = LR.slider(ctl, "correlation ρ", -0.95, 0.95, 0.05, st.rho, function (v) { st.rho = v; render(); });

    function setTo(v1, v2, rho) {
      st.v1 = v1; st.v2 = v2; st.rho = rho;
      s1.set(v1); s2.set(v2); s3.set(rho);
      render();
    }

    const P = { x0: 130, y0: 16, w: 370, h: H - 66, xmin: -4, xmax: 4, ymin: -4, ymax: 4, xlabel: "x₁", ylabel: "x₂" };
    const mu = [0, 0];

    function Sigma() {
      const cov = st.rho * Math.sqrt(st.v1 * st.v2);
      return [[st.v1, cov], [cov, st.v2]];
    }

    function render() {
      const S = Sigma();
      ctx.clearRect(0, 0, W, H);
      const map = LR.plot(ctx, P);

      // density heatmap (block size 3, normalized to the peak)
      const peak = GB.mvn2(0, 0, mu, S);
      const step = 3;
      for (let px = P.x0; px < P.x0 + P.w; px += step) {
        for (let py = P.y0; py < P.y0 + P.h; py += step) {
          const pt = map.inv(px + step / 2, py + step / 2);
          const v = GB.mvn2(pt.x, pt.y, mu, S) / peak;
          const c = ramp(Math.pow(v, 0.75));
          ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
          ctx.fillRect(px, py, step, step);
        }
      }
      ctx.strokeStyle = C.axis;
      ctx.lineWidth = 1.4;
      ctx.strokeRect(P.x0, P.y0, P.w, P.h);

      GB.ellipse(ctx, map, mu, S, 1, "#7a2e00", 2);
      GB.ellipse(ctx, map, mu, S, 2, "#7a2e00", 1.3, [4, 4]);
      GB.meanCross(ctx, map, mu, "#7a2e00");
      ctx.fillStyle = "#7a2e00";
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      const e = GB.eig2(S);
      ctx.fillText("1σ", map.sx(Math.sqrt(e.l1) * Math.cos(e.theta)) + 5, map.sy(Math.sqrt(e.l1) * Math.sin(e.theta)) - 5);

      const cov = S[0][1];
      ro.set("S", "[[" + LR.fmtF(S[0][0], 2) + ", " + LR.fmtF(cov, 2) + "], [" + LR.fmtF(cov, 2) + ", " + LR.fmtF(S[1][1], 2) + "]]");
      ro.set("det", LR.fmtF(GB.det2(S), 3));
      let shape;
      if (Math.abs(st.rho) < 0.02) {
        shape = Math.abs(st.v1 - st.v2) < 0.05 ? "circles (isotropic)" : "axis-aligned ellipses";
      } else {
        shape = "tilted ellipses, leaning " + (cov > 0 ? "up-right (cov > 0)" : "down-right (cov < 0)");
      }
      ro.set("shape", shape, C.orange);
      cv.setAttribute("aria-label", "Gaussian density with variances " + LR.fmtF(st.v1, 2) + " and " + LR.fmtF(st.v2, 2) +
        " and correlation " + LR.fmtF(st.rho, 2) + "; contours are " + shape);
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — a full GDA classifier in 2-D (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.gda2d = function (mount) {
    LR.header(
      mount,
      "Three classes, three Gaussians, one argmax",
      "Each class fitted by MLE; decision regions and boundaries computed live from the log-posteriors. Drag the query (or use arrow keys). The mode toggle re-fits the covariances three ways."
    );

    const data = GB.triData();
    const fits = data.map(GB.fit2d);
    let mode = "full";
    let q = { x: 5.0, y: 5.4 };

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const MODES = [
      { key: "full", label: "Full per-class (QDA)" },
      { key: "shared", label: "Shared (LDA)" },
      { key: "diag", label: "Shared diagonal (Naive Bayes)" },
    ];
    const btns = MODES.map(function (m) {
      return LR.button(grp, m.label, function () { mode = m.key; render(); }, m.key === mode ? "on" : "");
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 430, {
      aria: "Three-class scatter with fitted Gaussian ellipses, decision regions, and a draggable query point showing live posteriors",
    });
    cv.tabIndex = 0;

    // posterior bars
    const barBox = LR.el("div", "attn-weightbar");
    mount.appendChild(barBox);
    const NAMES = ["class A", "class B", "class C"];
    const bars = NAMES.map(function (n, k) {
      const item = LR.el("div", "wb-item");
      item.appendChild(LR.el("div", "wb-lab", "p(" + n + " | x)"));
      const track = LR.el("div", "wb-track");
      const fill = LR.el("div", "wb-fill");
      fill.style.background = GB.COLS[k];
      track.appendChild(fill);
      item.appendChild(track);
      const val = LR.el("div", "wb-val", "–");
      item.appendChild(val);
      barBox.appendChild(item);
      return { fill: fill, val: val };
    });

    const ro = LR.readout(mount, [
      { k: "verdict", label: "verdict" },
      { k: "params", label: "Gaussian params (k=3, d=2)" },
      { k: "priors", label: "priors (from counts)" },
    ]);

    const P = { x0: 56, y0: 16, w: W - 76, h: H - 66, xmin: 0, xmax: 10, ymin: 0, ymax: 10, xlabel: "feature x₁", ylabel: "feature x₂" };
    let map = null;

    function render() {
      btns.forEach((b, i) => b.classList.toggle("on", MODES[i].key === mode));
      ctx.clearRect(0, 0, W, H);
      map = LR.plot(ctx, P);
      const cls = GB.withMode(fits, mode);
      GB.regions(ctx, P, map, cls, { step: 3 });
      data.forEach(function (pts, k) { drawPts(ctx, map, pts, GB.COLS[k]); });
      cls.forEach(function (c, k) {
        GB.ellipse(ctx, map, c.mu, c.S, 1, GB.COLS[k], 2);
        GB.ellipse(ctx, map, c.mu, c.S, 2, GB.COLS[k], 1.1, [4, 4]);
        GB.meanCross(ctx, map, c.mu, GB.COLS[k]);
      });

      // query
      const px = map.sx(q.x), py = map.sy(q.y);
      LR.dot(ctx, px, py, 7, C.purple, "#fff");
      ctx.fillStyle = C.purple;
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("x = (" + LR.fmtF(q.x, 1) + ", " + LR.fmtF(q.y, 1) + ")", px + 10, py - 8);

      const r = GB.posteriors(q.x, q.y, cls);
      let bi = 0;
      r.post.forEach(function (v, k) { if (v > r.post[bi]) bi = k; });
      bars.forEach(function (b, k) {
        b.fill.style.width = Math.round(r.post[k] * 100) + "%";
        b.val.textContent = LR.fmtF(r.post[k], 3);
      });
      const count = GB.paramCount(3, 2, mode);
      ro.set("verdict", NAMES[bi], GB.COLS[bi]);
      ro.set("params", count + (mode === "full" ? " = 3 × (2 means + 3 cov)" : mode === "shared" ? " = 6 means + 3 shared cov" : " = 6 means + 2 shared variances") + ", plus 2 for the priors");
      ro.set("priors", "[" + cls.map((c) => LR.fmtF(c.prior, 2)).join(", ") + "]");
      cv.setAttribute("aria-label", "Query at " + LR.fmtF(q.x, 1) + ", " + LR.fmtF(q.y, 1) + " classified as " + NAMES[bi] +
        " with posterior " + LR.fmtF(r.post[bi], 2) + " under mode " + mode);
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
     Fig 5.2 — gda_predict, executed for real
     ══════════════════════════════════════════════════════════ */
  LR.figs.gdacode = function (mount) {
    LR.header(
      mount,
      "gda_predict, executed for real",
      "Runs the exact pipeline from the Python above on a fixed two-class toy dataset. Switch the covariance mode and re-run: the fits, log-scores, and verdict all recompute."
    );

    // fixed toy dataset (printed in the output so nothing is hidden)
    const C0 = [[1.0, 2.0], [1.5, 2.6], [0.8, 1.4], [2.0, 2.2], [1.4, 1.8], [1.3, 2.4]];
    const C1 = [[3.2, 0.8], [3.8, 1.4], [3.0, 1.9], [4.2, 1.0], [3.6, 0.4], [3.4, 1.2]];
    const QUERIES = [
      { name: "A (near the boundary)", x: [2.5, 2.3] },
      { name: "B (deep in class 0)", x: [1.2, 2.1] },
      { name: "C (deep in class 1)", x: [3.9, 0.9] },
    ];

    const bar = LR.controls(mount);
    let mode = "full";
    let qi = 0;
    const modeSel = document.createElement("select");
    [["full", "full per-class Σ (QDA)"], ["shared", "shared Σ (LDA)"], ["diag", "shared diagonal Σ (Naive Bayes)"]].forEach(function (o) {
      const opt = document.createElement("option");
      opt.value = o[0]; opt.textContent = o[1];
      modeSel.appendChild(opt);
    });
    modeSel.setAttribute("aria-label", "covariance mode");
    styleSel(modeSel);
    modeSel.addEventListener("change", function () { mode = modeSel.value; });
    const qSel = document.createElement("select");
    QUERIES.forEach(function (o, i) {
      const opt = document.createElement("option");
      opt.value = i; opt.textContent = "query " + o.name + " = [" + o.x.join(", ") + "]";
      qSel.appendChild(opt);
    });
    qSel.setAttribute("aria-label", "query point");
    styleSel(qSel);
    qSel.addEventListener("change", function () { qi = +qSel.value; });
    bar.appendChild(modeSel);
    bar.appendChild(qSel);
    LR.button(bar, "Run ▸", run, "primary");

    const out = LR.el("pre", "run-output attn-runout", "press Run to execute gda_fit + gda_predict");
    mount.appendChild(out);

    function toPts(A) { return A.map((r) => ({ x: r[0], y: r[1] })); }
    function fmtS(S) {
      return "[[" + LR.fmtF(S[0][0], 3) + ", " + LR.fmtF(S[0][1], 3) + "], [" + LR.fmtF(S[1][0], 3) + ", " + LR.fmtF(S[1][1], 3) + "]]";
    }

    function run() {
      const fits = [GB.fit2d(toPts(C0)), GB.fit2d(toPts(C1))];
      const cls = GB.withMode(fits, mode);
      const q = QUERIES[qi].x;
      const L = [];
      L.push(">>> X0 = " + JSON.stringify(C0));
      L.push(">>> X1 = " + JSON.stringify(C1));
      L.push(">>> classes = gda_fit(X, t, k=2)      # covariance mode: " + mode);
      cls.forEach(function (c, k) {
        L.push("  class " + k + ":  prior = " + LR.fmtF(c.prior, 3) +
          "   mu = [" + LR.fmtF(c.mu[0], 3) + ", " + LR.fmtF(c.mu[1], 3) + "]");
        L.push("            Sigma = " + fmtS(c.S));
      });
      L.push(">>> gda_predict(x=[" + q.join(", ") + "], classes)");
      const scores = cls.map(function (c, k) {
        const s = Math.log(c.prior) + GB.logMvn2(q[0], q[1], c.mu, c.S);
        L.push("  score " + k + " = log(prior) + log N(x; mu" + k + ", Sigma" + k + ") = " + LR.fmtF(s, 3));
        return s;
      });
      const win = scores[1] > scores[0] ? 1 : 0;
      L.push("  argmax -> class " + win);
      const r = GB.posteriors(q[0], q[1], cls);
      L.push("# normalized posteriors: [" + r.post.map((v) => LR.fmtF(v, 3)).join(", ") + "]");
      L.push("# evidence never computed: the argmax does not need it");
      out.textContent = L.join("\n");
    }
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.1 — Naive Bayes parameter-count demo
     ══════════════════════════════════════════════════════════ */
  LR.figs.nbparams = function (mount) {
    LR.header(
      mount,
      "2^D − 1 versus D + 1",
      "Slide the vocabulary size. The full joint's parameter count is exponential in D; Naive Bayes's is linear. The y axis is logarithmic, which is the only way to keep both curves on one plot."
    );

    let D = 20;

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 330, {
      aria: "Log-scale comparison of parameter counts: the full joint distribution grows exponentially with vocabulary size while Naive Bayes grows linearly",
    });

    const ro = LR.readout(mount, [
      { k: "full", label: "full joint ≈ 2^D − 1" },
      { k: "nb", label: "Naive Bayes: D + 1" },
      { k: "ratio", label: "discount factor" },
    ]);

    const ctl = LR.controls(mount);
    LR.slider(ctl, "vocabulary size D", 1, 50, 1, D, function (v) { D = v; render(); }, (v) => "D = " + v);

    const P = { x0: 64, y0: 16, w: W - 88, h: H - 66, xmin: 0, xmax: 50, ymin: 0, ymax: 16, yticks: [0, 3, 6, 9, 12, 15], xlabel: "vocabulary size D", ylabel: "parameters (log₁₀ scale)" };

    function render() {
      ctx.clearRect(0, 0, W, H);
      const map = LR.plot(ctx, P);

      // reference lines: a million-example dataset, a billion
      [[6, "one million"], [9, "one billion"]].forEach(function (ref) {
        ctx.strokeStyle = LR.C.faint;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(P.x0, map.sy(ref[0]));
        ctx.lineTo(P.x0 + P.w, map.sy(ref[0]));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = LR.C.faint;
        ctx.font = "10.5px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(ref[1], P.x0 + 6, map.sy(ref[0]) - 4);
      });

      // curves: log10(2^d - 1) and log10(d + 1)
      const curves = [
        { f: (d) => Math.log10(Math.pow(2, d) - 1), col: C.orange, lab: "2^D − 1 (full joint)" },
        { f: (d) => Math.log10(d + 1), col: C.green, lab: "D + 1 (Naive Bayes)" },
      ];
      curves.forEach(function (cu) {
        ctx.strokeStyle = cu.col;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        let first = true;
        for (let d = 1; d <= 50; d += 0.5) {
          const px = map.sx(d), py = map.sy(Math.max(0, cu.f(d)));
          if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      });
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.orange;
      ctx.textAlign = "right";
      ctx.fillText("2^D − 1 (full joint)", P.x0 + P.w - 8, map.sy(Math.log10(Math.pow(2, 46))) - 8);
      ctx.fillStyle = C.green;
      ctx.fillText("D + 1 (Naive Bayes)", P.x0 + P.w - 8, map.sy(Math.log10(51)) - 10);

      // marker at current D
      const full = Math.pow(2, D) - 1;
      ctx.strokeStyle = C.purple;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(map.sx(D), P.y0);
      ctx.lineTo(map.sx(D), P.y0 + P.h);
      ctx.stroke();
      ctx.setLineDash([]);
      LR.dot(ctx, map.sx(D), map.sy(Math.max(0, Math.log10(full))), 5, "#fff", C.orange);
      LR.dot(ctx, map.sx(D), map.sy(Math.log10(D + 1)), 5, "#fff", C.green);

      ro.set("full", full.toLocaleString("en-US"), C.orange);
      ro.set("nb", String(D + 1), C.green);
      ro.set("ratio", "×" + (full / (D + 1) >= 1e6 ? (full / (D + 1)).toExponential(2) : Math.round(full / (D + 1)).toLocaleString("en-US")));
      cv.setAttribute("aria-label", "At vocabulary size " + D + " the full joint needs " + full.toLocaleString("en-US") +
        " parameters per class and Naive Bayes needs " + (D + 1));
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.1 — the equal-posterior condition as a quadratic
     ══════════════════════════════════════════════════════════ */
  LR.figs.boundary1d = function (mount) {
    LR.header(
      mount,
      "Equal posteriors → ax² + bx + c = 0",
      "The coefficients come from expanding the two log Gaussians; the roots are the decision points. Unequal variances keep a alive (quadratic, up to two boundaries); equal variances kill it (linear, one boundary)."
    );

    const st = { mu0: 4, s0: 0.8, mu1 : 7, s1: 1.9, p1: 0.5 };
    const X0 = 0, X1 = 12;

    const { cv, ctx, W, H } = LR.canvas(mount, 660, 320, {
      aria: "Two prior-weighted Gaussians with shaded decision regions and the boundary points solved from the quadratic equation",
    });

    const ro = LR.readout(mount, [
      { k: "abc", label: "a, b, c (computed)" },
      { k: "roots", label: "boundary point(s)" },
      { k: "type", label: "boundary type" },
    ]);

    const ctl = LR.controls(mount);
    const sM0 = LR.slider(ctl, "μ₀", 1, 11, 0.25, st.mu0, function (v) { st.mu0 = v; render(); });
    const sS0 = LR.slider(ctl, "σ₀", 0.4, 3, 0.05, st.s0, function (v) { st.s0 = v; render(); });
    const sM1 = LR.slider(ctl, "μ₁", 1, 11, 0.25, st.mu1, function (v) { st.mu1 = v; render(); });
    const sS1 = LR.slider(ctl, "σ₁", 0.4, 3, 0.05, st.s1, function (v) { st.s1 = v; render(); });
    const sP1 = LR.slider(ctl, "p(C=1)", 0.05, 0.95, 0.05, st.p1, function (v) { st.p1 = v; render(); });
    LR.button(ctl, "Equalize variances", function () {
      const s = (st.s0 + st.s1) / 2;
      st.s0 = s; st.s1 = s;
      sS0.set(s); sS1.set(s);
      render();
    }, "primary");
    const msg = LR.msg(mount);

    function coeffs() {
      const v0 = st.s0 * st.s0, v1 = st.s1 * st.s1;
      return {
        a: 1 / (2 * v0) - 1 / (2 * v1),
        b: st.mu1 / v1 - st.mu0 / v0,
        c: (st.mu0 * st.mu0) / (2 * v0) - (st.mu1 * st.mu1) / (2 * v1) +
          Math.log(st.s0 / st.s1) + Math.log(st.p1 / (1 - st.p1)),
      };
    }
    function roots(k) {
      if (Math.abs(k.a) < 1e-9) return Math.abs(k.b) < 1e-12 ? [] : [-k.c / k.b];
      const d = k.b * k.b - 4 * k.a * k.c;
      if (d < 0) return [];
      const r = Math.sqrt(d);
      return [(-k.b - r) / (2 * k.a), (-k.b + r) / (2 * k.a)].sort((u, v) => u - v);
    }
    function score(k, x) {
      return k === 0
        ? (1 - st.p1) * GB.g1(x, st.mu0, st.s0 * st.s0)
        : st.p1 * GB.g1(x, st.mu1, st.s1 * st.s1);
    }

    const P = { x0: 56, y0: 16, w: W - 76, h: H - 66, xmin: X0, xmax: X1, ymin: 0, ymax: 0.4, xlabel: "x", ylabel: "p(x | C) · p(C)" };

    function render() {
      let peak = 0;
      for (let x = X0; x <= X1; x += 0.1) peak = Math.max(peak, score(0, x), score(1, x));
      P.ymax = peak * 1.15;
      ctx.clearRect(0, 0, W, H);
      const map = LR.plot(ctx, P);

      // decision band
      for (let px = P.x0; px < P.x0 + P.w; px += 2) {
        const x = map.inv(px, 0).x;
        ctx.fillStyle = score(1, x) > score(0, x) ? "rgba(232,89,12,0.14)" : "rgba(47,158,68,0.12)";
        ctx.fillRect(px, P.y0, 2, 8);
      }

      [0, 1].forEach(function (k) {
        ctx.strokeStyle = k === 0 ? C.green : C.orange;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        let first = true;
        for (let x = X0; x <= X1; x += 0.05) {
          const px = map.sx(x), py = map.sy(score(k, x));
          if (first) { ctx.moveTo(px, py); first = false; } else ctx.lineTo(px, py);
        }
        ctx.stroke();
      });

      const k = coeffs();
      const rs = roots(k).filter((r) => r > X0 && r < X1);
      rs.forEach(function (r) {
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1.7;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(map.sx(r), P.y0);
        ctx.lineTo(map.sx(r), P.y0 + P.h);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#333";
        ctx.font = "600 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("x = " + LR.fmtF(r, 2), map.sx(r), P.y0 + P.h - 6);
      });

      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.green;
      ctx.textAlign = "center";
      ctx.fillText("C = 0", map.sx(st.mu0), map.sy(score(0, st.mu0)) - 8);
      ctx.fillStyle = C.orange;
      ctx.fillText("C = 1", map.sx(st.mu1), map.sy(score(1, st.mu1)) - 8);

      const linear = Math.abs(k.a) < 1e-9;
      ro.set("abc", "a = " + LR.fmtF(k.a, 3) + ", b = " + LR.fmtF(k.b, 3) + ", c = " + LR.fmtF(k.c, 3));
      ro.set("roots", rs.length ? rs.map((r) => LR.fmtF(r, 2)).join(", ") : "none in view");
      ro.set("type", linear ? "linear (a = 0): one threshold" : "quadratic (a ≠ 0): up to two", linear ? C.green : C.orange);
      if (linear) {
        msg.show("σ₀ = σ₁, so the x² terms cancel: a = 0 and the boundary is the single threshold x = " + (rs.length ? LR.fmtF(rs[0], 2) : "–") + ". This is the 1-D shadow of the LDA cancellation.", "good");
      } else if (rs.length === 2) {
        msg.show("Two boundary points: the narrower class carves an interval out of the wider class's territory. That is what a quadratic boundary means in 1-D.", "info");
      } else {
        msg.show("The quadratic has " + (rs.length === 1 ? "one visible root" : "no real roots in view") + ": one class wins everywhere the eye can see. Nudge the priors or variances.", "info");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.2 — the QDA → LDA → Naive Bayes boundary morph
     ══════════════════════════════════════════════════════════ */
  LR.figs.morph = function (mount) {
    LR.header(
      mount,
      "One dataset, morphing covariance assumptions",
      "The slider blends each class's fitted Σ_k toward the pooled Σ; the boundary is recomputed at every step. Diagonalize zeroes the off-diagonals, snapping the ellipses upright."
    );

    const data = GB.morphData();
    const fits = data.map(GB.fit2d);
    const total = fits[0].n + fits[1].n;
    const pooled = GB.pool(fits);
    let t = 0;
    let diag = false;
    let anim = null;

    const bar = LR.controls(mount);
    const sT = LR.slider(bar, "share the covariance", 0, 1, 0.02, 0, function (v) { t = v; render(); }, (v) => LR.fmtF(v, 2));
    const dBtn = LR.button(bar, "Diagonalize: off", function () {
      diag = !diag;
      dBtn.textContent = "Diagonalize: " + (diag ? "on" : "off");
      dBtn.classList.toggle("on", diag);
      render();
    });
    LR.button(bar, "Animate ▸", function () {
      if (anim) { clearInterval(anim); anim = null; return; }
      if (LR.reducedMotion) { t = t < 0.5 ? 1 : 0; sT.set(t); render(); return; }
      const dir = t < 0.5 ? 1 : -1;
      anim = setInterval(function () {
        t = Math.min(1, Math.max(0, t + dir * 0.03));
        sT.set(t);
        render();
        if (t === 0 || t === 1) { clearInterval(anim); anim = null; }
      }, 40);
    }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 420, {
      aria: "Two-class scatter whose decision boundary morphs from quadratic to linear as the covariance interpolates from per-class to shared",
    });

    const ro = LR.readout(mount, [
      { k: "regime", label: "regime" },
      { k: "params", label: "Gaussian params (k=2, d=2)" },
    ]);

    const P = { x0: 56, y0: 16, w: W - 76, h: H - 66, xmin: 0, xmax: 10, ymin: 0, ymax: 10, xlabel: "feature x₁", ylabel: "feature x₂" };

    function lerpS(S) {
      const out = [
        [(1 - t) * S[0][0] + t * pooled[0][0], (1 - t) * S[0][1] + t * pooled[0][1]],
        [(1 - t) * S[1][0] + t * pooled[1][0], (1 - t) * S[1][1] + t * pooled[1][1]],
      ];
      return diag ? GB.diagOf(out) : out;
    }
    function classes() {
      return fits.map(function (f) {
        return { mu: f.mu, S: lerpS(f.S), prior: f.n / total };
      });
    }

    function regime() {
      if (!diag) {
        if (t >= 0.999) return { name: "LDA: shared Σ, linear boundary", count: GB.paramCount(2, 2, "shared") };
        if (t <= 0.001) return { name: "QDA: per-class Σ, quadratic boundary", count: GB.paramCount(2, 2, "full") };
        return { name: "between QDA and LDA (blend " + LR.fmtF(t, 2) + ")", count: null };
      }
      if (t >= 0.999) return { name: "Gaussian Naive Bayes, shared diagonal: linear, axis-aligned", count: GB.paramCount(2, 2, "diag") };
      return { name: "Gaussian Naive Bayes: per-class diagonal, axis-aligned portraits", count: 2 * 2 + 2 * 2 };
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const map = LR.plot(ctx, P);
      const cls = classes();
      GB.regions(ctx, P, map, cls, { step: 3 });
      data.forEach(function (pts, k) { drawPts(ctx, map, pts, GB.COLS[k]); });
      cls.forEach(function (c, k) {
        GB.ellipse(ctx, map, c.mu, c.S, 1, GB.COLS[k], 2);
        GB.ellipse(ctx, map, c.mu, c.S, 2, GB.COLS[k], 1.1, [4, 4]);
        GB.meanCross(ctx, map, c.mu, GB.COLS[k]);
      });
      const r = regime();
      ro.set("regime", r.name, C.orange);
      ro.set("params", r.count === null ? "–" : r.count + ", plus 1 for the prior");
      cv.setAttribute("aria-label", "Decision boundary under " + r.name);
    }
    render();
  };
})();
