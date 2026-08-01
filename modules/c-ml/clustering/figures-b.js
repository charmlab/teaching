/* ══════════════════════════════════════════════════════════════
   figures-b.js — Clustering lesson, mixture-model half
   Fig 3.1 soft · Fig 4.1 chicken · Fig 5.1 builder
   Fig 6.1 latent · Fig 6.2 emloop (signature) · Fig 7.1 vs
   Uses the shared clustering core LR.cl from figures-a.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const CL = LR.cl;

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — soft K-means: the stiffness dial
     ══════════════════════════════════════════════════════════ */
  LR.figs.soft = function (mount) {
    LR.header(
      mount,
      "Soft responsibilities under the stiffness β",
      "Every point's colour is its live softmax blend over the three centroids. Click a point to read its responsibilities; Iterate runs real soft update steps at the current β."
    );

    const X = CL.blobs(7);
    // start from a converged hard K-means so the centroids are sensible
    let km = null;
    for (let r = 0; r < 6; r++) {
      const run = CL.kmeansRun(X, CL.seedCentroids(X, 3, LR.rng(300 + r)));
      if (!km || run.J < km.J) km = run;
    }
    const M0 = km.M.map((m) => m.slice());
    let M = M0.map((m) => m.slice());
    let logb = -0.7; // beta = 10^logb ≈ 0.2
    let selected = 40;

    const bar = LR.controls(mount);
    const bs = LR.slider(bar, "stiffness β", -2, 1.4, 0.05, logb, function (v) {
      logb = v; render();
    }, (v) => LR.fmt(Math.pow(10, v), Math.pow(10, v) < 1 ? 3 : 1));
    LR.button(bar, "Iterate (one soft update)", function () {
      const beta = Math.pow(10, logb);
      const R = X.map((x) => CL.softResp(x, M, beta));
      M = CL.softUpdate(X, R, M);
      render();
    }, "primary");
    LR.button(bar, "Reset centroids", function () { M = M0.map((m) => m.slice()); render(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 420, {
      aria: "Scatter whose point colours blend three cluster colours according to softmax responsibilities controlled by a stiffness slider; click a point to inspect it",
    });

    // responsibility bars for the selected point
    const wb = LR.el("div", "attn-weightbar");
    mount.appendChild(wb);
    const items = [0, 1, 2].map(function (k) {
      const it = LR.el("div", "wb-item");
      it.appendChild(LR.el("span", "wb-lab", "cluster " + (k + 1)));
      const track = LR.el("div", "wb-track");
      const fill = LR.el("div", "wb-fill");
      fill.style.background = CL.COLS[k];
      track.appendChild(fill);
      it.appendChild(track);
      const val = LR.el("span", "wb-val", "–");
      it.appendChild(val);
      wb.appendChild(it);
      return { fill: fill, val: val };
    });
    const ro = LR.readout(mount, [
      { k: "beta", label: "β" },
      { k: "pt", label: "selected point" },
      { k: "sum", label: "Σk rk" },
    ]);

    let map = null;
    cv.addEventListener("click", function (e) {
      if (!map) return;
      const p = LR.evtXY(cv, W, H, e);
      let bi = -1, bd = Infinity;
      X.forEach(function (x, n) {
        const dx = p.x - map.sx(x[0]), dy = p.y - map.sy(x[1]);
        const d = dx * dx + dy * dy;
        if (d < bd) { bd = d; bi = n; }
      });
      if (bd < 400) { selected = bi; render(); }
    });

    function render() {
      const beta = Math.pow(10, logb);
      ctx.clearRect(0, 0, W, H);
      map = CL.frame(ctx, W, H);
      X.forEach(function (x, n) {
        const r = CL.softResp(x, M, beta);
        LR.dot(ctx, map.sx(x[0]), map.sy(x[1]), n === selected ? 6 : 4, CL.mix(r), n === selected ? C.text : null);
      });
      M.forEach((m, k) => CL.cross(ctx, map.sx(m[0]), map.sy(m[1]), CL.COLS[k]));

      const r = CL.softResp(X[selected], M, beta);
      items.forEach(function (it, k) {
        it.fill.style.width = (r[k] * 100).toFixed(1) + "%";
        it.val.textContent = LR.fmtF(r[k], 3);
      });
      ro.set("beta", LR.fmt(beta, beta < 1 ? 3 : 2), C.orange);
      ro.set("pt", "(" + LR.fmtF(X[selected][0], 2) + ", " + LR.fmtF(X[selected][1], 2) + ")");
      ro.set("sum", LR.fmtF(r[0] + r[1] + r[2], 3), C.green);
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — the chicken-and-egg, in two panels
     ══════════════════════════════════════════════════════════ */
  LR.figs.chicken = function (mount) {
    LR.header(
      mount,
      "Labels give parameters; parameters give labels",
      "1-D data, two Gaussian clusters. Each panel has the half of the puzzle the other one needs; EM alternates between them."
    );

    // 1-D dataset from two Gaussians (the true generative story)
    const rand = LR.rng(61);
    const xs = [], labs = [];
    for (let i = 0; i < 26; i++) { xs.push(2.4 + LR.gauss(rand) * 0.75); labs.push(0); }
    for (let i = 0; i < 22; i++) { xs.push(6.6 + LR.gauss(rand) * 0.95); labs.push(1); }
    // "given" parameters for the right panel (deliberately imperfect guesses)
    const given = [{ mu: 2.1, s2: 0.5, pi: 0.5 }, { mu: 6.9, s2: 1.1, pi: 0.5 }];

    let fitted = null;   // left panel's per-group MLE fits
    let coloured = false; // right panel state

    const bar = LR.controls(mount);
    const fitBtn = LR.button(bar, "Fit Gaussians by MLE (left)", function () {
      fitted = [0, 1].map(function (k) {
        const g = xs.filter((_, n) => labs[n] === k);
        const mu = g.reduce((a, b) => a + b, 0) / g.length;
        const s2 = g.reduce((a, b) => a + (b - mu) * (b - mu), 0) / g.length;
        return { mu: mu, s2: s2 };
      });
      render();
    }, "primary");
    LR.button(bar, "Colour by posterior (right)", function () { coloured = true; render(); });
    LR.button(bar, "Reset", function () { fitted = null; coloured = false; render(); });

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const leftBox = LR.el("div"), rightBox = LR.el("div");
    leftBox.appendChild(LR.el("div", "pane-label", "If you knew the labels → MLE per group"));
    rightBox.appendChild(LR.el("div", "pane-label", "If you knew the Gaussians → Bayes posterior"));
    grid.appendChild(leftBox); grid.appendChild(rightBox);
    const Lc = LR.canvas(leftBox, 400, 300, { aria: "Labelled one-dimensional points; a button fits each group's Gaussian by maximum likelihood" });
    const Rc = LR.canvas(rightBox, 400, 300, { aria: "Unlabelled one-dimensional points under two given Gaussians; a button colours each point by its posterior" });
    const note = LR.el("div", "fig-msg show info", "Each panel needs the other panel's output: labels in, parameters out on the left; parameters in, (soft) labels out on the right.");
    mount.appendChild(note);
    const ro = LR.readout(mount, [
      { k: "l", label: "left fit" },
      { k: "r", label: "right posterior at x = 4.4" },
    ]);

    function drawPanel(P, mode) {
      const ctx = P.ctx;
      ctx.clearRect(0, 0, P.W, P.H);
      const map = LR.plot(ctx, {
        x0: 44, y0: 12, w: P.W - 56, h: P.H - 54,
        xmin: 0, xmax: 10, ymin: 0, ymax: 0.6,
        xlabel: "x", ylabel: "density",
      });
      const params = mode === "left" ? fitted : given;
      // curves
      if (params) {
        params.forEach(function (g, k) {
          ctx.strokeStyle = CL.COLS[k];
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          for (let i = 0; i <= 160; i++) {
            const x = (i / 160) * 10;
            const y = CL.g1(x, g.mu, g.s2) * (mode === "left" ? 0.5 : g.pi) * 2; // scaled for visibility
            if (i === 0) ctx.moveTo(map.sx(x), map.sy(y));
            else ctx.lineTo(map.sx(x), map.sy(y));
          }
          ctx.stroke();
        });
      }
      // rug of points
      xs.forEach(function (x, n) {
        let col = "#9a9a9a";
        if (mode === "left") col = CL.COLS[labs[n]];
        else if (coloured) {
          const w0 = given[0].pi * CL.g1(x, given[0].mu, given[0].s2);
          const w1 = given[1].pi * CL.g1(x, given[1].mu, given[1].s2);
          col = CL.mix([w0 / (w0 + w1), w1 / (w0 + w1), 0]);
        }
        const jitter = ((n * 37) % 10) / 10;
        LR.dot(ctx, map.sx(x), map.sy(0.035 + jitter * 0.05), 3.6, col);
      });
    }

    function render() {
      drawPanel(Lc, "left");
      drawPanel(Rc, "right");
      ro.set("l", fitted
        ? "μ = (" + LR.fmtF(fitted[0].mu, 2) + ", " + LR.fmtF(fitted[1].mu, 2) + "), σ² = (" + LR.fmtF(fitted[0].s2, 2) + ", " + LR.fmtF(fitted[1].s2, 2) + ")"
        : "not fitted yet", fitted ? C.green : undefined);
      if (coloured) {
        const w0 = given[0].pi * CL.g1(4.4, given[0].mu, given[0].s2);
        const w1 = given[1].pi * CL.g1(4.4, given[1].mu, given[1].s2);
        ro.set("r", "p(C=1|x) = " + LR.fmtF(w0 / (w0 + w1), 3) + ", p(C=2|x) = " + LR.fmtF(w1 / (w0 + w1), 3), C.purple);
      } else {
        ro.set("r", "not computed yet");
      }
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — the mixture density builder (1-D and 2-D)
     ══════════════════════════════════════════════════════════ */
  LR.figs.builder = function (mount) {
    LR.header(
      mount,
      "Build a density out of Gaussians",
      "The solid curve/shading is the mixture p(x) = Σ πk N(x; μk, Σk), recomputed live from your knobs. The mixing weights always sum to 1."
    );

    let mode = "1d";
    const st1 = { pi1: 0.55, mu1: 3, s1: 0.8, mu2: 6.8, s2: 1.2 };
    const st2 = {
      pi1: 0.5,
      mus: [[3.2, 3.2], [6.8, 4.6]],
      spread: [1.0, 1.0],
      tilt: [0.5, -0.4],
    };

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const b1 = LR.button(group, "1-D curve", function () { mode = "1d"; sync(); });
    const b2 = LR.button(group, "2-D contours", function () { mode = "2d"; sync(); });
    LR.slider(bar, "π₁", 0.05, 0.95, 0.01, st1.pi1, function (v) { st1.pi1 = v; st2.pi1 = v; render(); });

    const bar1 = LR.controls(mount);
    const s1d = [
      LR.slider(bar1, "μ₁", 0.5, 9.5, 0.1, st1.mu1, function (v) { st1.mu1 = v; render(); }),
      LR.slider(bar1, "σ₁", 0.3, 2.5, 0.05, st1.s1, function (v) { st1.s1 = v; render(); }),
      LR.slider(bar1, "μ₂", 0.5, 9.5, 0.1, st1.mu2, function (v) { st1.mu2 = v; render(); }),
      LR.slider(bar1, "σ₂", 0.3, 2.5, 0.05, st1.s2, function (v) { st1.s2 = v; render(); }),
    ];
    const bar2 = LR.controls(mount);
    const s2d = [
      LR.slider(bar2, "spread₁", 0.5, 1.8, 0.05, st2.spread[0], function (v) { st2.spread[0] = v; render(); }),
      LR.slider(bar2, "tilt₁", -0.9, 0.9, 0.05, st2.tilt[0], function (v) { st2.tilt[0] = v; render(); }),
      LR.slider(bar2, "spread₂", 0.5, 1.8, 0.05, st2.spread[1], function (v) { st2.spread[1] = v; render(); }),
      LR.slider(bar2, "tilt₂", -0.9, 0.9, 0.05, st2.tilt[1], function (v) { st2.tilt[1] = v; render(); }),
    ];

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 400, {
      aria: "Mixture density builder: in 1-D a live curve of the two-component mixture, in 2-D a live density heatmap with draggable component means",
    });
    const ro = LR.readout(mount, [
      { k: "pis", label: "π = (π₁, π₂)" },
      { k: "int", label: "∫ p(x) dx (numeric)" },
    ]);

    function Sof(k) {
      const s = st2.spread[k], t = st2.tilt[k];
      // Σ = R diag(s², (0.45 s)²) Rᵀ with tilt angle t
      const c = Math.cos(t), n = Math.sin(t);
      const a = s * s, b = 0.45 * s * 0.45 * s;
      return [
        [a * c * c + b * n * n, (a - b) * c * n],
        [(a - b) * c * n, a * n * n + b * c * c],
      ];
    }

    // drag means in 2-D mode
    let map = null, dragK = -1;
    LR.drag(cv, W, H, {
      hit: function (p) {
        if (mode !== "2d" || !map) return false;
        for (let k = 0; k < 2; k++) {
          const dx = p.x - map.sx(st2.mus[k][0]), dy = p.y - map.sy(st2.mus[k][1]);
          if (dx * dx + dy * dy < 190) { dragK = k; return true; }
        }
        return false;
      },
      move: function (p) {
        if (dragK < 0) return;
        const w = map.inv(p.x, p.y);
        st2.mus[dragK] = [Math.max(0.3, Math.min(9.7, w.x)), Math.max(0.3, Math.min(7.7, w.y))];
        render();
      },
      up: function () { dragK = -1; },
    });

    function sync() {
      b1.classList.toggle("on", mode === "1d");
      b2.classList.toggle("on", mode === "2d");
      bar1.style.display = mode === "1d" ? "" : "none";
      bar2.style.display = mode === "2d" ? "" : "none";
      render();
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      if (mode === "1d") {
        const pis = [st1.pi1, 1 - st1.pi1];
        const comps = [{ mu: st1.mu1, s2: st1.s1 * st1.s1 }, { mu: st1.mu2, s2: st1.s2 * st1.s2 }];
        const mix = (x) => pis[0] * CL.g1(x, comps[0].mu, comps[0].s2) + pis[1] * CL.g1(x, comps[1].mu, comps[1].s2);
        let ymax = 0;
        for (let i = 0; i <= 200; i++) ymax = Math.max(ymax, mix((i / 200) * 10));
        const P = LR.plot(ctx, {
          x0: 50, y0: 14, w: W - 66, h: H - 58,
          xmin: 0, xmax: 10, ymin: 0, ymax: ymax * 1.15,
          xlabel: "x", ylabel: "density p(x)",
        });
        // components (dashed, weighted)
        comps.forEach(function (g, k) {
          ctx.strokeStyle = [C.green, C.purple][k];
          ctx.lineWidth = 1.8;
          ctx.setLineDash([5, 4]);
          ctx.beginPath();
          for (let i = 0; i <= 200; i++) {
            const x = (i / 200) * 10, y = pis[k] * CL.g1(x, g.mu, g.s2);
            if (i === 0) ctx.moveTo(P.sx(x), P.sy(y));
            else ctx.lineTo(P.sx(x), P.sy(y));
          }
          ctx.stroke();
          ctx.setLineDash([]);
        });
        // mixture
        ctx.strokeStyle = C.orange;
        ctx.lineWidth = 3;
        ctx.beginPath();
        for (let i = 0; i <= 240; i++) {
          const x = (i / 240) * 10, y = mix(x);
          if (i === 0) ctx.moveTo(P.sx(x), P.sy(y));
          else ctx.lineTo(P.sx(x), P.sy(y));
        }
        ctx.stroke();
        // numeric integral over the plotted range (trapezoid)
        let integ = 0;
        const dx = 10 / 400;
        for (let i = 0; i < 400; i++) integ += 0.5 * (mix(i * dx) + mix((i + 1) * dx)) * dx;
        ro.set("pis", "(" + LR.fmtF(pis[0], 2) + ", " + LR.fmtF(pis[1], 2) + "), sum = 1", C.orange);
        ro.set("int", LR.fmtF(integ, 4) + " (mass outside the plot is tiny)", C.green);
      } else {
        map = CL.frame(ctx, W, H);
        const pis = [st2.pi1, 1 - st2.pi1];
        const Ss = [Sof(0), Sof(1)];
        const nx = 90, ny = 64;
        let dmax = 0;
        const vals = [];
        for (let i = 0; i < nx; i++) {
          vals.push([]);
          for (let j = 0; j < ny; j++) {
            const x = (i + 0.5) * (10 / nx), y = (j + 0.5) * (8 / ny);
            const v = pis[0] * CL.mvn2([x, y], st2.mus[0], Ss[0]) + pis[1] * CL.mvn2([x, y], st2.mus[1], Ss[1]);
            vals[i].push(v);
            if (v > dmax) dmax = v;
          }
        }
        for (let i = 0; i < nx; i++) {
          for (let j = 0; j < ny; j++) {
            const t = Math.pow(vals[i][j] / dmax, 0.6);
            if (t < 0.02) continue;
            ctx.fillStyle = "rgba(232,89,12," + (0.6 * t).toFixed(3) + ")";
            const x0 = map.sx(i * (10 / nx)), x1 = map.sx((i + 1) * (10 / nx));
            const y0 = map.sy((j + 1) * (8 / ny)), y1 = map.sy(j * (8 / ny));
            ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
          }
        }
        [0, 1].forEach(function (k) {
          CL.ellipse(ctx, map, st2.mus[k], Ss[k], 1, [C.green, C.purple][k], 2.2);
          CL.ellipse(ctx, map, st2.mus[k], Ss[k], 2, [C.green, C.purple][k], 1.2, [4, 4]);
          CL.cross(ctx, map.sx(st2.mus[k][0]), map.sy(st2.mus[k][1]), [C.green, C.purple][k]);
        });
        ctx.fillStyle = C.muted;
        ctx.font = "600 11.5px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("drag the crosses to move μ₁, μ₂", 52, 26);
        ro.set("pis", "(" + LR.fmtF(pis[0], 2) + ", " + LR.fmtF(pis[1], 2) + "), sum = 1", C.orange);
        ro.set("int", "1 by construction (convex combination of densities)", C.green);
      }
    }

    sync();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.1 — the latent variable z, run forward
     ══════════════════════════════════════════════════════════ */
  LR.figs.latent = function (mount) {
    LR.header(
      mount,
      "The generative story: roll z, then draw x",
      "Each sample first picks a component from Categorical(π), then draws from that component's Gaussian. The colour of a point is its z, the thing real datasets erase."
    );

    let pis = [0.5, 0.3, 0.2];
    const mus = [[2.6, 2.4], [7.0, 3.0], [4.9, 6.0]];
    const Ss = [
      [[0.9, 0.45], [0.45, 0.5]],
      [[0.6, -0.3], [-0.3, 0.7]],
      [[1.0, 0.2], [0.2, 0.4]],
    ];
    let pts = [];       // {p, z}
    let observeOnly = false;
    let flashK = -1;
    const rand = LR.rng(2024);

    const bar = LR.controls(mount);
    LR.slider(bar, "π₁", 0.1, 0.8, 0.01, pis[0], function (v) {
      const rest = 1 - v;
      const old = pis[1] + pis[2];
      pis = [v, (pis[1] / old) * rest, (pis[2] / old) * rest];
      render();
    });
    LR.button(bar, "Sample a point", function () { sample(1, true); }, "primary");
    LR.button(bar, "Sample 100", function () { sample(100, false); });
    LR.button(bar, "Clear", function () { pts = []; flashK = -1; render(); });
    const obsLab = LR.el("label");
    const obsCb = document.createElement("input");
    obsCb.type = "checkbox";
    obsCb.setAttribute("aria-label", "Observe only x: hide the latent component colours");
    obsCb.addEventListener("change", function () { observeOnly = obsCb.checked; render(); });
    obsLab.appendChild(obsCb);
    obsLab.appendChild(document.createTextNode(" observe only x (hide z)"));
    bar.appendChild(obsLab);

    // categorical bar
    const catWrap = LR.el("div");
    catWrap.style.cssText = "display:flex;height:26px;border-radius:8px;overflow:hidden;margin-bottom:10px;border:1px solid #e3e3e3;";
    catWrap.setAttribute("aria-label", "Mixing weights of the three components shown as a stacked bar");
    const segs = [0, 1, 2].map(function (k) {
      const s = LR.el("div");
      s.style.cssText = "display:flex;align-items:center;justify-content:center;color:#fff;font:700 11.5px Inter,sans-serif;transition:width 0.2s,opacity 0.2s;";
      s.style.background = CL.COLS[k];
      catWrap.appendChild(s);
      return s;
    });
    mount.appendChild(catWrap);

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 400, {
      aria: "Points sampled from a three-component Gaussian mixture, coloured by their latent component unless the observe-only toggle hides it",
    });
    const ro = LR.readout(mount, [
      { k: "n", label: "points drawn" },
      { k: "emp", label: "empirical fractions" },
      { k: "pi", label: "true π" },
    ]);

    function sample(n, animate) {
      for (let i = 0; i < n; i++) {
        let u = rand(), z = 0, acc = 0;
        for (let k = 0; k < 3; k++) { acc += pis[k]; if (u <= acc) { z = k; break; } }
        pts.push({ p: CL.sampleMvn(mus[z], Ss[z], rand), z: z });
        if (animate) flashK = z;
      }
      render();
      if (animate && !LR.reducedMotion) {
        setTimeout(function () { flashK = -1; render(); }, 650);
      } else {
        flashK = -1;
      }
    }

    function render() {
      segs.forEach(function (s, k) {
        s.style.width = (pis[k] * 100).toFixed(1) + "%";
        s.textContent = "π" + (k + 1) + " = " + LR.fmtF(pis[k], 2);
        s.style.opacity = flashK === -1 || flashK === k ? "1" : "0.25";
      });
      ctx.clearRect(0, 0, W, H);
      const map = CL.frame(ctx, W, H);
      [0, 1, 2].forEach(function (k) {
        const col = observeOnly ? "#c3c3c3" : CL.COLS[k];
        CL.ellipse(ctx, map, mus[k], Ss[k], 1, col, flashK === k ? 3 : 1.8);
        CL.ellipse(ctx, map, mus[k], Ss[k], 2, col, 1, [4, 4]);
      });
      const counts = [0, 0, 0];
      pts.forEach(function (o, i) {
        counts[o.z]++;
        const isLast = i === pts.length - 1 && flashK !== -1;
        LR.dot(ctx, map.sx(o.p[0]), map.sy(o.p[1]), isLast ? 6 : 3.6,
          observeOnly ? "#8a8a8a" : CL.COLS[o.z], isLast ? C.text : null);
      });
      ro.set("n", String(pts.length));
      ro.set("emp", pts.length
        ? "(" + counts.map((c) => LR.fmtF(c / pts.length, 2)).join(", ") + ")"
        : "–", C.purple);
      ro.set("pi", "(" + pis.map((p) => LR.fmtF(p, 2)).join(", ") + ")", C.orange);
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.2 — EM for a GMM (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.emloop = function (mount) {
    LR.header(
      mount,
      "EM, step by step: infer γ, refit Θ, watch ℓ climb",
      "E-step paints points with their live responsibilities; M-step refits means, covariances, and mixing weights from them. The log-likelihood is evaluated after every M-step."
    );

    const X = CL.stretched(21);
    const K = 3;
    let st = null, phase = "e", iter = 0, llHist = [], converged = false;
    let initN = 0;
    let timer = null;

    const bar = LR.controls(mount);
    const eBtn = LR.button(bar, "E-step", doE, "primary");
    const mBtn = LR.button(bar, "M-step", doM);
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (timer) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      timer = setInterval(function () {
        if (converged) { stopPlay(); return; }
        if (phase === "e") doE(); else doM();
      }, LR.reducedMotion ? 1000 : 480);
    });
    LR.button(bar, "Re-initialize", function () { init(true); });
    LR.button(bar, "Seed from K-means", function () { init(false, true); });

    function stopPlay() {
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = "Play ▸▸";
    }

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const leftBox = LR.el("div"), rightBox = LR.el("div");
    leftBox.appendChild(LR.el("div", "pane-label", "Data, responsibilities, and components"));
    rightBox.appendChild(LR.el("div", "pane-label", "Log-likelihood ℓ(Θ) per iteration"));
    grid.appendChild(leftBox); grid.appendChild(rightBox);
    const Lc = LR.canvas(leftBox, 420, 350, { aria: "Scatter with three Gaussian component ellipses; point colours show soft responsibilities updated by the E-step, ellipses move on the M-step" });
    const Rc = LR.canvas(rightBox, 380, 350, { aria: "Curve of the mixture log-likelihood after each EM iteration, monotonically increasing" });

    const ro = LR.readout(mount, [
      { k: "it", label: "iteration" },
      { k: "phase", label: "next step" },
      { k: "ll", label: "log-likelihood" },
      { k: "dll", label: "Δℓ" },
    ]);

    function init(reseed, fromKmeans) {
      stopPlay();
      initN++;
      let mus;
      if (fromKmeans) {
        let km = null;
        for (let r = 0; r < 6; r++) {
          const run = CL.kmeansRun(X, CL.seedCentroids(X, K, LR.rng(70 + r)));
          if (!km || run.J < km.J) km = run;
        }
        mus = km.M.map((m) => m.slice());
      } else {
        mus = CL.seedCentroids(X, K, LR.rng(500 + initN * 23));
      }
      st = {
        pis: [1 / K, 1 / K, 1 / K],
        mus: mus,
        Ss: mus.map(() => [[1.2, 0], [0, 1.2]]),
        G: null,
      };
      phase = "e"; iter = 0; llHist = []; converged = false;
      ro.set("it", "0");
      ro.set("phase", "E-step", C.orange);
      ro.set("ll", "–"); ro.set("dll", "–");
      sync(); render();
    }

    function doE() {
      if (phase !== "e" || converged) return;
      st.G = CL.gammas(X, st.pis, st.mus, st.Ss);
      phase = "m";
      ro.set("phase", "M-step", C.purple);
      sync(); render();
    }

    function doM() {
      if (phase !== "m" || converged) return;
      const p = CL.mstep(X, st.G);
      st.pis = p.pis; st.mus = p.mus; st.Ss = p.Ss;
      iter++;
      const ll = CL.logLik(X, st.pis, st.mus, st.Ss);
      const prev = llHist.length ? llHist[llHist.length - 1] : -Infinity;
      llHist.push(ll);
      const d = ll - prev;
      if (isFinite(prev) && d < 1e-3) {
        converged = true;
        ro.set("phase", "converged (Δℓ ≈ 0)", C.green);
      } else {
        phase = "e";
        ro.set("phase", "E-step", C.orange);
      }
      ro.set("it", String(iter));
      ro.set("ll", LR.fmtF(ll, 2), C.orange);
      ro.set("dll", isFinite(prev) ? "+" + LR.fmtF(d, 4) : "–", C.green);
      sync(); render();
    }

    function sync() {
      eBtn.disabled = converged || phase !== "e";
      mBtn.disabled = converged || phase !== "m";
      eBtn.classList.toggle("primary", !converged && phase === "e");
      mBtn.classList.toggle("primary", !converged && phase === "m");
    }

    function render() {
      // left: data + components
      Lc.ctx.clearRect(0, 0, Lc.W, Lc.H);
      const map = LR.plot(Lc.ctx, {
        x0: 42, y0: 12, w: Lc.W - 54, h: Lc.H - 76,
        xmin: 0, xmax: 10, ymin: 0, ymax: 8,
        xlabel: "feature x₁", ylabel: "feature x₂",
      });
      st.mus.forEach(function (mu, k) {
        CL.ellipse(Lc.ctx, map, mu, st.Ss[k], 1, CL.COLS[k], 2.2);
        CL.ellipse(Lc.ctx, map, mu, st.Ss[k], 2, CL.COLS[k], 1.1, [4, 4]);
      });
      X.forEach(function (p, n) {
        const col = st.G ? CL.mix(st.G[n]) : "#9a9a9a";
        LR.dot(Lc.ctx, map.sx(p[0]), map.sy(p[1]), 3.4, col);
      });
      st.mus.forEach((mu, k) => CL.cross(Lc.ctx, map.sx(mu[0]), map.sy(mu[1]), CL.COLS[k]));
      // mixing-weight bars along the bottom
      const bx = 42, bw = Lc.W - 54, by = Lc.H - 22;
      let acc = 0;
      Lc.ctx.font = "700 10.5px Inter, sans-serif";
      st.pis.forEach(function (pi, k) {
        Lc.ctx.fillStyle = CL.COLS[k];
        Lc.ctx.fillRect(bx + acc * bw, by, pi * bw, 12);
        Lc.ctx.fillStyle = "#fff";
        Lc.ctx.textAlign = "center";
        if (pi > 0.08) Lc.ctx.fillText("π" + (k + 1) + "=" + LR.fmtF(pi, 2), bx + (acc + pi / 2) * bw, by + 10);
        acc += pi;
      });

      // right: log-likelihood curve
      Rc.ctx.clearRect(0, 0, Rc.W, Rc.H);
      const lo = llHist.length ? Math.min.apply(null, llHist) : -1;
      const hi = llHist.length ? Math.max.apply(null, llHist) : 0;
      const pad = Math.max((hi - lo) * 0.15, 1);
      const P = LR.plot(Rc.ctx, {
        x0: 60, y0: 14, w: Rc.W - 74, h: Rc.H - 58,
        xmin: 0, xmax: Math.max(10, llHist.length + 1), ymin: lo - pad, ymax: hi + pad,
        xlabel: "EM iteration", ylabel: "log-likelihood ℓ(Θ)",
      });
      if (llHist.length) {
        Rc.ctx.strokeStyle = C.orange;
        Rc.ctx.lineWidth = 2.4;
        Rc.ctx.beginPath();
        llHist.forEach((ll, i) => (i === 0 ? Rc.ctx.moveTo(P.sx(i + 1), P.sy(ll)) : Rc.ctx.lineTo(P.sx(i + 1), P.sy(ll))));
        Rc.ctx.stroke();
        llHist.forEach((ll, i) => LR.dot(Rc.ctx, P.sx(i + 1), P.sy(ll), 3.4, C.orange));
      } else {
        Rc.ctx.fillStyle = C.faint;
        Rc.ctx.font = "600 12px Inter, sans-serif";
        Rc.ctx.textAlign = "center";
        Rc.ctx.fillText("run an E-step and an M-step to start the curve", Rc.W / 2, Rc.H / 2);
      }
    }

    init(true);
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.1 — K-means vs GMM on stretched clusters
     ══════════════════════════════════════════════════════════ */
  LR.figs.vs = function (mount) {
    LR.header(
      mount,
      "One dataset, two rulers",
      "Both algorithms run live to convergence on the same elongated, tilted clusters. The toggle constrains the GMM to Σ = I with uniform π, which is exactly soft K-means."
    );

    const X = CL.stretched(21);
    const K = 3;
    let constrained = false;

    const bar = LR.controls(mount);
    const tog = LR.button(bar, "Constrain: Σk = I, uniform π", function () {
      constrained = !constrained;
      tog.classList.toggle("on", constrained);
      fit(); render();
    });
    LR.button(bar, "Re-run both", function () { seedShift++; fit(); render(); }, "primary");
    let seedShift = 0;

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const leftBox = LR.el("div"), rightBox = LR.el("div");
    leftBox.appendChild(LR.el("div", "pane-label", "K-means: hard, spherical, fixed distance"));
    const rightLab = LR.el("div", "pane-label", "GMM by EM: soft, elliptical, Mahalanobis");
    rightBox.appendChild(rightLab);
    grid.appendChild(leftBox); grid.appendChild(rightBox);
    const Lc = LR.canvas(leftBox, 400, 330, { aria: "K-means result on stretched clusters: hard colours and straight boundaries that cut across the elongated clumps" });
    const Rc = LR.canvas(rightBox, 400, 330, { aria: "GMM result on the same data: tilted ellipses and soft colours that follow the elongated clumps; a toggle collapses it to soft K-means" });
    const ro = LR.readout(mount, [
      { k: "km", label: "K-means J" },
      { k: "ll", label: "GMM log-likelihood" },
      { k: "mode", label: "GMM mode" },
    ]);

    let km = null, gm = null;

    function fit() {
      // K-means: best of 8 restarts
      km = null;
      for (let r = 0; r < 8; r++) {
        const run = CL.kmeansRun(X, CL.seedCentroids(X, K, LR.rng(800 + r * 3 + seedShift)));
        if (!km || run.J < km.J) km = run;
      }
      // GMM: EM from the K-means means
      gm = {
        pis: [1 / K, 1 / K, 1 / K],
        mus: km.M.map((m) => m.slice()),
        Ss: km.M.map(() => [[1, 0], [0, 1]]),
        G: null,
      };
      for (let i = 0; i < 80; i++) {
        gm.G = CL.gammas(X, gm.pis, gm.mus, gm.Ss);
        const p = CL.mstep(X, gm.G);
        if (constrained) {
          // frozen: identity covariances, uniform mixing → soft K-means
          gm.mus = p.mus;
          gm.Ss = gm.mus.map(() => [[1, 0], [0, 1]]);
          gm.pis = gm.mus.map(() => 1 / K);
        } else {
          gm.pis = p.pis; gm.mus = p.mus; gm.Ss = p.Ss;
        }
      }
      gm.G = CL.gammas(X, gm.pis, gm.mus, gm.Ss);
      gm.ll = CL.logLik(X, gm.pis, gm.mus, gm.Ss);
    }

    function panel(P, mode) {
      const ctx = P.ctx;
      ctx.clearRect(0, 0, P.W, P.H);
      const map = LR.plot(ctx, {
        x0: 42, y0: 12, w: P.W - 54, h: P.H - 54,
        xmin: 0, xmax: 10, ymin: 0, ymax: 8,
        xlabel: "feature x₁", ylabel: "feature x₂",
      });
      // decision tint
      const nx = 70, ny = 50;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
          const x = (i + 0.5) * (10 / nx), y = (j + 0.5) * (8 / ny);
          let best = 0;
          if (mode === "km") {
            let bd = Infinity;
            for (let k = 0; k < K; k++) {
              const d = CL.d2([x, y], km.M[k]);
              if (d < bd) { bd = d; best = k; }
            }
          } else {
            let bv = -Infinity;
            for (let k = 0; k < K; k++) {
              const v = gm.pis[k] * CL.mvn2([x, y], gm.mus[k], gm.Ss[k]);
              if (v > bv) { bv = v; best = k; }
            }
          }
          const rgb = CL.RGB[best];
          ctx.fillStyle = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ",0.08)";
          const x0 = map.sx(i * (10 / nx)), x1 = map.sx((i + 1) * (10 / nx));
          const y0 = map.sy((j + 1) * (8 / ny)), y1 = map.sy(j * (8 / ny));
          ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
        }
      }
      if (mode === "km") {
        X.forEach((p, n) => LR.dot(ctx, map.sx(p[0]), map.sy(p[1]), 3.4, CL.COLS[km.lab[n]]));
        km.M.forEach((m, k) => CL.cross(ctx, map.sx(m[0]), map.sy(m[1]), CL.COLS[k]));
      } else {
        gm.mus.forEach(function (mu, k) {
          CL.ellipse(ctx, map, mu, gm.Ss[k], 1, CL.COLS[k], 2.2);
          CL.ellipse(ctx, map, mu, gm.Ss[k], 2, CL.COLS[k], 1.1, [4, 4]);
        });
        X.forEach((p, n) => LR.dot(ctx, map.sx(p[0]), map.sy(p[1]), 3.4, CL.mix(gm.G[n])));
        gm.mus.forEach((mu, k) => CL.cross(ctx, map.sx(mu[0]), map.sy(mu[1]), CL.COLS[k]));
      }
    }

    function render() {
      panel(Lc, "km");
      panel(Rc, "gmm");
      rightLab.textContent = constrained
        ? "Constrained GMM: Σ = I, uniform π → soft K-means"
        : "GMM by EM: soft, elliptical, Mahalanobis";
      ro.set("km", LR.fmtF(km.J, 1), C.orange);
      ro.set("ll", LR.fmtF(gm.ll, 1), C.purple);
      ro.set("mode", constrained ? "collapsed to soft K-means" : "full covariances", constrained ? C.red : C.green);
    }

    fit(); render();
  };
})();
