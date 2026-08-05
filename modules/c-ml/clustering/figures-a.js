/* ══════════════════════════════════════════════════════════════
   figures-a.js — Clustering lesson, K-means half
   Fig 0.1 hook · Fig 1.1 flavours · Fig 2.1 kmloop (signature)
   Fig 2.2 restart · Fig 2.3 elbow · Fig 2.4 kmcode
   Also defines the shared clustering core (LR.cl) used by
   figures-b.js: distances, K-means steps, softmax
   responsibilities, 2-D Gaussian density/fit/ellipses, one EM
   step, log-likelihood, datasets, and responsibility colour
   blending. All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared clustering core (also used by figures-b.js) ─── */
  const CL = (LR.cl = {
    /* cluster colours: orange, green, purple, red, amber */
    COLS: [C.orange, C.green, C.purple, C.red, C.amber],
    RGB: [[232, 89, 12], [47, 158, 68], [112, 72, 232], [224, 49, 49], [240, 162, 2]],

    /* blend of cluster colours weighted by responsibilities */
    mix: function (resp) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < resp.length; k++) {
        r += resp[k] * CL.RGB[k][0];
        g += resp[k] * CL.RGB[k][1];
        b += resp[k] * CL.RGB[k][2];
      }
      return "rgb(" + Math.round(r) + "," + Math.round(g) + "," + Math.round(b) + ")";
    },

    /* squared Euclidean distance between [x,y] points */
    d2: function (a, b) {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
      return s;
    },

    /* ── K-means ────────────────────────────────────────────── */
    assign: function (X, M) {
      return X.map(function (x) {
        let best = 0, bd = Infinity;
        for (let k = 0; k < M.length; k++) {
          const d = CL.d2(x, M[k]);
          if (d < bd) { bd = d; best = k; }
        }
        return best;
      });
    },
    inertia: function (X, M, lab) {
      let J = 0;
      for (let n = 0; n < X.length; n++) J += CL.d2(X[n], M[lab[n]]);
      return J;
    },
    update: function (X, lab, M) {
      // empty clusters keep their old centroid
      return M.map(function (m, k) {
        let sx = 0, sy = 0, c = 0;
        for (let n = 0; n < X.length; n++) {
          if (lab[n] === k) { sx += X[n][0]; sy += X[n][1]; c++; }
        }
        return c > 0 ? [sx / c, sy / c] : m.slice();
      });
    },
    kmeansRun: function (X, M0, maxIter) {
      let M = M0.map((m) => m.slice());
      let lab = CL.assign(X, M), it = 0;
      for (it = 0; it < (maxIter || 100); it++) {
        M = CL.update(X, lab, M);
        const lab2 = CL.assign(X, M);
        let same = true;
        for (let n = 0; n < X.length; n++) if (lab2[n] !== lab[n]) { same = false; break; }
        lab = lab2;
        if (same) break;
      }
      return { M: M, lab: lab, J: CL.inertia(X, M, lab), iters: it + 1 };
    },
    seedCentroids: function (X, K, rand) {
      // pick K distinct data points as seeds
      const idx = [];
      while (idx.length < K) {
        const i = Math.floor(rand() * X.length);
        if (idx.indexOf(i) === -1) idx.push(i);
      }
      return idx.map((i) => X[i].slice());
    },

    /* ── soft K-means: softmax over negative distances ──────── */
    softResp: function (x, M, beta) {
      const s = M.map((m) => -beta * CL.d2(x, m));
      const mx = Math.max.apply(null, s);
      const e = s.map((v) => Math.exp(v - mx));
      const Z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / Z);
    },
    softUpdate: function (X, R, M) {
      return M.map(function (m, k) {
        let sx = 0, sy = 0, w = 0;
        for (let n = 0; n < X.length; n++) {
          sx += R[n][k] * X[n][0]; sy += R[n][k] * X[n][1]; w += R[n][k];
        }
        return w > 1e-12 ? [sx / w, sy / w] : m.slice();
      });
    },

    /* ── 2-D Gaussians (same forms as the Gaussian Bayes lesson) */
    det2: function (S) { return S[0][0] * S[1][1] - S[0][1] * S[1][0]; },
    inv2: function (S) {
      const d = CL.det2(S);
      return [[S[1][1] / d, -S[0][1] / d], [-S[1][0] / d, S[0][0] / d]];
    },
    logMvn2: function (p, mu, S) {
      const I = CL.inv2(S);
      const dx = p[0] - mu[0], dy = p[1] - mu[1];
      const q = dx * (I[0][0] * dx + I[0][1] * dy) + dy * (I[1][0] * dx + I[1][1] * dy);
      return -Math.log(2 * Math.PI) - 0.5 * Math.log(CL.det2(S)) - 0.5 * q;
    },
    mvn2: function (p, mu, S) { return Math.exp(CL.logMvn2(p, mu, S)); },
    g1: function (x, mu, s2) {
      return Math.exp(-((x - mu) * (x - mu)) / (2 * s2)) / Math.sqrt(2 * Math.PI * s2);
    },
    eig2: function (S) {
      const a = S[0][0], b = S[0][1], c = S[1][1];
      const tr2 = (a + c) / 2;
      const disc = Math.sqrt(((a - c) / 2) * ((a - c) / 2) + b * b);
      return { l1: tr2 + disc, l2: Math.max(tr2 - disc, 1e-9), theta: 0.5 * Math.atan2(2 * b, a - c) };
    },
    ellipse: function (ctx, map, mu, S, r, color, width, dash) {
      const e = CL.eig2(S);
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
    cross: function (ctx, px, py, color, size) {
      const s = size || 7;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(px - s, py); ctx.lineTo(px + s, py);
      ctx.moveTo(px, py - s); ctx.lineTo(px, py + s);
      ctx.stroke();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();
    },

    /* ── EM for a 2-D GMM ───────────────────────────────────── */
    gammas: function (X, pis, mus, Ss) {
      return X.map(function (p) {
        const w = mus.map((m, k) => pis[k] * CL.mvn2(p, m, Ss[k]));
        const s = w.reduce((a, b) => a + b, 0) || 1e-300;
        return w.map((v) => v / s);
      });
    },
    mstep: function (X, G) {
      const K = G[0].length, N = X.length;
      const Nk = Array.from({ length: K }, (_, k) => G.reduce((s, g) => s + g[k], 0));
      const mus = Array.from({ length: K }, function (_, k) {
        let sx = 0, sy = 0;
        for (let n = 0; n < N; n++) { sx += G[n][k] * X[n][0]; sy += G[n][k] * X[n][1]; }
        return [sx / Nk[k], sy / Nk[k]];
      });
      const Ss = Array.from({ length: K }, function (_, k) {
        let a = 0, b = 0, c = 0;
        for (let n = 0; n < N; n++) {
          const dx = X[n][0] - mus[k][0], dy = X[n][1] - mus[k][1];
          a += G[n][k] * dx * dx; b += G[n][k] * dx * dy; c += G[n][k] * dy * dy;
        }
        // tiny ridge keeps a collapsing component from going singular
        return [[a / Nk[k] + 1e-6, b / Nk[k]], [b / Nk[k], c / Nk[k] + 1e-6]];
      });
      return { pis: Nk.map((v) => v / N), mus: mus, Ss: Ss, Nk: Nk };
    },
    emStep: function (X, pis, mus, Ss) {
      const G = CL.gammas(X, pis, mus, Ss);
      const p = CL.mstep(X, G);
      return { pis: p.pis, mus: p.mus, Ss: p.Ss, G: G, ll: CL.logLik(X, p.pis, p.mus, p.Ss) };
    },
    logLik: function (X, pis, mus, Ss) {
      let ll = 0;
      for (const p of X) {
        let s = 0;
        for (let k = 0; k < pis.length; k++) s += pis[k] * CL.mvn2(p, mus[k], Ss[k]);
        ll += Math.log(Math.max(s, 1e-300));
      }
      return ll;
    },
    sampleMvn: function (mu, S, rand) {
      // Cholesky of a 2×2 SPD matrix
      const l11 = Math.sqrt(S[0][0]);
      const l21 = S[1][0] / l11;
      const l22 = Math.sqrt(Math.max(S[1][1] - l21 * l21, 1e-9));
      const z1 = LR.gauss(rand), z2 = LR.gauss(rand);
      return [mu[0] + l11 * z1, mu[1] + l21 * z1 + l22 * z2];
    },

    /* ── datasets (deterministic, in a 0..10 × 0..8 world) ──── */
    blobs: function (seed) {
      // three roundish clumps
      const rand = LR.rng(seed || 7);
      const spec = [
        { mu: [2.4, 2.2], s: 0.65, n: 32 },
        { mu: [7.3, 2.6], s: 0.75, n: 32 },
        { mu: [4.8, 5.9], s: 0.7, n: 32 },
      ];
      const X = [];
      spec.forEach(function (sp) {
        for (let i = 0; i < sp.n; i++) {
          X.push([sp.mu[0] + LR.gauss(rand) * sp.s, sp.mu[1] + LR.gauss(rand) * sp.s]);
        }
      });
      return X;
    },
    stretched: function (seed) {
      // three elongated, tilted clusters (the K-means killer)
      const rand = LR.rng(seed || 21);
      const spec = [
        { mu: [3.0, 2.0], A: [[1.5, 0], [0.95, 0.28]], n: 40 },
        { mu: [7.2, 3.4], A: [[1.3, 0], [-0.85, 0.3]], n: 40 },
        { mu: [4.6, 5.9], A: [[1.55, 0], [0.5, 0.3]], n: 40 },
      ];
      const X = [];
      spec.forEach(function (sp) {
        for (let i = 0; i < sp.n; i++) {
          const z1 = LR.gauss(rand), z2 = LR.gauss(rand);
          X.push([
            sp.mu[0] + sp.A[0][0] * z1 + sp.A[0][1] * z2,
            sp.mu[1] + sp.A[1][0] * z1 + sp.A[1][1] * z2,
          ]);
        }
      });
      return X;
    },

    /* standard scatter frame for the 0..10 × 0..8 world */
    frame: function (ctx, W, H, title) {
      return LR.plot(ctx, {
        x0: 46, y0: 14, w: W - 60, h: H - 58,
        xmin: 0, xmax: 10, ymin: 0, ymax: 8,
        xlabel: "feature x₁", ylabel: "feature x₂", title: title,
      });
    },
  });

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — the hook: groups + a density, found live by EM
     ══════════════════════════════════════════════════════════ */
  LR.figs.hook = function (mount) {
    LR.header(
      mount,
      "Unlabeled points, and what this lecture can do with them",
      "A real EM fit runs when you press the button; nothing is staged. Colours are responsibilities, ellipses are components, hollow points are samples from the fitted density."
    );

    let seed = 7;
    let X = CL.blobs(seed);
    const K = 3;
    let st = null; // {pis, mus, Ss, G, ll}
    let samples = [];
    let timer = null;
    let iter = 0;

    const bar = LR.controls(mount);
    const { cv, ctx, W, H } = LR.canvas(mount, 640, 430, {
      aria: "Scatter of unlabeled points; a mixture of Gaussians is fitted live, colouring the points and drawing component ellipses",
    });
    const ro = LR.readout(mount, [
      { k: "state", label: "state" },
      { k: "it", label: "EM iterations" },
      { k: "ll", label: "log-likelihood" },
    ]);

    function initEM() {
      const rand = LR.rng(seed * 31 + 5);
      const mus = CL.seedCentroids(X, K, rand);
      st = {
        pis: [1 / K, 1 / K, 1 / K],
        mus: mus,
        Ss: mus.map(() => [[1.3, 0], [0, 1.3]]),
        G: null, ll: NaN,
      };
      iter = 0;
    }

    function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }

    function fit() {
      stopTimer();
      samples = [];
      initEM();
      let prev = -Infinity;
      const tick = function () {
        const r = CL.emStep(X, st.pis, st.mus, st.Ss);
        st = r; iter++;
        const done = Math.abs(r.ll - prev) < 1e-4 || iter >= 80;
        prev = r.ll;
        ro.set("state", done ? "converged" : "fitting…", done ? C.green : C.orange);
        ro.set("it", String(iter));
        ro.set("ll", LR.fmtF(r.ll, 2), C.orange);
        render();
        if (done) stopTimer();
      };
      if (LR.reducedMotion) {
        // no animation: run to convergence in one go
        for (let i = 0; i < 80; i++) {
          const r = CL.emStep(X, st.pis, st.mus, st.Ss);
          const done = Math.abs(r.ll - (st.ll || -Infinity)) < 1e-4 && i > 0;
          st = r; iter++;
          if (done) break;
        }
        ro.set("state", "converged", C.green);
        ro.set("it", String(iter));
        ro.set("ll", LR.fmtF(st.ll, 2), C.orange);
        render();
      } else {
        tick();
        timer = setInterval(tick, 240);
      }
    }

    LR.button(bar, "Find the groups", fit, "primary");
    LR.button(bar, "Sample new points", function () {
      if (!st || !st.G) { ro.set("state", "fit first, then sample", C.red); return; }
      const rand = LR.rng(Date.now() % 100000);
      for (let i = 0; i < 25; i++) {
        // ancestral sampling: z ~ Categorical(pi), then x ~ N(mu_z, S_z)
        let u = rand(), z = 0, acc = 0;
        for (let k = 0; k < K; k++) { acc += st.pis[k]; if (u <= acc) { z = k; break; } }
        samples.push({ p: CL.sampleMvn(st.mus[z], st.Ss[z], rand), z: z });
      }
      ro.set("state", samples.length + " generated points", C.purple);
      render();
    });
    LR.button(bar, "Re-scatter", function () {
      stopTimer();
      seed = (seed * 7 + 3) % 1000 + 2;
      X = CL.blobs(seed);
      st = null; samples = []; iter = 0;
      ro.set("state", "unlabeled"); ro.set("it", "–"); ro.set("ll", "–");
      render();
    });
    LR.button(bar, "Reset", function () {
      stopTimer();
      st = null; samples = []; iter = 0;
      ro.set("state", "unlabeled"); ro.set("it", "–"); ro.set("ll", "–");
      render();
    });

    function render() {
      ctx.clearRect(0, 0, W, H);
      const map = CL.frame(ctx, W, H);
      // components
      if (st && st.G) {
        st.mus.forEach(function (mu, k) {
          CL.ellipse(ctx, map, mu, st.Ss[k], 1, CL.COLS[k], 2.2);
          CL.ellipse(ctx, map, mu, st.Ss[k], 2, CL.COLS[k], 1.2, [4, 4]);
        });
      }
      // data points
      X.forEach(function (p, n) {
        const col = st && st.G ? CL.mix(st.G[n]) : "#9a9a9a";
        LR.dot(ctx, map.sx(p[0]), map.sy(p[1]), 4, col);
      });
      // generated samples (hollow)
      samples.forEach(function (s) {
        LR.dot(ctx, map.sx(s.p[0]), map.sy(s.p[1]), 4.5, "#fff", CL.COLS[s.z]);
      });
      // centroid crosses
      if (st && st.G) st.mus.forEach((mu, k) => CL.cross(ctx, map.sx(mu[0]), map.sy(mu[1]), CL.COLS[k]));
    }

    ro.set("state", "unlabeled");
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — three flavours of unsupervised learning
     ══════════════════════════════════════════════════════════ */
  LR.figs.flavours = function (mount) {
    LR.header(
      mount,
      "Reduce, cluster, or estimate the density",
      "Same dataset each time. The toggle changes which summary we compute; every view is calculated live from the points."
    );

    const X = CL.blobs(19);
    let mode = "cluster";

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = {};
    [["reduce", "Reduce (PCA)"], ["cluster", "Cluster (K-means)"], ["density", "Density (mixture)"]].forEach(function (m) {
      btns[m[0]] = LR.button(group, m[1], function () {
        mode = m[0];
        Object.keys(btns).forEach((k) => btns[k].classList.toggle("on", k === mode));
        render();
      });
    });
    btns[mode].classList.add("on");

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 400, {
      aria: "The same scatter summarized three ways: a principal axis, K-means cluster colours, or a mixture density heatmap",
    });
    const note = LR.el("div", "fig-sub");
    mount.appendChild(note);

    // precompute the three summaries once (all real)
    // PCA: mean + leading eigenvector of the covariance
    const mean = [0, 0];
    X.forEach((p) => { mean[0] += p[0] / X.length; mean[1] += p[1] / X.length; });
    let sxx = 0, sxy = 0, syy = 0;
    X.forEach((p) => {
      const dx = p[0] - mean[0], dy = p[1] - mean[1];
      sxx += dx * dx / X.length; sxy += dx * dy / X.length; syy += dy * dy / X.length;
    });
    const eg = CL.eig2([[sxx, sxy], [sxy, syy]]);
    const dir = [Math.cos(eg.theta), Math.sin(eg.theta)];

    // K-means: best of 8 restarts, K=3
    let km = null;
    for (let r = 0; r < 8; r++) {
      const run = CL.kmeansRun(X, CL.seedCentroids(X, 3, LR.rng(100 + r)));
      if (!km || run.J < km.J) km = run;
    }

    // GMM: EM from the K-means solution
    let gm = { pis: [1 / 3, 1 / 3, 1 / 3], mus: km.M.map((m) => m.slice()), Ss: km.M.map(() => [[0.8, 0], [0, 0.8]]) };
    for (let i = 0; i < 40; i++) {
      const r = CL.emStep(X, gm.pis, gm.mus, gm.Ss);
      gm = { pis: r.pis, mus: r.mus, Ss: r.Ss, G: r.G };
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const map = CL.frame(ctx, W, H);

      if (mode === "density") {
        // heatmap of the fitted mixture density
        const nx = 90, ny = 64;
        let dmax = 0;
        const vals = [];
        for (let i = 0; i < nx; i++) {
          vals.push([]);
          for (let j = 0; j < ny; j++) {
            const x = (i + 0.5) * (10 / nx), y = (j + 0.5) * (8 / ny);
            let v = 0;
            for (let k = 0; k < 3; k++) v += gm.pis[k] * CL.mvn2([x, y], gm.mus[k], gm.Ss[k]);
            vals[i].push(v);
            if (v > dmax) dmax = v;
          }
        }
        for (let i = 0; i < nx; i++) {
          for (let j = 0; j < ny; j++) {
            const t = Math.pow(vals[i][j] / dmax, 0.6);
            if (t < 0.02) continue;
            ctx.fillStyle = "rgba(232,89,12," + (0.55 * t).toFixed(3) + ")";
            const x0 = map.sx(i * (10 / nx)), x1 = map.sx((i + 1) * (10 / nx));
            const y0 = map.sy((j + 1) * (8 / ny)), y1 = map.sy(j * (8 / ny));
            ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
          }
        }
        X.forEach((p) => LR.dot(ctx, map.sx(p[0]), map.sy(p[1]), 3, "#333"));
        note.textContent = "Density estimation: p(x) = Σ πk N(x; μk, Σk) fitted by EM, shaded bright where the model expects data. The summary is a distribution over the whole space.";
      } else if (mode === "cluster") {
        X.forEach((p, n) => LR.dot(ctx, map.sx(p[0]), map.sy(p[1]), 4, CL.COLS[km.lab[n]]));
        km.M.forEach((m, k) => CL.cross(ctx, map.sx(m[0]), map.sy(m[1]), CL.COLS[k]));
        note.textContent = "Clustering: each point is summarized by a prototypical example, the centroid it belongs to (crosses). Found by a real K-means run, J = " + LR.fmtF(km.J, 1) + ".";
      } else {
        // projection segment through the mean along the principal direction
        const L = 4.6;
        const a = [mean[0] - dir[0] * L, mean[1] - dir[1] * L];
        const b = [mean[0] + dir[0] * L, mean[1] + dir[1] * L];
        ctx.strokeStyle = C.purple;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(map.sx(a[0]), map.sy(a[1]));
        ctx.lineTo(map.sx(b[0]), map.sy(b[1]));
        ctx.stroke();
        X.forEach(function (p) {
          const t = (p[0] - mean[0]) * dir[0] + (p[1] - mean[1]) * dir[1];
          const pr = [mean[0] + t * dir[0], mean[1] + t * dir[1]];
          ctx.strokeStyle = "rgba(112,72,232,0.25)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(map.sx(p[0]), map.sy(p[1]));
          ctx.lineTo(map.sx(pr[0]), map.sy(pr[1]));
          ctx.stroke();
          LR.dot(ctx, map.sx(p[0]), map.sy(p[1]), 3, "#bbb");
          LR.dot(ctx, map.sx(pr[0]), map.sy(pr[1]), 3.4, C.purple);
        });
        note.textContent = "Dimensionality reduction: each point is summarized by one number, its coordinate along the leading principal direction (computed live from the covariance), the PCA move from Lecture 18.";
      }
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — the K-means loop (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.kmloop = function (mount) {
    LR.header(
      mount,
      "K-means: drag the seeds, then alternate the two steps",
      "Assign colours each point by its nearest centroid; Update slides each centroid to its cluster mean. The tinted background is the current nearest-centroid partition of the plane."
    );

    const X = CL.blobs(7);
    let K = 3;
    let M = [], lab = null, phase = "assign", steps = 0, converged = false;
    let trails = [];
    let seedCount = 0;
    let timer = null;

    const bar = LR.controls(mount);
    const kSlider = LR.slider(bar, "K", 2, 5, 1, K, function (v) {
      K = Math.round(v);
      reset(true);
    }, (v) => String(Math.round(v)));
    const assignBtn = LR.button(bar, "Assign step", doAssign, "primary");
    const updateBtn = LR.button(bar, "Update step", doUpdate);
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (timer) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      timer = setInterval(function () {
        if (converged) { stopPlay(); return; }
        if (phase === "assign") doAssign(); else doUpdate();
      }, LR.reducedMotion ? 1100 : 550);
    });
    LR.button(bar, "Re-seed", function () { reset(true); });

    function stopPlay() {
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = "Play ▸▸";
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 640, 430, {
      aria: "K-means demo: draggable centroid crosses over a scatter; buttons alternate assignment and update steps while the inertia readout falls",
    });
    const ro = LR.readout(mount, [
      { k: "step", label: "steps taken" },
      { k: "J", label: "inertia J" },
      { k: "st", label: "status" },
    ]);

    function reset(reseed) {
      stopPlay();
      if (reseed) {
        seedCount++;
        M = CL.seedCentroids(X, K, LR.rng(40 + seedCount * 17 + K));
      }
      lab = null; phase = "assign"; steps = 0; converged = false;
      trails = M.map((m) => [m.slice()]);
      syncButtons();
      ro.set("step", "0");
      ro.set("J", "–");
      ro.set("st", "drag the crosses, then Assign");
      render();
    }

    function doAssign() {
      if (phase !== "assign" || converged) return;
      const old = lab;
      lab = CL.assign(X, M);
      steps++;
      let same = old !== null;
      if (old) for (let n = 0; n < X.length; n++) if (lab[n] !== old[n]) { same = false; break; }
      if (same) {
        converged = true;
        ro.set("st", "assignments unchanged → converged", C.green);
      } else {
        ro.set("st", "points recoloured to nearest centroid");
      }
      phase = "update";
      ro.set("step", String(steps));
      ro.set("J", LR.fmtF(CL.inertia(X, M, lab), 2), C.orange);
      syncButtons();
      render();
    }

    function doUpdate() {
      if (phase !== "update" || converged) return;
      M = CL.update(X, lab, M);
      M.forEach((m, k) => trails[k].push(m.slice()));
      steps++;
      phase = "assign";
      ro.set("step", String(steps));
      ro.set("J", LR.fmtF(CL.inertia(X, M, lab), 2), C.orange);
      ro.set("st", "centroids moved to their cluster means");
      syncButtons();
      render();
    }

    function syncButtons() {
      assignBtn.disabled = converged || phase !== "assign";
      updateBtn.disabled = converged || phase !== "update";
      assignBtn.classList.toggle("primary", !converged && phase === "assign");
      updateBtn.classList.toggle("primary", !converged && phase === "update");
    }

    // drag the centroids
    let dragK = -1;
    let map = null;
    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!map) return false;
        for (let k = 0; k < M.length; k++) {
          const dx = p.x - map.sx(M[k][0]), dy = p.y - map.sy(M[k][1]);
          if (dx * dx + dy * dy < 190) { dragK = k; return true; }
        }
        return false;
      },
      move: function (p) {
        if (dragK < 0) return;
        const w = map.inv(p.x, p.y);
        M[dragK] = [Math.max(0, Math.min(10, w.x)), Math.max(0, Math.min(8, w.y))];
        trails[dragK] = [M[dragK].slice()];
        // moving a seed reopens the loop
        converged = false; phase = "assign"; lab = null; steps = 0;
        ro.set("step", "0"); ro.set("J", "–"); ro.set("st", "seeds moved; Assign to start");
        syncButtons();
        render();
      },
      up: function () { dragK = -1; },
    });

    function render() {
      ctx.clearRect(0, 0, W, H);
      map = CL.frame(ctx, W, H);
      // nearest-centroid background tint (Voronoi partition, coarse grid)
      const nx = 88, ny = 62;
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
          const x = (i + 0.5) * (10 / nx), y = (j + 0.5) * (8 / ny);
          let best = 0, bd = Infinity;
          for (let k = 0; k < M.length; k++) {
            const d = CL.d2([x, y], M[k]);
            if (d < bd) { bd = d; best = k; }
          }
          const rgb = CL.RGB[best];
          ctx.fillStyle = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ",0.07)";
          const x0 = map.sx(i * (10 / nx)), x1 = map.sx((i + 1) * (10 / nx));
          const y0 = map.sy((j + 1) * (8 / ny)), y1 = map.sy(j * (8 / ny));
          ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
        }
      }
      // centroid trails
      trails.forEach(function (tr, k) {
        if (tr.length < 2) return;
        ctx.strokeStyle = CL.COLS[k];
        ctx.lineWidth = 1.4;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        tr.forEach((m, i) => (i === 0 ? ctx.moveTo(map.sx(m[0]), map.sy(m[1])) : ctx.lineTo(map.sx(m[0]), map.sy(m[1]))));
        ctx.stroke();
        ctx.setLineDash([]);
      });
      // points
      X.forEach(function (p, n) {
        const col = lab ? CL.COLS[lab[n]] : "#9a9a9a";
        LR.dot(ctx, map.sx(p[0]), map.sy(p[1]), 4, col);
      });
      // centroids
      M.forEach((m, k) => CL.cross(ctx, map.sx(m[0]), map.sy(m[1]), CL.COLS[k], 8));
    }

    reset(true);
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.2 — local minima and random restarts
     ══════════════════════════════════════════════════════════ */
  LR.figs.restart = function (mount) {
    LR.header(
      mount,
      "Different starts, different endings",
      "Each restart runs K-means (K = 3) to convergence from fresh random seeds. Converged is not the same as correct: compare the final inertias."
    );

    const X = CL.blobs(33);
    let run = null;
    let history = [];
    let seedN = 0;

    const bar = LR.controls(mount);
    LR.button(bar, "Random restart", function () {
      seedN++;
      const M0 = CL.seedCentroids(X, 3, LR.rng(900 + seedN * 13));
      run = CL.kmeansRun(X, M0);
      history.push(run.J);
      render();
    }, "primary");
    LR.button(bar, "Clear history", function () { history = []; run = null; render(); });

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const leftBox = LR.el("div");
    const rightBox = LR.el("div");
    leftBox.appendChild(LR.el("div", "pane-label", "This restart's converged clustering"));
    rightBox.appendChild(LR.el("div", "pane-label", "Final inertia of every restart"));
    grid.appendChild(leftBox); grid.appendChild(rightBox);

    const L = LR.canvas(leftBox, 400, 330, { aria: "Converged K-means clustering for the latest random restart" });
    const R = LR.canvas(rightBox, 400, 330, { aria: "Bar chart of final inertia values across restarts; the best run is starred" });
    const ro = LR.readout(mount, [
      { k: "J", label: "this run's final J" },
      { k: "best", label: "best J so far" },
      { k: "n", label: "restarts" },
    ]);

    function render() {
      // left: clustering
      L.ctx.clearRect(0, 0, L.W, L.H);
      const map = LR.plot(L.ctx, {
        x0: 42, y0: 12, w: L.W - 54, h: L.H - 54,
        xmin: 0, xmax: 10, ymin: 0, ymax: 8,
        xlabel: "feature x₁", ylabel: "feature x₂",
      });
      X.forEach(function (p, n) {
        const col = run ? CL.COLS[run.lab[n]] : "#9a9a9a";
        LR.dot(L.ctx, map.sx(p[0]), map.sy(p[1]), 3.6, col);
      });
      if (run) run.M.forEach((m, k) => CL.cross(L.ctx, map.sx(m[0]), map.sy(m[1]), CL.COLS[k]));

      // right: J history bars
      R.ctx.clearRect(0, 0, R.W, R.H);
      const ymax = history.length ? Math.max.apply(null, history) * 1.15 : 100;
      const bmap = LR.plot(R.ctx, {
        x0: 46, y0: 12, w: R.W - 58, h: R.H - 54,
        xmin: 0, xmax: Math.max(8, history.length + 1), ymin: 0, ymax: ymax,
        xlabel: "restart #", ylabel: "final inertia J",
        xticks: Array.from({ length: Math.max(8, history.length + 1) }, (_, i) => i + 1).filter((v) => v % 1 === 0),
      });
      const best = history.length ? Math.min.apply(null, history) : null;
      history.forEach(function (J, i) {
        const x0 = bmap.sx(i + 0.65), x1 = bmap.sx(i + 1.35);
        const isBest = Math.abs(J - best) < 1e-9;
        R.ctx.fillStyle = isBest ? C.green : (J > best * 1.25 ? C.red : C.orange);
        R.ctx.fillRect(x0, bmap.sy(J), x1 - x0, bmap.sy(0) - bmap.sy(J));
        if (isBest) {
          R.ctx.fillStyle = C.green;
          R.ctx.font = "700 13px Inter, sans-serif";
          R.ctx.textAlign = "center";
          R.ctx.fillText("★", (x0 + x1) / 2, bmap.sy(J) - 5);
        }
      });

      ro.set("J", run ? LR.fmtF(run.J, 2) : "–", C.orange);
      ro.set("best", best !== null ? LR.fmtF(best, 2) : "–", C.green);
      ro.set("n", String(history.length));
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.3 — the elbow method
     ══════════════════════════════════════════════════════════ */
  LR.figs.elbow = function (mount) {
    LR.header(
      mount,
      "Inertia vs K: find the elbow",
      "For each K, the best final inertia over 6 real K-means restarts. Click a point on the curve (or use the slider) to see that K's clustering."
    );

    const X = CL.blobs(19);
    const Ks = [1, 2, 3, 4, 5, 6, 7, 8];
    const results = Ks.map(function (K) {
      let best = null;
      for (let r = 0; r < 6; r++) {
        const run = CL.kmeansRun(X, CL.seedCentroids(X, K, LR.rng(K * 100 + r * 7 + 1)));
        if (!best || run.J < best.J) best = run;
      }
      return best;
    });

    let sel = 3;

    const bar = LR.controls(mount);
    const slider = LR.slider(bar, "show clustering for K", 1, 8, 1, sel, function (v) {
      sel = Math.round(v);
      render();
    }, (v) => String(Math.round(v)));

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const leftBox = LR.el("div"), rightBox = LR.el("div");
    leftBox.appendChild(LR.el("div", "pane-label", "Inertia J vs number of clusters K"));
    rightBox.appendChild(LR.el("div", "pane-label", "The clustering at the selected K"));
    grid.appendChild(leftBox); grid.appendChild(rightBox);
    const Lc = LR.canvas(leftBox, 400, 330, { aria: "Elbow curve of inertia against K with a marked elbow at K equals 3; points are clickable" });
    const Rc = LR.canvas(rightBox, 400, 330, { aria: "Scatter coloured by the best K-means clustering for the selected K" });
    const ro = LR.readout(mount, [
      { k: "K", label: "K" },
      { k: "J", label: "best J" },
      { k: "drop", label: "drop vs K−1" },
    ]);

    let cmap = null;
    Lc.cv.style.cursor = "pointer";
    Lc.cv.addEventListener("click", function (e) {
      const p = LR.evtXY(Lc.cv, Lc.W, Lc.H, e);
      let bi = -1, bd = Infinity;
      Ks.forEach(function (K, i) {
        const dx = p.x - cmap.sx(K), dy = p.y - cmap.sy(results[i].J);
        const d = dx * dx + dy * dy;
        if (d < bd) { bd = d; bi = i; }
      });
      if (bd < 700) { sel = Ks[bi]; slider.set(sel); render(); }
    });

    function render() {
      const Jmax = results[0].J * 1.1;
      Lc.ctx.clearRect(0, 0, Lc.W, Lc.H);
      cmap = LR.plot(Lc.ctx, {
        x0: 48, y0: 14, w: Lc.W - 62, h: Lc.H - 56,
        xmin: 0.5, xmax: 8.5, ymin: 0, ymax: Jmax,
        xlabel: "K (number of clusters)", ylabel: "inertia J",
        xticks: Ks,
      });
      // elbow annotation at K=3
      const ex = cmap.sx(3), ey = cmap.sy(results[2].J);
      Lc.ctx.strokeStyle = C.green;
      Lc.ctx.setLineDash([4, 4]);
      Lc.ctx.beginPath();
      Lc.ctx.moveTo(ex, ey);
      Lc.ctx.lineTo(ex + 44, ey - 44);
      Lc.ctx.stroke();
      Lc.ctx.setLineDash([]);
      Lc.ctx.fillStyle = C.green;
      Lc.ctx.font = "700 12px Inter, sans-serif";
      Lc.ctx.textAlign = "left";
      Lc.ctx.fillText("the elbow", ex + 48, ey - 48);
      // curve
      Lc.ctx.strokeStyle = C.orange;
      Lc.ctx.lineWidth = 2.4;
      Lc.ctx.beginPath();
      Ks.forEach((K, i) => (i === 0 ? Lc.ctx.moveTo(cmap.sx(K), cmap.sy(results[i].J)) : Lc.ctx.lineTo(cmap.sx(K), cmap.sy(results[i].J))));
      Lc.ctx.stroke();
      Ks.forEach(function (K, i) {
        LR.dot(Lc.ctx, cmap.sx(K), cmap.sy(results[i].J), K === sel ? 6.5 : 4.5, K === sel ? C.text : "#fff", C.orange);
      });

      // right: clustering at sel
      const run = results[sel - 1];
      Rc.ctx.clearRect(0, 0, Rc.W, Rc.H);
      const map = LR.plot(Rc.ctx, {
        x0: 42, y0: 12, w: Rc.W - 54, h: Rc.H - 54,
        xmin: 0, xmax: 10, ymin: 0, ymax: 8,
        xlabel: "feature x₁", ylabel: "feature x₂",
      });
      X.forEach(function (p, n) {
        LR.dot(Rc.ctx, map.sx(p[0]), map.sy(p[1]), 3.6, CL.COLS[run.lab[n] % CL.COLS.length]);
      });
      run.M.forEach((m, k) => CL.cross(Rc.ctx, map.sx(m[0]), map.sy(m[1]), CL.COLS[k % CL.COLS.length]));

      ro.set("K", String(sel));
      ro.set("J", LR.fmtF(run.J, 2), C.orange);
      ro.set("drop", sel > 1 ? LR.fmtF(results[sel - 2].J - run.J, 2) : "–", C.green);
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.4 — kmeans_step, executed live on the worked example
     ══════════════════════════════════════════════════════════ */
  LR.figs.kmcode = function (mount) {
    LR.header(
      mount,
      "kmeans_step on the worked example, for real",
      "The four points are fixed; the starting centroids are yours to edit. Each Step runs one genuine assignment + update and prints what happened."
    );

    const X = [[1, 1], [2, 1], [4, 3], [5, 4]];
    const DEF = [[0, 0], [4, 4]];
    let M = DEF.map((m) => m.slice());
    let prevLab = null;
    let stepN = 0;

    const bar = LR.controls(mount);
    const inputs = [];
    [["m₁ x", 0, 0], ["m₁ y", 0, 1], ["m₂ x", 1, 0], ["m₂ y", 1, 1]].forEach(function (spec) {
      const lab = LR.el("label");
      lab.appendChild(document.createTextNode(spec[0]));
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "0.5"; inp.value = DEF[spec[1]][spec[2]];
      inp.style.width = "64px";
      inp.setAttribute("aria-label", "initial centroid " + spec[0]);
      inp.addEventListener("input", function () {
        const v = parseFloat(inp.value);
        if (isFinite(v)) { M[spec[1]][spec[2]] = v; resetRun("centroids edited; press Step"); }
      });
      lab.appendChild(inp);
      bar.appendChild(lab);
      inputs.push({ inp: inp, i: spec[1], j: spec[2] });
    });
    LR.button(bar, "Step", doStep, "primary");
    LR.button(bar, "Reset", function () {
      M = DEF.map((m) => m.slice());
      inputs.forEach((o) => (o.inp.value = DEF[o.i][o.j]));
      resetRun("reset to the worked example's seeds");
    });

    const out = LR.el("pre", "attn-runout");
    out.setAttribute("aria-live", "polite");
    mount.appendChild(out);

    function resetRun(msg) {
      prevLab = null; stepN = 0;
      out.textContent = ">>> X = [(1,1), (2,1), (4,3), (5,4)]\n>>> centroids = [(" +
        M[0].join(", ") + "), (" + M[1].join(", ") + ")]\n" + (msg ? "# " + msg : "# press Step to run kmeans_step once");
    }

    function doStep() {
      const lab = CL.assign(X, M);
      const J = CL.inertia(X, M, lab);
      const d2s = X.map((x) => M.map((m) => CL.d2(x, m)));
      let same = prevLab !== null;
      if (prevLab) for (let n = 0; n < X.length; n++) if (lab[n] !== prevLab[n]) { same = false; break; }
      stepN++;
      let s = ">>> assign, centroids, J = kmeans_step(X, centroids)   # step " + stepN + "\n";
      s += "squared distances (to m1, m2):\n";
      X.forEach(function (x, n) {
        s += "  x" + (n + 1) + " = (" + x.join(", ") + ")  ->  (" + LR.fmt(d2s[n][0], 2) + ", " + LR.fmt(d2s[n][1], 2) + ")  ->  cluster " + (lab[n] + 1) + "\n";
      });
      s += "assign = [" + lab.map((l) => l + 1).join(", ") + "]\n";
      s += "J (inertia after assignment) = " + LR.fmt(J, 4) + "\n";
      if (same) {
        s += "# assignments identical to the previous step -> CONVERGED\n";
        s += "centroids stay at [(" + M[0].map((v) => LR.fmt(v, 3)).join(", ") + "), (" + M[1].map((v) => LR.fmt(v, 3)).join(", ") + ")]";
      } else {
        M = CL.update(X, lab, M);
        s += "new centroids = [(" + M[0].map((v) => LR.fmt(v, 3)).join(", ") + "), (" + M[1].map((v) => LR.fmt(v, 3)).join(", ") + ")]\n";
        s += "J recomputed at the new centroids = " + LR.fmt(CL.inertia(X, M, lab), 4);
      }
      prevLab = lab;
      out.textContent = s;
    }

    resetRun();
  };
})();
