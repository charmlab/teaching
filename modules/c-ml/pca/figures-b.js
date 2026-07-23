/* ══════════════════════════════════════════════════════════════
   figures-b.js — PCA lesson, payoff half
   Fig 0.1 hook · Fig 3.3 pcacode · Fig 4.1 duality (signature)
   Fig 5.1 scree (signature) · Fig 6.1 digits · Fig 7.1 randproj
   Also defines the synthetic digit engine: a parametric family of
   ring-shaped digit images, a real 196-D PCA computed in the
   browser via power iteration on the Gram matrix (with deflation),
   and project/reconstruct in pixel space. All numbers computed
   live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const P = LR.pca;

  /* ══════════════════════════════════════════════════════════
     The digit engine (shared by Fig 0.1 and Fig 6.1)
     14 x 14 = 196 pixels. 90 synthetic digit variants are
     generated from a few latent knobs (size, slant, thickness,
     position); PCA runs on the real 196-D pixel vectors.
     Eigenpairs come from power iteration with deflation on the
     90 x 90 Gram matrix (the dual trick), then are mapped back
     to pixel space: if (Xc Xcᵀ/(N-1)) is too big, use Xcᵀ Xc.
     ══════════════════════════════════════════════════════════ */
  const G = 14, D = G * G, NDIG = 90, MCOMP = 40;

  function makeDigit(rx, ry, rot, th, dx, dy) {
    const img = new Array(D);
    const cx = G / 2 + dx, cy = G / 2 + dy;
    const co = Math.cos(rot), si = Math.sin(rot);
    const avg = (rx + ry) / 2;
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const x = j + 0.5 - cx, y = i + 0.5 - cy;
        const xr = co * x + si * y, yr = -si * x + co * y;
        const r = Math.sqrt((xr / rx) * (xr / rx) + (yr / ry) * (yr / ry));
        const d = Math.abs(r - 1) * avg;
        img[i * G + j] = Math.exp(-(d * d) / (th * th));
      }
    }
    return img;
  }

  function digitModel() {
    if (digitModel.__m) return digitModel.__m;

    const rand = LR.rng(18657);
    const imgs = [];
    for (let n = 0; n < NDIG; n++) {
      imgs.push(makeDigit(
        3.2 + 2.0 * rand(),        // rx
        4.0 + 2.0 * rand(),        // ry
        -0.35 + 0.7 * rand(),      // rot
        0.9 + 0.8 * rand(),        // thickness
        -1 + 2 * rand(),           // dx
        -1 + 2 * rand()            // dy
      ));
    }

    // mean and centered data
    const mean = new Array(D).fill(0);
    imgs.forEach((im) => { for (let p = 0; p < D; p++) mean[p] += im[p] / NDIG; });
    const Xc = imgs.map((im) => im.map((v, p) => v - mean[p]));

    // total variance = trace(Σ) = Σ ||xc||² / (N-1)
    let totVar = 0;
    Xc.forEach((r) => { for (let p = 0; p < D; p++) totVar += r[p] * r[p]; });
    totVar /= NDIG - 1;

    // Gram matrix (N x N)
    const Gm = [];
    for (let i = 0; i < NDIG; i++) {
      Gm.push(new Array(NDIG));
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let p = 0; p < D; p++) s += Xc[i][p] * Xc[j][p];
        Gm[i][j] = s;
        if (j < i) Gm[j][i] = s;
      }
    }

    // power iteration with deflation: top MCOMP eigenvectors of Gm
    const vecs = [], mus = [];
    for (let k = 0; k < MCOMP; k++) {
      let v = Array.from({ length: NDIG }, () => rand() - 0.5);
      for (let it = 0; it < 90; it++) {
        // orthogonalize against previous eigenvectors (deflation)
        vecs.forEach(function (w) {
          let d = 0;
          for (let i = 0; i < NDIG; i++) d += v[i] * w[i];
          for (let i = 0; i < NDIG; i++) v[i] -= d * w[i];
        });
        // v ← Gm v
        const nv = new Array(NDIG).fill(0);
        for (let i = 0; i < NDIG; i++) {
          const row = Gm[i];
          let s = 0;
          for (let j = 0; j < NDIG; j++) s += row[j] * v[j];
          nv[i] = s;
        }
        let n = 0;
        for (let i = 0; i < NDIG; i++) n += nv[i] * nv[i];
        n = Math.sqrt(n) || 1;
        for (let i = 0; i < NDIG; i++) v[i] = nv[i] / n;
      }
      // Rayleigh quotient μ = vᵀ G v
      let mu = 0;
      for (let i = 0; i < NDIG; i++) {
        let s = 0;
        for (let j = 0; j < NDIG; j++) s += Gm[i][j] * v[j];
        mu += v[i] * s;
      }
      vecs.push(v); mus.push(mu);
    }

    // map back to pixel space: u = Xcᵀ v / ‖·‖, λ = μ / (N-1)
    const comps = [], lams = [];
    for (let k = 0; k < MCOMP; k++) {
      const u = new Array(D).fill(0);
      for (let i = 0; i < NDIG; i++) {
        const vi = vecs[k][i];
        if (vi === 0) continue;
        const row = Xc[i];
        for (let p = 0; p < D; p++) u[p] += vi * row[p];
      }
      let n = 0;
      for (let p = 0; p < D; p++) n += u[p] * u[p];
      n = Math.sqrt(n) || 1;
      for (let p = 0; p < D; p++) u[p] /= n;
      comps.push(u);
      lams.push(mus[k] / (NDIG - 1));
    }

    function reconstruct(img, K) {
      const rec = mean.slice();
      for (let k = 0; k < K; k++) {
        const u = comps[k];
        let z = 0;
        for (let p = 0; p < D; p++) z += (img[p] - mean[p]) * u[p];
        for (let p = 0; p < D; p++) rec[p] += z * u[p];
      }
      return rec;
    }
    function varKept(K) {
      let s = 0;
      for (let k = 0; k < K; k++) s += lams[k];
      return s / totVar;
    }
    function mse(a, b) {
      let s = 0;
      for (let p = 0; p < D; p++) s += (a[p] - b[p]) * (a[p] - b[p]);
      return s / D;
    }

    digitModel.__m = { imgs, mean, comps, lams, totVar, reconstruct, varKept, mse };
    return digitModel.__m;
  }

  function drawImg(ctx, img, x0, y0, cell, label) {
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const v = Math.max(0, Math.min(1, img[i * G + j]));
        const g = Math.round(255 * (1 - v));
        ctx.fillStyle = "rgb(" + g + "," + g + "," + g + ")";
        ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
      }
    }
    ctx.strokeStyle = "#d8d8d8"; ctx.lineWidth = 1;
    ctx.strokeRect(x0 - 0.5, y0 - 0.5, G * cell + 1, G * cell + 1);
    if (label) {
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "center";
      ctx.fillText(label, x0 + (G * cell) / 2, y0 + G * cell + 18);
    }
  }

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — the hook: a digit from K directions
     ══════════════════════════════════════════════════════════ */
  LR.figs.hook = function (mount) {
    LR.header(
      mount,
      "196 pixels from K directions",
      "Left: the original image, one number per pixel. Right: its reconstruction from only the top K principal components."
    );

    const m = digitModel();
    let K = 4, idx = 3;

    const bar = LR.controls(mount);
    const sl = LR.slider(bar, "components kept K", 1, MCOMP, 1, K, function (v) { K = Math.round(v); render(); }, (v) => String(Math.round(v)));
    LR.button(bar, "Another digit ⟲", function () { idx = (idx + 1) % NDIG; render(); });

    const cell = 13, gap = 70;
    const W = 2 * G * cell + gap + 60, H = G * cell + 40;
    const { ctx } = LR.canvas(mount, W, H, {
      aria: "A synthetic digit image next to its reconstruction from the top K principal components",
    });

    const ro = LR.readout(mount, [
      { k: "dims", label: "dimensions kept" },
      { k: "var", label: "variance kept" },
    ]);

    function render() {
      sl.set(K);
      ctx.clearRect(0, 0, W, H);
      const img = m.imgs[idx];
      const rec = m.reconstruct(img, K);
      const x0 = 30;
      drawImg(ctx, img, x0, 6, cell, "original (196 numbers)");
      ctx.font = "800 18px Inter, sans-serif";
      ctx.fillStyle = C.faint; ctx.textAlign = "center";
      ctx.fillText("→", x0 + G * cell + gap / 2, 6 + (G * cell) / 2);
      drawImg(ctx, rec, x0 + G * cell + gap, 6, cell, "rebuilt from K = " + K);
      ro.set("dims", K + " / " + D + "  (" + LR.fmtF((100 * K) / D, 1) + "%)");
      ro.set("var", LR.fmtF(100 * m.varKept(K), 1) + "%", C.orange);
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.3 — live pca_fit on the worked dataset
     ══════════════════════════════════════════════════════════ */
  LR.figs.pcacode = function (mount) {
    LR.header(
      mount,
      "pca_fit, executed for real",
      "The worked example's five points go through the exact pipeline from the code block above."
    );

    // the worked dataset, one point per row (Python convention)
    const X = [[2, 2], [3, 2], [4, 3], [5, 4], [6, 4]];
    let K = 1;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const k1 = LR.button(group, "K = 1", function () { K = 1; k1.classList.add("on"); k2.classList.remove("on"); }, "on");
    const k2 = LR.button(group, "K = 2", function () { K = 2; k2.classList.add("on"); k1.classList.remove("on"); });
    LR.button(bar, "Run ▸", run, "primary");

    const out = LR.el("pre", "attn-runout", "press Run to execute pca_fit(X, K) on the worked dataset…");
    out.setAttribute("aria-live", "polite");
    mount.appendChild(out);

    function f(n, d) { return LR.fmtF(n, d === undefined ? 4 : d); }

    function run() {
      const { m, Xc } = P.center2(X);
      const S = P.cov2c(Xc);
      const e = P.eig2(S);
      // sign convention: first entry of u1 non-negative (matches the worked example)
      const u1 = e.u1[0] < 0 ? [-e.u1[0], -e.u1[1]] : e.u1;
      const u2 = [-u1[1], u1[0]];
      const dirs = [u1, u2].slice(0, K);
      const Z = Xc.map((p) => dirs.map((u) => p[0] * u[0] + p[1] * u[1]));
      const rec = P.projectRec(Xc, { u1, u2, l1: e.l1, l2: e.l2 }, K);
      const err = P.recErr(Xc, rec);
      const discard = K === 1 ? e.l2 : 0;
      const evr = (K === 1 ? e.l1 : e.l1 + e.l2) / (e.l1 + e.l2);

      const lines = [];
      lines.push(">>> mean, UK, lam = pca_fit(X, K=" + K + ")");
      lines.push("mean  = [" + f(m[0], 1) + " " + f(m[1], 1) + "]");
      lines.push("Sigma =");
      lines.push("  [[" + f(S[0][0], 2) + " " + f(S[0][1], 2) + "]");
      lines.push("   [" + f(S[1][0], 2) + " " + f(S[1][1], 2) + "]]");
      lines.push("lam   = [" + f(e.l1) + " " + f(e.l2) + "]   # sorted, largest first");
      lines.push("UK    =");
      if (K === 1) {
        lines.push("  [[" + f(u1[0]) + "]");
        lines.push("   [" + f(u1[1]) + "]]");
      } else {
        lines.push("  [[" + f(u1[0]) + " " + f(u2[0]) + "]");
        lines.push("   [" + f(u1[1]) + " " + f(u2[1]) + "]]");
      }
      lines.push("");
      lines.push(">>> Z = project(X, mean, UK)          # " + K + " coordinate" + (K > 1 ? "s" : "") + " per point");
      lines.push("Z     = [" + Z.map((z) => z.map((v) => f(v)).join(" ")).join("  |  ") + "]");
      lines.push("");
      lines.push(">>> Xh = reconstruct(Z, mean, UK)");
      lines.push("reconstruction error = " + f(err) + "   # Frobenius² / (N-1)");
      lines.push("sum of discarded lam = " + f(discard) + "   # they match" + (K === 2 ? " (nothing discarded)" : ""));
      lines.push("");
      lines.push(">>> explained_variance_ratio(lam, K=" + K + ")");
      lines.push(f(evr) + (K === 1 ? "   # 97.9% of the variance in 1 of 2 dimensions" : "   # all of it: K = D"));
      out.textContent = lines.join("\n");
    }
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — the variance/reconstruction duality (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.duality = function (mount) {
    LR.header(
      mount,
      "One dial, two labels",
      "Drag the direction. Variance captured and reconstruction error trade off exactly; their sum never moves."
    );

    const Xc = P.center2(P.cloud0()).Xc;
    const S = P.cov2c(Xc);
    const e = P.eig2(S);
    const tot = e.l1 + e.l2;
    let theta = Math.atan2(e.u1[1], e.u1[0]) + 0.85;

    const bar = LR.controls(mount);
    const sl = LR.slider(bar, "direction angle θ", -90, 90, 1, Math.round((theta * 180) / Math.PI), function (v) {
      theta = (v * Math.PI) / 180; render();
    }, (v) => LR.fmt(v, 0) + "°");
    LR.button(bar, "Snap to u1 (best)", function () { theta = Math.atan2(e.u1[1], e.u1[0]); render(); });
    LR.button(bar, "Snap to u2 (worst)", function () { theta = Math.atan2(e.u2[1], e.u2[0]); render(); });
    let rseed = 5;
    LR.button(bar, "Random direction", function () {
      const r = LR.rng(rseed++);
      theta = -Math.PI / 2 + Math.PI * r();
      render();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 700, 430, {
      aria: "Point cloud with a draggable projection direction; drag the handle or use the angle slider",
    });

    // duality bars (HTML, reusing the weight-bar styles)
    const bars = LR.el("div", "attn-weightbar");
    const rows = [
      { lab: "variance kept", color: C.green },
      { lab: "recon. error", color: C.red },
    ].map(function (r) {
      const item = LR.el("div", "wb-item");
      item.appendChild(LR.el("span", "wb-lab", r.lab));
      const track = LR.el("div", "wb-track");
      const fill = LR.el("div", "wb-fill");
      fill.style.background = r.color;
      track.appendChild(fill);
      item.appendChild(track);
      const val = LR.el("span", "wb-val", "–");
      item.appendChild(val);
      bars.appendChild(item);
      return { fill, val };
    });
    mount.appendChild(bars);

    const ro = LR.readout(mount, [
      { k: "sum", label: "kept + error" },
      { k: "tot", label: "total variance" },
      { k: "best", label: "λ1 (max possible)" },
      { k: "worst", label: "λ2 (min error)" },
    ]);
    const msg = LR.msg(mount);

    const Pl = { x0: 52, y0: 16, w: W - 76, h: H - 72, xmin: -5, xmax: 5, ymin: -3.6, ymax: 3.6, xlabel: "x1 (centered)", ylabel: "x2 (centered)" };
    const HANDLE = 3.4;

    function render() {
      sl.set(Math.round((theta * 180) / Math.PI));
      ctx.clearRect(0, 0, W, H);
      const w = [Math.cos(theta), Math.sin(theta)];
      // variance along w from the sample covariance: wᵀ S w
      const kept = w[0] * (S[0][0] * w[0] + S[0][1] * w[1]) + w[1] * (S[1][0] * w[0] + S[1][1] * w[1]);
      const err = tot - kept;

      const { sx, sy } = LR.plot(ctx, Pl);
      ctx.save();
      ctx.beginPath(); ctx.rect(Pl.x0, Pl.y0, Pl.w, Pl.h); ctx.clip();

      // candidate line
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(sx(-6 * w[0]), sy(-6 * w[1]));
      ctx.lineTo(sx(6 * w[0]), sy(6 * w[1]));
      ctx.stroke();

      // residual segments (thin)
      ctx.strokeStyle = "rgba(61,61,61,0.35)"; ctx.lineWidth = 1;
      Xc.forEach(function (p) {
        const z = p[0] * w[0] + p[1] * w[1];
        ctx.beginPath();
        ctx.moveTo(sx(p[0]), sy(p[1]));
        ctx.lineTo(sx(z * w[0]), sy(z * w[1]));
        ctx.stroke();
      });

      // u1 reference (dashed)
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "rgba(102,102,102,0.6)"; ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(sx(-6 * e.u1[0]), sy(-6 * e.u1[1]));
      ctx.lineTo(sx(6 * e.u1[0]), sy(6 * e.u1[1]));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "left";
      ctx.fillText("u1", sx(4.4 * e.u1[0]) + 4, sy(4.4 * e.u1[1]));

      Xc.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 2.8, "rgba(17,17,17,0.55)"));

      // drag handle
      LR.dot(ctx, sx(HANDLE * w[0]), sy(HANDLE * w[1]), 8, C.orange, "#ffffff");
      ctx.restore();

      rows[0].fill.style.width = (100 * kept / tot) + "%";
      rows[0].val.textContent = LR.fmtF(kept, 3);
      rows[1].fill.style.width = (100 * err / tot) + "%";
      rows[1].val.textContent = LR.fmtF(err, 3);

      ro.set("sum", LR.fmtF(kept + err, 3), C.orange);
      ro.set("tot", LR.fmtF(tot, 3));
      ro.set("best", LR.fmtF(e.l1, 3), C.green);
      ro.set("worst", LR.fmtF(e.l2, 3), C.red);

      const d1 = Math.abs(w[0] * e.u1[0] + w[1] * e.u1[1]);
      if (d1 > 0.9995) {
        msg.show("This is PCA's pick: variance captured hits its maximum λ1 = " + LR.fmtF(e.l1, 3) + " exactly where reconstruction error hits its minimum λ2 = " + LR.fmtF(e.l2, 3) + ". Two objectives, one direction.", "good");
      } else if (d1 < 0.033) {
        msg.show("The worst direction: this is u2, where the kept variance bottoms out at λ2 and the error peaks at λ1.", "bad");
      } else {
        msg.hide();
      }
    }

    LR.drag(cv, W, H, {
      hit: function (p) {
        const sx = (x) => Pl.x0 + ((x - Pl.xmin) / (Pl.xmax - Pl.xmin)) * Pl.w;
        const sy = (y) => Pl.y0 + Pl.h - ((y - Pl.ymin) / (Pl.ymax - Pl.ymin)) * Pl.h;
        const w = [Math.cos(theta), Math.sin(theta)];
        return Math.hypot(p.x - sx(HANDLE * w[0]), p.y - sy(HANDLE * w[1])) < 26 ||
               Math.hypot(p.x - sx(-HANDLE * w[0]), p.y - sy(-HANDLE * w[1])) < 26;
      },
      move: function (p) {
        const x = Pl.xmin + ((p.x - Pl.x0) / Pl.w) * (Pl.xmax - Pl.xmin);
        const y = Pl.ymin + ((Pl.y0 + Pl.h - p.y) / Pl.h) * (Pl.ymax - Pl.ymin);
        if (Math.hypot(x, y) < 0.2) return;
        theta = Math.atan2(y, x);
        if (theta > Math.PI / 2) theta -= Math.PI;
        if (theta < -Math.PI / 2) theta += Math.PI;
        render();
      },
    });

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — scree plot / explained variance (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.scree = function (mount) {
    LR.header(
      mount,
      "Reading the spectrum",
      "Bars: sorted eigenvalues (left axis). Curve: cumulative explained variance (right axis). Drag the divider to choose K."
    );

    const SPECTRA = {
      fast: { name: "Fast decay", lam: [6.0, 2.2, 1.0, 0.35, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02] },
      flat: { name: "Flat decay", lam: [1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.10] },
    };
    let mode = "fast";
    let K = 3;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bFast = LR.button(group, "Fast decay", function () { mode = "fast"; bFast.classList.add("on"); bFlat.classList.remove("on"); render(); }, "on");
    const bFlat = LR.button(group, "Flat decay", function () { mode = "flat"; bFlat.classList.add("on"); bFast.classList.remove("on"); render(); });
    const sl = LR.slider(bar, "components kept K", 1, 10, 1, K, function (v) { K = Math.round(v); render(); }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 700, 400, {
      aria: "Scree plot with a draggable K divider; bars show sorted eigenvalues and a curve shows cumulative explained variance",
    });
    cv.tabIndex = 0;
    cv.setAttribute("aria-label", "Scree plot. Drag the divider, or use left and right arrow keys to change K");

    const ro = LR.readout(mount, [
      { k: "kept", label: "explained variance at K" },
      { k: "err", label: "recon. error (Σ discarded λ)" },
      { k: "rec", label: "smallest K reaching 90%" },
    ]);

    const Pl = { x0: 56, y0: 22, w: W - 116, h: H - 78 };

    function render() {
      sl.set(K);
      ctx.clearRect(0, 0, W, H);
      const lam = SPECTRA[mode].lam;
      const Dd = lam.length;
      const tot = lam.reduce((a, b) => a + b, 0);
      const lmax = lam[0] * 1.12;

      const bx = (i) => Pl.x0 + (i / Dd) * Pl.w;         // left edge of bar i (0-based)
      const by = (v) => Pl.y0 + Pl.h - (v / lmax) * Pl.h;
      const cy = (f) => Pl.y0 + Pl.h - f * Pl.h;         // cumulative axis 0..1

      // frame
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1.4;
      ctx.strokeRect(Pl.x0, Pl.y0, Pl.w, Pl.h);

      // left ticks (eigenvalue)
      ctx.font = "11px Inter, sans-serif";
      ctx.fillStyle = C.faint; ctx.textAlign = "right";
      LR.ticks(0, lmax, 4).forEach(function (t) {
        ctx.fillText(LR.fmt(t, 1), Pl.x0 - 7, by(t) + 3.5);
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(Pl.x0, by(t)); ctx.lineTo(Pl.x0 + Pl.w, by(t)); ctx.stroke();
      });
      // right ticks (cumulative %)
      ctx.textAlign = "left";
      [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
        ctx.fillStyle = C.purple;
        ctx.fillText(Math.round(100 * f) + "%", Pl.x0 + Pl.w + 8, cy(f) + 3.5);
      });

      // bars
      const bw = (Pl.w / Dd) * 0.62;
      lam.forEach(function (l, i) {
        ctx.fillStyle = i < K ? C.orange : "#d9d9d9";
        ctx.fillRect(bx(i) + (Pl.w / Dd - bw) / 2, by(l), bw, Pl.y0 + Pl.h - by(l));
      });

      // 90% line on the cumulative axis
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = C.purple; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(Pl.x0, cy(0.9)); ctx.lineTo(Pl.x0 + Pl.w, cy(0.9)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "700 11px Inter, sans-serif";
      ctx.fillStyle = C.purple; ctx.textAlign = "left";
      ctx.fillText("90% target", Pl.x0 + 6, cy(0.9) - 5);

      // cumulative curve
      ctx.strokeStyle = C.purple; ctx.lineWidth = 2.2;
      ctx.beginPath();
      let c = 0, K90 = Dd;
      lam.forEach(function (l, i) {
        c += l;
        const f = c / tot;
        if (f >= 0.9 - 1e-9 && K90 === Dd) K90 = i + 1;
        const px = bx(i) + Pl.w / Dd / 2;
        if (i === 0) ctx.moveTo(px, cy(f));
        else ctx.lineTo(px, cy(f));
      });
      ctx.stroke();
      c = 0;
      lam.forEach(function (l, i) {
        c += l;
        LR.dot(ctx, bx(i) + Pl.w / Dd / 2, cy(c / tot), 3.4, C.purple, "#ffffff");
      });

      // 90% recommendation marker
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillStyle = C.purple; ctx.textAlign = "center";
      ctx.fillText("K₉₀ = " + K90, bx(K90 - 1) + Pl.w / Dd / 2, Pl.y0 + 14);

      // draggable K divider
      const dx = bx(K);
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(dx, Pl.y0); ctx.lineTo(dx, Pl.y0 + Pl.h); ctx.stroke();
      ctx.fillStyle = C.orange;
      ctx.beginPath();
      ctx.moveTo(dx - 7, Pl.y0 + Pl.h + 2); ctx.lineTo(dx + 7, Pl.y0 + Pl.h + 2); ctx.lineTo(dx, Pl.y0 + Pl.h + 11);
      ctx.closePath(); ctx.fill();
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("keep " + K + " | drop " + (Dd - K), dx, Pl.y0 + Pl.h + 26);

      // axis labels
      ctx.fillStyle = C.text;
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("component index (sorted by eigenvalue)", Pl.x0 + Pl.w / 2, H - 8);
      ctx.save();
      ctx.translate(14, Pl.y0 + Pl.h / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText("eigenvalue λᵢ", 0, 0);
      ctx.restore();
      ctx.save();
      ctx.translate(W - 10, Pl.y0 + Pl.h / 2); ctx.rotate(Math.PI / 2);
      ctx.fillStyle = C.purple;
      ctx.fillText("cumulative explained variance", 0, 0);
      ctx.restore();

      const keptSum = lam.slice(0, K).reduce((a, b) => a + b, 0);
      ro.set("kept", LR.fmtF((100 * keptSum) / tot, 1) + "%", C.orange);
      ro.set("err", LR.fmtF(tot - keptSum, 2) + " of " + LR.fmtF(tot, 2), C.red);
      ro.set("rec", String(K90), C.purple);
    }

    LR.drag(cv, W, H, {
      move: function (p) {
        const i = Math.round(((p.x - Pl.x0) / Pl.w) * 10);
        const nk = Math.max(1, Math.min(10, i));
        if (nk !== K) { K = nk; render(); }
      },
    });
    cv.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { K = Math.max(1, K - 1); render(); e.preventDefault(); }
      if (e.key === "ArrowRight") { K = Math.min(10, K + 1); render(); e.preventDefault(); }
    });

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.1 — compression and denoising on the digit set
     ══════════════════════════════════════════════════════════ */
  LR.figs.digits = function (mount) {
    LR.header(
      mount,
      "Compression and denoising, one eigendecomposition",
      "The pipeline only ever sees the input image. With noise on, watch the reconstruction land closer to the clean original than its own input."
    );

    const m = digitModel();
    let K = 15, idx = 7, noisy = false, sigma = 0.25;

    const bar = LR.controls(mount);
    const slK = LR.slider(bar, "components kept K", 1, MCOMP, 1, K, function (v) { K = Math.round(v); render(); }, (v) => String(Math.round(v)));
    const noiseBtn = LR.button(bar, "Noise: off", function () {
      noisy = !noisy;
      noiseBtn.textContent = noisy ? "Noise: on" : "Noise: off";
      noiseBtn.classList.toggle("on", noisy);
      render();
    });
    noiseBtn.setAttribute("aria-pressed", "false");
    LR.slider(bar, "noise level σ", 0.05, 0.5, 0.05, sigma, function (v) { sigma = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Another digit ⟲", function () { idx = (idx + 1) % NDIG; render(); });

    const cell = 11, gap = 46;
    const W = 3 * G * cell + 2 * gap + 40, H = G * cell + 44;
    const { ctx } = LR.canvas(mount, W, H, {
      aria: "Clean digit, pipeline input which may be noisy, and PCA reconstruction from K components",
    });

    const ro = LR.readout(mount, [
      { k: "var", label: "variance kept" },
      { k: "ein", label: "input vs clean (MSE)" },
      { k: "erec", label: "reconstruction vs clean (MSE)" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      slK.set(K);
      if (noiseBtn) noiseBtn.setAttribute("aria-pressed", String(noisy));
      ctx.clearRect(0, 0, W, H);
      const clean = m.imgs[idx];
      let input = clean;
      if (noisy) {
        const rand = LR.rng(900 + idx);
        input = clean.map((v) => v + sigma * LR.gauss(rand));
      }
      const rec = m.reconstruct(input, K);

      const x0 = 20;
      drawImg(ctx, clean, x0, 6, cell, "clean original");
      drawImg(ctx, input, x0 + G * cell + gap, 6, cell, noisy ? "input (noise σ = " + LR.fmtF(sigma, 2) + ")" : "input (no noise)");
      drawImg(ctx, rec, x0 + 2 * (G * cell + gap), 6, cell, "reconstruction, K = " + K);

      const ein = m.mse(input, clean);
      const erec = m.mse(rec, clean);
      ro.set("var", LR.fmtF(100 * m.varKept(K), 1) + "%", C.orange);
      ro.set("ein", LR.fmtF(ein, 4));
      ro.set("erec", LR.fmtF(erec, 4), erec < ein ? C.green : C.red);

      if (noisy && erec < ein) {
        msg.show("Denoised: the reconstruction (" + LR.fmtF(erec, 4) + ") is closer to the clean original than the noisy input it came from (" + LR.fmtF(ein, 4) + "). The discarded " + (196 - K) + " low-eigenvalue directions carried mostly noise.", "good");
      } else if (noisy) {
        msg.show("At this K the kept subspace admits too much noise (or too little signal). Try K near 10 to 20: enough for the signal, little room for the noise.", "info");
      } else {
        msg.show("Compression view: " + K + " numbers per image instead of 196, keeping " + LR.fmtF(100 * m.varKept(K), 1) + "% of the variance. Toggle the noise to switch hats.", "info");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.1 — random vs principal projection
     ══════════════════════════════════════════════════════════ */
  LR.figs.randproj = function (mount) {
    LR.header(
      mount,
      "Fitted versus drawn",
      "Both maps cost one dot product per point. Only one of them looked at the data first."
    );

    const Xc = P.center2(P.cloud0()).Xc;
    const S = P.cov2c(Xc);
    const e = P.eig2(S);
    const tot = e.l1 + e.l2;
    let rseed = 42;
    let r = drawRandom();

    function drawRandom() {
      const rand = LR.rng(rseed++);
      const th = 2 * Math.PI * rand();
      return [Math.cos(th), Math.sin(th)];
    }
    function varAlong(w) {
      return w[0] * (S[0][0] * w[0] + S[0][1] * w[1]) + w[1] * (S[1][0] * w[0] + S[1][1] * w[1]);
    }

    const bar = LR.controls(mount);
    LR.button(bar, "New random direction 🎲", function () { r = drawRandom(); render(); }, "primary");

    const { ctx, W, H } = LR.canvas(mount, 700, 520, {
      aria: "Point cloud with its principal direction and a random direction, and two one-dimensional strips showing both projections",
    });

    const ro = LR.readout(mount, [
      { k: "vp", label: "variance kept · principal" },
      { k: "vr", label: "variance kept · random" },
      { k: "frac", label: "random / best" },
    ]);

    function strip(y0, pts1d, color, label, vkept) {
      const x0 = 70, w = W - 120;
      const lo = -5, hi = 5;
      const sx = (v) => x0 + ((v - lo) / (hi - lo)) * w;
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + w, y0); ctx.stroke();
      [-4, -2, 0, 2, 4].forEach(function (t) {
        ctx.strokeStyle = "#cfcfcf";
        ctx.beginPath(); ctx.moveTo(sx(t), y0 - 3); ctx.lineTo(sx(t), y0 + 3); ctx.stroke();
        ctx.font = "10.5px Inter, sans-serif";
        ctx.fillStyle = C.faint; ctx.textAlign = "center";
        ctx.fillText(LR.fmt(t, 0), sx(t), y0 + 16);
      });
      pts1d.forEach(function (v) {
        ctx.globalAlpha = 0.55;
        LR.dot(ctx, sx(v), y0 - 7, 3, color);
        ctx.globalAlpha = 1;
      });
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = color; ctx.textAlign = "left";
      ctx.fillText(label + "  ·  variance " + LR.fmtF(vkept, 3), x0, y0 - 20);
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const vr = varAlong(r);

      const Pl = { x0: 52, y0: 14, w: W - 76, h: 300, xmin: -5, xmax: 5, ymin: -3.6, ymax: 3.6, xlabel: "", ylabel: "x2" };
      const { sx, sy } = LR.plot(ctx, Pl);
      ctx.save();
      ctx.beginPath(); ctx.rect(Pl.x0, Pl.y0, Pl.w, Pl.h); ctx.clip();
      Xc.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 2.8, "rgba(17,17,17,0.5)"));
      // principal direction
      ctx.strokeStyle = C.green; ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(sx(-6 * e.u1[0]), sy(-6 * e.u1[1]));
      ctx.lineTo(sx(6 * e.u1[0]), sy(6 * e.u1[1]));
      ctx.stroke();
      // random direction
      ctx.strokeStyle = C.purple; ctx.lineWidth = 2.6;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(sx(-6 * r[0]), sy(-6 * r[1]));
      ctx.lineTo(sx(6 * r[0]), sy(6 * r[1]));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "left";
      ctx.fillText("u1 (fitted)", Pl.x0 + 10, Pl.y0 + 18);
      ctx.fillStyle = C.purple;
      ctx.fillText("R (drawn)", Pl.x0 + 10, Pl.y0 + 36);
      ctx.restore();

      strip(392, Xc.map((p) => p[0] * e.u1[0] + p[1] * e.u1[1]), C.green, "z = u1ᵀx (principal)", e.l1);
      strip(478, Xc.map((p) => p[0] * r[0] + p[1] * r[1]), C.purple, "y = Rᵀx (random)", vr);

      ro.set("vp", LR.fmtF(e.l1, 3) + "  (" + LR.fmtF((100 * e.l1) / tot, 1) + "% of total)", C.green);
      ro.set("vr", LR.fmtF(vr, 3) + "  (" + LR.fmtF((100 * vr) / tot, 1) + "% of total)", C.purple);
      ro.set("frac", LR.fmtF(vr / e.l1, 3));
    }
    render();
  };
})();
