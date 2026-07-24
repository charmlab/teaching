/* ══════════════════════════════════════════════════════════════
   figures-a.js — Autoencoders & VAEs lesson, sections 0–3
   Fig 0.1 arch (signature) · Fig 1.1 linearpca · Fig 2.1 manifold
   (signature) · Fig 3.1 sparse · Fig 3.2 denoise
   Also defines the shared autoencoder core (LR.ae) used by
   figures-b.js: matrix helpers, PCA on real covariances, the toy
   glyph decoder for the latent-space figures, Gaussian sampling,
   and the closed-form KL term. All numbers computed live.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared autoencoder core (also used by figures-b.js) ── */
  const AE = (LR.ae = {
    matmul: function (A, B) {
      return A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
    },
    transpose: function (A) {
      return A[0].map((_, j) => A.map((r) => r[j]));
    },
    sub2: function (a, b) { return [a[0] - b[0], a[1] - b[1]]; },
    dot2: function (a, b) { return a[0] * b[0] + a[1] * b[1]; },
    nrm2: function (a) { return Math.hypot(a[0], a[1]); },

    // closed-form KL( N(mu, diag(sigma^2)) || N(0, I) ), summed over dims.
    // Flagged addition: the source states the KL regularizer to N(0, I);
    // this standard closed form is how implementations compute it.
    klGauss: function (mu, sigma) {
      let s = 0;
      for (let i = 0; i < mu.length; i++) {
        s += 0.5 * (mu[i] * mu[i] + sigma[i] * sigma[i] - 1 - Math.log(sigma[i] * sigma[i]));
      }
      return s;
    },

    /* ── colour ramps (single hue, matching the series) ────── */
    ramp: function (v) {
      // v in [0,1] → white → orange
      const t = Math.max(0, Math.min(1, v));
      const r = Math.round(255 + (26 - 255) * t);
      const g = Math.round(255 + (26 - 255) * t);
      const b = Math.round(255 + (26 - 255) * t);
      return "rgb(" + r + "," + g + "," + b + ")";
    },
    rampSigned: function (v, vmax) {
      // v in [-vmax, vmax] → purple ← white → orange
      const t = Math.max(-1, Math.min(1, v / (vmax || 1)));
      if (t >= 0) return AE.ramp(t);
      const u = -t;
      const r = Math.round(255 + (150 - 255) * u);
      const g = Math.round(255 + (150 - 255) * u);
      const b = Math.round(255 + (150 - 255) * u);
      return "rgb(" + r + "," + g + "," + b + ")";
    },
    txtOn: function (bg) {
      const m = bg.match(/(\d+),(\d+),(\d+)/);
      if (!m) return "#111111";
      const lum = 0.299 * +m[1] + 0.587 * +m[2] + 0.114 * +m[3];
      return lum > 150 ? "#111111" : "#ffffff";
    },

    /* ── matrix table renderer (same conventions as the series) ── */
    matrix: function (M, opts) {
      opts = opts || {};
      const box = LR.el("div", "mtx-box");
      if (opts.label) box.appendChild(LR.el("div", "mtx-cap", opts.label));
      const tbl = LR.el("table", "mtx" + (opts.big ? " big" : ""));
      let vmax = opts.vmax;
      if (!vmax && opts.heat) {
        vmax = 0;
        M.forEach((r) => r.forEach((v) => { if (isFinite(v)) vmax = Math.max(vmax, Math.abs(v)); }));
        vmax = vmax || 1;
      }
      if (opts.cols) {
        const tr = LR.el("tr");
        tr.appendChild(LR.el("th", null, ""));
        opts.cols.forEach((c) => tr.appendChild(LR.el("th", null, c)));
        tbl.appendChild(tr);
      }
      M.forEach(function (row, i) {
        const tr = LR.el("tr");
        if (opts.rows) tr.appendChild(LR.el("th", null, opts.rows[i]));
        else if (opts.cols) tr.appendChild(LR.el("th", null, ""));
        row.forEach(function (v, j) {
          const td = LR.el("td");
          td.textContent = LR.fmtF(v, opts.d === undefined ? 2 : opts.d);
          if (opts.heat === "pos") {
            const bg = AE.ramp(v / vmax);
            td.style.background = bg;
            td.style.color = AE.txtOn(bg);
          } else if (opts.heat === "signed") {
            const bg = AE.rampSigned(v, vmax);
            td.style.background = bg;
            td.style.color = AE.txtOn(bg);
          }
          if (opts.onCell) opts.onCell(i, j, td);
          tr.appendChild(td);
        });
        tbl.appendChild(tr);
      });
      box.appendChild(tbl);
      if (opts.sub) box.appendChild(LR.el("div", "mtx-sub", opts.sub));
      return box;
    },
    op: function (sym) { return LR.el("div", "mtx-op", sym); },

    /* ── the toy glyph world for the latent-space figures ─────
       Three families of 16×16 synthetic glyphs (disc, cross, wedge),
       each with one shape parameter s in [0, 1]. The "decoder" blends
       parametric renders by latent proximity to each family's segment.
       Flagged addition: this constructed decoder stands in for a trained
       one so the latent geometry is exact and every decode is computed;
       the source specifies the failure modes, not this toy world. */
    G: 16,
    band: function (d, w) { return Math.max(0, Math.min(1, 1.4 * (w - Math.abs(d)))); },
    renderDisc: function (s) {
      const G = AE.G, img = new Float32Array(G * G);
      const r = 2.4 + 3.4 * s;
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++) {
          const d = Math.hypot(i - 7.5, j - 7.5);
          img[i * G + j] = Math.max(0, Math.min(1, 1.3 * (r - d)));
        }
      return img;
    },
    renderCross: function (s) {
      const G = AE.G, img = new Float32Array(G * G);
      const w = 0.9 + 1.6 * s, L = 6.4;
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++) {
          const dy = i - 7.5, dx = j - 7.5;
          const v = Math.max(
            AE.band(dx, w) * AE.band(dy, L),
            AE.band(dy, w) * AE.band(dx, L)
          );
          img[i * G + j] = v;
        }
      return img;
    },
    renderWedge: function (s) {
      const G = AE.G, img = new Float32Array(G * G);
      const h = 10 + 3 * s, base = 5 + 4 * s, top = 2.2;
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++) {
          const y = i - top, dx = j - 7.5;
          if (y < 0 || y > h) { img[i * G + j] = 0; continue; }
          const half = (y / h) * base;
          img[i * G + j] = Math.max(0, Math.min(1, 1.4 * (half - Math.abs(dx))));
        }
      return img;
    },

    CLUSTERS: [
      { name: "disc",  mu: [-2.2, 1.7],  dir: [0.9, 0.44],  colorKey: "orange" },
      { name: "cross", mu: [2.3, 1.9],   dir: [0.5, -0.87], colorKey: "green" },
      { name: "wedge", mu: [0.2, -2.35], dir: [0.97, 0.26], colorKey: "purple" },
    ],
    renderers: function () { return [AE.renderDisc, AE.renderCross, AE.renderWedge]; },

    // layout(beta): beta = 0 is the raw deterministic latent (spread
    // clusters, gaps); beta = 1 pulls every mean toward the origin and
    // widens sigma toward 1, the N(0, I)-matched latent.
    layout: function (beta) {
      const shrink = 1 - 0.78 * beta;
      return {
        beta: beta,
        halfLen: 1.15,
        tau: 0.55,
        sigmaEnc: 0.25 + 0.75 * beta,
        clusters: AE.CLUSTERS.map(function (c) {
          return { name: c.name, mu: [c.mu[0] * shrink, c.mu[1] * shrink], dir: c.dir.slice(), colorKey: c.colorKey };
        }),
      };
    },

    // deterministic "training set" latents: 16 per cluster with a shape
    // parameter s that varies along the cluster's direction
    latents: function (layout) {
      const rand = LR.rng(1907);
      const out = [];
      layout.clusters.forEach(function (c, ci) {
        const perp = [-c.dir[1], c.dir[0]];
        for (let k = 0; k < 16; k++) {
          const u = -1 + (2 * k) / 15 + (rand() - 0.5) * 0.08;
          const off = LR.gauss(rand) * 0.15;
          out.push({
            c: ci,
            u: u,
            s: (u + 1) / 2,
            z: [
              c.mu[0] + c.dir[0] * u * layout.halfLen + perp[0] * off,
              c.mu[1] + c.dir[1] * u * layout.halfLen + perp[1] * off,
            ],
          });
        }
      });
      return out;
    },

    // the decoder's off-manifold behaviour: far from every training code
    // its output collapses toward a blurry mean glyph (the network's bias),
    // modelled as a smoothed average of the three families
    mush: function () {
      if (AE._mush) return AE._mush;
      const G = AE.G;
      const R = AE.renderers();
      const raw = new Float32Array(G * G);
      R.forEach(function (r) {
        const part = r(0.5);
        for (let p = 0; p < G * G; p++) raw[p] += part[p] / 3;
      });
      // one box-blur pass so it reads as smeared, not crisp
      const img = new Float32Array(G * G);
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++) {
          let s = 0, n = 0;
          for (let di = -1; di <= 1; di++)
            for (let dj = -1; dj <= 1; dj++) {
              const ii = i + di, jj = j + dj;
              if (ii < 0 || jj < 0 || ii >= G || jj >= G) continue;
              s += raw[ii * G + jj]; n += 1;
            }
          img[i * G + j] = 0.85 * (s / n);
        }
      AE._mush = img;
      return img;
    },
    BG_W: 0.08, // constant unnormalized weight of the off-manifold component

    // the constructed decoder: proximity-weighted blend of the three
    // parametric renderers plus the off-manifold background component.
    // Every decode is a real computation.
    decode: function (z, layout) {
      const R = AE.renderers();
      const ws = [], ss = [];
      layout.clusters.forEach(function (c) {
        const rel = AE.sub2(z, c.mu);
        const u = Math.max(-layout.halfLen, Math.min(layout.halfLen, AE.dot2(rel, c.dir)));
        const q = [c.mu[0] + c.dir[0] * u, c.mu[1] + c.dir[1] * u];
        const d = AE.nrm2(AE.sub2(z, q));
        ws.push(Math.exp(-(d * d) / (2 * layout.tau * layout.tau)));
        ss.push((u / layout.halfLen + 1) / 2);
      });
      const clusterSum = ws.reduce((a, b) => a + b, 0);
      ws.push(AE.BG_W);
      const wsum = clusterSum + AE.BG_W;
      const wn = ws.map((w) => w / wsum);
      const G = AE.G, img = new Float32Array(G * G);
      for (let ci = 0; ci < 3; ci++) {
        if (wn[ci] < 1e-4) continue;
        const part = R[ci](Math.max(0, Math.min(1, ss[ci])));
        for (let p = 0; p < G * G; p++) img[p] += wn[ci] * part[p];
      }
      if (wn[3] > 1e-4) {
        const bg = AE.mush();
        for (let p = 0; p < G * G; p++) img[p] += wn[3] * bg[p];
      }
      let dominant = 0;
      for (let ci = 1; ci < 3; ci++) if (wn[ci] > wn[dominant]) dominant = ci;
      return { img: img, wn: wn, conf: Math.min(1, clusterSum), realism: wn[dominant], dominant: dominant, s: ss };
    },

    drawGlyph: function (ctx, img, x0, y0, size, opts) {
      opts = opts || {};
      const G = AE.G, cell = size / G;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(x0, y0, size, size);
      for (let i = 0; i < G; i++)
        for (let j = 0; j < G; j++) {
          const v = Math.max(0, Math.min(1, img[i * G + j]));
          if (v < 0.02) continue;
          ctx.fillStyle = "rgba(17,17,17," + v.toFixed(3) + ")";
          ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.4, cell + 0.4);
        }
      ctx.strokeStyle = opts.border || "#d0d0d0";
      ctx.lineWidth = opts.borderW || 1;
      ctx.strokeRect(x0, y0, size, size);
    },

    clusterColor: function (ci) {
      return [C.orange, C.green, C.purple][ci];
    },
  });

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — encoder, bottleneck, decoder (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.arch = function (mount) {
    LR.header(
      mount,
      "The machine: squeeze, then rebuild",
      "An 8-dimensional signal flows through the encoder, narrows to K latent numbers, and is rebuilt by the decoder. Slide K and watch the reconstruction error move; press Play to see the flow."
    );

    /* dataset: 160 smooth signals sampled at 8 points, built from three
       shaped components with decaying variance plus small noise, so the
       covariance spectrum decays and the K-slider has real consequences.
       Flagged addition: the signal family is ours; the errors are computed. */
    const D = 8, N = 160;
    const rand = LR.rng(657);
    const ts = Array.from({ length: D }, (_, j) => j / (D - 1));
    const basis = [
      ts.map((t) => Math.sin(Math.PI * t)),
      ts.map((t) => Math.cos(Math.PI * t)),
      ts.map((t) => Math.sin(2 * Math.PI * t)),
    ];
    const data = [];
    for (let n = 0; n < N; n++) {
      const a = LR.gauss(rand) * 1.6, b = LR.gauss(rand) * 1.0, c = LR.gauss(rand) * 0.55;
      data.push(ts.map((_, j) =>
        a * basis[0][j] + b * basis[1][j] + c * basis[2][j] + LR.gauss(rand) * 0.18
      ));
    }
    const covOut = LR.cov(data);
    const mean = covOut.mean;
    const eig = LR.eigSym(covOut.C);       // real 8×8 eigendecomposition
    const totalVar = eig.values.reduce((a, b) => a + b, 0);

    let K = 2;
    let sampleIdx = 3;
    let playing = !LR.reducedMotion;
    let phase = 0;

    function reconstruct(x, k) {
      const xc = x.map((v, j) => v - mean[j]);
      const out = mean.slice();
      for (let i = 0; i < k; i++) {
        const u = eig.vectors[i];
        const z = xc.reduce((s, v, j) => s + v * u[j], 0);
        for (let j = 0; j < D; j++) out[j] += z * u[j];
      }
      return out;
    }
    function avgError(k) {
      // mean squared reconstruction error over the whole dataset (real)
      let s = 0;
      data.forEach(function (x) {
        const r = reconstruct(x, k);
        x.forEach((v, j) => { s += (v - r[j]) * (v - r[j]); });
      });
      return s / N;
    }
    function tailEig(k) {
      let s = 0;
      for (let i = k; i < D; i++) s += eig.values[i];
      return s;
    }

    const bar = LR.controls(mount);
    LR.slider(bar, "bottleneck width K", 1, 8, 1, K, function (v) { K = v; draw(); update(); }, (v) => "K = " + v);
    LR.button(bar, "Next sample ▸", function () { sampleIdx = (sampleIdx + 1) % N; draw(); update(); });
    const playBtn = LR.button(bar, playing ? "Pause ❚❚" : "Play ▸▸", function () {
      playing = !playing;
      playBtn.textContent = playing ? "Pause ❚❚" : "Play ▸▸";
      if (playing) tick();
    });

    const cvo = LR.canvas(mount, 840, 320, {
      aria: "Autoencoder diagram: eight input nodes feed an encoder that narrows to K latent nodes, then a decoder expands back to eight outputs; input and reconstruction signals are plotted on either side",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;

    const ro = LR.readout(mount, [
      { k: "err", label: "mean ‖x − x̃‖² (dataset)" },
      { k: "tail", label: "Σ trailing λᵢ" },
      { k: "kept", label: "variance kept" },
    ]);
    const msg = LR.msg(mount);

    function colYs(count, cy, gap) {
      const ys = [];
      for (let i = 0; i < count; i++) ys.push(cy + (i - (count - 1) / 2) * gap);
      return ys;
    }
    function bez(p0, p1, p2, p3, u) {
      const v = 1 - u;
      return v * v * v * p0 + 3 * v * v * u * p1 + 3 * v * u * u * p2 + u * u * u * p3;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const cy = 150;
      const layers = [
        { n: D, x: 150, label: "x  (8)" },
        { n: 5, x: 272, label: "f" },
        { n: K, x: 420, label: "z  (K = " + K + ")" },
        { n: 5, x: 568, label: "gθ" },
        { n: D, x: 690, label: "x̃  (8)" },
      ];
      const ys = layers.map((L) => colYs(L.n, cy, L.n > 6 ? 26 : 34));

      // edges
      ctx.lineWidth = 1;
      for (let l = 0; l < layers.length - 1; l++) {
        ys[l].forEach(function (y1) {
          ys[l + 1].forEach(function (y2) {
            const mid = (layers[l].x + layers[l + 1].x) / 2;
            ctx.strokeStyle = l < 2 ? "rgba(26,26,26,0.15)" : "rgba(102,102,102,0.15)";
            ctx.beginPath();
            ctx.moveTo(layers[l].x, y1);
            ctx.bezierCurveTo(mid, y1, mid, y2, layers[l + 1].x, y2);
            ctx.stroke();
          });
        });
      }

      // flowing pulses (paused under reduced motion or by button)
      if (playing) {
        const t = phase % 1;
        const seg = Math.floor(phase % 4);
        const l = seg;
        ys[l].forEach(function (y1, i) {
          const y2 = ys[l + 1][i % ys[l + 1].length];
          const mid = (layers[l].x + layers[l + 1].x) / 2;
          const bx = bez(layers[l].x, mid, mid, layers[l + 1].x, t);
          const by = bez(y1, y1, y2, y2, t);
          LR.dot(ctx, bx, by, 3, l < 2 ? C.orange : C.green);
        });
      }

      // nodes
      layers.forEach(function (L, l) {
        ys[l].forEach(function (y) {
          const isLatent = l === 2;
          LR.dot(ctx, L.x, y, isLatent ? 9 : 6.5, isLatent ? C.purple : "#ffffff", isLatent ? C.purple : "#444444");
        });
        ctx.fillStyle = l === 2 ? C.purple : C.text;
        ctx.font = "700 12.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(L.label, L.x, 290);
      });

      // stage labels
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.orange;
      ctx.textAlign = "center";
      ctx.fillText("encoder  z = f(x)", (layers[0].x + layers[2].x) / 2, 22);
      ctx.fillStyle = C.green;
      ctx.fillText("decoder  x̃ = gθ(z)", (layers[2].x + layers[4].x) / 2, 22);
      ctx.fillStyle = C.purple;
      ctx.fillText("bottleneck", layers[2].x, 40);

      // input / reconstruction mini-plots
      const x = data[sampleIdx];
      const r = reconstruct(x, K);
      miniSignal(22, 58, 96, 190, [{ v: x, col: C.text, w: 2 }], "input x");
      miniSignal(724, 58, 96, 190, [{ v: x, col: "#c8c8c8", w: 1.6 }, { v: r, col: C.orange, w: 2.2 }], "x (grey) vs x̃");
    }

    function miniSignal(x0, y0, w, h, series, label) {
      ctx.strokeStyle = "#dddddd";
      ctx.lineWidth = 1;
      ctx.strokeRect(x0, y0, w, h);
      ctx.fillStyle = LR.C.muted;
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x0 + w / 2, y0 + h + 14);
      const lo = -3.4, hi = 3.4;
      series.forEach(function (s) {
        ctx.strokeStyle = s.col;
        ctx.lineWidth = s.w;
        ctx.beginPath();
        s.v.forEach(function (v, j) {
          const px = x0 + 6 + (j / (D - 1)) * (w - 12);
          const py = y0 + h - ((v - lo) / (hi - lo)) * h;
          if (j === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      });
    }

    function update() {
      const err = avgError(K);
      const tail = tailEig(K);
      const kept = 1 - tail / totalVar;
      ro.set("err", LR.fmtF(err, 3), C.orange);
      ro.set("tail", LR.fmtF(tail, 3) + "  (they agree: the linear optimum is Σ trailing λ)");
      ro.set("kept", Math.round(kept * 100) + "%", kept > 0.9 ? C.green : C.amber);
      if (K <= 2) {
        msg.show("A tight bottleneck: 8 numbers forced through " + K + ". Compression is lossy, and the error above is exactly what was thrown away.", "info");
      } else if (K >= 7) {
        msg.show("Almost no bottleneck: with K near 8 the network can nearly copy its input. Reconstruction is easy and the code is nearly useless as a summary.", "bad");
      } else {
        msg.hide();
      }
    }

    function tick() {
      if (!playing) return;
      phase += 0.02;
      if (phase > 4) phase = 0;
      draw();
      requestAnimationFrame(tick);
    }

    draw();
    update();
    if (playing) tick();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — a linear autoencoder converges to PCA
     ══════════════════════════════════════════════════════════ */
  LR.figs.linearpca = function (mount) {
    LR.header(
      mount,
      "Train a linear autoencoder, watch PCA appear",
      "Gradient descent on W_f and W_g really runs in your browser. The product W_gW_f marches toward the PCA projection U₁U₁ᵀ, while the factors themselves stay free to change."
    );

    // dataset: 2-D cloud stretched along [1,1]/√2 (seeded, real covariance)
    const rand = LR.rng(41);
    const u0 = [1 / Math.SQRT2, 1 / Math.SQRT2];
    const perp0 = [-u0[1], u0[0]];
    const pts = [];
    for (let i = 0; i < 44; i++) {
      const s = LR.gauss(rand) * 1.8;
      const n = LR.gauss(rand) * 0.5;
      pts.push([s * u0[0] + n * perp0[0], s * u0[1] + n * perp0[1]]);
    }
    const covOut = LR.cov(pts);
    const eig = LR.eigSym(covOut.C);
    const u1 = eig.vectors[0][1] >= 0 ? eig.vectors[0] : eig.vectors[0].map((v) => -v);
    const P = [
      [u1[0] * u1[0], u1[0] * u1[1]],
      [u1[1] * u1[0], u1[1] * u1[1]],
    ];

    // linear AE parameters: w encodes (1×2), v decodes (2×1)
    let w, v, steps;
    function resetAE() {
      const r2 = LR.rng(99);
      w = [0.9 * (r2() - 0.2), -0.7 * r2()];
      v = [0.3 * r2() - 0.6, 0.9 * r2()];
      steps = 0;
    }
    resetAE();

    let aFac = 1; // the invertible reparameterization A (a scalar, since K = 1)

    function loss() {
      let s = 0;
      pts.forEach(function (p) {
        const z = w[0] * p[0] + w[1] * p[1];
        const e0 = p[0] - v[0] * z, e1 = p[1] - v[1] * z;
        s += e0 * e0 + e1 * e1;
      });
      return s / pts.length;
    }
    function gdStep(lr) {
      let gw0 = 0, gw1 = 0, gv0 = 0, gv1 = 0;
      pts.forEach(function (p) {
        const z = w[0] * p[0] + w[1] * p[1];
        const e0 = p[0] - v[0] * z, e1 = p[1] - v[1] * z;
        const ev = e0 * v[0] + e1 * v[1];
        gw0 += (-2 * ev * p[0]) / pts.length;
        gw1 += (-2 * ev * p[1]) / pts.length;
        gv0 += (-2 * e0 * z) / pts.length;
        gv1 += (-2 * e1 * z) / pts.length;
      });
      w[0] -= lr * gw0; w[1] -= lr * gw1;
      v[0] -= lr * gv0; v[1] -= lr * gv1;
    }

    const bar = LR.controls(mount);
    LR.button(bar, "Train 1 step", function () { gdStep(0.08); steps += 1; render(); });
    LR.button(bar, "Train 200 steps ▸", function () {
      for (let i = 0; i < 200; i++) gdStep(0.08);
      steps += 200; render();
    }, "primary");
    LR.button(bar, "Reset ⟲", function () { resetAE(); aCtl.set(1); aFac = 1; render(); });
    const aCtl = LR.slider(bar, "refactor A", 0.25, 4, 0.05, 1, function (val) { aFac = val; render(); }, (val) => "A = " + LR.fmtF(val, 2));

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left);
    grid.appendChild(right);

    const cvo = LR.canvas(left, 400, 330, {
      aria: "Scatter of 2-D data with the PCA direction as a dashed green line and the linear autoencoder's decoder direction as a solid orange line, converging as training runs",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;
    right.appendChild(LR.el("div", "pane-label", "the product vs the PCA projection"));
    const mtxBox = LR.el("div", "mtx-stage");
    right.appendChild(mtxBox);

    const ro = LR.readout(mount, [
      { k: "steps", label: "GD steps" },
      { k: "loss", label: "reconstruction loss" },
      { k: "gap", label: "max |W_gW_f − U₁U₁ᵀ|" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      // display factors after the invertible refactor A (product unchanged)
      const wd = [aFac * w[0], aFac * w[1]];
      const vd = [v[0] / aFac, v[1] / aFac];
      const prod = [
        [vd[0] * wd[0], vd[0] * wd[1]],
        [vd[1] * wd[0], vd[1] * wd[1]],
      ];
      let gap = 0;
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++) gap = Math.max(gap, Math.abs(prod[i][j] - P[i][j]));

      ctx.clearRect(0, 0, W, H);
      const Pl = { x0: 46, y0: 14, w: W - 62, h: H - 58, xmin: -4, xmax: 4, ymin: -3.3, ymax: 3.3, xlabel: "x₁", ylabel: "x₂" };
      const sc = LR.plot(ctx, Pl);
      const sx = sc.sx, sy = sc.sy;

      // PCA direction (dashed green)
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(sx(-4 * u1[0]), sy(-4 * u1[1]));
      ctx.lineTo(sx(4 * u1[0]), sy(4 * u1[1]));
      ctx.stroke();
      ctx.setLineDash([]);

      // decoder span (solid orange): the AE's learned 1-D subspace
      const vn = Math.hypot(vd[0], vd[1]) || 1e-9;
      const vv = [vd[0] / vn, vd[1] / vn];
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(sx(-4 * vv[0]), sy(-4 * vv[1]));
      ctx.lineTo(sx(4 * vv[0]), sy(4 * vv[1]));
      ctx.stroke();

      // reconstructions
      pts.forEach(function (p) {
        const z = wd[0] * p[0] + wd[1] * p[1];
        const rx = vd[0] * z, ry = vd[1] * z;
        ctx.strokeStyle = "rgba(61,61,61,0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx(p[0]), sy(p[1]));
        ctx.lineTo(sx(rx), sy(ry));
        ctx.stroke();
      });
      pts.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 3.4, C.text));
      pts.forEach(function (p) {
        const z = wd[0] * p[0] + wd[1] * p[1];
        LR.dot(ctx, sx(vd[0] * z), sy(vd[1] * z), 2.8, "#ffffff", C.orange);
      });

      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillStyle = C.green;
      ctx.textAlign = "left";
      ctx.fillText("u₁ (PCA)", Pl.x0 + 8, Pl.y0 + 16);
      ctx.fillStyle = C.orange;
      ctx.fillText("span(W_g) (autoencoder)", Pl.x0 + 8, Pl.y0 + 32);

      mtxBox.innerHTML = "";
      const row = LR.el("div", "mtx-row");
      mtxBox.appendChild(row);
      row.appendChild(AE.matrix([wd], { label: "W_f (1×2) · A", d: 3 }));
      row.appendChild(AE.matrix([[vd[0]], [vd[1]]], { label: "W_g (2×1) · A⁻¹", d: 3 }));
      const row2 = LR.el("div", "mtx-row");
      mtxBox.appendChild(row2);
      row2.appendChild(AE.matrix(prod, { label: "W_gW_f", d: 3, heat: "signed", vmax: 0.8 }));
      row2.appendChild(AE.op("vs"));
      row2.appendChild(AE.matrix(P, { label: "U₁U₁ᵀ (target)", d: 3, heat: "signed", vmax: 0.8 }));

      ro.set("steps", String(steps));
      ro.set("loss", LR.fmtF(loss(), 4) + "  (optimum = λ₂ = " + LR.fmtF(eig.values[1], 4) + ")", C.orange);
      ro.set("gap", LR.fmtF(gap, 4), gap < 0.02 ? C.green : C.amber);

      if (gap < 0.02) {
        msg.show("Converged: W_gW_f matches U₁U₁ᵀ to two decimals, and the loss sits at λ₂. Now slide A: the two factors change and the product, the subspace, and the loss do not. The factorization is not unique.", "good");
      } else if (steps === 0) {
        msg.show("Untrained: the orange subspace starts wherever the random init put it. Train and watch it rotate onto the PCA direction.", "info");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — flat subspace vs curved manifold (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.manifold = function (mount) {
    LR.header(
      mount,
      "Bend the data until the flat model breaks",
      "PCA must answer with a straight line. A nonlinear autoencoder can answer with a curve. Both reconstruction errors are computed live; slide the bend and watch them separate."
    );

    let bend = 0.85;
    let view = 2; // 0 = PCA, 1 = curved, 2 = both

    // build the dataset for the current bend (seeded, so scrubbing is stable)
    function makeData(b) {
      const rand = LR.rng(29);
      const R = 1.62 / b; // arc length held roughly fixed as the bend changes
      const pts = [];
      for (let i = 0; i < 52; i++) {
        const t = -1 + (2 * i) / 51 + (rand() - 0.5) * 0.02;
        const phi = b * Math.PI * t * 0.98;
        const radial = [Math.sin(phi), -Math.cos(phi)];
        const noise = LR.gauss(rand) * 0.22;
        pts.push([
          R * Math.sin(phi) + noise * radial[0],
          R - R * Math.cos(phi) + noise * radial[1],
        ]);
      }
      // centre the cloud
      const m = LR.cov(pts).mean;
      return pts.map((p) => [p[0] - m[0], p[1] - m[1]]);
    }

    // Kasa circle fit: linear least squares, the "nonlinear decoder" arc
    function fitCircle(pts) {
      let sxx = 0, sxy = 0, syy = 0, sx = 0, sy = 0, sxr = 0, syr = 0, sr = 0;
      const n = pts.length;
      pts.forEach(function (p) {
        const r2 = p[0] * p[0] + p[1] * p[1];
        sxx += p[0] * p[0]; sxy += p[0] * p[1]; syy += p[1] * p[1];
        sx += p[0]; sy += p[1];
        sxr += p[0] * r2; syr += p[1] * r2; sr += r2;
      });
      const sol = LR.solve(
        [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]],
        [sxr, syr, sr]
      );
      const cx = sol[0] / 2, cy = sol[1] / 2;
      const r = Math.sqrt(Math.max(1e-9, sol[2] + cx * cx + cy * cy));
      return { cx: cx, cy: cy, r: r };
    }

    const bar = LR.controls(mount);
    LR.slider(bar, "bend", 0.12, 1, 0.01, bend, function (v) { bend = v; render(); }, (v) => LR.fmtF(v, 2));
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bts = [
      LR.button(grp, "PCA line", function () { view = 0; render(); }),
      LR.button(grp, "nonlinear AE", function () { view = 1; render(); }),
      LR.button(grp, "both", function () { view = 2; render(); }, "on"),
    ];

    const cvo = LR.canvas(mount, 760, 470, {
      aria: "Scatter of points along a curved arc; a straight PCA line misses the curve while a fitted circular arc follows it; residual segments show each model's reconstruction error",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;
    const ro = LR.readout(mount, [
      { k: "flat", label: "MSE, PCA (flat, K=1)" },
      { k: "curved", label: "MSE, nonlinear AE (K=1)" },
      { k: "ratio", label: "flat ÷ curved" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      bts.forEach((b, i) => b.classList.toggle("on", i === view));
      const pts = makeData(bend);
      const eig = LR.eigSym(LR.cov(pts).C);
      const u1 = eig.vectors[0];

      // flat reconstruction: project onto u1
      let mseFlat = 0;
      const flatRec = pts.map(function (p) {
        const z = p[0] * u1[0] + p[1] * u1[1];
        const r = [z * u1[0], z * u1[1]];
        mseFlat += (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2;
        return r;
      });
      mseFlat /= pts.length;

      // curved reconstruction: radial projection onto the fitted circle
      const circ = fitCircle(pts);
      let mseCurv = 0;
      const curvRec = pts.map(function (p) {
        const dx = p[0] - circ.cx, dy = p[1] - circ.cy;
        const d = Math.hypot(dx, dy) || 1e-9;
        const r = [circ.cx + (circ.r * dx) / d, circ.cy + (circ.r * dy) / d];
        mseCurv += (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2;
        return r;
      });
      mseCurv /= pts.length;

      ctx.clearRect(0, 0, W, H);
      const Pl = {
        x0: 60, y0: 16, w: 656, h: 380,
        xmin: -6.56, xmax: 6.56, ymin: -3.8, ymax: 3.8,   // equal aspect on both axes
        xlabel: "x₁", ylabel: "x₂",
      };
      const sc = LR.plot(ctx, Pl);
      const sx = sc.sx, sy = sc.sy;

      // residuals for the flat model
      ctx.lineWidth = 1;
      if (view === 0 || view === 2) {
        ctx.strokeStyle = "rgba(61,61,61,0.4)";
        pts.forEach(function (p, i) {
          ctx.beginPath();
          ctx.moveTo(sx(p[0]), sy(p[1]));
          ctx.lineTo(sx(flatRec[i][0]), sy(flatRec[i][1]));
          ctx.stroke();
        });
      }

      // PCA line
      if (view === 0 || view === 2) {
        ctx.strokeStyle = C.green;
        ctx.lineWidth = 2.4;
        ctx.setLineDash(view === 2 ? [7, 5] : []);
        ctx.beginPath();
        ctx.moveTo(sx(-7 * u1[0]), sy(-7 * u1[1]));
        ctx.lineTo(sx(7 * u1[0]), sy(7 * u1[1]));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // fitted circle (the curved 1-D manifold)
      if (view === 1 || view === 2) {
        ctx.strokeStyle = C.orange;
        ctx.lineWidth = 2.6;
        ctx.save();
        ctx.beginPath();
        ctx.rect(Pl.x0, Pl.y0, Pl.w, Pl.h);
        ctx.clip();
        ctx.beginPath();
        ctx.arc(sx(circ.cx), sy(circ.cy), (circ.r / (Pl.xmax - Pl.xmin)) * Pl.w, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // points and curved reconstructions
      pts.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 3.6, C.text));
      if (view === 1 || view === 2) {
        pts.forEach(function (p, i) {
          LR.dot(ctx, sx(curvRec[i][0]), sy(curvRec[i][1]), 2.6, "#ffffff", C.orange);
        });
      }

      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "left";
      if (view !== 1) { ctx.fillStyle = C.green; ctx.fillText("PCA: best flat line", Pl.x0 + 10, Pl.y0 + 18); }
      if (view !== 0) { ctx.fillStyle = C.orange; ctx.fillText("nonlinear AE: best curve", Pl.x0 + 10, Pl.y0 + 36); }

      ro.set("flat", LR.fmtF(mseFlat, 3), C.green);
      ro.set("curved", LR.fmtF(mseCurv, 3), C.orange);
      ro.set("ratio", LR.fmtF(mseFlat / Math.max(mseCurv, 1e-9), 1) + "×", mseFlat / mseCurv > 3 ? C.red : C.text);

      if (bend < 0.25) {
        msg.show("Nearly straight data: the flat line and the curve agree, and PCA is the right tool. Nonlinearity buys nothing here.", "good");
      } else if (mseFlat / mseCurv > 5) {
        msg.show("The manifold is one-dimensional, but curved. One latent number (position along the arc) reconstructs it almost perfectly; the best flat line pays " + LR.fmtF(mseFlat / mseCurv, 0) + "× more error for the same K.", "info");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.2 — the denoising autoencoder
     ══════════════════════════════════════════════════════════ */
  LR.figs.denoise = function (mount) {
    LR.header(
      mount,
      "Corrupt, then recover",
      "A clean signal from a 2-parameter family is corrupted by noise you control. The denoiser maps the corrupted input back to the family; the projection is computed live, nothing is staged."
    );

    const D = 40;
    const ts = Array.from({ length: D }, (_, j) => j / (D - 1));
    const b1 = ts.map((t) => Math.sin(2 * Math.PI * t));
    const b2 = ts.map((t) => Math.cos(2 * Math.PI * t));

    let sigSeed = 7, noiseSeed = 3, sigma = 0.55;

    function makeClean(seed) {
      const rand = LR.rng(seed);
      const a = (rand() - 0.5) * 3, b = (rand() - 0.5) * 3;
      return { a: a, b: b, v: ts.map((_, j) => a * b1[j] + b * b2[j]) };
    }
    function makeNoise(seed, s) {
      const rand = LR.rng(seed * 977 + 13);
      return ts.map(() => LR.gauss(rand) * s);
    }
    function denoise(x) {
      // least-squares fit of (a, b) in the clean family: the manifold projection
      let s11 = 0, s12 = 0, s22 = 0, r1 = 0, r2 = 0;
      for (let j = 0; j < D; j++) {
        s11 += b1[j] * b1[j]; s12 += b1[j] * b2[j]; s22 += b2[j] * b2[j];
        r1 += b1[j] * x[j]; r2 += b2[j] * x[j];
      }
      const sol = LR.solve([[s11, s12], [s12, s22]], [r1, r2]);
      return ts.map((_, j) => sol[0] * b1[j] + sol[1] * b2[j]);
    }
    function mse(x, y) {
      let s = 0;
      for (let j = 0; j < D; j++) s += (x[j] - y[j]) * (x[j] - y[j]);
      return s / D;
    }

    const bar = LR.controls(mount);
    LR.slider(bar, "noise σ", 0, 1.2, 0.05, sigma, function (v) { sigma = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "New signal", function () { sigSeed = (sigSeed * 31 + 7) % 100000; render(); });
    LR.button(bar, "New noise ▸", function () { noiseSeed += 1; render(); }, "primary");

    const cvo = LR.canvas(mount, 760, 340, {
      aria: "Three signals plotted over time: the clean original, its noisy corruption, and the denoised output which nearly recovers the original",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;
    const ro = LR.readout(mount, [
      { k: "noisy", label: "MSE(x̃, x)" },
      { k: "den", label: "MSE(denoised, x)" },
      { k: "gain", label: "error removed" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      const clean = makeClean(sigSeed);
      const noise = makeNoise(noiseSeed, sigma);
      const noisy = clean.v.map((v, j) => v + noise[j]);
      const den = denoise(noisy);

      ctx.clearRect(0, 0, W, H);
      const Pl = { x0: 56, y0: 16, w: W - 80, h: H - 70, xmin: 0, xmax: 1, ymin: -3.6, ymax: 3.6, xlabel: "t", ylabel: "signal value" };
      const sc = LR.plot(ctx, Pl);
      const sx = sc.sx, sy = sc.sy;

      function curve(v, col, wdt, dash) {
        ctx.strokeStyle = col;
        ctx.lineWidth = wdt;
        if (dash) ctx.setLineDash(dash);
        ctx.beginPath();
        v.forEach(function (val, j) {
          if (j === 0) ctx.moveTo(sx(ts[j]), sy(val));
          else ctx.lineTo(sx(ts[j]), sy(val));
        });
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // noisy as dots + faint line
      ctx.strokeStyle = "rgba(61,61,61,0.35)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      noisy.forEach(function (val, j) {
        if (j === 0) ctx.moveTo(sx(ts[j]), sy(val));
        else ctx.lineTo(sx(ts[j]), sy(val));
      });
      ctx.stroke();
      noisy.forEach((val, j) => LR.dot(ctx, sx(ts[j]), sy(val), 2.4, C.red));
      curve(clean.v, "#bbbbbb", 2, [6, 5]);
      curve(den, C.orange, 2.6);

      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.red; ctx.fillText("x̃ (corrupted input)", Pl.x0 + 10, Pl.y0 + 16);
      ctx.fillStyle = "#999999"; ctx.fillText("x (clean, hidden from the model)", Pl.x0 + 10, Pl.y0 + 32);
      ctx.fillStyle = C.orange; ctx.fillText("denoiser output", Pl.x0 + 10, Pl.y0 + 48);

      const eN = mse(noisy, clean.v), eD = mse(den, clean.v);
      ro.set("noisy", LR.fmtF(eN, 3), C.red);
      ro.set("den", LR.fmtF(eD, 3), C.orange);
      const gain = eN > 1e-9 ? 1 - eD / eN : 0;
      ro.set("gain", Math.round(Math.max(0, gain) * 100) + "%", gain > 0.6 ? C.green : C.amber);

      if (sigma === 0) {
        msg.show("No corruption: input equals output and there is nothing to learn. Turn the noise up; the interesting mapping is x̃ → x.", "info");
      } else if (gain > 0.85) {
        msg.show("The denoiser never sees the clean x. It succeeds because clean signals live on a low-dimensional family, and mapping to the nearest family member averages the noise away: under squared loss the optimal output is E[x | x̃].", "good");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — the sparse autoencoder's code
     ══════════════════════════════════════════════════════════ */
  LR.figs.sparse = function (mount) {
    LR.header(
      mount,
      "Sparsity is a constraint on activations",
      "Eight decoder atoms could reconstruct this signal densely. Raise the sparsity penalty and watch most latent activations get pushed to exactly zero; the code is re-solved live (40 ISTA iterations per frame)."
    );

    const D = 40, K = 8;
    const ts = Array.from({ length: D }, (_, j) => j / (D - 1));
    // dictionary: 8 normalized Gaussian bumps (the "decoder columns")
    const atoms = [];
    for (let k = 0; k < K; k++) {
      const cpos = (k + 0.5) / K;
      let a = ts.map((t) => Math.exp(-((t - cpos) ** 2) / (2 * 0.055 * 0.055)));
      const nrm = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
      a = a.map((v) => v / nrm);
      atoms.push(a);
    }
    // target: two atoms plus small noise (seeded)
    const rand = LR.rng(11);
    const target = ts.map((_, j) => 1.25 * atoms[2][j] + 0.85 * atoms[5][j] + LR.gauss(rand) * 0.02);

    // Lipschitz constant of DᵀD for the ISTA step size (real eigenvalue)
    const Gm = Array.from({ length: K }, (_, a) =>
      Array.from({ length: K }, (_, b) => atoms[a].reduce((s, v, j) => s + v * atoms[b][j], 0))
    );
    const Lmax = LR.eigSym(Gm).values[0];
    const eta = 0.9 / Lmax;

    let lambda = 0;

    function solveCode(lam) {
      const z = new Array(K).fill(0);
      for (let it = 0; it < 40; it++) {
        const rec = new Array(D).fill(0);
        for (let k = 0; k < K; k++)
          for (let j = 0; j < D; j++) rec[j] += z[k] * atoms[k][j];
        const grad = new Array(K).fill(0);
        for (let k = 0; k < K; k++)
          for (let j = 0; j < D; j++) grad[k] += atoms[k][j] * (rec[j] - target[j]);
        for (let k = 0; k < K; k++) {
          const t = z[k] - eta * grad[k];
          const thr = eta * lam;
          z[k] = Math.sign(t) * Math.max(0, Math.abs(t) - thr); // soft threshold
        }
      }
      return z;
    }

    const bar = LR.controls(mount);
    LR.slider(bar, "sparsity penalty λ", 0, 0.4, 0.01, lambda, function (v) { lambda = v; render(); }, (v) => LR.fmtF(v, 2));

    const cvo = LR.canvas(mount, 760, 330, {
      aria: "Left: bar chart of the eight latent activations, most of which become zero as the sparsity penalty rises. Right: the reconstructed signal against the target",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;
    const ro = LR.readout(mount, [
      { k: "nz", label: "non-zero activations" },
      { k: "err", label: "reconstruction MSE" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      const z = solveCode(lambda);
      const rec = new Array(D).fill(0);
      for (let k = 0; k < K; k++)
        for (let j = 0; j < D; j++) rec[j] += z[k] * atoms[k][j];
      let err = 0;
      for (let j = 0; j < D; j++) err += (rec[j] - target[j]) ** 2;
      err /= D;
      const nz = z.filter((v) => Math.abs(v) > 1e-3).length;

      ctx.clearRect(0, 0, W, H);

      // left: activation bars
      const L = { x0: 56, y0: 16, w: 280, h: H - 70, xmin: -0.5, xmax: 7.5, ymin: -0.4, ymax: 1.6, xlabel: "latent coordinate k", ylabel: "activation zₖ", xticks: [0, 1, 2, 3, 4, 5, 6, 7] };
      const pl = LR.plot(ctx, L);
      const zero = pl.sy(0);
      z.forEach(function (v, k) {
        const bw = 22;
        const px = pl.sx(k) - bw / 2;
        const py = pl.sy(v);
        ctx.fillStyle = Math.abs(v) > 1e-3 ? C.orange : "#dcdcdc";
        ctx.fillRect(px, Math.min(zero, py), bw, Math.abs(zero - py) || 1.6);
      });
      ctx.strokeStyle = C.axis;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L.x0, zero); ctx.lineTo(L.x0 + L.w, zero); ctx.stroke();

      // right: reconstruction
      const Rp = { x0: 420, y0: 16, w: W - 450, h: H - 70, xmin: 0, xmax: 1, ymin: -0.2, ymax: 1.1, xlabel: "t", ylabel: "signal" };
      const pr = LR.plot(ctx, Rp);
      ctx.strokeStyle = "#bbbbbb";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      target.forEach(function (v, j) {
        if (j === 0) ctx.moveTo(pr.sx(ts[j]), pr.sy(v));
        else ctx.lineTo(pr.sx(ts[j]), pr.sy(v));
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      rec.forEach(function (v, j) {
        if (j === 0) ctx.moveTo(pr.sx(ts[j]), pr.sy(v));
        else ctx.lineTo(pr.sx(ts[j]), pr.sy(v));
      });
      ctx.stroke();
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "#999999"; ctx.fillText("target x", Rp.x0 + 8, Rp.y0 + 16);
      ctx.fillStyle = C.orange; ctx.fillText("reconstruction W_g z", Rp.x0 + 8, Rp.y0 + 32);

      ro.set("nz", nz + " of " + K, nz <= 3 ? C.green : C.amber);
      ro.set("err", LR.fmtF(err, 4), C.orange);

      if (lambda === 0) {
        msg.show("No penalty: the code spreads across several coordinates even though two atoms generated the signal. Dense codes reconstruct well but say little.", "info");
      } else if (nz === 2) {
        msg.show("Two active coordinates, six exact zeros: the code found the two atoms that actually generated the signal. The constraint acts on the activations z, not on the weights, which is the defining feature of a sparse autoencoder.", "good");
      } else if (nz < 2) {
        msg.show("Too much penalty: the code is sparser than the truth and reconstruction visibly suffers. Sparsity is a trade, not a free lunch.", "bad");
      } else {
        msg.hide();
      }
    }
    render();
  };
})();
