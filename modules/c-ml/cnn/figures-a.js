/* ══════════════════════════════════════════════════════════════
   figures-a.js — CNN lesson figures, sections 0–3
   Fig 0.1 params · Fig 2.1 detector · Fig 3.1 conv1d
   Fig 3.2 conv2d (signature figure)
   Also defines the shared convolution core (LR.cnn) used by
   figures-b.js: conv2d, output sizing, pooling, kernels, ramps.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared CNN core (also used by figures-b.js) ────────── */
  const CNN = (LR.cnn = {
    // output-size formula, exact (may be fractional if S does not divide)
    outSize: function (W, K, P, S) {
      return (W - K + 2 * P) / S + 1;
    },

    // zero-pad a 2-D array with P rings of zeros
    pad: function (img, P) {
      if (!P) return img;
      const W = img[0].length;
      const blank = new Array(W + 2 * P).fill(0);
      const out = [];
      for (let i = 0; i < P; i++) out.push(blank.slice());
      img.forEach(function (row) {
        out.push(new Array(P).fill(0).concat(row, new Array(P).fill(0)));
      });
      for (let i = 0; i < P; i++) out.push(blank.slice());
      return out;
    },

    // 2-D convolution as CNNs compute it: the sliding dot product
    // (cross-correlation; see the flagged note in the lesson text).
    conv2d: function (img, ker, S, P) {
      S = S || 1;
      const im = CNN.pad(img, P || 0);
      const K = ker.length;
      const Ho = Math.floor((im.length - K) / S) + 1;
      const Wo = Math.floor((im[0].length - K) / S) + 1;
      const out = [];
      for (let i = 0; i < Ho; i++) {
        const row = [];
        for (let j = 0; j < Wo; j++) {
          let s = 0;
          for (let a = 0; a < K; a++)
            for (let b = 0; b < K; b++) s += im[i * S + a][j * S + b] * ker[a][b];
          row.push(s);
        }
        out.push(row);
      }
      return out;
    },

    // full true 2-D convolution of two kernels (for the linear-collapse
    // demo: corr(corr(x,a),b) = corr(x, convFullK(a,b)) exactly)
    convFullK: function (a, b) {
      const Ka = a.length, Kb = b.length, n = Ka + Kb - 1;
      const out = [];
      for (let u = 0; u < n; u++) {
        const row = [];
        for (let v = 0; v < n; v++) {
          let s = 0;
          for (let p = 0; p < Kb; p++) {
            for (let q = 0; q < Kb; q++) {
              const i = u - p, j = v - q;
              if (i >= 0 && i < Ka && j >= 0 && j < Ka) s += a[i][j] * b[p][q];
            }
          }
          row.push(s);
        }
        out.push(row);
      }
      return out;
    },

    relu: function (m) {
      return m.map((row) => row.map((v) => Math.max(0, v)));
    },

    // non-overlapping pooling, stride = size
    pool: function (m, size, mode) {
      const n = Math.floor(m.length / size);
      const out = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
          let best = -Infinity, sum = 0;
          for (let a = 0; a < size; a++)
            for (let b = 0; b < size; b++) {
              const v = m[i * size + a][j * size + b];
              if (v > best) best = v;
              sum += v;
            }
          row.push(mode === "avg" ? sum / (size * size) : best);
        }
        out.push(row);
      }
      return out;
    },

    maxAbs: function (m) {
      let mx = 0;
      m.forEach((row) => row.forEach((v) => { if (Math.abs(v) > mx) mx = Math.abs(v); }));
      return mx;
    },

    /* the lesson's shared test image: a bright square on dark background
       (5 rows × 4 cols of 1s), so edge kernels have real edges to find */
    IMG8: [
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ],

    /* flagged addition: preset kernel values (Prewitt-style edge filters,
       1/9 box blur, 5-centre sharpen) are standard textbook examples chosen
       by me; the source slides name no specific kernels. */
    KER: {
      vedge: { name: "Vertical edge", k: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
      hedge: { name: "Horizontal edge", k: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
      blur: { name: "Blur", k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] },
      sharpen: { name: "Sharpen", k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    },

    /* colour helpers: grayscale for images, a restrained diverging ramp
       for feature maps (orange = positive, purple = negative, white = 0) */
    gray: function (v) {
      const s = Math.round(245 - Math.max(0, Math.min(1, v)) * 215);
      return "rgb(" + s + "," + s + "," + s + ")";
    },
    ramp: function (v, vmax) {
      if (!vmax) return "#ffffff";
      const a = Math.min(1, Math.abs(v) / vmax) * 0.85;
      return v >= 0 ? "rgba(26,26,26," + a.toFixed(3) + ")" : "rgba(150,150,150," + a.toFixed(3) + ")";
    },

    fmtV: function (v) {
      if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
      return (Math.round(v * 10) / 10).toFixed(1);
    },
    fmtK: function (v) {
      if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
      return LR.fmtF(v, 2).replace("0.", ".");
    },

    // draw a numeric grid; opts: {fill(v), num(v)→string|null, numColor(v), stroke, font}
    grid: function (ctx, x0, y0, cell, g, o) {
      o = o || {};
      ctx.save();
      ctx.font = o.font || "600 " + Math.max(9, Math.min(13, Math.round(cell * 0.4))) + "px 'JetBrains Mono', Menlo, monospace";
      ctx.textAlign = "center";
      for (let r = 0; r < g.length; r++) {
        for (let c = 0; c < g[r].length; c++) {
          const v = g[r][c];
          const x = x0 + c * cell, y = y0 + r * cell;
          ctx.fillStyle = o.fill ? o.fill(v, r, c) : CNN.gray(v);
          ctx.fillRect(x, y, cell - 1, cell - 1);
          ctx.strokeStyle = o.stroke || "#d8d8d8";
          ctx.lineWidth = 0.8;
          ctx.strokeRect(x, y, cell - 1, cell - 1);
          if (o.num) {
            const t = o.num(v, r, c);
            if (t !== null && t !== undefined) {
              ctx.fillStyle = o.numColor ? o.numColor(v, r, c) : C.text;
              ctx.fillText(t, x + cell / 2, y + cell / 2 + 4);
            }
          }
        }
      }
      ctx.restore();
    },

    label: function (ctx, x, y, text, color) {
      ctx.save();
      ctx.font = "600 12px Inter, -apple-system, sans-serif";
      ctx.fillStyle = color || C.muted;
      ctx.textAlign = "left";
      ctx.fillText(text, x, y);
      ctx.restore();
    },
  });

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — the parameter-count shock
     FC params = (side²·3)·width + width  vs  conv = 3²·3·width + width.
     Both computed live from the section-4 formulas.
     ════════════════════════════════════════════════════════════ */
  LR.figs.params = function (mount) {
    LR.header(
      mount,
      "Count the weights before training anything",
      "One hidden layer on an RGB image. Fully connected: every unit sees every pixel. Convolutional: the same number of units as 3×3 shared filters."
    );

    let side = 200, width = 1000;

    const bar = LR.controls(mount);
    const sSide = LR.slider(bar, "image side (px)", 20, 500, 10, side, function (v) {
      side = Math.round(v); draw();
    }, (v) => String(Math.round(v)));
    LR.slider(bar, "layer width (units / filters)", 100, 2000, 100, width, function (v) {
      width = Math.round(v); draw();
    }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Fully connected parameter count growing with image size while the convolutional count stays flat",
    });
    const ro = LR.readout(mount, [
      { k: "in", label: "input values (side²·3)" },
      { k: "fc", label: "fully connected params" },
      { k: "conv", label: "conv params (K=3)" },
      { k: "ratio", label: "FC ÷ conv" },
    ]);

    function fcCount(n, h) { return 3 * n * n * h + h; }        // weights + biases
    function convCount(h) { return 9 * 3 * h + h; }             // K²·J·M + M

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 64, y0: 20, w: W - 92, h: H - 76,
        xmin: 0, xmax: 500, ymin: 3, ymax: 10,
        xlabel: "image side (pixels)", ylabel: "parameters (log₁₀ scale)",
      };
      const { sx, sy } = LR.plot(ctx, P);

      // FC curve across image sizes at the current width
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let n = 20; n <= 500; n += 5) {
        const y = Math.log10(fcCount(n, width));
        n === 20 ? ctx.moveTo(sx(n), sy(y)) : ctx.lineTo(sx(n), sy(y));
      }
      ctx.stroke();

      // conv line: flat, no image-size dependence
      const cy = Math.log10(convCount(width));
      ctx.strokeStyle = C.green; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(sx(20), sy(cy)); ctx.lineTo(sx(500), sy(cy)); ctx.stroke();

      // markers at the current image side
      const fcv = fcCount(side, width), cvv = convCount(width);
      LR.dot(ctx, sx(side), sy(Math.log10(fcv)), 6, C.orange, "#ffffff");
      LR.dot(ctx, sx(side), sy(cy), 6, C.green, "#ffffff");
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(0,0,0,0.25)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(side), sy(3)); ctx.lineTo(sx(side), sy(Math.log10(fcv))); ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = "600 12.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.orange;
      ctx.fillText("fully connected: 3·side²·width + width", P.x0 + 12, sy(Math.log10(fcCount(430, width))) - 12);
      ctx.fillStyle = C.green;
      ctx.fillText("convolutional: 3²·3·width + width (flat)", P.x0 + 12, sy(cy) - 10);

      ro.set("in", (3 * side * side).toLocaleString("en-CA"));
      ro.set("fc", fcv.toLocaleString("en-CA"), C.orange);
      ro.set("conv", cvv.toLocaleString("en-CA"), C.green);
      ro.set("ratio", Math.round(fcv / cvv).toLocaleString("en-CA") + "× more", C.red);
    }
    draw();
    sSide.set(side);
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — weights as pattern detectors: z = wᵀx + b
     Editable patch and filter as accessible button cells.
     ════════════════════════════════════════════════════════════ */
  LR.figs.detector = function (mount) {
    LR.header(
      mount,
      "The dot product is a match score",
      "Click cells to edit. Patch values cycle 0 → 1 → 3 → 5; filter weights cycle −1 → 0 → 1. The response is recomputed term by term."
    );

    const XCYC = [0, 1, 3, 5], WCYC = [-1, 0, 1];
    let x = [5, 5, 5, 5, 5, 5];
    let w = [1, 1, 1, 1, 1, 1];
    let b = 0;

    const bar = LR.controls(mount);
    LR.button(bar, "x₁ = [5,5,5,5,5,5]", function () { x = [5, 5, 5, 5, 5, 5]; sync(); }, "small");
    LR.button(bar, "x₂ = [0,0,0,5,5,5]", function () { x = [0, 0, 0, 5, 5, 5]; sync(); }, "small");
    LR.button(bar, "w = all ones", function () { w = [1, 1, 1, 1, 1, 1]; sync(); }, "small");
    LR.button(bar, "w = step detector", function () { w = [-1, -1, -1, 1, 1, 1]; sync(); }, "small");
    LR.slider(bar, "bias b", -20, 20, 1, b, function (v) { b = Math.round(v); sync(); }, (v) => String(Math.round(v)));

    function cellRow(vals, cyc, label, kind) {
      const row = LR.el("div", "vec-cells");
      row.appendChild(LR.el("span", "vec-lab", label));
      const btns = vals.map(function (_, i) {
        const btn = LR.el("button", "vec-cell");
        btn.type = "button";
        btn.setAttribute("aria-label", kind + " value " + (i + 1) + " of 6, click to cycle");
        btn.addEventListener("click", function () {
          const arr = kind === "patch" ? x : w;
          let idx = cyc.indexOf(arr[i]);
          if (idx < 0) idx = 0;
          arr[i] = cyc[(idx + 1) % cyc.length];
          sync();
        });
        row.appendChild(btn);
        return btn;
      });
      mount.appendChild(row);
      return btns;
    }

    const xBtns = cellRow(x, XCYC, "x =", "patch");
    const wBtns = cellRow(w, WCYC, "w =", "filter");

    const calc = LR.el("div", "det-calc");
    mount.appendChild(calc);

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 74, {
      aria: "Bar showing the detector response z between minus thirty and plus fifty",
    });
    const ro = LR.readout(mount, [{ k: "z", label: "response z = wᵀx + b" }]);
    const msg = LR.msg(mount);

    function sync() {
      // paint cells
      xBtns.forEach(function (btn, i) {
        btn.textContent = String(x[i]);
        btn.style.background = CNN.gray(x[i] / 5);
        btn.style.color = x[i] >= 3 ? "#ffffff" : C.text;
      });
      wBtns.forEach(function (btn, i) {
        btn.textContent = String(w[i]);
        btn.style.background = w[i] === 0 ? "#ffffff" : w[i] > 0 ? "rgba(26,26,26,0.75)" : "rgba(150,150,150,0.75)";
        btn.style.color = w[i] === 0 ? C.text : "#ffffff";
      });

      // real computation, term by term
      let z = b;
      const terms = x.map(function (xi, i) { z += w[i] * xi; return "(" + w[i] + ")(" + xi + ")"; });
      calc.innerHTML =
        "z = " + terms.join(" + ") + (b !== 0 ? " + " + b : "") +
        " = <b style='color:" + (z >= 0 ? C.orange : C.purple) + "'>" + z + "</b>";

      // bar
      ctx.clearRect(0, 0, W, H);
      const zmin = -30, zmax = 50;
      const px = (v) => 30 + ((v - zmin) / (zmax - zmin)) * (W - 60);
      ctx.strokeStyle = "#cccccc"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(30, 40); ctx.lineTo(W - 30, 40); ctx.stroke();
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(px(0), 26); ctx.lineTo(px(0), 54); ctx.stroke();
      ctx.fillStyle = z >= 0 ? "rgba(26,26,26,0.8)" : "rgba(150,150,150,0.8)";
      const x0p = px(Math.min(0, z)), x1p = px(Math.max(0, z));
      ctx.fillRect(x0p, 30, Math.max(1.5, x1p - x0p), 20);
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "center";
      ctx.fillText("0", px(0), 68);
      ctx.fillText("30 (perfect match for w = 1s, x = 5s)", px(30), 68);

      ro.set("z", String(z), z >= 0 ? C.orange : C.purple);

      const dotOnly = z - b;
      if (String(w) === "1,1,1,1,1,1" && String(x) === "5,5,5,5,5,5") {
        msg.show("The lecture's aligned patch: wᵀx = 30, the maximum this filter can score. The patch points exactly along w.", "good");
      } else if (String(w) === "1,1,1,1,1,1" && String(x) === "0,0,0,5,5,5") {
        msg.show("The half-lit patch scores wᵀx = 15, half the aligned response. Misalignment costs activation.", "info");
      } else if (dotOnly === 0) {
        msg.show("wᵀx = 0: this patch is invisible to this filter. (Try the step detector on a flat patch: perfectly balanced.)", "info");
      } else {
        msg.hide();
      }
    }
    sync();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the 1-D convolution stepper
     (x * w)[t] = Σ_τ x[t−τ] w[τ], stepped one output at a time.
     ════════════════════════════════════════════════════════════ */
  LR.figs.conv1d = function (mount) {
    LR.header(
      mount,
      "Slide, multiply, add",
      "The kernel is flipped (that is the t − τ) and slid along the zero-padded input. Each stop produces one output: the sum of the aligned products."
    );

    const PRESETS = {
      A: { x: [1, 1], w: [2, -2], label: "[1, 1] * [2, −2]" },
      B: { x: [1, 2, 3, 1], w: [1, 0, -1], label: "[1, 2, 3, 1] * [1, 0, −1]" },
    };
    let cur = "A";
    let t = -1;         // current output index; −1 = not started
    let playing = null;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const pBtns = {};
    Object.keys(PRESETS).forEach(function (key) {
      pBtns[key] = LR.button(group, PRESETS[key].label, function () {
        cur = key; t = -1; stopPlay();
        for (const k2 in pBtns) pBtns[k2].classList.toggle("on", k2 === cur);
        draw();
      }, "small" + (key === cur ? " on" : ""));
    });
    LR.button(bar, "Step ▸", step, "primary");
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, LR.reducedMotion ? 1300 : 900);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); t = -1; draw(); });
    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Step-through animation of a one-dimensional convolution: flipped kernel sliding along the padded input, products and running sum shown at each step",
    });

    function outLen() {
      const p = PRESETS[cur];
      return p.x.length + p.w.length - 1;
    }
    function yAt(ti) {
      const p = PRESETS[cur];
      let s = 0;
      for (let tau = 0; tau < p.w.length; tau++) {
        const xi = ti - tau;
        if (xi >= 0 && xi < p.x.length) s += p.x[xi] * p.w[tau];
      }
      return s;
    }

    function step() {
      if (t >= outLen() - 1) { stopPlay(); return; }
      t += 1;
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const p = PRESETS[cur];
      const n = p.x.length, k = p.w.length, m = outLen();
      const cell = 52;
      // padded x occupies indices −(k−1) … n−1+(k−1)
      const total = n + 2 * (k - 1);
      const x0 = (W - total * cell) / 2;
      const xRowY = 46, wRowY = 118, outRowY = 246;
      const colX = (idx) => x0 + (idx + (k - 1)) * cell;   // idx in x coordinates

      CNN.label(ctx, x0, xRowY - 12, "input x (zero-padded)");
      ctx.save();
      ctx.font = "700 15px 'JetBrains Mono', Menlo, monospace";
      ctx.textAlign = "center";
      for (let i = -(k - 1); i <= n - 1 + (k - 1); i++) {
        const inside = i >= 0 && i < n;
        const v = inside ? p.x[i] : 0;
        const active = t >= 0 && i <= t && i > t - k;
        ctx.fillStyle = active ? "rgba(26,26,26,0.14)" : inside ? "#ffffff" : "#fafafa";
        ctx.fillRect(colX(i), xRowY, cell - 2, 44);
        ctx.strokeStyle = active ? C.orange : "#d0d0d0";
        ctx.lineWidth = active ? 2 : 1;
        ctx.strokeRect(colX(i), xRowY, cell - 2, 44);
        ctx.fillStyle = inside ? C.text : "#b5b5b5";
        ctx.fillText(String(v), colX(i) + cell / 2 - 1, xRowY + 28);
      }

      // flipped kernel row, aligned so that w[τ] sits under x[t−τ]
      if (t >= 0) {
        CNN.label(ctx, x0, wRowY - 12, "kernel w, flipped, at stop t = " + t);
        for (let tau = 0; tau < k; tau++) {
          const i = t - tau;   // x-index this kernel entry multiplies
          ctx.fillStyle = "rgba(150,150,150,0.12)";
          ctx.fillRect(colX(i), wRowY, cell - 2, 44);
          ctx.strokeStyle = C.purple; ctx.lineWidth = 1.6;
          ctx.strokeRect(colX(i), wRowY, cell - 2, 44);
          ctx.fillStyle = C.purple;
          ctx.fillText("w[" + tau + "]=" + p.w[tau], colX(i) + cell / 2 - 1, wRowY + 28);
          LR.arrow(ctx, colX(i) + cell / 2, wRowY - 2, colX(i) + cell / 2, xRowY + 48, "rgba(150,150,150,0.5)", 1.4);
        }

        // the products, written out
        const parts = [];
        let sum = 0;
        for (let tau = 0; tau < k; tau++) {
          const xi = t - tau;
          const xv = xi >= 0 && xi < n ? p.x[xi] : 0;
          parts.push("(" + xv + ")(" + p.w[tau] + ")");
          sum += xv * p.w[tau];
        }
        ctx.font = "600 14.5px 'JetBrains Mono', Menlo, monospace";
        ctx.textAlign = "center";
        ctx.fillStyle = C.text;
        ctx.fillText("y[" + t + "] = " + parts.join(" + ") + " = " + sum, W / 2, wRowY + 82);
      } else {
        CNN.label(ctx, x0, wRowY + 20, "press Step to place the kernel at its first stop");
      }
      ctx.restore();

      // output row
      CNN.label(ctx, x0, outRowY - 12, "output y = x * w  (" + n + " + " + k + " − 1 = " + m + " entries)");
      ctx.save();
      ctx.font = "700 15px 'JetBrains Mono', Menlo, monospace";
      ctx.textAlign = "center";
      const ox0 = (W - m * cell) / 2;
      for (let i = 0; i < m; i++) {
        const done = t >= i;
        ctx.fillStyle = i === t ? "rgba(26,26,26,0.16)" : "#ffffff";
        ctx.fillRect(ox0 + i * cell, outRowY, cell - 2, 44);
        ctx.strokeStyle = i === t ? C.orange : "#d0d0d0";
        ctx.lineWidth = i === t ? 2 : 1;
        ctx.strokeRect(ox0 + i * cell, outRowY, cell - 2, 44);
        ctx.fillStyle = done ? C.text : "#c9c9c9";
        ctx.fillText(done ? String(yAt(i)) : "·", ox0 + i * cell + cell / 2 - 1, outRowY + 28);
      }
      ctx.restore();
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.2 — 2-D convolution: the signature figure
     Drag or sweep a 3×3 kernel over the 8×8 image; every product,
     sum, and feature-map value is computed live.
     ════════════════════════════════════════════════════════════ */
  LR.figs.conv2d = function (mount) {
    LR.header(
      mount,
      "One kernel, every location",
      "Drag the highlighted patch across the image, or press Sweep. Click kernel cells to edit them (values cycle −2…2). The feature map fills with real sums."
    );

    const IMG = CNN.IMG8;
    const N = 8, K = 3, NO = N - K + 1;   // 6×6 output, stride 1, no padding

    let ker = CNN.KER.vedge.k.map((r) => r.slice());
    let kerName = "Vertical edge";
    let pos = 7;                  // output index 0..35 (i*6+j), start inside
    let revealed = new Array(NO * NO).fill(false);
    let playing = null;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const kBtns = {};
    Object.keys(CNN.KER).forEach(function (key) {
      kBtns[key] = LR.button(group, CNN.KER[key].name, function () {
        ker = CNN.KER[key].k.map((r) => r.slice());
        kerName = CNN.KER[key].name;
        for (const k2 in kBtns) kBtns[k2].classList.toggle("on", k2 === key);
        revealed = new Array(NO * NO).fill(false);
        revealed[pos] = true;
        draw();
      }, "small" + (key === "vedge" ? " on" : ""));
    });
    LR.button(bar, "Step ▸", function () { stopPlay(); stepOnce(); }, "");
    const playBtn = LR.button(bar, "Sweep ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(stepOnce, LR.reducedMotion ? 700 : 220);
    }, "primary");
    LR.button(bar, "Reset ⟲", function () {
      stopPlay(); pos = 0; revealed = new Array(NO * NO).fill(false); revealed[0] = true; draw();
    });
    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Sweep ▸▸";
    }
    function stepOnce() {
      if (!revealed[pos]) { revealed[pos] = true; draw(); return; }
      if (pos >= NO * NO - 1 && revealed[NO * NO - 1]) { stopPlay(); return; }
      pos = Math.min(NO * NO - 1, pos + 1);
      revealed[pos] = true;
      draw();
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 480, {
      aria: "Two-dimensional convolution: a draggable three by three kernel over an eight by eight image, with elementwise products, their sum, and the feature map filling cell by cell",
    });
    const ro = LR.readout(mount, [
      { k: "pos", label: "kernel at output (i, j)" },
      { k: "sum", label: "patch · kernel" },
      { k: "done", label: "feature map" },
    ]);

    // layout
    const CI = 32, IX = 36, IY = 78;            // input grid
    const CK = 36, KX = 350, KY = 78;           // kernel
    const PX = 350, PY = 250;                   // products
    const CO = 36, OX = 560, OY = 78;           // output

    const KCYC = [-2, -1, 0, 1, 2];

    function outVals() { return CNN.conv2d(IMG, ker, 1, 0); }

    LR.drag(cv, W, H, {
      hit: function (p) {
        // kernel edit click
        if (p.x >= KX && p.x < KX + K * CK && p.y >= KY && p.y < KY + K * CK) {
          const c = Math.floor((p.x - KX) / CK), r = Math.floor((p.y - KY) / CK);
          let idx = KCYC.indexOf(ker[r][c]);
          if (idx < 0) {
            // snap non-cycle values (e.g. blur 1/9) to the nearest cycle entry
            let best = 0, bd = Infinity;
            KCYC.forEach(function (v, i2) { const d = Math.abs(v - ker[r][c]); if (d < bd) { bd = d; best = i2; } });
            idx = best;
          }
          ker[r][c] = KCYC[(idx + 1) % KCYC.length];
          kerName = "Custom";
          for (const k2 in kBtns) kBtns[k2].classList.remove("on");
          draw();
          return false; // no drag
        }
        return p.x >= IX && p.x < IX + N * CI && p.y >= IY && p.y < IY + N * CI;
      },
      down: movePatch,
      move: movePatch,
    });
    function movePatch(p) {
      const c = Math.floor((p.x - IX) / CI), r = Math.floor((p.y - IY) / CI);
      const i = Math.max(0, Math.min(NO - 1, r - 1));
      const j = Math.max(0, Math.min(NO - 1, c - 1));
      stopPlay();
      pos = i * NO + j;
      revealed[pos] = true;
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const out = outVals();
      const vmax = Math.max(CNN.maxAbs(out), 1e-9);
      const i = Math.floor(pos / NO), j = pos % NO;

      // input
      CNN.label(ctx, IX, IY - 10, "input image, 8 × 8");
      CNN.grid(ctx, IX, IY, CI, IMG, {
        num: (v) => CNN.fmtV(v),
        numColor: (v) => (v > 0.5 ? "#ffffff" : "#9a9a9a"),
      });
      // receptive patch highlight
      ctx.strokeStyle = C.orange; ctx.lineWidth = 3;
      ctx.strokeRect(IX + j * CI - 1, IY + i * CI - 1, K * CI, K * CI);

      // kernel
      CNN.label(ctx, KX, KY - 10, "kernel: " + kerName + " (click to edit)");
      CNN.grid(ctx, KX, KY, CK, ker, {
        fill: (v) => CNN.ramp(v, 2),
        num: (v) => CNN.fmtK(v),
        numColor: () => C.text,
        stroke: "#c9c9c9",
      });

      // elementwise products for the current patch
      const prods = [];
      let sum = 0;
      for (let a = 0; a < K; a++) {
        const row = [];
        for (let b2 = 0; b2 < K; b2++) {
          const v = IMG[i + a][j + b2] * ker[a][b2];
          row.push(v);
          sum += v;
        }
        prods.push(row);
      }
      CNN.label(ctx, PX, PY - 10, "elementwise products");
      const pmax = Math.max(CNN.maxAbs(prods), 1e-9);
      CNN.grid(ctx, PX, PY, CK, prods, {
        fill: (v) => CNN.ramp(v, pmax),
        num: (v) => CNN.fmtV(v),
        numColor: () => C.text,
        stroke: "#c9c9c9",
      });
      ctx.save();
      ctx.font = "700 15px 'JetBrains Mono', Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = sum >= 0 ? C.orange : C.purple;
      ctx.fillText("sum = " + CNN.fmtV(sum), PX, PY + K * CK + 26);
      ctx.restore();

      // arrow products → output cell
      LR.arrow(ctx, PX + K * CK + 14, PY + (K * CK) / 2, OX + j * CO - 6, OY + i * CO + CO / 2, "rgba(26,26,26,0.55)", 1.8);

      // output feature map
      CNN.label(ctx, OX, OY - 10, "feature map, 6 × 6");
      CNN.grid(ctx, OX, OY, CO, out, {
        fill: function (v, r, c) {
          return revealed[r * NO + c] ? CNN.ramp(v, vmax) : "#fafafa";
        },
        num: function (v, r, c) { return revealed[r * NO + c] ? CNN.fmtV(v) : "·"; },
        numColor: function (v, r, c) { return revealed[r * NO + c] ? C.text : "#c9c9c9"; },
      });
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.5;
      ctx.strokeRect(OX + j * CO - 1, OY + i * CO - 1, CO, CO);

      // legend for the ramp
      ctx.save();
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.orange; ctx.fillText("■ positive response", OX, OY + NO * CO + 20);
      ctx.fillStyle = C.purple; ctx.fillText("■ negative response", OX, OY + NO * CO + 36);
      ctx.restore();

      const nDone = revealed.filter(Boolean).length;
      ro.set("pos", "(" + i + ", " + j + ")");
      ro.set("sum", CNN.fmtV(sum), sum >= 0 ? C.orange : C.purple);
      ro.set("done", nDone + " / " + NO * NO + " cells filled", nDone === NO * NO ? C.green : undefined);
    }

    revealed[pos] = true;
    draw();
  };
})();
