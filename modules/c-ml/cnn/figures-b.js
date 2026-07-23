/* ══════════════════════════════════════════════════════════════
   figures-b.js — CNN lesson figures, sections 3.3–5
   Fig 3.3 padstride · Fig 4.1 filterbank · Fig 4.2 codeex
   Fig 5.1 nonlinear · Fig 5.2 pooling · Fig 5.3 rfgrowth
   Uses the shared convolution core (LR.cnn) from figures-a.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const CNN = LR.cnn;

  /* ════════════════════════════════════════════════════════════
     Fig 3.3 — the padding-and-stride explorer
     Ŵ = (W − K + 2P)/S + 1 computed live over a fixed 7×7 input.
     ════════════════════════════════════════════════════════════ */
  LR.figs.padstride = function (mount) {
    LR.header(
      mount,
      "Feel the sizing formula",
      "A 7×7 input, your choice of kernel size, padding, and stride. The kernel steps through every position it can legally take; each stop fills one output cell."
    );

    // fixed 7×7 input: a bright 3×3 block, values 0/1 (kernel = all ones,
    // so each output is the count of bright pixels under it: easy to verify)
    const IMG = [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
    ];
    const Wn = 7;

    let K = 3, P = 1, S = 1;
    let idx = 0, playing = null;

    const bar = LR.controls(mount);
    LR.slider(bar, "K (kernel)", 1, 5, 1, K, function (v) { K = Math.round(v); reset(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "P (padding)", 0, 2, 1, P, function (v) { P = Math.round(v); reset(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "S (stride)", 1, 3, 1, S, function (v) { S = Math.round(v); reset(); }, (v) => String(Math.round(v)));
    LR.button(bar, "Step ▸", function () { stopPlay(); step(); }, "");
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, LR.reducedMotion ? 650 : 240);
    }, "primary");
    LR.button(bar, "Reset ⟲", reset);
    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Padding and stride explorer: kernel positions animated over a padded seven by seven input with the live output size",
    });
    const ro = LR.readout(mount, [
      { k: "formula", label: "Ŵ = (W − K + 2P)/S + 1" },
      { k: "out", label: "output size" },
    ]);
    const msg = LR.msg(mount);

    function nPos() { return Math.max(0, Math.floor((Wn + 2 * P - K) / S)) + 1; }

    function step() {
      const n = nPos();
      if (idx >= n * n - 1) { stopPlay(); return; }
      idx += 1;
      draw();
    }
    function reset() {
      stopPlay();
      idx = 0;
      draw();
    }

    function patchSum(i, j) {
      const im = CNN.pad(IMG, P);
      let s = 0;
      for (let a = 0; a < K; a++)
        for (let b = 0; b < K; b++) s += im[i * S + a][j * S + b];
      return s;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const n = nPos();
      const exact = CNN.outSize(Wn, K, P, S);
      const whole = Math.abs(exact - Math.round(exact)) < 1e-9;
      const i = Math.floor(idx / n), j = idx % n;

      // padded input, left
      const total = Wn + 2 * P;
      const ci = Math.min(30, Math.floor(300 / total));
      const IXX = 40, IYY = 70;
      CNN.label(ctx, IXX, IYY - 12, "input 7 × 7, padding " + P + " (grey ring = zeros)");
      const im = CNN.pad(IMG, P);
      CNN.grid(ctx, IXX, IYY, ci, im, {
        fill: function (v, r, c) {
          const inPad = r < P || c < P || r >= total - P || c >= total - P;
          return inPad ? "#f3f3f3" : CNN.gray(v);
        },
        num: (v) => (ci >= 22 ? CNN.fmtV(v) : null),
        numColor: (v) => (v > 0.5 ? "#ffffff" : "#adadad"),
      });
      // kernel box
      ctx.strokeStyle = C.orange; ctx.lineWidth = 3;
      ctx.strokeRect(IXX + j * S * ci - 1, IYY + i * S * ci - 1, K * ci, K * ci);

      // output grid, right
      const co = Math.min(30, Math.floor(280 / Math.max(1, n)));
      const OXX = 480, OYY = 70;
      CNN.label(ctx, OXX, OYY - 12, "output " + n + " × " + n + " (values = bright pixels seen)");
      const outGrid = [];
      for (let r = 0; r < n; r++) {
        const row = [];
        for (let c = 0; c < n; c++) row.push(r * n + c <= idx ? patchSum(r, c) : null);
        outGrid.push(row);
      }
      const filledMax = Math.max(1, K * K);
      CNN.grid(ctx, OXX, OYY, co, outGrid, {
        fill: (v) => (v === null ? "#fafafa" : CNN.ramp(v, filledMax)),
        num: (v) => (co >= 20 ? (v === null ? "·" : CNN.fmtV(v)) : null),
        numColor: (v) => (v === null ? "#c9c9c9" : C.text),
      });
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.5;
      ctx.strokeRect(OXX + j * co - 1, OYY + i * co - 1, co, co);

      LR.arrow(ctx, IXX + total * ci + 18, IYY + (total * ci) / 2, OXX - 18, OYY + (n * co) / 2, "rgba(0,0,0,0.3)", 2);

      const fstr = "(" + Wn + " − " + K + " + 2·" + P + ")/" + S + " + 1 = " + LR.fmt(exact, 2);
      ro.set("formula", fstr, whole ? C.green : C.amber);
      ro.set("out", n + " × " + n + "  (" + n * n + " kernel stops)", C.text);
      if (!whole) {
        msg.show("The formula gives " + LR.fmt(exact, 2) + ", not a whole number: with stride " + S + " the last position does not land flush against the edge. Implementations floor it, so " + n + " positions fit per row.", "info");
      } else {
        msg.hide();
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — filter bank → feature-map stack
     Three real convolutions of the same input; depth out = M.
     ════════════════════════════════════════════════════════════ */
  LR.figs.filterbank = function (mount) {
    LR.header(
      mount,
      "One layer, three detectors, three maps",
      "The same 8×8 input convolved with three filters (all values computed live). The maps stack into a 6×6×3 volume: the next layer's input."
    );

    const IMG = CNN.IMG8;
    const FILTERS = [
      { key: "vedge", color: C.orange },
      { key: "hedge", color: C.purple },
      { key: "blur", color: C.green },
    ];
    const maps = FILTERS.map((f) => CNN.conv2d(IMG, CNN.KER[f.key].k, 1, 0));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 480, {
      aria: "Filter bank: three kernels convolve one input image, producing three feature maps that stack into a volume of depth three",
    });
    const ro = LR.readout(mount, [
      { k: "depth", label: "output depth I" },
      { k: "kd", label: "kernel depth" },
      { k: "ranges", label: "map value ranges" },
    ]);

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // input, left middle
      const CI = 24, IXX = 24, IYY = 140;
      CNN.label(ctx, IXX, IYY - 12, "input, 8 × 8 × 1");
      CNN.grid(ctx, IXX, IYY, CI, IMG, {});

      const rows = [46, 196, 346];
      const KX = 288, MX = 420, CK = 20, CM = 20;

      FILTERS.forEach(function (f, m) {
        const y = rows[m];
        const ker = CNN.KER[f.key].k;
        const vmax = Math.max(CNN.maxAbs(maps[m]), 1e-9);

        // arrow input → kernel
        LR.arrow(ctx, IXX + 8 * CI + 10, IYY + 4 * CI, KX - 12, y + 1.5 * CK, "rgba(0,0,0,0.25)", 1.6);

        CNN.label(ctx, KX, y - 8, "filter " + (m + 1), f.color);
        CNN.grid(ctx, KX, y, CK, ker, {
          fill: (v) => CNN.ramp(v, Math.max(CNN.maxAbs(ker), 1e-9)),
          num: (v) => CNN.fmtK(v),
          numColor: () => C.text,
          stroke: "#c9c9c9",
          font: "600 9px 'JetBrains Mono', Menlo, monospace",
        });
        CNN.label(ctx, KX - 8, y + 3 * CK + 16, CNN.KER[f.key].name, f.color);

        LR.arrow(ctx, KX + 3 * CK + 10, y + 1.5 * CK, MX - 10, y + 1.5 * CK, f.color, 1.8);

        CNN.label(ctx, MX, y - 8, "map " + (m + 1) + " · 6 × 6");
        CNN.grid(ctx, MX, y, CM, maps[m], {
          fill: (v) => CNN.ramp(v, vmax),
          stroke: "#e2e2e2",
        });

        // arrow map → its slice in the stack
        LR.arrow(ctx, MX + 6 * CM + 8, y + 3 * CM, 596 + m * 14 + 8, 160 + m * 14 + 30, "rgba(0,0,0,0.22)", 1.4);
      });

      // the stack: three offset slices = one 6×6×3 volume
      const SX = 596, SY = 160, CS = 22;
      CNN.label(ctx, SX - 4, SY - 22, "stacked: 6 × 6 × 3 volume");
      for (let m = FILTERS.length - 1; m >= 0; m--) {
        const off = m * 14;
        const vmax = Math.max(CNN.maxAbs(maps[m]), 1e-9);
        CNN.grid(ctx, SX + off, SY + off, CS, maps[m], {
          fill: (v) => CNN.ramp(v, vmax),
          stroke: "#cfcfcf",
        });
        ctx.strokeStyle = FILTERS[m].color;
        ctx.lineWidth = 2;
        ctx.strokeRect(SX + off - 1, SY + off - 1, 6 * CS + 1, 6 * CS + 1);
      }
      ctx.save();
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("depth = number of filters", SX - 4, SY + 6 * CS + 46);
      ctx.fillText("next layer's kernels: K × K × 3", SX - 4, SY + 6 * CS + 64);
      ctx.restore();

      const ranges = maps
        .map(function (mp, m2) {
          let lo = Infinity, hi = -Infinity;
          mp.forEach((r) => r.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
          return "[" + CNN.fmtV(lo) + ", " + CNN.fmtV(hi) + "]";
        })
        .join("  ");
      ro.set("depth", "3 (= M, the number of filters)", C.orange);
      ro.set("kd", "1 (= J, the input depth; RGB would force 3)");
      ro.set("ranges", ranges);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.2 — interactive conv2d code exercise
     The nested-loop conv2d really runs on a fixed 6×6 image.
     ════════════════════════════════════════════════════════════ */
  LR.figs.codeex = function (mount) {
    LR.header(
      mount,
      "Run conv2d yourself",
      "Pick a kernel, padding, and stride; the nested loops from the lesson execute against the fixed image and print the real output."
    );

    const IMG6 = [
      [0, 0, 0, 0, 0, 0],
      [0, 9, 9, 9, 0, 0],
      [0, 9, 9, 9, 0, 0],
      [0, 9, 9, 9, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
    ];
    const CHOICES = {
      "vertical edge": CNN.KER.vedge.k,
      "blur": CNN.KER.blur.k,
      "sharpen": CNN.KER.sharpen.k,
    };

    const CODE =
      "img = [[0,0,0,0,0,0],\n" +
      "       [0,9,9,9,0,0],\n" +
      "       [0,9,9,9,0,0],\n" +
      "       [0,9,9,9,0,0],\n" +
      "       [0,0,0,0,0,0],\n" +
      "       [0,0,0,0,0,0]]\n" +
      "\n" +
      "kernels = {\n" +
      "  'vertical edge': [[1,0,-1],[1,0,-1],[1,0,-1]],\n" +
      "  'blur':          [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],\n" +
      "  'sharpen':       [[0,-1,0],[-1,5,-1],[0,-1,0]],\n" +
      "}\n" +
      "\n" +
      "out = conv2d(img, kernels[choice], stride=S, pad=P)   # the loops above";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);

    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);

    let choice = "vertical edge", padv = 0, stridev = 1;
    function makeSelect(labelText, options, oninput) {
      const lab = LR.el("label", "", labelText + " ");
      const sel = document.createElement("select");
      options.forEach(function (o) {
        const opt = document.createElement("option");
        opt.value = o; opt.textContent = o;
        sel.appendChild(opt);
      });
      sel.style.cssText = "font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid #cccccc";
      sel.setAttribute("aria-label", labelText);
      sel.addEventListener("change", function () { oninput(sel.value); });
      lab.appendChild(sel);
      controls.appendChild(lab);
      return sel;
    }
    makeSelect("kernel", Object.keys(CHOICES), (v) => { choice = v; });
    makeSelect("pad", ["0", "1"], (v) => { padv = parseInt(v, 10); });
    makeSelect("stride", ["1", "2"], (v) => { stridev = parseInt(v, 10); });
    LR.button(controls, "Run ▸", run, "primary small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; conv2d(img, kernels['vertical edge'], stride=1, pad=0)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    // mini canvas: input + latest output as heat grids
    const { cv, ctx, W, H } = LR.canvas(mount, 560, 220, {
      aria: "The fixed six by six input image and the most recent convolution output",
    });
    cv.style.marginTop = "14px";

    function fmtCell(v) {
      if (Math.abs(v - Math.round(v)) < 1e-9) {
        const s = String(Math.round(v));
        return s.length >= 4 ? s : (" ".repeat(4 - s.length) + s);
      }
      const s = v.toFixed(1);
      return s.length >= 4 ? s : (" ".repeat(4 - s.length) + s);
    }

    function run() {
      const ker = CHOICES[choice];
      const res = CNN.conv2d(IMG6, ker, stridev, padv);
      const exact = CNN.outSize(6, 3, padv, stridev);
      const whole = Math.abs(exact - Math.round(exact)) < 1e-9;
      const n = res.length;

      let text =
        "&gt;&gt;&gt; conv2d(img, kernels['" + choice + "'], stride=" + stridev + ", pad=" + padv + ")\n" +
        "output size: (6 − 3 + 2·" + padv + ")/" + stridev + " + 1 = " + LR.fmt(exact, 2) +
        (whole ? "" : "  → floors to " + n) + "  →  " + n + " × " + n + "\n";
      res.forEach(function (row, i) {
        text += (i === 0 ? "[[" : " [") + row.map(fmtCell).join(",") + (i === n - 1 ? "]]" : "],") + "\n";
      });
      out.innerHTML = text;
      draw(res);
    }

    function draw(res) {
      ctx.clearRect(0, 0, W, H);
      const CI = 28;
      CNN.label(ctx, 20, 24, "img (6 × 6)");
      CNN.grid(ctx, 20, 34, CI, IMG6, {
        fill: (v) => CNN.gray(v / 9),
        num: (v) => String(v),
        numColor: (v) => (v > 4 ? "#ffffff" : "#9a9a9a"),
      });
      if (res) {
        const vmax = Math.max(CNN.maxAbs(res), 1e-9);
        const co = Math.min(28, Math.floor(170 / res.length));
        CNN.label(ctx, 300, 24, "out (" + res.length + " × " + res.length + ")");
        CNN.grid(ctx, 300, 34, co, res, {
          fill: (v) => CNN.ramp(v, vmax),
          num: (v) => (co >= 24 ? CNN.fmtV(v) : null),
          numColor: () => C.text,
        });
        LR.arrow(ctx, 20 + 6 * CI + 14, 34 + 3 * CI, 292, 34 + 3 * CI, "rgba(0,0,0,0.3)", 2);
      }
    }
    draw(null);
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — why nonlinearity: the linear-collapse demo
     Two conv layers vs the single composed 5×5 kernel, both real.
     ════════════════════════════════════════════════════════════ */
  LR.figs.nonlinear = function (mount) {
    LR.header(
      mount,
      "Stack two convolutions, get one back",
      "Top path: edge detect, activation, blur (two 3×3 convolutions). Bottom path: one 5×5 kernel computed by convolving the two kernels together. Toggle the activation and compare."
    );

    const IMG = CNN.IMG8;
    const k1 = CNN.KER.vedge.k;   // produces negatives → ReLU actually bites
    const k2 = CNN.KER.blur.k;
    const kc = CNN.convFullK(k1, k2);   // the single equivalent kernel (5×5)

    let act = "identity";

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bI = LR.button(group, "activation: identity", function () {
      act = "identity"; bI.classList.add("on"); bR.classList.remove("on"); draw();
    }, "small on");
    const bR = LR.button(group, "activation: ReLU", function () {
      act = "relu"; bR.classList.add("on"); bI.classList.remove("on"); draw();
    }, "small");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Two stacked convolutions compared against a single composed kernel, with the maximum difference between the two outputs",
    });
    const ro = LR.readout(mount, [{ k: "diff", label: "max |two-pass − one-pass|" }]);
    const msg = LR.msg(mount);

    function draw() {
      ctx.clearRect(0, 0, W, H);

      const L1 = CNN.conv2d(IMG, k1, 1, 0);                    // 6×6
      const A1 = act === "relu" ? CNN.relu(L1) : L1;
      const L2 = CNN.conv2d(A1, k2, 1, 0);                     // 4×4
      const Ld = CNN.conv2d(IMG, kc, 1, 0);                    // 4×4 via one 5×5 kernel

      let diff = 0;
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) diff = Math.max(diff, Math.abs(L2[i][j] - Ld[i][j]));

      // ── top path ──
      const y1 = 64;
      CNN.label(ctx, 20, y1 - 14, "two-pass: conv (edge) → " + (act === "relu" ? "ReLU" : "identity") + " → conv (blur)", C.text);
      const CI = 17;
      CNN.grid(ctx, 20, y1, CI, IMG, {});
      LR.arrow(ctx, 20 + 8 * CI + 8, y1 + 4 * CI, 196, y1 + 4 * CI, "rgba(0,0,0,0.28)", 1.6);

      CNN.label(ctx, 200, y1 - 2, "3×3 edge");
      CNN.grid(ctx, 200, y1 + 6, 15, k1, { fill: (v) => CNN.ramp(v, Math.max(CNN.maxAbs(k1), 1e-9)), stroke: "#c9c9c9" });

      LR.arrow(ctx, 252, y1 + 4 * CI, 288, y1 + 4 * CI, "rgba(0,0,0,0.28)", 1.6);
      const a1max = Math.max(CNN.maxAbs(A1), 1e-9);
      CNN.label(ctx, 292, y1 - 2, act === "relu" ? "ReLU(z₁), 6×6" : "z₁, 6×6");
      CNN.grid(ctx, 292, y1 + 6, 19, A1, { fill: (v) => CNN.ramp(v, a1max), stroke: "#e2e2e2" });

      LR.arrow(ctx, 292 + 6 * 19 + 8, y1 + 4 * CI, 448, y1 + 4 * CI, "rgba(0,0,0,0.28)", 1.6);
      CNN.label(ctx, 452, y1 - 2, "3×3 blur");
      CNN.grid(ctx, 452, y1 + 6, 15, k2, { fill: (v) => CNN.ramp(v, Math.max(CNN.maxAbs(k2), 1e-9)), stroke: "#c9c9c9" });

      LR.arrow(ctx, 504, y1 + 4 * CI, 540, y1 + 4 * CI, "rgba(0,0,0,0.28)", 1.6);
      const l2max = Math.max(CNN.maxAbs(L2), CNN.maxAbs(Ld), 1e-9);
      CNN.label(ctx, 544, y1 - 2, "output, 4×4");
      CNN.grid(ctx, 544, y1 + 6, 34, L2, {
        fill: (v) => CNN.ramp(v, l2max),
        num: (v) => CNN.fmtV(v),
        numColor: () => C.text,
      });

      // ── bottom path ──
      const y2 = 292;
      CNN.label(ctx, 20, y2 - 14, "one-pass: a single 5×5 kernel = edge * blur (computed, not designed)", C.text);
      CNN.grid(ctx, 20, y2, CI, IMG, {});
      LR.arrow(ctx, 20 + 8 * CI + 8, y2 + 4 * CI, 268, y2 + 4 * CI, "rgba(0,0,0,0.28)", 1.6);

      const kcmax = Math.max(CNN.maxAbs(kc), 1e-9);
      CNN.label(ctx, 272, y2 - 2, "5×5 composed kernel");
      CNN.grid(ctx, 272, y2 + 6, 22, kc, {
        fill: (v) => CNN.ramp(v, kcmax),
        num: (v) => CNN.fmtK(v),
        numColor: () => C.text,
        stroke: "#c9c9c9",
        font: "600 8px 'JetBrains Mono', Menlo, monospace",
      });

      LR.arrow(ctx, 272 + 5 * 22 + 10, y2 + 4 * CI, 540, y2 + 4 * CI, "rgba(0,0,0,0.28)", 1.6);
      CNN.label(ctx, 544, y2 - 2, "output, 4×4");
      CNN.grid(ctx, 544, y2 + 6, 34, Ld, {
        fill: (v) => CNN.ramp(v, l2max),
        num: (v) => CNN.fmtV(v),
        numColor: () => C.text,
      });

      const exact = diff < 1e-9;
      ro.set("diff", exact ? "0 (exact, to machine precision)" : LR.fmtF(diff, 3), exact ? C.green : C.red);
      if (exact) {
        msg.show("With the identity activation the two paths agree everywhere: the two convolutions collapsed into one 5×5 convolution. Depth bought nothing.", "info");
      } else {
        msg.show("With ReLU between the layers the outputs disagree (max difference " + LR.fmtF(diff, 3) + "). No single kernel reproduces the two-pass result: the network is now genuinely nonlinear.", "good");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — pooling: max vs average, and translation invariance
     Shift the activation blob and count which pooled cells change.
     ════════════════════════════════════════════════════════════ */
  LR.figs.pooling = function (mount) {
    LR.header(
      mount,
      "Summarize, then stop caring where exactly",
      "An activation blob (a detector firing), shifted right pixel by pixel. Compare the pooled map against the unshifted reference: changed cells are outlined in red."
    );

    const N = 8;
    function makeMap(shift) {
      const m = Array.from({ length: N }, () => new Array(N).fill(0));
      // asymmetric blob so average pooling has no accidental symmetry to hide behind
      const bump = [[0.2, 0.5, 0.3], [0.6, 1.0, 0.7], [0.1, 0.4, 0.8]];
      for (let a = 0; a < 3; a++)
        for (let b = 0; b < 3; b++) m[1 + a][1 + shift + b] = bump[a][b];
      return m;
    }

    let mode = "max", size = 2, shift = 1;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bM = LR.button(group, "max pooling", function () {
      mode = "max"; bM.classList.add("on"); bA.classList.remove("on"); draw();
    }, "small on");
    const bA = LR.button(group, "average pooling", function () {
      mode = "avg"; bA.classList.add("on"); bM.classList.remove("on"); draw();
    }, "small");
    LR.slider(bar, "pool size", 2, 4, 2, size, function (v) { size = Math.round(v); draw(); }, (v) => Math.round(v) + "×" + Math.round(v));
    LR.slider(bar, "shift (px right)", 0, 3, 1, shift, function (v) { shift = Math.round(v); draw(); }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Pooling demo: an activation map with a shifting blob, its pooled reference, and the pooled shifted map with changed cells outlined",
    });
    const ro = LR.readout(mount, [
      { k: "max", label: "cells changed (max)" },
      { k: "avg", label: "cells changed (avg)" },
      { k: "pk", label: "strongest pooled response (max | avg)" },
    ]);
    const msg = LR.msg(mount);

    function countChanged(a, b) {
      let n = 0;
      for (let i = 0; i < a.length; i++)
        for (let j = 0; j < a.length; j++) if (Math.abs(a[i][j] - b[i][j]) > 1e-9) n++;
      return n;
    }
    function peak(m) {
      let mx = 0;
      m.forEach((r) => r.forEach((v) => { if (v > mx) mx = v; }));
      return mx;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const map0 = makeMap(0), mapS = makeMap(shift);
      const ref = CNN.pool(map0, size, mode), cur = CNN.pool(mapS, size, mode);
      const refMax = CNN.pool(map0, size, "max"), curMax = CNN.pool(mapS, size, "max");
      const refAvg = CNN.pool(map0, size, "avg"), curAvg = CNN.pool(mapS, size, "avg");
      const chMax = countChanged(refMax, curMax);
      const chAvg = countChanged(refAvg, curAvg);
      const n = ref.length;

      // activation map
      const CI = 26;
      CNN.label(ctx, 30, 52, "activation map (blob shifted " + shift + " px)");
      CNN.grid(ctx, 30, 62, CI, mapS, {
        fill: (v) => CNN.ramp(v, 1),
        num: (v) => (v > 0 ? CNN.fmtV(v) : null),
        numColor: () => C.text,
      });
      // pooling group gridlines
      ctx.strokeStyle = "rgba(0,0,0,0.45)";
      ctx.lineWidth = 1.6;
      for (let g = 0; g <= N; g += size) {
        ctx.beginPath(); ctx.moveTo(30 + g * CI, 62); ctx.lineTo(30 + g * CI, 62 + N * CI); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(30, 62 + g * CI); ctx.lineTo(30 + N * CI, 62 + g * CI); ctx.stroke();
      }

      const co = size === 2 ? 44 : 72;
      // reference pooled (shift 0)
      CNN.label(ctx, 330, 52, "pooled, unshifted (reference)");
      CNN.grid(ctx, 330, 62, co, ref, {
        fill: (v) => CNN.ramp(v, 1),
        num: (v) => LR.fmtF(v, 2),
        numColor: () => C.text,
      });

      // current pooled
      CNN.label(ctx, 580, 52, "pooled, shifted " + shift + " px (" + mode + ")");
      CNN.grid(ctx, 580, 62, co, cur, {
        fill: (v) => CNN.ramp(v, 1),
        num: (v) => LR.fmtF(v, 2),
        numColor: () => C.text,
      });
      // outline changed cells
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          if (Math.abs(cur[i][j] - ref[i][j]) > 1e-9) {
            ctx.strokeStyle = C.red; ctx.lineWidth = 2.5;
            ctx.strokeRect(580 + j * co, 62 + i * co, co - 1, co - 1);
          }

      const pkM = peak(curMax), pkA = peak(curAvg), pkA0 = peak(refAvg);
      ro.set("max", chMax + " of " + n * n, chMax === 0 ? C.green : C.amber);
      ro.set("avg", chAvg + " of " + n * n, chAvg === 0 ? C.green : C.red);
      ro.set("pk", LR.fmtF(pkM, 2) + " | " + LR.fmtF(pkA, 3), C.text);

      if (shift === 0) {
        msg.show("No shift, no change, by definition. Now slide the blob one pixel right and watch the counters and the peak responses.", "info");
      } else if (Math.abs(pkM - 1) < 1e-9 && Math.abs(pkA - pkA0) > 1e-9) {
        msg.show("The blob moved " + shift + " px. Max pooling's strongest response is still exactly 1.00: the detector keeps firing at full strength (and only " + chMax + " cell" + (chMax === 1 ? "" : "s") + " changed vs " + chAvg + " for average). Average's strongest response drifted from " + LR.fmtF(pkA0, 3) + " to " + LR.fmtF(pkA, 3) + " because every activation in the group dilutes it. That gap is approximate translation invariance.", "good");
      } else {
        msg.show("Max changed " + chMax + " cells, average changed " + chAvg + ". Max only reacts when the peak crosses a group boundary; average reacts to any mass moving. Bigger pooling groups tolerate bigger shifts (try pool size 4).", "info");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.3 — receptive-field growth through the stack
     conv 3×3 s1 → pool 2×2 s2 → conv 3×3 s1, regions computed
     exactly by back-projecting index ranges.
     ════════════════════════════════════════════════════════════ */
  LR.figs.rfgrowth = function (mount) {
    LR.header(
      mount,
      "What a deep unit can see",
      "Click any unit in any layer. The exact patch of the original image it depends on is computed by walking the architecture backwards."
    );

    // architecture: sizes and the op that produced each layer
    const LAYERS = [
      { name: "input", sub: "12 × 12", n: 12, cell: 15 },
      { name: "conv 3×3, stride 1", sub: "10 × 10", n: 10, cell: 15, K: 3, S: 1 },
      { name: "pool 2×2, stride 2", sub: "5 × 5", n: 5, cell: 24, K: 2, S: 2 },
      { name: "conv 3×3, stride 1", sub: "3 × 3", n: 3, cell: 30, K: 3, S: 1 },
    ];
    const XPOS = [24, 250, 470, 650];
    const YTOP = 60;

    let sel = { layer: 3, i: 1, j: 1 };

    const bar = LR.controls(mount);
    [1, 2, 3].forEach(function (d) {
      LR.button(bar, "centre unit, depth " + d, function () {
        const n = LAYERS[d].n;
        sel = { layer: d, i: Math.floor(n / 2), j: Math.floor(n / 2) };
        draw();
      }, "small" + (d === 3 ? "" : ""));
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 320, {
      aria: "Receptive field growth: click a unit in any layer to highlight the region of the original image it depends on",
    });
    const ro = LR.readout(mount, [
      { k: "unit", label: "selected unit" },
      { k: "rf", label: "receptive field on the input" },
    ]);

    // back-project an index range [lo, hi] at layer L to layer L−1
    function down(range, L) {
      const op = LAYERS[L];
      return [range[0] * op.S, range[1] * op.S + op.K - 1];
    }
    // ranges (rows, cols) at every layer from `sel` down to the input
    function regions() {
      const out = {};
      let ri = [sel.i, sel.i], rj = [sel.j, sel.j];
      out[sel.layer] = { ri: ri.slice(), rj: rj.slice() };
      for (let L = sel.layer; L >= 1; L--) {
        ri = down(ri, L);
        rj = down(rj, L);
        // clip to layer bounds (kernel overhang cannot exceed the grid)
        const n = LAYERS[L - 1].n;
        ri = [Math.max(0, ri[0]), Math.min(n - 1, ri[1])];
        rj = [Math.max(0, rj[0]), Math.min(n - 1, rj[1])];
        out[L - 1] = { ri: ri.slice(), rj: rj.slice() };
      }
      return out;
    }

    LR.drag(cv, W, H, {
      hit: function (p) {
        for (let L = 1; L < LAYERS.length; L++) {
          const g = LAYERS[L], x0 = XPOS[L], y0 = YTOP + gridYOff(L);
          if (p.x >= x0 && p.x < x0 + g.n * g.cell && p.y >= y0 && p.y < y0 + g.n * g.cell) {
            sel = {
              layer: L,
              i: Math.floor((p.y - y0) / g.cell),
              j: Math.floor((p.x - x0) / g.cell),
            };
            draw();
            return false;
          }
        }
        return false;
      },
    });

    function gridYOff(L) {
      // vertically centre the smaller grids against the input grid
      const h0 = LAYERS[0].n * LAYERS[0].cell;
      return (h0 - LAYERS[L].n * LAYERS[L].cell) / 2;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const reg = regions();

      LAYERS.forEach(function (g, L) {
        const x0 = XPOS[L], y0 = YTOP + gridYOff(L);
        CNN.label(ctx, x0, y0 - 20, "layer " + L + (L === 0 ? " (image)" : ""), C.text);
        CNN.label(ctx, x0, y0 - 6, g.name + " · " + g.sub, C.faint);

        const blank = Array.from({ length: g.n }, () => new Array(g.n).fill(0));
        CNN.grid(ctx, x0, y0, g.cell, blank, {
          fill: function (v, r, c2) {
            const rg = reg[L];
            if (rg && r >= rg.ri[0] && r <= rg.ri[1] && c2 >= rg.rj[0] && c2 <= rg.rj[1]) {
              return L === sel.layer ? "rgba(26,26,26,0.75)" : L === 0 ? "rgba(26,26,26,0.28)" : "rgba(173,173,173,0.30)";
            }
            return "#fbfbfb";
          },
          stroke: "#dcdcdc",
        });
        // region outline
        const rg = reg[L];
        if (rg && L <= sel.layer) {
          ctx.strokeStyle = C.orange;
          ctx.lineWidth = L === 0 ? 3 : 2;
          ctx.strokeRect(
            x0 + rg.rj[0] * g.cell - 1,
            y0 + rg.ri[0] * g.cell - 1,
            (rg.rj[1] - rg.rj[0] + 1) * g.cell,
            (rg.ri[1] - rg.ri[0] + 1) * g.cell
          );
        }
        if (L > 0) {
          const px = XPOS[L - 1] + LAYERS[L - 1].n * LAYERS[L - 1].cell;
          LR.arrow(ctx, px + 8, YTOP + (LAYERS[0].n * LAYERS[0].cell) / 2, x0 - 8, YTOP + (LAYERS[0].n * LAYERS[0].cell) / 2, "rgba(0,0,0,0.25)", 1.6);
        }
      });

      const r0 = reg[0];
      const rh = r0.ri[1] - r0.ri[0] + 1, rw = r0.rj[1] - r0.rj[0] + 1;
      ro.set("unit", "layer " + sel.layer + ", position (" + sel.i + ", " + sel.j + ")");
      ro.set("rf", rw + " × " + rh + " pixels of the original image", C.orange);
    }
    draw();
  };
})();
