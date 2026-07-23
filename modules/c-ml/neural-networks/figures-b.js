/* ══════════════════════════════════════════════════════════════
   figures-b.js — Neural Networks lesson figures, sections 4–6
   Fig 4.1 builder · Fig 5.1 collapse · Fig 5.2 activations
   Fig 5.3 capacity · Fig 6.1 xornet · Fig 6.2 codeex
   Uses the shared network core (LR.nn) from figures-a.js.
   All forward passes and the capacity demo's training are real.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const NN = LR.nn;

  const POS = C.green, NEG = C.purple;
  const POS_SOFT = "rgba(102,102,102,0.16)", NEG_SOFT = "rgba(150,150,150,0.13)";

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — network builder / forward pass (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.builder = function (mount) {
    LR.header(
      mount,
      "Build it, then push numbers through it",
      "2 inputs, one hidden layer, 1 output, sigmoid units. Every value shown is genuinely computed from the weights."
    );

    let H = 3;
    let x = [1.0, 0.0];
    let net = null;       // {W1, b1, W2, b2}
    let preset = "A";
    let stage = 3;        // 0 none, 1 inputs, 2 hidden, 3 output
    let timers = [];

    function randomNet(seed) {
      const rand = LR.rng(seed);
      const r = () => Math.round((rand() * 4 - 2) * 10) / 10;
      const W1 = [], b1 = [];
      for (let i = 0; i < H; i++) { W1.push([r(), r()]); b1.push(r()); }
      const W2 = [[]], b2 = [Math.round((rand() * 2 - 1) * 10) / 10];
      for (let i = 0; i < H; i++) W2[0].push(r());
      return { W1, b1, W2, b2 };
    }
    // sharp-sigmoid XOR: scaled version of the section-6 construction
    function xorNet() {
      return { W1: [[8, 8], [8, 8]], b1: [-4, -12], W2: [[8, -8]], b2: [-4] };
    }
    function applyPreset() {
      if (preset === "XOR") { H = 2; wS.set(2); net = xorNet(); }
      else net = randomNet(preset === "A" ? 3 : 8);
    }

    const bar = LR.controls(mount);
    const wS = LR.slider(bar, "hidden width", 1, 5, 1, H, function (v) {
      H = Math.round(v);
      if (preset === "XOR" && H !== 2) { preset = "A"; syncPresetBtns(); }
      applyPreset();
      runInstant();
    }, (v) => String(Math.round(v)));
    LR.slider(bar, "input x₁", 0, 1, 0.05, x[0], function (v) { x[0] = v; runInstant(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar, "input x₂", 0, 1, 0.05, x[1], function (v) { x[1] = v; runInstant(); }, (v) => LR.fmtF(v, 2));

    const bar2 = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar2.appendChild(group);
    const pbtns = {};
    [["A", "Preset A"], ["B", "Preset B"], ["XOR", "XOR (sharp sigmoid)"]].forEach(function (p) {
      pbtns[p[0]] = LR.button(group, p[1], function () {
        preset = p[0];
        syncPresetBtns();
        applyPreset();
        runInstant();
      }, "small" + (p[0] === "A" ? " on" : ""));
    });
    function syncPresetBtns() {
      for (const k in pbtns) pbtns[k].classList.toggle("on", k === preset);
    }
    LR.button(bar2, "Animate forward pass ▸", runAnimated, "primary");

    const { cv, ctx, W, H: CH } = LR.canvas(mount, 820, 430, {
      aria: "Diagram of a small multi-layer perceptron whose node activations light up layer by layer during a forward pass",
    });
    const ro = LR.readout(mount, [
      { k: "params", label: "parameters (Nᵢ × Nᵢ₋₁ + Nᵢ per layer)" },
      { k: "y", label: "prediction y" },
    ]);

    function fw() {
      return NN.forward(x, [
        { W: net.W1, b: net.b1, act: NN.sigmoid },
        { W: net.W2, b: net.b2, act: NN.sigmoid },
      ]);
    }

    function nodePos() {
      const colX = [150, 410, 670];
      const ys = (n) => {
        const out = [];
        for (let i = 0; i < n; i++) out.push(215 + (i - (n - 1) / 2) * Math.min(86, 300 / Math.max(1, n - 1) || 86));
        return out;
      };
      return { inputs: ys(2).map((y, i) => ({ x: colX[0], y })), hidden: ys(H).map((y) => ({ x: colX[1], y })), out: [{ x: colX[2], y: 215 }] };
    }

    function heatFill(a) {
      // activation in [0,1] → white to orange
      return "rgba(26,26,26," + (0.06 + 0.66 * a).toFixed(3) + ")";
    }

    function draw() {
      ctx.clearRect(0, 0, W, CH);
      const pos = nodePos();
      const F = fw();
      const R = 25;

      // edges input→hidden
      function edge(p1, p2, w, lit) {
        ctx.strokeStyle = (w >= 0 ? "rgba(102,102,102," : "rgba(61,61,61,") + (lit ? 0.85 : 0.3) + ")";
        ctx.lineWidth = 1 + 2.4 * Math.min(Math.abs(w), 2.5) / 2.5;
        ctx.beginPath(); ctx.moveTo(p1.x + R, p1.y); ctx.lineTo(p2.x - R, p2.y); ctx.stroke();
      }
      for (let i = 0; i < H; i++)
        for (let j = 0; j < 2; j++)
          edge(pos.inputs[j], pos.hidden[i], net.W1[i][j], stage >= 2);
      for (let i = 0; i < H; i++)
        edge(pos.hidden[i], pos.out[0], net.W2[0][i], stage >= 3);

      // nodes
      function node(p, label, val, lit, sub) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, R, 0, Math.PI * 2);
        ctx.fillStyle = lit ? heatFill(val) : "#ffffff";
        ctx.fill();
        ctx.strokeStyle = lit ? C.orange : "#bbbbbb";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.font = "700 13px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = lit && val > 0.55 ? "#ffffff" : C.text;
        ctx.textAlign = "center";
        ctx.fillText(lit ? LR.fmtF(val, 2) : "·", p.x, p.y + 4.5);
        ctx.font = "600 12px Inter, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.fillText(label, p.x, p.y - R - 8);
        if (sub) { ctx.font = "500 10.5px Inter, sans-serif"; ctx.fillText(sub, p.x, p.y + R + 15); }
      }
      node(pos.inputs[0], "x₁", x[0], stage >= 1);
      node(pos.inputs[1], "x₂", x[1], stage >= 1);
      for (let i = 0; i < H; i++)
        node(pos.hidden[i], "h" + (i + 1), F.acts[1][i], stage >= 2, stage >= 2 ? "z = " + LR.fmtF(F.zs[0][i], 2) : "");
      node(pos.out[0], "y", F.y[0], stage >= 3, stage >= 3 ? "z = " + LR.fmtF(F.zs[1][0], 2) : "");

      // column labels
      ctx.font = "700 12px Inter, sans-serif"; ctx.fillStyle = C.faint; ctx.textAlign = "center";
      ctx.fillText("input layer", 150, 44);
      ctx.fillText("hidden layer (" + H + " unit" + (H > 1 ? "s" : "") + ")", 410, 44);
      ctx.fillText("output layer", 670, 44);
      ctx.font = "500 11px Inter, sans-serif";
      ctx.fillText("green edge: w > 0 · red edge: w < 0 · thickness = |w|", W / 2, CH - 12);

      const p1 = H * 2, p2 = H;
      ro.set("params", "layer 1: " + H + "×2 = " + p1 + " + " + H + " biases · layer 2: 1×" + H + " = " + p2 + " + 1 · total " + (p1 + H + p2 + 1));
      ro.set("y", stage >= 3 ? LR.fmtF(F.y[0], 3) : "…", C.orange);
    }

    function clearTimers() { timers.forEach(clearTimeout); timers = []; }
    function runInstant() { clearTimers(); stage = 3; draw(); }
    function runAnimated() {
      clearTimers();
      if (LR.reducedMotion) { runInstant(); return; }
      stage = 0; draw();
      [1, 2, 3].forEach(function (s, i) {
        timers.push(setTimeout(function () { stage = s; draw(); }, 380 * (i + 1)));
      });
    }

    applyPreset();
    runInstant();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — linear-collapse demonstrator
     ════════════════════════════════════════════════════════════ */
  LR.figs.collapse = function (mount) {
    LR.header(
      mount,
      "Two layers, one map (until you add the nonlinearity)",
      "A fixed 2-2-1 network. The dashed line is the single affine layer computed by multiplying the weight matrices. Identity: they agree everywhere. Tanh: they part ways."
    );

    // fixed weights for the 2-2-1 network
    const W1 = [[1.2, -0.9], [0.7, 1.1]], b1 = [0.3, -0.4];
    const w2 = [1.4, -1.2], b2 = 0.2;

    let actName = "identity", gain = 1.6;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const abtns = {};
    [["identity", "identity (linear)"], ["tanh", "tanh"]].forEach(function (a) {
      abtns[a[0]] = LR.button(group, a[1], function () {
        actName = a[0];
        for (const k in abtns) abtns[k].classList.toggle("on", k === actName);
        draw();
      }, "small" + (a[0] === "identity" ? " on" : ""));
    });
    LR.slider(bar, "hidden gain s", 0.5, 3.5, 0.05, gain, function (v) { gain = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 440, {
      aria: "Decision regions of a two-layer network compared against its collapsed single affine layer, with an activation toggle",
    });
    const ro = LR.readout(mount, [
      { k: "affine", label: "collapsed affine map w̃, b̃" },
      { k: "diff", label: "max |two-layer − affine| on the plane" },
      { k: "verdict", label: "did the layers collapse?" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -2, xmax: 2, ymin: -2, ymax: 2, xlabel: "x₁", ylabel: "x₂" };

    // two-layer map with the chosen activation and gain on the hidden layer
    function f2(x, y) {
      const act = actName === "identity" ? NN.identity : NN.tanh;
      const h1 = act(gain * (W1[0][0] * x + W1[0][1] * y + b1[0]));
      const h2 = act(gain * (W1[1][0] * x + W1[1][1] * y + b1[1]));
      return w2[0] * h1 + w2[1] * h2 + b2;
    }
    // the collapsed affine map (exact when the activation is the identity)
    function affineParams() {
      const wt = [
        w2[0] * gain * W1[0][0] + w2[1] * gain * W1[1][0],
        w2[0] * gain * W1[0][1] + w2[1] * gain * W1[1][1],
      ];
      const bt = w2[0] * gain * b1[0] + w2[1] * gain * b1[1] + b2;
      return { wt, bt };
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const { wt, bt } = affineParams();

      // shading by the sign of the real two-layer map + max difference
      const BLK = 7;
      let maxdiff = 0;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const v = f2(wx, wy);
          const d = Math.abs(v - (wt[0] * wx + wt[1] * wy + bt));
          if (d > maxdiff) maxdiff = d;
          ctx.fillStyle = v > 0 ? POS_SOFT : NEG_SOFT;
          ctx.fillRect(px, py, BLK, BLK);
        }
      }

      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // true zero contour of the two-layer map (scan columns for sign changes)
      ctx.fillStyle = C.orange;
      const NX = 240, NY = 240;
      for (let i = 0; i <= NX; i++) {
        const wx = P.xmin + ((P.xmax - P.xmin) * i) / NX;
        let prev = f2(wx, P.ymin);
        for (let j = 1; j <= NY; j++) {
          const wy = P.ymin + ((P.ymax - P.ymin) * j) / NY;
          const cur = f2(wx, wy);
          if ((prev > 0) !== (cur > 0)) {
            const frac = Math.abs(prev) / (Math.abs(prev) + Math.abs(cur));
            const yc = wy - (1 - frac) * (P.ymax - P.ymin) / NY;
            ctx.fillRect(sx(wx) - 1.4, sy(yc) - 1.4, 2.8, 2.8);
          }
          prev = cur;
        }
      }

      // dashed collapsed-affine boundary
      ctx.strokeStyle = C.text; ctx.lineWidth = 2; ctx.setLineDash([7, 5]);
      ctx.beginPath();
      if (Math.abs(wt[1]) > Math.abs(wt[0])) {
        ctx.moveTo(sx(P.xmin), sy(-(wt[0] * P.xmin + bt) / wt[1]));
        ctx.lineTo(sx(P.xmax), sy(-(wt[0] * P.xmax + bt) / wt[1]));
      } else {
        ctx.moveTo(sx(-(wt[1] * P.ymin + bt) / wt[0]), sy(P.ymin));
        ctx.lineTo(sx(-(wt[1] * P.ymax + bt) / wt[0]), sy(P.ymax));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // legend
      ctx.font = "600 11.5px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.orange; ctx.fillText("— two-layer boundary (real)", P.x0 + 10, P.y0 + 16);
      ctx.fillStyle = C.text; ctx.fillText("- - collapsed single affine layer w̃ᵀx + b̃ = 0", P.x0 + 10, P.y0 + 32);

      ro.set("affine", "w̃ = [" + LR.fmtF(wt[0], 2) + ", " + LR.fmtF(wt[1], 2) + "], b̃ = " + LR.fmtF(bt, 2));
      ro.set("diff", maxdiff < 1e-9 ? "0 (exactly)" : LR.fmtF(maxdiff, 3), maxdiff < 1e-9 ? C.green : C.red);
      ro.set(
        "verdict",
        actName === "identity" ? "yes: two layers = one affine map, nothing gained" : "no: tanh made the composition genuinely nonlinear",
        actName === "identity" ? C.purple : C.green
      );
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — activation function explorer
     ════════════════════════════════════════════════════════════ */
  LR.figs.activations = function (mount) {
    LR.header(
      mount,
      "Sigmoid, tanh, ReLU",
      "Slide the probe. The grey bands are where sigmoid and tanh have saturated: nearly flat, nearly unresponsive. ReLU never flattens on the right."
    );

    let z = 2;
    const bar = LR.controls(mount);
    LR.slider(bar, "z", -6, 6, 0.1, z, function (v) { z = v; draw(); }, (v) => LR.fmtF(v, 1));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 420, {
      aria: "Plots of sigmoid, tanh, and ReLU with a movable probe reading all three values",
    });
    const ro = LR.readout(mount, [
      { k: "sig", label: "σ(z), range (0, 1)" },
      { k: "tanh", label: "tanh(z), range (−1, 1)" },
      { k: "relu", label: "ReLU(z), range [0, ∞)" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -6, xmax: 6, ymin: -1.6, ymax: 3, xlabel: "z (pre-activation)", ylabel: "activation output" };

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // saturation bands for sigmoid/tanh: |z| > 4
      ctx.fillStyle = "rgba(0,0,0,0.05)";
      ctx.fillRect(P.x0, P.y0, sx(-4) - P.x0, P.h);
      ctx.fillRect(sx(4), P.y0, P.x0 + P.w - sx(4), P.h);
      ctx.font = "600 11px Inter, sans-serif"; ctx.fillStyle = C.faint; ctx.textAlign = "center";
      ctx.fillText("saturated", (P.x0 + sx(-4)) / 2, P.y0 + 18);
      ctx.fillText("saturated (σ, tanh only)", (sx(4) + P.x0 + P.w) / 2, P.y0 + 18);

      // zero line
      ctx.strokeStyle = "#d5d5d5"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(P.x0, sy(0)); ctx.lineTo(P.x0 + P.w, sy(0)); ctx.stroke();

      // curves
      function curve(fn, color) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.6;
        ctx.beginPath();
        for (let i = 0; i <= 300; i++) {
          const zz = P.xmin + ((P.xmax - P.xmin) * i) / 300;
          const v = fn(zz);
          i === 0 ? ctx.moveTo(sx(zz), sy(v)) : ctx.lineTo(sx(zz), sy(v));
        }
        ctx.stroke();
      }
      curve(NN.sigmoid, C.orange);
      curve(NN.tanh, C.purple);
      curve(NN.relu, C.green);

      // probe line + dots
      ctx.strokeStyle = C.text; ctx.lineWidth = 1.4; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(sx(z), P.y0); ctx.lineTo(sx(z), P.y0 + P.h); ctx.stroke();
      ctx.setLineDash([]);
      LR.dot(ctx, sx(z), sy(NN.sigmoid(z)), 6, C.orange, "#ffffff");
      LR.dot(ctx, sx(z), sy(NN.tanh(z)), 6, C.purple, "#ffffff");
      if (NN.relu(z) <= P.ymax) LR.dot(ctx, sx(z), sy(NN.relu(z)), 6, C.green, "#ffffff");
      ctx.restore();

      // legend
      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.orange; ctx.fillText("— sigmoid", P.x0 + 10, P.y0 + P.h - 44);
      ctx.fillStyle = C.purple; ctx.fillText("— tanh", P.x0 + 10, P.y0 + P.h - 28);
      ctx.fillStyle = C.green; ctx.fillText("— ReLU", P.x0 + 10, P.y0 + P.h - 12);

      ro.set("sig", LR.fmtF(NN.sigmoid(z), 3), C.orange);
      ro.set("tanh", LR.fmtF(NN.tanh(z), 3), C.purple);
      ro.set("relu", LR.fmtF(NN.relu(z), 3), C.green);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.3 — depth-and-width capacity demo (real training)
     ════════════════════════════════════════════════════════════ */
  LR.figs.capacity = function (mount) {
    LR.header(
      mount,
      "Capacity, trained live in your browser",
      "The same curvy dataset, fit by gradient descent every time you move a slider. Four points are deliberately mislabeled; watch when the network starts believing them."
    );

    // dataset: curvy true boundary t = [x2 > 1.05 sin(1.6 x1)] + 4 label flips
    const rand = LR.rng(11);
    const data = [];
    for (let i = 0; i < 76; i++) {
      const x = -2.2 + rand() * 4.4, y = -2.2 + rand() * 4.4;
      data.push({ x, y, t: y > 1.05 * Math.sin(1.6 * x) ? 1 : 0, flip: false });
    }
    [5, 23, 41, 60].forEach(function (i) { data[i].t = 1 - data[i].t; data[i].flip = true; });

    let H = 2, L = 1;
    let net = null, acc = 0, loss = 0, memorized = 0;
    let pending = null;

    const bar = LR.controls(mount);
    LR.slider(bar, "hidden units per layer", 0, 3, 1, 1, function (v) {
      H = [1, 2, 4, 8][Math.round(v)];
      schedule();
    }, (v) => String([1, 2, 4, 8][Math.round(v)]));
    LR.slider(bar, "hidden layers", 1, 2, 1, L, function (v) { L = Math.round(v); schedule(); }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H: CH } = LR.canvas(mount, 820, 460, {
      aria: "Two-class dataset with the decision boundary of a freshly trained neural network whose width and depth are set by sliders",
    });
    const ro = LR.readout(mount, [
      { k: "arch", label: "architecture" },
      { k: "params", label: "parameters" },
      { k: "acc", label: "training accuracy" },
      { k: "mem", label: "mislabeled points memorized" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 60, y0: 20, w: W - 84, h: CH - 72, xmin: -2.4, xmax: 2.4, ymin: -2.4, ymax: 2.4, xlabel: "x₁", ylabel: "x₂" };

    /* ---- a tiny trainable MLP (tanh hidden, sigmoid output) ----
       Training uses gradient descent with momentum. Backprop itself is
       next lecture's topic; here it is only the engine behind the curtain. */
    function train() {
      const sizes = [2].concat(new Array(L).fill(H)).concat([1]);
      const r = LR.rng(7 + H * 13 + L * 101);
      const Ws = [], bs = [], vW = [], vb = [];
      for (let l = 0; l < sizes.length - 1; l++) {
        const nin = sizes[l], nout = sizes[l + 1];
        const s = 1.4 / Math.sqrt(nin);
        Ws.push(Array.from({ length: nout }, () => Array.from({ length: nin }, () => LR.gauss(r) * s)));
        bs.push(new Array(nout).fill(0));
        vW.push(Array.from({ length: nout }, () => new Array(nin).fill(0)));
        vb.push(new Array(nout).fill(0));
      }
      const NL = Ws.length, N = data.length;
      const lr = 0.35, mom = 0.9, EPOCHS = 1400;

      for (let ep = 0; ep < EPOCHS; ep++) {
        // gradient accumulators
        const gW = Ws.map((Wl) => Wl.map((row) => new Array(row.length).fill(0)));
        const gb = bs.map((bl) => new Array(bl.length).fill(0));
        for (const p of data) {
          // forward, caching activations
          const acts = [[p.x, p.y]];
          for (let l = 0; l < NL; l++) {
            const z = NN.affine(acts[l], Ws[l], bs[l]);
            acts.push(z.map(l === NL - 1 ? NN.sigmoid : NN.tanh));
          }
          // backward (cross-entropy + sigmoid output → delta = y − t)
          let delta = [acts[NL][0] - p.t];
          for (let l = NL - 1; l >= 0; l--) {
            for (let i = 0; i < delta.length; i++) {
              gb[l][i] += delta[i];
              for (let j = 0; j < acts[l].length; j++) gW[l][i][j] += delta[i] * acts[l][j];
            }
            if (l > 0) {
              const nd = new Array(acts[l].length).fill(0);
              for (let j = 0; j < nd.length; j++) {
                let s = 0;
                for (let i = 0; i < delta.length; i++) s += Ws[l][i][j] * delta[i];
                nd[j] = s * (1 - acts[l][j] * acts[l][j]); // tanh'
              }
              delta = nd;
            }
          }
        }
        // momentum update
        for (let l = 0; l < NL; l++) {
          for (let i = 0; i < Ws[l].length; i++) {
            vb[l][i] = mom * vb[l][i] - (lr / N) * gb[l][i];
            bs[l][i] += vb[l][i];
            for (let j = 0; j < Ws[l][i].length; j++) {
              vW[l][i][j] = mom * vW[l][i][j] - (lr / N) * gW[l][i][j];
              Ws[l][i][j] += vW[l][i][j];
            }
          }
        }
      }

      net = { Ws, bs, NL };
      // final metrics
      let ok = 0, mem = 0, lo = 0;
      for (const p of data) {
        const y = predict(p.x, p.y);
        const pred = y > 0.5 ? 1 : 0;
        if (pred === p.t) { ok++; if (p.flip) mem++; }
        lo += -(p.t * Math.log(Math.max(y, 1e-12)) + (1 - p.t) * Math.log(Math.max(1 - y, 1e-12)));
      }
      acc = ok / data.length; memorized = mem; loss = lo / data.length;
    }

    function predict(x, y) {
      let h = [x, y];
      for (let l = 0; l < net.NL; l++) {
        const z = NN.affine(h, net.Ws[l], net.bs[l]);
        h = z.map(l === net.NL - 1 ? NN.sigmoid : NN.tanh);
      }
      return h[0];
    }

    function paramCount() {
      const sizes = [2].concat(new Array(L).fill(H)).concat([1]);
      let n = 0;
      for (let l = 0; l < sizes.length - 1; l++) n += sizes[l + 1] * sizes[l] + sizes[l + 1];
      return n;
    }

    function schedule() {
      if (pending) clearTimeout(pending);
      msg.show("training…", "info");
      pending = setTimeout(function () { train(); draw(); }, 120);
    }

    function draw() {
      ctx.clearRect(0, 0, W, CH);

      const BLK = 7;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          ctx.fillStyle = predict(wx, wy) > 0.5 ? POS_SOFT : NEG_SOFT;
          ctx.fillRect(px, py, BLK, BLK);
        }
      }

      const SC = LR.plot(ctx, P);
      data.forEach(function (p) {
        LR.dot(ctx, SC.sx(p.x), SC.sy(p.y), p.flip ? 6.5 : 5, p.t ? POS : NEG, p.flip ? C.red : "#ffffff");
      });

      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.red; ctx.textAlign = "left";
      ctx.fillText("red ring = deliberately mislabeled", P.x0 + 10, P.y0 + 16);

      const archStr = ["2"].concat(new Array(L).fill(String(H))).concat(["1"]).join(" → ");
      ro.set("arch", archStr);
      ro.set("params", String(paramCount()));
      ro.set("acc", LR.fmtF(acc * 100, 1) + "%  (loss " + LR.fmtF(loss, 3) + ")", acc > 0.97 ? C.amber : C.green);
      ro.set("mem", memorized + " / 4", memorized === 0 ? C.green : C.red);

      if (H === 1 && L === 1) {
        msg.show("One hidden unit: the boundary is almost a single line again. Underfit: the sine wiggle is invisible to it.", "info");
      } else if (memorized >= 3) {
        msg.show("High capacity: the network now carves pockets around the mislabeled points. Perfectly separating noisy training data is memorization, not learning.", "bad");
      } else if (acc > 0.9) {
        msg.show("The boundary tracks the true sine-shaped class edge while still ignoring most of the label noise. This is the capacity sweet spot.", "good");
      } else {
        msg.hide();
      }
    }

    train();
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — the XOR network walkthrough (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.xornet = function (mount) {
    LR.header(
      mount,
      "OR and AND compose into XOR",
      "Hard-threshold units with the lecture's exact weights. Pick an input and watch the real arithmetic light the network up."
    );

    const bar = LR.controls(mount);
    const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    inputs.forEach(function (inp) {
      LR.button(bar, "x = (" + inp[0] + ", " + inp[1] + ")", function () { run(inp, null); }, "small");
    });
    LR.button(bar, "Step through all four ▸", stepAll, "primary");

    const grid = LR.el("div", "stepper-grid");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left); grid.appendChild(right);

    const { cv, ctx, W, H } = LR.canvas(left, 430, 330, {
      aria: "Diagram of the two-hidden-unit XOR network with units lighting up as each input is evaluated",
    });

    // truth table (HTML, filled live)
    const tbl = LR.el("table", "ttable");
    tbl.innerHTML =
      "<thead><tr><th>x₁</th><th>x₂</th><th>h₁ (OR)</th><th>h₂ (AND)</th><th>y</th><th>XOR</th></tr></thead>" +
      "<tbody>" +
      inputs.map((inp) => "<tr data-in='" + inp.join("") + "'><td>" + inp[0] + "</td><td>" + inp[1] + "</td><td>·</td><td>·</td><td>·</td><td>" + (inp[0] ^ inp[1]) + "</td></tr>").join("") +
      "</tbody>";
    right.appendChild(tbl);
    const narrate = LR.el("div", "stepper-narrate", "Pick an input on the left, or step through all four.");
    right.appendChild(narrate);

    const nodes = {
      x1: { x: 70, y: 95 }, x2: { x: 70, y: 235 },
      h1: { x: 215, y: 95 }, h2: { x: 215, y: 235 },
      y: { x: 370, y: 165 },
    };
    const R = 23;

    let cur = null, stage = 0, timers = [];
    function clearTimers() { timers.forEach(clearTimeout); timers = []; }

    function compute(inp) {
      const F = NN.forward(inp, NN.XOR.layers());
      return { h1: F.acts[1][0], h2: F.acts[1][1], y: F.y[0], z: F.zs };
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const F = cur ? compute(cur) : null;

      function edge(a, b, label, lit) {
        ctx.strokeStyle = lit ? C.orange : "#c9c9c9";
        ctx.lineWidth = lit ? 2.4 : 1.6;
        ctx.beginPath(); ctx.moveTo(a.x + R, a.y); ctx.lineTo(b.x - R, b.y); ctx.stroke();
        ctx.font = "700 11.5px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = lit ? C.orange : C.faint;
        ctx.textAlign = "center";
        ctx.fillText(label, (a.x + b.x) / 2, (a.y + b.y) / 2 - 7);
      }
      edge(nodes.x1, nodes.h1, "1", stage >= 2);
      edge(nodes.x2, nodes.h1, "1", stage >= 2);
      edge(nodes.x1, nodes.h2, "1", stage >= 2);
      edge(nodes.x2, nodes.h2, "1", stage >= 2);
      edge(nodes.h1, nodes.y, "1", stage >= 3);
      edge(nodes.h2, nodes.y, "−1", stage >= 3);

      function node(p, name, role, bias, val, lit) {
        ctx.beginPath(); ctx.arc(p.x, p.y, R, 0, Math.PI * 2);
        ctx.fillStyle = lit ? (val > 0.5 ? C.orange : "#ffffff") : "#ffffff";
        ctx.fill();
        ctx.strokeStyle = lit ? C.orange : "#bbbbbb"; ctx.lineWidth = 2; ctx.stroke();
        ctx.font = "700 13px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = lit && val > 0.5 ? "#ffffff" : C.text;
        ctx.textAlign = "center";
        ctx.fillText(lit ? String(val) : "·", p.x, p.y + 4.5);
        ctx.font = "700 12px Inter, sans-serif"; ctx.fillStyle = C.text;
        ctx.fillText(name, p.x, p.y - R - 18);
        ctx.font = "600 10.5px Inter, sans-serif"; ctx.fillStyle = C.muted;
        ctx.fillText(role, p.x, p.y - R - 6);
        if (bias) ctx.fillText(bias, p.x, p.y + R + 13);
      }
      node(nodes.x1, "x₁", "input", "", cur ? cur[0] : 0, stage >= 1);
      node(nodes.x2, "x₂", "input", "", cur ? cur[1] : 0, stage >= 1);
      node(nodes.h1, "h₁", "OR", "b = −0.5", F ? F.h1 : 0, stage >= 2);
      node(nodes.h2, "h₂", "AND", "b = −1.5", F ? F.h2 : 0, stage >= 2);
      node(nodes.y, "y", "h₁ AND NOT h₂", "b = −0.5", F ? F.y : 0, stage >= 3);
    }

    function fillRow(inp) {
      const F = compute(inp);
      const row = tbl.querySelector("tr[data-in='" + inp.join("") + "']");
      const tds = row.querySelectorAll("td");
      tds[2].textContent = F.h1;
      tds[3].textContent = F.h2;
      tds[4].innerHTML = "<b>" + F.y + "</b>";
      tds[4].className = F.y === (inp[0] ^ inp[1]) ? "t-pos" : "t-neg";
      tbl.querySelectorAll("tr").forEach((r) => r.classList.remove("hl"));
      row.classList.add("hl");
    }

    function narrateFor(inp) {
      const F = compute(inp);
      return (
        "<b>x = (" + inp[0] + ", " + inp[1] + ")</b>: " +
        "h₁ = 𝕀[" + inp[0] + "+" + inp[1] + "−0.5 > 0] = 𝕀[" + LR.fmtF(F.z[0][0], 1) + " > 0] = " + F.h1 +
        " · h₂ = 𝕀[" + LR.fmtF(F.z[0][1], 1) + " > 0] = " + F.h2 +
        " · y = 𝕀[" + F.h1 + "−" + F.h2 + "−0.5 > 0] = 𝕀[" + LR.fmtF(F.z[1][0], 1) + " > 0] = <b>" + F.y + "</b>. " +
        "XOR says " + (inp[0] ^ inp[1]) + " ✓"
      );
    }

    function run(inp, done) {
      clearTimers();
      cur = inp;
      narrate.innerHTML = narrateFor(inp);
      if (LR.reducedMotion) {
        stage = 3; draw(); fillRow(inp);
        if (done) done();
        return;
      }
      stage = 1; draw();
      timers.push(setTimeout(function () { stage = 2; draw(); }, 420));
      timers.push(setTimeout(function () { stage = 3; draw(); fillRow(inp); if (done) done(); }, 840));
    }

    function stepAll() {
      clearTimers();
      let i = 0;
      const next = function () {
        if (i >= inputs.length) {
          narrate.innerHTML = "<b>All four rows verified.</b> OR and AND, two linearly-computable concepts, composed into XOR, which no single line could draw.";
          return;
        }
        const inp = inputs[i++];
        run(inp, function () { timers.push(setTimeout(next, 500)); });
      };
      next();
    }

    stage = 0;
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.2 — the forward pass as a live code exercise
     ════════════════════════════════════════════════════════════ */
  LR.figs.codeex = function (mount) {
    LR.header(
      mount,
      "Run the network yourself",
      "The forward pass below really executes. The hidden layer is fixed to the OR and AND detectors; the output unit's weights are yours to set. Break XOR, then repair it."
    );

    const CODE =
      "step = lambda z: (z > 0).astype(float)   # hard threshold\n" +
      "\n" +
      "W1 = np.array([[1., 1.],                 # h1: x1 + x2 - 0.5 > 0   (OR)\n" +
      "               [1., 1.]])                # h2: x1 + x2 - 1.5 > 0   (AND)\n" +
      "b1 = np.array([-0.5, -1.5])\n" +
      "\n" +
      "W2 = np.array([[v1, v2]])                # output unit: yours to set\n" +
      "b2 = np.array([c])\n" +
      "\n" +
      "def forward(x, layers):\n" +
      "    h = x\n" +
      "    for W, b, act in layers:\n" +
      "        h = act(W @ h + b)               # affine, then nonlinearity\n" +
      "    return h\n" +
      "\n" +
      "y = forward(x, [(W1, b1, step), (W2, b2, step)])";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);

    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);

    let v1 = 1, v2 = -1, c = -0.5;
    let xin = [1, 0];

    const xLab = LR.el("label", "", "x = ");
    const sel = document.createElement("select");
    [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(function (inp) {
      const o = document.createElement("option");
      o.value = inp.join(",");
      o.textContent = "(" + inp[0] + ", " + inp[1] + ")";
      if (inp[0] === 1 && inp[1] === 0) o.selected = true;
      sel.appendChild(o);
    });
    sel.style.cssText = "font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid #cccccc";
    sel.addEventListener("change", function () { xin = sel.value.split(",").map(Number); });
    xLab.appendChild(sel);
    controls.appendChild(xLab);
    LR.slider(controls, "v₁ (weight on h₁)", -2, 2, 0.25, v1, (v) => { v1 = v; }, (v) => LR.fmtF(v, 2));
    LR.slider(controls, "v₂ (weight on h₂)", -2, 2, 0.25, v2, (v) => { v2 = v; }, (v) => LR.fmtF(v, 2));
    LR.slider(controls, "c (output bias)", -2, 2, 0.25, c, (v) => { c = v; }, (v) => LR.fmtF(v, 2));
    LR.button(controls, "Run ▸", run, "primary small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; forward(np.array([1, 0]), layers)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    function netFor(v1_, v2_, c_) {
      return [
        { W: [[1, 1], [1, 1]], b: [-0.5, -1.5], act: NN.step },
        { W: [[v1_, v2_]], b: [c_], act: NN.step },
      ];
    }

    function run() {
      const layers = netFor(v1, v2, c);
      const F = NN.forward(xin, layers);
      let txt =
        "&gt;&gt;&gt; forward(np.array([" + xin[0] + ", " + xin[1] + "]), layers)   # v1=" + LR.fmt(v1) + ", v2=" + LR.fmt(v2) + ", c=" + LR.fmt(c) + "\n" +
        "hidden: h = [" + F.acts[1].join(", ") + "]   (h1 = OR, h2 = AND)\n" +
        "output: y = " + F.y[0] + "\n\n" +
        "full truth table with your output weights:\n";
      let okAll = 0;
      [[0, 0], [0, 1], [1, 0], [1, 1]].forEach(function (inp) {
        const G = NN.forward(inp, layers);
        const want = inp[0] ^ inp[1];
        const ok = G.y[0] === want;
        if (ok) okAll++;
        txt += "  (" + inp[0] + "," + inp[1] + ") → h=(" + G.acts[1].join(",") + ") → y=" + G.y[0] + "  target " + want + "  " + (ok ? "✓" : "✗") + "\n";
      });
      txt += okAll === 4
        ? "<b>matches XOR: 4/4 ✓</b>"
        : "matches XOR on " + okAll + "/4 rows. The classic fix: v1=1, v2=−1, c=−0.5 (fire when h1 on, h2 off).";
      out.innerHTML = txt;
    }
    run();
  };
})();
