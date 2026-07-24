/* ══════════════════════════════════════════════════════════════
   figures-a.js — Backpropagation lesson figures, sections 3–4
   Fig 3.1 graph · Fig 3.2 inspector · Fig 3.3 fanout
   Fig 4.1 mlp · Fig 4.2 backex
   Shared backprop core (LR.bp) lives here and is reused by
   figures-b.js (trainer, early stopping). All numbers computed
   live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared backprop / autodiff core ────────────────────────
     The same forward/backward math drives the graph figures, the
     MLP stepper, the trainer, and the early-stopping demo. */
  const BP = (LR.bp = {
    sig: function (z) { return 1 / (1 + Math.exp(-z)); },

    // logistic least squares: z = wx+b, y = σ(z), L = ½(y−t)²
    // returns every node value and every error signal (bar notation)
    logistic: function (x, w, b, t) {
      const z = w * x + b;
      const y = BP.sig(z);
      const L = 0.5 * (y - t) * (y - t);
      const ds = y * (1 - y);            // σ'(z) = σ(z)(1−σ(z))
      const Lbar = 1;
      const ybar = y - t;
      const zbar = ybar * ds;
      const wbar = zbar * x;
      const bbar = zbar;
      return { x, w, b, t, z, y, L, ds, Lbar, ybar, zbar, wbar, bbar };
    },

    // adds the L2 regularizer: R = ½w², Lreg = L + λR  (fan-out at w)
    regularized: function (x, w, b, t, lam) {
      const g = BP.logistic(x, w, b, t);
      const R = 0.5 * w * w;
      const Lreg = g.L + lam * R;
      const Rbar = lam;                  // ∂Lreg/∂R
      const wbarZ = g.zbar * x;          // contribution through z
      const wbarR = Rbar * w;            // contribution through R
      return Object.assign({}, g, { lam, R, Lreg, Rbar, wbarZ, wbarR, wbar: wbarZ + wbarR });
    },

    // central finite difference (the universal gradient unit test)
    numGrad: function (f, v, eps) {
      eps = eps || 1e-5;
      return (f(v + eps) - f(v - eps)) / (2 * eps);
    },

    /* generic tiny MLP: d inputs → H sigmoid hidden → 1 output
       (linear or sigmoid), squared loss. Used by Fig 4.3 and 6.3. */
    mlpInit: function (H, d, seed, scale, outSig) {
      const rand = LR.rng(seed);
      const net = { H, d, outSig: !!outSig };
      net.W1 = Array.from({ length: H }, function () {
        const row = new Array(d);
        for (let j = 0; j < d; j++) row[j] = LR.gauss(rand) * scale;
        return row;
      });
      net.b1 = Array.from({ length: H }, () => 0);
      net.w2 = Array.from({ length: H }, () => LR.gauss(rand) * scale);
      net.b2 = 0;
      return net;
    },
    // variant matching Fig 6.3's tuned init (biases drawn, small output weights)
    mlp1Init: function (H, seed, scale) {
      const rand = LR.rng(seed);
      return {
        H, d: 1, outSig: false,
        W1: Array.from({ length: H }, () => [LR.gauss(rand) * scale]),
        b1: Array.from({ length: H }, () => LR.gauss(rand) * scale),
        w2: Array.from({ length: H }, () => LR.gauss(rand) * 0.5),
        b2: 0,
      };
    },
    mlpForward: function (net, x) {
      const H = net.H, d = net.d;
      const z = new Array(H), h = new Array(H);
      for (let i = 0; i < H; i++) {
        let s = net.b1[i];
        for (let j = 0; j < d; j++) s += net.W1[i][j] * x[j];
        z[i] = s;
        h[i] = BP.sig(s);
      }
      let u = net.b2;
      for (let i = 0; i < H; i++) u += net.w2[i] * h[i];
      return { z, h, u, y: net.outSig ? BP.sig(u) : u };
    },
    mlpLoss: function (net, X, T) {
      let s = 0;
      for (let k = 0; k < X.length; k++) {
        const r = BP.mlpForward(net, X[k]).y - T[k];
        s += 0.5 * r * r;
      }
      return s / X.length;
    },
    // one full-batch gradient step via backprop; returns mean loss BEFORE the step
    mlpBatchStep: function (net, X, T, lr) {
      const H = net.H, d = net.d, n = X.length;
      const gW1 = net.W1.map(() => new Array(d).fill(0));
      const gb1 = new Array(H).fill(0);
      const gw2 = new Array(H).fill(0);
      let gb2 = 0, loss = 0;
      for (let k = 0; k < n; k++) {
        const f = BP.mlpForward(net, X[k]);
        const ybar = f.y - T[k];                       // dL/dy
        loss += 0.5 * ybar * ybar;
        const ubar = net.outSig ? ybar * f.y * (1 - f.y) : ybar; // through output σ if any
        for (let i = 0; i < H; i++) {
          gw2[i] += ubar * f.h[i];                     // w̄²_i = ū·h_i
          const zbar = ubar * net.w2[i] * f.h[i] * (1 - f.h[i]); // h̄_i σ'(z_i)
          for (let j = 0; j < d; j++) gW1[i][j] += zbar * X[k][j];
          gb1[i] += zbar;
        }
        gb2 += ubar;
      }
      for (let i = 0; i < H; i++) {
        net.w2[i] -= (lr * gw2[i]) / n;
        net.b1[i] -= (lr * gb1[i]) / n;
        for (let j = 0; j < d; j++) net.W1[i][j] -= (lr * gW1[i][j]) / n;
      }
      net.b2 -= (lr * gb2) / n;
      return loss / n;
    },
  });

  /* ── graph drawing helpers (shared by Figs 3.1–3.3, 4.1) ── */
  function trimEdge(a, b, pad) {
    const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
    const ux = dx / d, uy = dy / d;
    return {
      x1: a.x + ux * (a.r + 2), y1: a.y + uy * (a.r + 2),
      x2: b.x - ux * (b.r + (pad === undefined ? 5 : pad)), y2: b.y - uy * (b.r + (pad === undefined ? 5 : pad)),
      ux, uy,
    };
  }
  function fwdArrow(ctx, a, b, o) {
    o = o || {};
    const e = trimEdge(a, b);
    LR.arrow(ctx, e.x1, e.y1, e.x2, e.y2, o.lit ? C.orange : (o.color || "#9a9a9a"), o.lit ? 2.4 : 1.7);
    if (o.label) {
      const mx = (e.x1 + e.x2) / 2 + e.uy * 14, my = (e.y1 + e.y2) / 2 - e.ux * 14;
      ctx.font = (o.lit ? "700 " : "600 ") + "11px Inter, sans-serif";
      ctx.fillStyle = o.lit ? C.orange : C.faint;
      ctx.textAlign = "center";
      ctx.fillText(o.label, mx, my + 4);
    }
    return e;
  }
  // curved dashed arrow for the backward pass (child → parent)
  function backArrow(ctx, child, parent, o) {
    o = o || {};
    const bend = o.bend === undefined ? 34 : o.bend;
    const e = trimEdge(child, parent, 8);
    const mx = (e.x1 + e.x2) / 2, my = (e.y1 + e.y2) / 2;
    const nx = e.uy * bend, ny = -e.ux * bend;
    const cx = mx + nx, cy = my + ny;
    ctx.save();
    ctx.strokeStyle = o.color || C.orange;
    ctx.fillStyle = o.color || C.orange;
    ctx.lineWidth = o.width || 2.1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(e.x1, e.y1);
    ctx.quadraticCurveTo(cx, cy, e.x2, e.y2);
    ctx.stroke();
    ctx.setLineDash([]);
    const ang = Math.atan2(e.y2 - cy, e.x2 - cx);
    ctx.beginPath();
    ctx.moveTo(e.x2, e.y2);
    ctx.lineTo(e.x2 - 8 * Math.cos(ang - 0.42), e.y2 - 8 * Math.sin(ang - 0.42));
    ctx.lineTo(e.x2 - 8 * Math.cos(ang + 0.42), e.y2 - 8 * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
    if (o.label) {
      ctx.font = "700 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(o.label, cx, cy + (ny < 0 ? -4 : 12));
    }
    ctx.restore();
  }
  function drawNode(ctx, n, o) {
    o = o || {};
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = o.lit ? "#f5f5f5" : "#ffffff";
    ctx.fill();
    ctx.lineWidth = o.lit ? 2.6 : 1.6;
    ctx.strokeStyle = o.lit ? C.orange : o.dim ? "#c8c8c8" : C.axis;
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillStyle = o.dim ? C.faint : C.text;
    ctx.font = "italic 700 15px Georgia, serif";
    ctx.fillText(n.label, n.x, n.y + (o.val != null ? -3 : 5));
    if (o.val != null) {
      ctx.font = "700 11px 'JetBrains Mono', Menlo, monospace";
      ctx.fillStyle = C.muted;
      ctx.fillText(o.val, n.x, n.y + 13);
    }
    if (o.bar != null) {
      ctx.font = "700 12px 'JetBrains Mono', Menlo, monospace";
      ctx.fillStyle = C.orange;
      ctx.fillText(o.bar, n.x, n.y - n.r - 7);
    }
  }

  /* logistic-least-squares graph layout (Figs 3.1 and 3.2) */
  function logisticLayout() {
    return {
      x: { x: 100, y: 78, r: 24, label: "x" },
      w: { x: 100, y: 168, r: 24, label: "w" },
      b: { x: 100, y: 258, r: 24, label: "b" },
      z: { x: 350, y: 168, r: 26, label: "z" },
      y: { x: 530, y: 168, r: 26, label: "y" },
      L: { x: 700, y: 168, r: 26, label: "L" },
      t: { x: 530, y: 272, r: 22, label: "t" },
    };
  }

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the computation graph, forward then backward
     ════════════════════════════════════════════════════════════ */
  LR.figs.graph = function (mount) {
    LR.header(
      mount,
      "Values forward, error signals backward",
      "The logistic-least-squares graph. Step it: three forward moves fill the values, four backward moves build every gradient. Sliders re-run everything live."
    );

    let xv = 2.0, wv = 0.5, bv = -0.5, tv = 1;
    let step = 0; // 0 inputs · 1 z · 2 y · 3 L · 4 L̄ · 5 ȳ · 6 z̄ · 7 w̄,b̄
    let g = BP.logistic(xv, wv, bv, tv);

    const bar = LR.controls(mount);
    const stepBtn = LR.button(bar, "Step ▸", doStep, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(doStep, LR.reducedMotion ? 1400 : 850);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); step = 0; update(); });
    LR.slider(bar, "x", -3, 3, 0.1, xv, function (v) { xv = v; update(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "w", -2, 2, 0.1, wv, function (v) { wv = v; update(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "b", -2, 2, 0.1, bv, function (v) { bv = v; update(); }, (v) => LR.fmtF(v, 1));
    const tg = LR.el("div", "toggle-group");
    bar.appendChild(tg);
    const t0 = LR.button(tg, "t = 0", function () { tv = 0; t0.classList.add("on"); t1.classList.remove("on"); update(); }, "small");
    const t1 = LR.button(tg, "t = 1", function () { tv = 1; t1.classList.add("on"); t0.classList.remove("on"); update(); }, "small on");

    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Computation graph of logistic least squares; stepping runs the forward pass left to right, then the backward pass right to left",
    });
    const narrate = LR.el("div", "stepper-narrate", "");
    mount.appendChild(narrate);
    const ro = LR.readout(mount, [
      { k: "L", label: "loss L" },
      { k: "wbar", label: "w̄ = ∂L/∂w" },
      { k: "bbar", label: "b̄ = ∂L/∂b" },
    ]);

    const N = logisticLayout();
    const f2 = (v) => LR.fmtF(v, 2), f3 = (v) => LR.fmtF(v, 3), f4 = (v) => LR.fmtF(v, 4);

    function doStep() {
      if (step >= 7) { stopPlay(); return; }
      step += 1;
      update();
      if (step >= 7) stopPlay();
    }

    function narrateFor() {
      switch (step) {
        case 0: return "Inputs set: x = " + f2(g.x) + ", w = " + f2(g.w) + ", b = " + f2(g.b) + ", t = " + g.t + ". Press <b>Step</b> to run the forward pass.";
        case 1: return "<b>Forward.</b> z = wx + b = (" + f2(g.w) + ")(" + f2(g.x) + ") + (" + f2(g.b) + ") = <b>" + f3(g.z) + "</b>";
        case 2: return "<b>Forward.</b> y = σ(z) = σ(" + f3(g.z) + ") = <b>" + f3(g.y) + "</b>";
        case 3: return "<b>Forward done.</b> L = ½(y − t)² = ½(" + f3(g.y) + " − " + g.t + ")² = <b>" + f4(g.L) + "</b>. Step again for the backward pass.";
        case 4: return "<b>Backward.</b> Seed the sweep at the top: L̄ = ∂L/∂L = <b>1</b>.";
        case 5: return "<b>Backward.</b> ȳ = L̄ · (y − t) = <b>" + f3(g.ybar) + "</b>. The edge y → L contributed its local derivative, y − t.";
        case 6: return "<b>Backward.</b> z̄ = ȳ · σ′(z) = (" + f3(g.ybar) + ")(" + f3(g.ds) + ") = <b>" + f4(g.zbar) + "</b>, using σ′(z) = y(1 − y).";
        case 7: return "<b>Done.</b> w̄ = z̄·x = (" + f4(g.zbar) + ")(" + f2(g.x) + ") = <b>" + f4(g.wbar) + "</b> and b̄ = z̄·1 = <b>" + f4(g.bbar) + "</b>. z̄ was computed once and reused by both.";
      }
      return "";
    }

    function update() {
      g = BP.logistic(xv, wv, bv, tv);
      narrate.innerHTML = narrateFor();
      ro.set("L", step >= 3 ? f4(g.L) : "–");
      ro.set("wbar", step >= 7 ? f4(g.wbar) : "–", C.orange);
      ro.set("bbar", step >= 7 ? f4(g.bbar) : "–", C.orange);
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      // forward edges (lit while their head node is being computed)
      fwdArrow(ctx, N.x, N.z, { lit: step === 1 });
      fwdArrow(ctx, N.w, N.z, { lit: step === 1 });
      fwdArrow(ctx, N.b, N.z, { lit: step === 1 });
      fwdArrow(ctx, N.z, N.y, { lit: step === 2, label: "σ" });
      fwdArrow(ctx, N.y, N.L, { lit: step === 3 });
      fwdArrow(ctx, N.t, N.L, { lit: step === 3 });

      // backward arrows accumulate as the sweep proceeds
      if (step >= 5) backArrow(ctx, N.L, N.y, { label: "× (y−t)" });
      if (step >= 6) backArrow(ctx, N.y, N.z, { label: "× σ′(z)" });
      if (step >= 7) {
        backArrow(ctx, N.z, N.w, { label: "× x", bend: 40 });
        backArrow(ctx, N.z, N.b, { label: "× 1", bend: -40 });
      }

      drawNode(ctx, N.x, { val: f2(g.x), dim: false });
      drawNode(ctx, N.w, { val: f2(g.w), lit: step === 7, bar: step >= 7 ? "w̄ " + f3(g.wbar) : null });
      drawNode(ctx, N.b, { val: f2(g.b), lit: step === 7, bar: step >= 7 ? "b̄ " + f3(g.bbar) : null });
      drawNode(ctx, N.z, { val: step >= 1 ? f3(g.z) : "?", lit: step === 1 || step === 6, bar: step >= 6 ? "z̄ " + f3(g.zbar) : null });
      drawNode(ctx, N.y, { val: step >= 2 ? f3(g.y) : "?", lit: step === 2 || step === 5, bar: step >= 5 ? "ȳ " + f3(g.ybar) : null });
      drawNode(ctx, N.L, { val: step >= 3 ? f4(g.L) : "?", lit: step === 3 || step === 4, bar: step >= 4 ? "L̄ 1" : null });
      drawNode(ctx, N.t, { val: String(g.t), dim: true });

      // direction legend
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.faint;
      ctx.fillText("forward: values →", 620, 26);
      ctx.fillStyle = C.orange;
      ctx.fillText("← backward: error signals", 620, 44);
    }
    update();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.2 — the local-derivative inspector
     ════════════════════════════════════════════════════════════ */
  LR.figs.inspector = function (mount) {
    LR.header(
      mount,
      "Every edge owns one derivative",
      "Hover an edge on the graph (or use the buttons) to see its local derivative and how it multiplies into the running error signal."
    );

    let xv = 2.0, wv = 0.5, bv = -0.5, tv = 1;
    let g = BP.logistic(xv, wv, bv, tv);
    let sel = "wz";

    const f2 = (v) => LR.fmtF(v, 2), f3 = (v) => LR.fmtF(v, 3), f4 = (v) => LR.fmtF(v, 4);

    const EDGES = [
      { id: "xz", a: "x", b: "z", chip: "x → z",
        text: () => "Edge <b>x → z</b> (z = wx + b). Local derivative: ∂z/∂x = w = <b>" + f2(g.w) + "</b>. Backprop would use it as x̄ = z̄·w = (" + f4(g.zbar) + ")(" + f2(g.w) + ") = <b>" + f4(g.zbar * g.w) + "</b>. We never update the input x, but its error signal exists all the same." },
      { id: "wz", a: "w", b: "z", chip: "w → z",
        text: () => "Edge <b>w → z</b> (z = wx + b). Local derivative: ∂z/∂w = x = <b>" + f2(g.x) + "</b>. Used in the backward pass as w̄ = z̄·x = (" + f4(g.zbar) + ")(" + f2(g.x) + ") = <b>" + f4(g.wbar) + "</b>." },
      { id: "bz", a: "b", b: "z", chip: "b → z",
        text: () => "Edge <b>b → z</b> (z = wx + b). Local derivative: ∂z/∂b = 1. Used as b̄ = z̄·1 = <b>" + f4(g.bbar) + "</b>. The bias inherits z's error signal unchanged." },
      { id: "zy", a: "z", b: "y", chip: "z → y",
        text: () => "Edge <b>z → y</b> (y = σ(z)). Local derivative: ∂y/∂z = σ′(z) = y(1−y) = (" + f3(g.y) + ")(" + f3(1 - g.y) + ") = <b>" + f4(g.ds) + "</b>. Used as z̄ = ȳ·σ′(z) = (" + f4(g.ybar) + ")(" + f4(g.ds) + ") = <b>" + f4(g.zbar) + "</b>." },
      { id: "yL", a: "y", b: "L", chip: "y → L",
        text: () => "Edge <b>y → L</b> (L = ½(y−t)²). Local derivative: ∂L/∂y = y − t = <b>" + f4(g.ybar) + "</b>. Used as ȳ = L̄·(y−t) = (1)(" + f4(g.ybar) + ") = <b>" + f4(g.ybar) + "</b>. The whole backward pass starts from this edge." },
      { id: "tL", a: "t", b: "L", chip: "t → L",
        text: () => "Edge <b>t → L</b> (L = ½(y−t)²). Local derivative: ∂L/∂t = t − y = <b>" + f4(g.t - g.y) + "</b>. Perfectly well defined, and never used: the target is data, not a parameter, so no gradient step ever touches it." },
    ];

    const bar = LR.controls(mount);
    const chips = {};
    EDGES.forEach(function (e) {
      chips[e.id] = LR.button(bar, e.chip, function () { sel = e.id; syncChips(); update(); }, "small" + (e.id === sel ? " on" : ""));
    });
    LR.slider(bar, "x", -3, 3, 0.1, xv, function (v) { xv = v; update(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "w", -2, 2, 0.1, wv, function (v) { wv = v; update(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "b", -2, 2, 0.1, bv, function (v) { bv = v; update(); }, (v) => LR.fmtF(v, 1));

    function syncChips() {
      EDGES.forEach((e) => chips[e.id].classList.toggle("on", e.id === sel));
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Computation graph whose edges can be selected to inspect their local derivatives",
    });
    const panel = LR.el("div", "stepper-narrate", "");
    mount.appendChild(panel);

    const N = logisticLayout();
    let segs = {}; // edge id → trimmed segment, for hover hit-testing

    function distToSeg(p, s) {
      const vx = s.x2 - s.x1, vy = s.y2 - s.y1;
      const len2 = vx * vx + vy * vy;
      let u = ((p.x - s.x1) * vx + (p.y - s.y1) * vy) / len2;
      u = Math.max(0, Math.min(1, u));
      return Math.hypot(p.x - (s.x1 + u * vx), p.y - (s.y1 + u * vy));
    }
    LR.drag(cv, W, H, {
      hit: () => false,
      hover: function (p) {
        let best = null, bd = 14;
        for (const id in segs) {
          const d = distToSeg(p, segs[id]);
          if (d < bd) { bd = d; best = id; }
        }
        if (best && best !== sel) { sel = best; syncChips(); update(); }
      },
    });

    function update() {
      g = BP.logistic(xv, wv, bv, tv);
      const e = EDGES.find((e) => e.id === sel);
      panel.innerHTML = e.text();
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      segs.xz = fwdArrow(ctx, N.x, N.z, { lit: sel === "xz" });
      segs.wz = fwdArrow(ctx, N.w, N.z, { lit: sel === "wz" });
      segs.bz = fwdArrow(ctx, N.b, N.z, { lit: sel === "bz" });
      segs.zy = fwdArrow(ctx, N.z, N.y, { lit: sel === "zy", label: "σ" });
      segs.yL = fwdArrow(ctx, N.y, N.L, { lit: sel === "yL" });
      segs.tL = fwdArrow(ctx, N.t, N.L, { lit: sel === "tL" });

      drawNode(ctx, N.x, { val: LR.fmtF(g.x, 2) });
      drawNode(ctx, N.w, { val: LR.fmtF(g.w, 2), bar: "w̄ " + LR.fmtF(g.wbar, 3) });
      drawNode(ctx, N.b, { val: LR.fmtF(g.b, 2), bar: "b̄ " + LR.fmtF(g.bbar, 3) });
      drawNode(ctx, N.z, { val: LR.fmtF(g.z, 3), bar: "z̄ " + LR.fmtF(g.zbar, 3) });
      drawNode(ctx, N.y, { val: LR.fmtF(g.y, 3), bar: "ȳ " + LR.fmtF(g.ybar, 3) });
      drawNode(ctx, N.L, { val: LR.fmtF(g.L, 4), bar: "L̄ 1" });
      drawNode(ctx, N.t, { val: String(g.t), dim: true });

      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.faint;
      ctx.fillText("full backward pass shown; hover an edge to zoom in on its one job", 100, 26);
    }
    update();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.3 — fan-out: two backward contributions sum at w
     ════════════════════════════════════════════════════════════ */
  LR.figs.fanout = function (mount) {
    LR.header(
      mount,
      "Fan-out: w feeds two children",
      "L2-regularized regression: Lreg = L + λR with R = ½w². Fixed x = 2, b = −0.5, t = 1. Step the backward pass and watch two contributions converge on w."
    );

    let wv = 0.5, lam = 0.1;
    let step = 0; // 0 idle · 1 L̄reg · 2 L̄,R̄ · 3 ȳ · 4 z̄ · 5 contributions · 6 sum
    let g = BP.regularized(2, wv, -0.5, 1, lam);

    const bar = LR.controls(mount);
    LR.button(bar, "Step ◂", doStep, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(doStep, LR.reducedMotion ? 1400 : 850);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); step = 0; update(); });
    LR.slider(bar, "w", -2, 2, 0.1, wv, function (v) { wv = v; update(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "λ (decay strength)", 0, 1, 0.05, lam, function (v) { lam = v; update(); }, (v) => LR.fmtF(v, 2));

    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }
    function doStep() {
      if (step >= 6) { stopPlay(); return; }
      step += 1;
      update();
      if (step >= 6) stopPlay();
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 370, {
      aria: "Computation graph where the weight w fans out to the prediction path and the regularizer; the backward pass sums both contributions",
    });
    const narrate = LR.el("div", "stepper-narrate", "");
    mount.appendChild(narrate);
    const ro = LR.readout(mount, [
      { k: "c1", label: "via z: z̄·x" },
      { k: "c2", label: "via R: R̄·w" },
      { k: "sum", label: "w̄ = sum" },
    ]);

    const N = {
      x: { x: 95, y: 75, r: 22, label: "x" },
      w: { x: 95, y: 190, r: 24, label: "w" },
      b: { x: 95, y: 300, r: 22, label: "b" },
      z: { x: 300, y: 130, r: 25, label: "z" },
      y: { x: 445, y: 130, r: 25, label: "y" },
      L: { x: 585, y: 130, r: 25, label: "L" },
      t: { x: 445, y: 40, r: 19, label: "t" },
      R: { x: 300, y: 280, r: 25, label: "R" },
      Lr: { x: 715, y: 205, r: 27, label: "Lreg" },
    };
    const f2 = (v) => LR.fmtF(v, 2), f3 = (v) => LR.fmtF(v, 3), f4 = (v) => LR.fmtF(v, 4);

    function narrateFor() {
      switch (step) {
        case 0: return "Forward pass already done (values under each node). w has <b>fan-out 2</b>: it feeds z = wx + b and R = ½w². Press <b>Step ◂</b>.";
        case 1: return "Seed at the final loss: L̄reg = ∂Lreg/∂Lreg = <b>1</b>.";
        case 2: return "Lreg = L + λR has two parents. Local derivatives: ∂Lreg/∂L = 1 so L̄ = <b>1</b>; ∂Lreg/∂R = λ so R̄ = <b>" + f3(g.Rbar) + "</b>. Both branches now carry gradient.";
        case 3: return "Down the prediction branch: ȳ = L̄·(y − t) = <b>" + f3(g.ybar) + "</b>.";
        case 4: return "z̄ = ȳ·σ′(z) = (" + f3(g.ybar) + ")(" + f3(g.ds) + ") = <b>" + f4(g.zbar) + "</b>.";
        case 5: return "Two contributions arrive at w. Through z: z̄·x = (" + f4(g.zbar) + ")(2) = <b>" + f4(g.wbarZ) + "</b>. Through R: R̄·w = (" + f3(g.Rbar) + ")(" + f2(g.w) + ") = <b>" + f4(g.wbarR) + "</b>.";
        case 6: return "<b>Sum them.</b> w̄ = z̄·x + R̄·w = " + f4(g.wbarZ) + " + " + f4(g.wbarR) + " = <b>" + f4(g.wbar) + "</b>. That is the multivariate chain rule: one term per child, added.";
      }
      return "";
    }

    function update() {
      g = BP.regularized(2, wv, -0.5, 1, lam);
      narrate.innerHTML = narrateFor();
      ro.set("c1", step >= 5 ? f4(g.wbarZ) : "–", C.purple);
      ro.set("c2", step >= 5 ? f4(g.wbarR) : "–", C.green);
      ro.set("sum", step >= 6 ? f4(g.wbar) : "–", C.orange);
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      fwdArrow(ctx, N.x, N.z, {});
      fwdArrow(ctx, N.w, N.z, {});
      fwdArrow(ctx, N.b, N.z, {});
      fwdArrow(ctx, N.z, N.y, { label: "σ" });
      fwdArrow(ctx, N.y, N.L, {});
      fwdArrow(ctx, N.t, N.L, {});
      fwdArrow(ctx, N.w, N.R, { label: "½w²" });
      fwdArrow(ctx, N.L, N.Lr, {});
      fwdArrow(ctx, N.R, N.Lr, { label: "× λ" });

      if (step >= 2) {
        backArrow(ctx, N.Lr, N.L, { label: "× 1" });
        backArrow(ctx, N.Lr, N.R, { label: "× λ", bend: -40, color: C.green });
      }
      if (step >= 3) backArrow(ctx, N.L, N.y, { label: "× (y−t)" });
      if (step >= 4) backArrow(ctx, N.y, N.z, { label: "× σ′(z)" });
      if (step >= 5) {
        backArrow(ctx, N.z, N.w, { label: "× x", bend: 42, color: C.purple });
        backArrow(ctx, N.R, N.w, { label: "× w", bend: 42, color: C.green });
      }

      drawNode(ctx, N.x, { val: "2.00", dim: true });
      drawNode(ctx, N.w, {
        val: f2(g.w), lit: step >= 5,
        bar: step >= 6 ? "w̄ " + f3(g.wbar) : step >= 5 ? f3(g.wbarZ) + " + " + f3(g.wbarR) : null,
      });
      drawNode(ctx, N.b, { val: "−0.50", dim: true, bar: step >= 5 ? "b̄ " + f3(g.bbar) : null });
      drawNode(ctx, N.z, { val: f3(g.z), lit: step === 4, bar: step >= 4 ? "z̄ " + f3(g.zbar) : null });
      drawNode(ctx, N.y, { val: f3(g.y), lit: step === 3, bar: step >= 3 ? "ȳ " + f3(g.ybar) : null });
      drawNode(ctx, N.L, { val: f4(g.L), lit: step === 2, bar: step >= 2 ? "L̄ 1" : null });
      drawNode(ctx, N.t, { val: "1", dim: true });
      drawNode(ctx, N.R, { val: f3(g.R), lit: step === 2, bar: step >= 2 ? "R̄ " + f3(g.Rbar) : null });
      drawNode(ctx, N.Lr, { val: f4(g.Lreg), lit: step === 1, bar: step >= 1 ? "1" : null });
    }
    update();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — backprop on a two-layer MLP (2-3-1), step by step
     ════════════════════════════════════════════════════════════ */
  LR.figs.mlp = function (mount) {
    LR.header(
      mount,
      "The algorithm on a real network",
      "A 2-3-1 MLP with fixed weights (printed on the edges). Forward fills activations layer by layer; backward flows error signals layer by layer."
    );

    // weights hand-picked for legible round numbers (flagged: chosen for the
    // figure, not from the source; the equations are the source's)
    const W1 = [[1.0, -1.0], [-0.5, 1.5], [0.5, 0.5]];
    const B1 = [0.0, 0.5, -0.5];
    const W2 = [1.5, -1.0, 0.5];
    const B2 = 0.2;
    const Hn = 3;

    let x1 = 1.0, x2 = -1.0, tv = 1;
    let step = 0; // 0 inputs · 1 z · 2 h · 3 y · 4 L · 5 ȳ · 6 h̄,w̄² · 7 z̄ · 8 w̄¹
    let S = compute();

    function compute() {
      const x = [x1, x2];
      const z = W1.map((r, i) => r[0] * x[0] + r[1] * x[1] + B1[i]);
      const h = z.map(BP.sig);
      const y = W2[0] * h[0] + W2[1] * h[1] + W2[2] * h[2] + B2;
      const ybar = y - tv;
      const L = 0.5 * ybar * ybar;
      const w2bar = h.map((hi) => ybar * hi);
      const hbar = W2.map((w) => ybar * w);          // h̄_i = Σ_k ȳ_k w_ki (one output)
      const ds = h.map((hi) => hi * (1 - hi));       // σ'(z_i)
      const zbar = hbar.map((hb, i) => hb * ds[i]);  // z̄_i = h̄_i σ'(z_i)
      const W1bar = zbar.map((zb) => [zb * x[0], zb * x[1]]); // w̄_ij = z̄_i x_j
      return { x, z, h, y, L, ybar, w2bar, hbar, ds, zbar, W1bar };
    }

    const bar = LR.controls(mount);
    LR.button(bar, "Step ▸", doStep, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(doStep, LR.reducedMotion ? 1500 : 950);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); step = 0; update(); });
    LR.slider(bar, "x₁", -2, 2, 0.5, x1, function (v) { x1 = v; update(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "x₂", -2, 2, 0.5, x2, function (v) { x2 = v; update(); }, (v) => LR.fmtF(v, 1));
    const tg = LR.el("div", "toggle-group");
    bar.appendChild(tg);
    const t0 = LR.button(tg, "t = 0", function () { tv = 0; t0.classList.add("on"); t1.classList.remove("on"); update(); }, "small");
    const t1 = LR.button(tg, "t = 1", function () { tv = 1; t1.classList.add("on"); t0.classList.remove("on"); update(); }, "small on");

    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }
    function doStep() {
      if (step >= 8) { stopPlay(); return; }
      step += 1;
      update();
      if (step >= 8) stopPlay();
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 420, {
      aria: "Two-layer MLP diagram; stepping runs the forward pass and then the backward pass layer by layer with all values shown",
    });
    const narrate = LR.el("div", "stepper-narrate", "");
    mount.appendChild(narrate);
    const vars = LR.el("div", "stepper-vars", "");
    mount.appendChild(vars);

    const NX = [{ x: 95, y: 145, r: 22, label: "x₁" }, { x: 95, y: 285, r: 22, label: "x₂" }];
    const NH = [
      { x: 350, y: 90, r: 32, label: "h₁" },
      { x: 350, y: 215, r: 32, label: "h₂" },
      { x: 350, y: 340, r: 32, label: "h₃" },
    ];
    const NY = { x: 600, y: 215, r: 26, label: "y" };
    const NL = { x: 735, y: 215, r: 24, label: "L" };
    const NT = { x: 600, y: 345, r: 19, label: "t" };
    const f2 = (v) => LR.fmtF(v, 2), f3 = (v) => LR.fmtF(v, 3);
    const vec = (a, d) => "[" + a.map((v) => LR.fmtF(v, d === undefined ? 3 : d)).join(", ") + "]";

    function narrateFor() {
      switch (step) {
        case 0: return "Inputs x = [" + f2(x1) + ", " + f2(x2) + "], target t = " + tv + ". First-layer weights sit on the edges. Press <b>Step</b>.";
        case 1: return "<b>Forward, layer 1.</b> z₁ = (" + f2(W1[0][0]) + ")(" + f2(x1) + ") + (" + f2(W1[0][1]) + ")(" + f2(x2) + ") + " + f2(B1[0]) + " = <b>" + f3(S.z[0]) + "</b>; likewise z₂ = <b>" + f3(S.z[1]) + "</b>, z₃ = <b>" + f3(S.z[2]) + "</b>.";
        case 2: return "<b>Forward, activations.</b> h = σ(z) = " + vec(S.h) + ".";
        case 3: return "<b>Forward, layer 2.</b> y = (" + f2(W2[0]) + ")(" + f3(S.h[0]) + ") + (" + f2(W2[1]) + ")(" + f3(S.h[1]) + ") + (" + f2(W2[2]) + ")(" + f3(S.h[2]) + ") + " + f2(B2) + " = <b>" + f3(S.y) + "</b>.";
        case 4: return "<b>Forward done.</b> L = ½(y − t)² = ½(" + f3(S.y) + " − " + tv + ")² = <b>" + LR.fmtF(S.L, 4) + "</b>. Step again to go backward.";
        case 5: return "<b>Backward.</b> ȳ = y − t = <b>" + f3(S.ybar) + "</b>. One error signal at the output; it will now fan back across every hidden unit.";
        case 6: return "<b>Backward, layer 2.</b> h̄ᵢ = ȳ·wᵢ⁽²⁾: h̄₁ = (" + f3(S.ybar) + ")(" + f2(W2[0]) + ") = <b>" + f3(S.hbar[0]) + "</b>, h̄₂ = <b>" + f3(S.hbar[1]) + "</b>, h̄₃ = <b>" + f3(S.hbar[2]) + "</b>. Also w̄ᵢ⁽²⁾ = ȳ·hᵢ = " + vec(S.w2bar) + ".";
        case 7: return "<b>Backward, through σ.</b> z̄ᵢ = h̄ᵢ·σ′(zᵢ) with σ′(z) = h(1−h) = " + vec(S.ds) + ", giving z̄ = " + vec(S.zbar) + ".";
        case 8: return "<b>Done.</b> w̄ᵢⱼ⁽¹⁾ = z̄ᵢ·xⱼ, e.g. w̄₁₁ = (" + f3(S.zbar[0]) + ")(" + f2(x1) + ") = <b>" + f3(S.W1bar[0][0]) + "</b>. Every gradient of both layers came from one forward and one backward sweep.";
      }
      return "";
    }

    function update() {
      S = compute();
      narrate.innerHTML = narrateFor();
      const rows = [];
      rows.push('<span class="var-name">z</span> = ' + (step >= 1 ? vec(S.z) : "–") + '&nbsp;&nbsp;<span class="var-name">h</span> = ' + (step >= 2 ? vec(S.h) : "–"));
      rows.push('<span class="var-name">y</span> = ' + (step >= 3 ? f3(S.y) : "–") + '&nbsp;&nbsp;<span class="var-name">L</span> = ' + (step >= 4 ? LR.fmtF(S.L, 4) : "–") + '&nbsp;&nbsp;<span class="var-name">ȳ</span> = ' + (step >= 5 ? f3(S.ybar) : "–"));
      rows.push('<span class="var-name">w̄⁽²⁾</span> = ' + (step >= 6 ? vec(S.w2bar) : "–") + '&nbsp;&nbsp;<span class="var-name">h̄</span> = ' + (step >= 6 ? vec(S.hbar) : "–"));
      rows.push('<span class="var-name">z̄</span> = ' + (step >= 7 ? vec(S.zbar) : "–"));
      rows.push('<span class="var-name">W̄⁽¹⁾</span> = ' + (step >= 8 ? "[" + S.W1bar.map((r) => vec(r)).join(", ") + "]" : "–"));
      vars.innerHTML = rows.join("<br>");
      draw();
    }

    function wLabel(ctx, a, b, txt, frac, lit) {
      const x = a.x + (b.x - a.x) * frac, y = a.y + (b.y - a.y) * frac;
      ctx.font = "600 10.5px 'JetBrains Mono', Menlo, monospace";
      ctx.fillStyle = lit ? C.orange : "#8a8a8a";
      ctx.textAlign = "center";
      ctx.fillText(txt, x, y - 5);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      // layer-1 edges; at the last step the labels become the actual
      // weight gradients w̄_ij = z̄_i x_j, computed live
      for (let i = 0; i < Hn; i++) {
        for (let j = 0; j < 2; j++) {
          fwdArrow(ctx, NX[j], NH[i], { lit: step === 1 || step === 8 });
          const txt = step >= 8 ? "w̄ " + f3(S.W1bar[i][j]) : f2(W1[i][j]);
          wLabel(ctx, NX[j], NH[i], txt, 0.34 + j * 0.12, step === 1 || step === 8);
        }
      }
      // layer-2 edges
      for (let i = 0; i < Hn; i++) {
        fwdArrow(ctx, NH[i], NY, { lit: step === 3 || step === 6 });
        wLabel(ctx, NH[i], NY, f2(W2[i]), 0.5, step === 3 || step === 6);
      }
      fwdArrow(ctx, NY, NL, { lit: step === 4 });
      fwdArrow(ctx, NT, NL, { lit: step === 4 });

      if (step >= 5) backArrow(ctx, NL, NY, { label: "× (y−t)" });
      if (step >= 6) for (let i = 0; i < Hn; i++) backArrow(ctx, NY, NH[i], { label: "× w" + ["₁", "₂", "₃"][i] + "⁽²⁾", bend: i === 1 ? 30 : 34 });

      // nodes
      drawNode(ctx, NX[0], { val: f2(x1) });
      drawNode(ctx, NX[1], { val: f2(x2) });
      for (let i = 0; i < Hn; i++) {
        const n = NH[i];
        const bars = [];
        if (step >= 6) bars.push("h̄ " + f3(S.hbar[i]));
        if (step >= 7) bars.push("z̄ " + f3(S.zbar[i]));
        drawNode(ctx, n, { val: "", lit: step === 1 || step === 2 || step === 6 || step === 7, bar: bars.length ? bars.join("  ") : null });
        ctx.font = "700 10.5px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText(step >= 1 ? "z " + f2(S.z[i]) : "z ?", n.x, n.y + 12);
        ctx.fillText(step >= 2 ? "h " + f2(S.h[i]) : "h ?", n.x, n.y + 24);
      }
      drawNode(ctx, NY, { val: step >= 3 ? f3(S.y) : "?", lit: step === 3 || step === 5, bar: step >= 5 ? "ȳ " + f3(S.ybar) : null });
      drawNode(ctx, NL, { val: step >= 4 ? LR.fmtF(S.L, 4) : "?", lit: step === 4 });
      drawNode(ctx, NT, { val: String(tv), dim: true });

      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.faint;
      ctx.fillText("b⁽¹⁾ = [0.0, 0.5, −0.5],  b⁽²⁾ = 0.2 (biases not drawn)", 96, 26);
    }
    update();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.2 — backward() as a rerunnable code exercise with a
     finite-difference gradient check
     ════════════════════════════════════════════════════════════ */
  LR.figs.backex = function (mount) {
    LR.header(
      mount,
      "Run backward(), then audit it",
      "The function below executes with your inputs; every run is checked against central finite differences computed on the spot."
    );

    const CODE =
      "def backward(x, w, b, t):\n" +
      "    z = w * x + b                 # forward pass\n" +
      "    y = 1 / (1 + np.exp(-z))\n" +
      "    L = 0.5 * (y - t) ** 2\n" +
      "    y_bar = y - t                 # dL/dy\n" +
      "    z_bar = y_bar * y * (1 - y)   # dL/dz, sigma'(z) = y(1-y)\n" +
      "    w_bar = z_bar * x             # dL/dw\n" +
      "    b_bar = z_bar                 # dL/db\n" +
      "    return w_bar, b_bar";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);
    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);

    let xv = 2.0, wv = 0.5, bv = -0.5, tv = 1;
    LR.slider(controls, "x", -3, 3, 0.25, xv, (v) => { xv = v; }, (v) => LR.fmtF(v, 2));
    LR.slider(controls, "w", -2, 2, 0.25, wv, (v) => { wv = v; }, (v) => LR.fmtF(v, 2));
    LR.slider(controls, "b", -2, 2, 0.25, bv, (v) => { bv = v; }, (v) => LR.fmtF(v, 2));
    const lab = LR.el("label", "", "t = ");
    const sel = document.createElement("select");
    [0, 1].forEach(function (v) {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      if (v === 1) o.selected = true;
      sel.appendChild(o);
    });
    sel.style.cssText = "font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid #cccccc";
    sel.addEventListener("change", () => { tv = parseInt(sel.value, 10); });
    lab.appendChild(sel);
    controls.appendChild(lab);
    LR.button(controls, "Run ▸", run, "primary small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; backward(x=2.00, w=0.50, b=-0.50, t=1)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    function run() {
      const g = BP.logistic(xv, wv, bv, tv);
      // real central differences, computed right here
      const eps = 1e-5;
      const ndw = BP.numGrad((w) => BP.logistic(xv, w, bv, tv).L, wv, eps);
      const ndb = BP.numGrad((b) => BP.logistic(xv, wv, b, tv).L, bv, eps);
      const dw = Math.abs(ndw - g.wbar), db = Math.abs(ndb - g.bbar);
      const okW = dw < 1e-7, okB = db < 1e-7;
      out.innerHTML =
        "&gt;&gt;&gt; backward(x=" + LR.fmtF(xv, 2) + ", w=" + LR.fmtF(wv, 2) + ", b=" + LR.fmtF(bv, 2) + ", t=" + tv + ")\n" +
        "forward:   z = " + LR.fmtF(BP.logistic(xv, wv, bv, tv).z, 6) + "   y = " + LR.fmtF(g.y, 6) + "   L = " + LR.fmtF(g.L, 6) + "\n" +
        "backward:  w_bar = " + LR.fmtF(g.wbar, 8) + "   b_bar = " + LR.fmtF(g.bbar, 8) + "\n" +
        "check vs central differences (eps = 1e-5):\n" +
        "  dL/dw ≈ " + LR.fmtF(ndw, 8) + "   |diff| = " + dw.toExponential(1) + "  " + (okW ? "✓" : "✗ MISMATCH") + "\n" +
        "  dL/db ≈ " + LR.fmtF(ndb, 8) + "   |diff| = " + db.toExponential(1) + "  " + (okB ? "✓" : "✗ MISMATCH");
    }
  };
})();
