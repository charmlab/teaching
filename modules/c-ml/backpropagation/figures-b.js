/* ══════════════════════════════════════════════════════════════
   figures-b.js — Backpropagation lesson figures, sections 4–6
   Fig 4.3 trainer · Fig 5.1 sgdrace · Fig 6.1 augment
   Fig 6.2 decay · Fig 6.3 earlystop
   Uses the shared backprop core (LR.bp) from figures-a.js.
   All gradients are real backprop gradients; all curves are
   really trained. Deterministic via LR.rng seeds.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const BP = LR.bp;

  /* ════════════════════════════════════════════════════════════
     Fig 4.3 — gradient descent on a network: the trainer
     ════════════════════════════════════════════════════════════ */
  LR.figs.trainer = function (mount) {
    LR.header(
      mount,
      "Backprop trains a network, live",
      "XOR-pattern data that no linear model can separate. A 2-6-1 MLP learns it by repeated forward-backward-step, exactly the algorithm of this section."
    );

    // data: four blobs in an XOR pattern (deterministic)
    const rand = LR.rng(33);
    const PTS = [], T = [];
    [[1.2, 1.2, 1], [3.8, 3.8, 1], [1.2, 3.8, 0], [3.8, 1.2, 0]].forEach(function (c) {
      for (let i = 0; i < 10; i++) {
        PTS.push({ x: c[0] + LR.gauss(rand) * 0.5, y: c[1] + LR.gauss(rand) * 0.5, cls: c[2] });
        T.push(c[2]);
      }
    });
    // standardized copies for training (inputs roughly in [-1, 1])
    const X = PTS.map((p) => [(p.x - 2.5) / 1.5, (p.y - 2.5) / 1.5]);

    let lr = 2.5, net, epoch, losses;
    function resetNet() {
      net = BP.mlpInit(6, 2, 5, 1.0, true); // sigmoid output, squared loss
      epoch = 0;
      losses = [BP.mlpLoss(net, X, T)];
    }
    resetNet();

    const bar = LR.controls(mount);
    LR.button(bar, "Step ×10 ▸", function () { runEpochs(10); render(); }, "primary");
    let raf = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (raf) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      const perFrame = LR.reducedMotion ? 4 : 15;
      const tick = function () {
        runEpochs(perFrame);
        render();
        if (epoch < 20000) raf = requestAnimationFrame(tick);
        else stopPlay();
      };
      raf = requestAnimationFrame(tick);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); resetNet(); render(); });
    LR.slider(bar, "α (learning rate)", 0.2, 5, 0.1, lr, function (v) { lr = v; }, (v) => LR.fmtF(v, 1));

    function stopPlay() {
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      playBtn.textContent = "Play ▸▸";
    }
    function runEpochs(n) {
      for (let i = 0; i < n; i++) {
        const L = BP.mlpBatchStep(net, X, T, lr);
        epoch += 1;
        if (epoch % 5 === 0 || epoch < 20) losses.push(L);
      }
    }

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Data + decision boundary"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Training loss (log scale)"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 340, { aria: "Scatter of XOR-pattern data with the network's decision boundary updating as it trains" });
    const RC = LR.canvas(rb, 400, 340, { aria: "Training loss curve falling as backprop-driven gradient descent runs" });

    const ro = LR.readout(mount, [
      { k: "epoch", label: "epoch" },
      { k: "loss", label: "loss J" },
      { k: "acc", label: "train accuracy" },
    ]);

    const P = { x0: 46, y0: 14, w: LC.W - 62, h: LC.H - 58, xmin: 0, xmax: 5, ymin: 0, ymax: 5, xlabel: "x₁", ylabel: "x₂" };
    const CLS_COLOR = { 1: C.green, 0: C.purple };

    function render() {
      // ---- left: boundary + data ----
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      const BLK = 7;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const y = BP.mlpForward(net, [(wx - 2.5) / 1.5, (wy - 2.5) / 1.5]).y;
          const a = Math.min(0.30, Math.abs(y - 0.5) * 0.7 + 0.05);
          ctx.fillStyle = y > 0.5 ? "rgba(102,102,102," + a + ")" : "rgba(150,150,150," + a + ")";
          ctx.fillRect(px, py, BLK, BLK);
        }
      }
      const SC = LR.plot(ctx, P);
      PTS.forEach(function (p) {
        LR.dot(ctx, SC.sx(p.x), SC.sy(p.y), 5, CLS_COLOR[p.cls], "#ffffff");
      });

      // ---- right: loss curve (log10) ----
      const c2 = RC.ctx;
      c2.clearRect(0, 0, RC.W, RC.H);
      const lo = Math.log10(Math.max(1e-5, Math.min.apply(null, losses)));
      const P2 = {
        x0: 52, y0: 14, w: RC.W - 68, h: RC.H - 58,
        xmin: 0, xmax: Math.max(50, epoch), ymin: Math.min(-3.5, Math.floor(lo)), ymax: 0,
        xlabel: "epoch", ylabel: "log₁₀ J",
      };
      const S2 = LR.plot(c2, P2);
      c2.strokeStyle = C.orange;
      c2.lineWidth = 2.2;
      c2.beginPath();
      const step = Math.max(1, Math.floor(losses.length / 300));
      for (let i = 0; i < losses.length; i += step) {
        const ep = i < 20 ? i : (i - 20) * 5 + 20; // matches the logging schedule
        const lx = S2.sx(Math.min(ep, P2.xmax));
        const ly = S2.sy(Math.max(P2.ymin, Math.log10(Math.max(1e-9, losses[i]))));
        i === 0 ? c2.moveTo(lx, ly) : c2.lineTo(lx, ly);
      }
      c2.stroke();

      let acc = 0;
      X.forEach(function (p, i) { if ((BP.mlpForward(net, p).y > 0.5 ? 1 : 0) === T[i]) acc++; });
      ro.set("epoch", String(epoch));
      ro.set("loss", LR.fmtF(losses[losses.length - 1], 4), C.orange);
      ro.set("acc", LR.fmtF((100 * acc) / X.length, 0) + "%", acc === X.length ? C.green : undefined);
    }
    render();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — batch vs mini-batch vs SGD on a real loss surface
     ════════════════════════════════════════════════════════════ */
  LR.figs.sgdrace = function (mount) {
    LR.header(
      mount,
      "Three optimizers, one surface",
      "The exact quadratic loss surface of a 40-point linear regression. Every path uses real per-example gradients; only the number of examples per update differs."
    );

    const pts = LR.houseData(40, 11, 0.9);
    const N = pts.length;
    const opt = LR.fit1d(pts);                       // true minimum
    const Jmin = LR.loss1d(pts, opt.w0, opt.w1);
    // Hessian of J = 1/(2N) Σ (w0 + w1 x − t)²  →  [[1, x̄], [x̄, x²̄]]
    let mx = 0, mxx = 0;
    pts.forEach((p) => { mx += p.x; mxx += p.x * p.x; });
    mx /= N; mxx /= N;
    const a = 1, bH = mx, cH = mxx;
    const tr2 = (a + cH) / 2, disc = Math.sqrt(tr2 * tr2 - (a * cH - bH * bH));
    const l1 = tr2 + disc, l2 = tr2 - disc;          // eigenvalues (l1 large)
    const v1 = (function () { const vx = bH, vy = l1 - a, n2 = Math.hypot(vx, vy); return { x: vx / n2, y: vy / n2 }; })();
    const v2 = { x: -v1.y, y: v1.x };

    const START = [4.5, 3.5];
    let lr = 0.1, S = 8, stepN = 0;
    let runs;
    function gradOn(w, idxs) {
      let g0 = 0, g1 = 0;
      for (const i of idxs) {
        const r = w[0] + w[1] * pts[i].x - pts[i].t;
        g0 += r; g1 += r * pts[i].x;
      }
      return [g0 / idxs.length, g1 / idxs.length];
    }
    const ALL = Array.from({ length: N }, (_, i) => i);
    function sampleIdx(rng, k) {
      // k distinct indices, deterministic partial Fisher-Yates
      const arr = ALL.slice();
      for (let i = 0; i < k; i++) {
        const j = i + Math.floor(rng() * (N - i));
        const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      }
      return arr.slice(0, k);
    }
    function resetRuns() {
      stepN = 0;
      runs = [
        { name: "batch (S = " + N + ")", color: C.green, w: START.slice(), path: [START.slice()], J: [LR.loss1d(pts, START[0], START[1])], pick: () => ALL },
        { name: "mini-batch", color: C.amber, w: START.slice(), path: [START.slice()], J: [LR.loss1d(pts, START[0], START[1])], rng: LR.rng(8), pick: function () { return sampleIdx(this.rng, S); } },
        { name: "SGD (S = 1)", color: C.purple, w: START.slice(), path: [START.slice()], J: [LR.loss1d(pts, START[0], START[1])], rng: LR.rng(7), pick: function () { return [Math.floor(this.rng() * N)]; } },
      ];
    }
    resetRuns();

    const bar = LR.controls(mount);
    LR.button(bar, "Step ▸", function () { doSteps(1); render(); }, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(function () {
        doSteps(1);
        render();
        if (stepN >= 300) stopPlay();
      }, LR.reducedMotion ? 300 : 90);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); resetRuns(); render(); });
    LR.slider(bar, "α (learning rate)", 0.02, 0.2, 0.01, lr, function (v) { lr = v; }, (v) => LR.fmtF(v, 2));
    LR.slider(bar, "S (mini-batch size)", 2, 40, 1, S, function (v) {
      S = Math.round(v);
      stopPlay(); resetRuns(); render(); // deterministic restart so paths are comparable
    }, (v) => String(Math.round(v)));

    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }
    function doSteps(n) {
      for (let s = 0; s < n; s++) {
        if (stepN >= 300) return;
        runs.forEach(function (r) {
          const g = gradOn(r.w, r.pick());
          r.w = [r.w[0] - lr * g[0], r.w[1] - lr * g[1]];
          r.path.push(r.w.slice());
          r.J.push(LR.loss1d(pts, r.w[0], r.w[1]));
        });
        stepN += 1;
      }
    }

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Weight space (w₀, w₁) with J contours"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Full-data loss J after each update"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 430, 350, { aria: "Contour plot of the quadratic loss with the paths of batch, mini-batch, and stochastic gradient descent" });
    const RC = LR.canvas(rb, 370, 350, { aria: "Loss versus update count for the three optimizers; smaller batches give bumpier curves" });

    const ro = LR.readout(mount, [
      { k: "step", label: "updates" },
      { k: "jb", label: "J batch" },
      { k: "jm", label: "J mini" },
      { k: "js", label: "J sgd" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 52, y0: 16, w: LC.W - 70, h: LC.H - 60, xmin: -2.5, xmax: 5.5, ymin: -1.0, ymax: 4.2, xlabel: "w₀ (intercept)", ylabel: "w₁ (slope)" };

    function render() {
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      const SC = LR.plot(ctx, P);
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // exact contour ellipses of the quadratic: J = Jmin + ½ Δw'HΔw
      ctx.strokeStyle = "#d8d8d8";
      ctx.lineWidth = 1.2;
      [0.05, 0.25, 0.8, 2, 5, 10, 20, 40].forEach(function (lev) {
        const r1 = Math.sqrt((2 * lev) / l1), r2 = Math.sqrt((2 * lev) / l2);
        ctx.beginPath();
        for (let i = 0; i <= 72; i++) {
          const u = (i / 72) * Math.PI * 2;
          const dw0 = r1 * Math.cos(u) * v1.x + r2 * Math.sin(u) * v2.x;
          const dw1 = r1 * Math.cos(u) * v1.y + r2 * Math.sin(u) * v2.y;
          const px = SC.sx(opt.w0 + dw0), py = SC.sy(opt.w1 + dw1);
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.stroke();
      });

      // paths
      runs.forEach(function (r) {
        ctx.strokeStyle = r.color;
        ctx.lineWidth = 1.9;
        ctx.beginPath();
        r.path.forEach(function (w, i) {
          const px = SC.sx(w[0]), py = SC.sy(w[1]);
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.stroke();
        const last = r.path[r.path.length - 1];
        LR.dot(ctx, SC.sx(last[0]), SC.sy(last[1]), 5, r.color, "#ffffff");
      });
      ctx.restore();

      // start + optimum
      LR.dot(ctx, SC.sx(START[0]), SC.sy(START[1]), 5.5, C.text, "#ffffff");
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "left";
      ctx.fillText("start", SC.sx(START[0]) + 9, SC.sy(START[1]) + 4);
      LR.dot(ctx, SC.sx(opt.w0), SC.sy(opt.w1), 5.5, "#ffffff", C.red);
      ctx.fillStyle = C.red;
      ctx.fillText("minimum", SC.sx(opt.w0) + 9, SC.sy(opt.w1) + 4);

      // legend
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.fillText("— batch (S=40)", P.x0 + 8, P.y0 + 16);
      ctx.fillStyle = C.amber; ctx.fillText("— mini-batch (S=" + S + ")", P.x0 + 8, P.y0 + 32);
      ctx.fillStyle = C.purple; ctx.fillText("— SGD (S=1)", P.x0 + 8, P.y0 + 48);

      // ---- right: loss curves ----
      const c2 = RC.ctx;
      c2.clearRect(0, 0, RC.W, RC.H);
      const P2 = {
        x0: 52, y0: 16, w: RC.W - 68, h: RC.H - 60,
        xmin: 0, xmax: Math.max(40, stepN), ymin: Math.floor(Math.log10(Jmin)), ymax: 2,
        xlabel: "update", ylabel: "log₁₀ J",
      };
      const S2 = LR.plot(c2, P2);
      c2.save();
      c2.beginPath(); c2.rect(P2.x0, P2.y0, P2.w, P2.h); c2.clip();
      runs.forEach(function (r) {
        c2.strokeStyle = r.color;
        c2.lineWidth = 1.8;
        c2.beginPath();
        r.J.forEach(function (J, i) {
          const px = S2.sx(i), py = S2.sy(Math.log10(Math.max(1e-6, J)));
          i === 0 ? c2.moveTo(px, py) : c2.lineTo(px, py);
        });
        c2.stroke();
      });
      // Jmin reference
      c2.strokeStyle = C.red; c2.setLineDash([4, 4]); c2.lineWidth = 1.2;
      c2.beginPath(); c2.moveTo(S2.sx(0), S2.sy(Math.log10(Jmin))); c2.lineTo(S2.sx(P2.xmax), S2.sy(Math.log10(Jmin))); c2.stroke();
      c2.setLineDash([]);
      c2.font = "600 10.5px Inter, sans-serif";
      c2.fillStyle = C.red; c2.textAlign = "left";
      c2.fillText("J at the minimum", P2.x0 + 6, S2.sy(Math.log10(Jmin)) - 5);
      c2.restore();

      ro.set("step", String(stepN));
      ro.set("jb", LR.fmtF(runs[0].J[runs[0].J.length - 1], 3), C.green);
      ro.set("jm", LR.fmtF(runs[1].J[runs[1].J.length - 1], 3), C.amber);
      ro.set("js", LR.fmtF(runs[2].J[runs[2].J.length - 1], 3), C.purple);

      if (stepN >= 60) {
        msg.show(
          "Same learning rate, same start. Batch glides; SGD jitters around the valley yet keeps descending, because each noisy gradient is unbiased: right on average, wrong step by step. Slide S and the mini-batch path morphs between the two. Per epoch, SGD makes " + N + " updates for the FLOPs of one batch update.",
          "info"
        );
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — data augmentation gallery
     ════════════════════════════════════════════════════════════ */
  LR.figs.augment = function (mount) {
    LR.header(
      mount,
      "Manufacture training data",
      "One labelled digit, four augmented clones. Toggle the transforms; each clone draws its own random shift, angle, and noise."
    );

    // 8×8 "7" (same digit style as the kNN lesson's vectorizer)
    const G = [
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, .9, .9, .9, .9, .9, .7, 0],
      [0, 0, 0, 0, 0, .8, .9, 0],
      [0, 0, 0, 0, .7, .9, .2, 0],
      [0, 0, 0, .6, .9, .3, 0, 0],
      [0, 0, .4, .9, .5, 0, 0, 0],
      [0, 0, .8, .9, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ];

    let doShift = true, doRotate = true, doFlip = false, doNoise = true;
    let nonce = 1;

    const bar = LR.controls(mount);
    function toggle(label, get, set) {
      const btn = LR.button(bar, label, function () {
        set(!get());
        btn.classList.toggle("on", get());
        draw();
      }, "small" + (get() ? " on" : ""));
      return btn;
    }
    toggle("Shift", () => doShift, (v) => (doShift = v));
    toggle("Rotate", () => doRotate, (v) => (doRotate = v));
    toggle("Flip ↔", () => doFlip, (v) => (doFlip = v));
    toggle("Noise", () => doNoise, (v) => (doNoise = v));
    LR.button(bar, "Resample ⟲", function () { nonce += 1; draw(); }, "primary small");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 240, {
      aria: "A training digit seven and four augmented copies with random shifts, rotations, optional mirroring, and noise",
    });
    const msg = LR.msg(mount);

    // offscreen 8×8 pixel buffer
    const off = document.createElement("canvas");
    off.width = 8; off.height = 8;
    const offCtx = off.getContext("2d");

    function renderDigit(noiseSig, rand) {
      const img = offCtx.createImageData(8, 8);
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          let v = G[r][c];
          if (noiseSig > 0) v = Math.max(0, Math.min(1, v + LR.gauss(rand) * noiseSig));
          const shade = Math.round(248 - v * 235);
          const k = (r * 8 + c) * 4;
          img.data[k] = shade; img.data[k + 1] = shade; img.data[k + 2] = shade; img.data[k + 3] = 255;
        }
      }
      offCtx.putImageData(img, 0, 0);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.imageSmoothingEnabled = false;
      const SZ = 120, Y0 = 52;

      // original
      renderDigit(0, null);
      ctx.drawImage(off, 40, Y0, SZ, SZ);
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1.4;
      ctx.strokeRect(40, Y0, SZ, SZ);
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "center";
      ctx.fillText("original, label “7”", 40 + SZ / 2, Y0 - 10);

      LR.arrow(ctx, 178, Y0 + SZ / 2, 216, Y0 + SZ / 2, C.muted, 2);

      // four augmented clones, each with its own deterministic draw
      for (let s = 0; s < 4; s++) {
        const rand = LR.rng(1000 + nonce * 17 + s);
        const dx = doShift ? (rand() - 0.5) * 26 : 0;
        const dy = doShift ? (rand() - 0.5) * 26 : 0;
        const ang = doRotate ? ((rand() - 0.5) * 50 * Math.PI) / 180 : 0;
        renderDigit(doNoise ? 0.16 : 0, rand);
        const cx = 250 + s * 140 + SZ / 2, cy = Y0 + SZ / 2;
        ctx.save();
        ctx.translate(cx + dx * 0.4, cy + dy * 0.4);
        ctx.rotate(ang);
        if (doFlip) ctx.scale(-1, 1);
        ctx.drawImage(off, -SZ / 2, -SZ / 2, SZ, SZ);
        ctx.restore();
        ctx.strokeStyle = "#cfcfcf"; ctx.lineWidth = 1;
        ctx.strokeRect(250 + s * 140, Y0, SZ, SZ);
        ctx.fillStyle = C.muted;
        ctx.font = "600 11px Inter, sans-serif";
        const tags = [];
        if (doShift) tags.push("shift");
        if (doRotate) tags.push("rot");
        if (doFlip) tags.push("flip");
        if (doNoise) tags.push("noise");
        ctx.fillText(tags.length ? tags.join("+") : "copy", 250 + s * 140 + SZ / 2, Y0 + SZ + 18);
      }

      if (doFlip) {
        msg.show(
          "Careful: a mirrored 7 is not a 7 anymore. Horizontal flips preserve labels for cars and cats, not for digits or text. Valid transforms are a property of the task. (And whatever you pick: training data only, never the test set.)",
          "bad"
        );
      } else {
        msg.show(
          "Shifts, small rotations, and noise all keep a 7 a 7, so every clone is a free labelled example. Augment the training set only; the test set must stay untouched reality.",
          "good"
        );
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.2 — weight decay taming a degree-9 polynomial
     ════════════════════════════════════════════════════════════ */
  LR.figs.decay = function (mount) {
    LR.header(
      mount,
      "Small weights cannot oscillate wildly",
      "A degree-9 polynomial fit to 10 noisy points, with an L2 penalty λ. The coefficients below are the actual fitted values."
    );

    // data: noisy sine on [-1, 1], unevenly sampled (uneven spacing is what
    // lets the degree-9 interpolant swing wildly between points)
    const rand = LR.rng(9);
    const xs = [], ts = [];
    for (let i = 0; i < 10; i++) xs.push(-1 + 2 * rand());
    xs.sort((a, b) => a - b);
    for (let i = 0; i < 10; i++) ts.push(Math.sin(Math.PI * xs[i]) + LR.gauss(rand) * 0.25);
    const DEG = 9;

    let loglam = -8;
    const bar = LR.controls(mount);
    LR.slider(bar, "log₁₀ λ", -8, 0, 0.25, loglam, function (v) { loglam = v; draw(); },
      (v) => "10^" + LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 380, {
      aria: "Polynomial fit whose wild oscillations calm down as the weight decay strength increases",
    });
    const ro = LR.readout(mount, [
      { k: "lam", label: "λ" },
      { k: "shrink", label: "per-step factor (1 − αλ), α = 0.1" },
      { k: "maxw", label: "max |wⱼ|" },
    ]);
    const tblWrap = LR.el("div", "wtable-wrap");
    mount.appendChild(tblWrap);

    const P = { x0: 56, y0: 18, w: W - 76, h: H - 62, xmin: -1, xmax: 1, ymin: -2.2, ymax: 2.2, xlabel: "x", ylabel: "t" };

    function draw() {
      const lam = Math.pow(10, loglam);
      const w = LR.polyfit(xs, ts, DEG, lam);   // real ridge solution

      ctx.clearRect(0, 0, W, H);
      const SC = LR.plot(ctx, P);
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // true function (what the noise hides)
      ctx.strokeStyle = "#bbbbbb";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const x = -1 + (2 * i) / 200;
        const y = Math.sin(Math.PI * x);
        i === 0 ? ctx.moveTo(SC.sx(x), SC.sy(y)) : ctx.lineTo(SC.sx(x), SC.sy(y));
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // fitted polynomial
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let i = 0; i <= 400; i++) {
        const x = -1 + (2 * i) / 400;
        const y = LR.polyval(w, x);
        i === 0 ? ctx.moveTo(SC.sx(x), SC.sy(y)) : ctx.lineTo(SC.sx(x), SC.sy(y));
      }
      ctx.stroke();
      ctx.restore();

      xs.forEach((x, i) => LR.dot(ctx, SC.sx(x), SC.sy(ts[i]), 5, C.text));

      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "#9a9a9a";
      ctx.fillText("- - true function", P.x0 + 10, P.y0 + 16);
      ctx.fillStyle = C.orange;
      ctx.fillText("— degree-9 fit, λ = " + lam.toExponential(1), P.x0 + 10, P.y0 + 33);

      // coefficients table
      const maxw = Math.max.apply(null, w.map((v) => Math.abs(v)));
      let html = "<table class='wtable'><tr>";
      for (let j = 0; j <= DEG; j++) html += "<th>w<sub>" + j + "</sub></th>";
      html += "</tr><tr>";
      for (let j = 0; j <= DEG; j++) {
        const big = Math.abs(w[j]) > 10;
        const txt = Math.abs(w[j]) >= 1000 ? w[j].toExponential(1) : LR.fmtF(w[j], Math.abs(w[j]) >= 10 ? 1 : 2);
        html += "<td" + (big ? " class='w-big'" : "") + ">" + txt + "</td>";
      }
      html += "</tr></table>";
      tblWrap.innerHTML = html;

      ro.set("lam", lam.toExponential(2));
      ro.set("shrink", LR.fmtF(1 - 0.1 * lam, 4), C.orange);
      ro.set("maxw", maxw >= 1000 ? maxw.toExponential(1) : LR.fmtF(maxw, 1), maxw > 10 ? C.red : C.green);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.3 — early stopping on a genuinely trained network
     ════════════════════════════════════════════════════════════ */
  LR.figs.earlystop = function (mount) {
    LR.header(
      mount,
      "Stop at the turn",
      "A 1-16-1 network really trained for 48,000 epochs on 10 noisy points, snapshotted every 240 epochs. Drag the stop line, watch the fit, then lock in your call."
    );

    /* train for real (deterministic; ~half a second, done once on figure load) */
    const NTR = 10, NOISE = 0.4, EPOCHS = 48000, LOG = 240, LRATE = 0.8;
    const r2 = LR.rng(99);
    const xtr = [], ttr = [];
    for (let i = 0; i < NTR; i++) {
      const x = i / (NTR - 1);
      xtr.push(x);
      ttr.push(Math.sin(2 * Math.PI * x) + LR.gauss(r2) * NOISE);
    }
    const xvl = [], tvl = [];
    for (let i = 0; i < 40; i++) {
      const x = r2();
      xvl.push(x);
      tvl.push(Math.sin(2 * Math.PI * x) + LR.gauss(r2) * NOISE);
    }
    const net = BP.mlp1Init(16, 5, 3.0);
    const Xtr = xtr.map((x) => [x]), Xvl = xvl.map((x) => [x]);
    const trL = [], vlL = [], snaps = [];
    for (let e = 0; e <= EPOCHS; e++) {
      if (e % LOG === 0) {
        trL.push(BP.mlpLoss(net, Xtr, ttr));
        vlL.push(BP.mlpLoss(net, Xvl, tvl));
        snaps.push({ W1: net.W1.map((r) => r.slice()), b1: net.b1.slice(), w2: net.w2.slice(), b2: net.b2 });
      }
      BP.mlpBatchStep(net, Xtr, ttr, LRATE);
    }
    let mi = 1;
    for (let i = 1; i < vlL.length; i++) if (vlL[i] < vlL[mi]) mi = i;

    let sel = 4;         // reader's stop line (log index)
    let locked = false;

    const bar = LR.controls(mount);
    LR.button(bar, "Lock in my stop ▸", function () {
      locked = true;
      const d = sel - mi;
      if (Math.abs(d) <= 8) {
        msg.show("✓ Good call. The validation minimum is at epoch " + mi * LOG + " (val loss " + LR.fmtF(vlL[mi], 4) + "); you stopped at epoch " + sel * LOG + " with val loss " + LR.fmtF(vlL[sel], 4) + ". Training longer would only have memorized noise.", "good");
      } else if (d < 0) {
        msg.show("Too early. At epoch " + sel * LOG + " validation loss was still falling (" + LR.fmtF(vlL[sel], 4) + " vs the eventual minimum " + LR.fmtF(vlL[mi], 4) + " at epoch " + mi * LOG + "). The network was still learning real structure; stopping here underfits.", "bad");
      } else {
        msg.show("Too late. Validation bottomed out at epoch " + mi * LOG + " (" + LR.fmtF(vlL[mi], 4) + ") and has risen to " + LR.fmtF(vlL[sel], 4) + " by your stop. Training loss kept falling, but that extra fit is noise memorization. Look at the wiggles in the right panel.", "bad");
      }
      render();
    }, "primary");
    LR.button(bar, "Reset ⟲", function () { locked = false; sel = 4; msg.hide(); render(); });

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Loss curves (drag the stop line)"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "The network's fit at your stop epoch"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 430, 350, { aria: "Training and validation loss versus epoch with a draggable early-stopping line" });
    const RC = LR.canvas(rb, 370, 350, { aria: "The network's fitted curve at the chosen stopping epoch, over the noisy training points" });

    const ro = LR.readout(mount, [
      { k: "ep", label: "your stop epoch" },
      { k: "tr", label: "train loss" },
      { k: "vl", label: "val loss" },
    ]);
    const msg = LR.msg(mount);

    // x axis: log10(epoch), from the first logged point
    const X0 = Math.log10(LOG), X1 = Math.log10(EPOCHS);
    const YMAX = Math.min(0.6, Math.max.apply(null, vlL.slice(2)) * 1.15);
    const P = { x0: 56, y0: 16, w: LC.W - 74, h: LC.H - 60, xmin: X0, xmax: X1, ymin: 0, ymax: YMAX, xlabel: "epoch (log scale)", ylabel: "loss", xticks: [] };
    let SC = null;

    LR.drag(LC.cv, LC.W, LC.H, {
      hit: (p) => SC && p.x > P.x0 - 10 && p.x < P.x0 + P.w + 10,
      move: function (p) {
        const lx = P.xmin + ((p.x - P.x0) / P.w) * (P.xmax - P.xmin);
        const ep = Math.pow(10, Math.max(X0, Math.min(X1, lx)));
        sel = Math.max(1, Math.min(vlL.length - 1, Math.round(ep / LOG)));
        render();
      },
      down: function (p) {
        const lx = P.xmin + ((p.x - P.x0) / P.w) * (P.xmax - P.xmin);
        const ep = Math.pow(10, Math.max(X0, Math.min(X1, lx)));
        sel = Math.max(1, Math.min(vlL.length - 1, Math.round(ep / LOG)));
        render();
      },
    });

    function curve(ctx, arr, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      let started = false;
      for (let i = 1; i < arr.length; i++) {
        const px = SC.sx(Math.log10(i * LOG));
        const py = SC.sy(Math.min(P.ymax, arr[i]));
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    function render() {
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      SC = LR.plot(ctx, P);
      // custom log ticks
      ctx.font = "10.5px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.textAlign = "center";
      [300, 1000, 3000, 10000, 30000].forEach(function (ep) {
        const px = SC.sx(Math.log10(ep));
        ctx.fillText(ep >= 1000 ? ep / 1000 + "k" : String(ep), px, P.y0 + P.h + 16);
        ctx.strokeStyle = "#f0f0f0";
        ctx.beginPath(); ctx.moveTo(px, P.y0); ctx.lineTo(px, P.y0 + P.h); ctx.stroke();
      });

      curve(ctx, trL, C.green);
      curve(ctx, vlL, C.orange);

      // stop line
      const sx = SC.sx(Math.log10(sel * LOG));
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(sx, P.y0); ctx.lineTo(sx, P.y0 + P.h); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "center";
      ctx.fillText("stop here", sx, P.y0 + 12);

      // reveal the true minimum once locked
      if (locked) {
        const mx2 = SC.sx(Math.log10(mi * LOG));
        LR.dot(ctx, mx2, SC.sy(vlL[mi]), 6, "#ffffff", C.red);
        ctx.fillStyle = C.red;
        ctx.fillText("val minimum", mx2, SC.sy(vlL[mi]) - 12);
      }

      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.green; ctx.fillText("— training loss", P.x0 + P.w - 118, P.y0 + 16);
      ctx.fillStyle = C.orange; ctx.fillText("— validation loss", P.x0 + P.w - 118, P.y0 + 32);

      // ---- right: snapshot fit ----
      const c2 = RC.ctx;
      c2.clearRect(0, 0, RC.W, RC.H);
      const P2 = { x0: 46, y0: 16, w: RC.W - 62, h: RC.H - 60, xmin: 0, xmax: 1, ymin: -2, ymax: 2, xlabel: "x", ylabel: "t" };
      const S2 = LR.plot(c2, P2);
      c2.save();
      c2.beginPath(); c2.rect(P2.x0, P2.y0, P2.w, P2.h); c2.clip();
      // true function
      c2.strokeStyle = "#bbbbbb"; c2.lineWidth = 1.4; c2.setLineDash([5, 4]);
      c2.beginPath();
      for (let i = 0; i <= 150; i++) {
        const x = i / 150, y = Math.sin(2 * Math.PI * x);
        i === 0 ? c2.moveTo(S2.sx(x), S2.sy(y)) : c2.lineTo(S2.sx(x), S2.sy(y));
      }
      c2.stroke();
      c2.setLineDash([]);
      // snapshot network
      const sn = snaps[sel];
      const snNet = { H: 16, d: 1, outSig: false, W1: sn.W1, b1: sn.b1, w2: sn.w2, b2: sn.b2 };
      c2.strokeStyle = C.orange; c2.lineWidth = 2.4;
      c2.beginPath();
      for (let i = 0; i <= 250; i++) {
        const x = i / 250;
        const y = BP.mlpForward(snNet, [x]).y;
        i === 0 ? c2.moveTo(S2.sx(x), S2.sy(y)) : c2.lineTo(S2.sx(x), S2.sy(y));
      }
      c2.stroke();
      c2.restore();
      xtr.forEach((x, i) => LR.dot(c2, S2.sx(x), S2.sy(ttr[i]), 4.5, C.text));
      c2.font = "600 11px Inter, sans-serif";
      c2.fillStyle = C.muted; c2.textAlign = "left";
      c2.fillText("epoch " + sel * LOG, P2.x0 + 8, P2.y0 + 16);

      ro.set("ep", String(sel * LOG));
      ro.set("tr", LR.fmtF(trL[sel], 4), C.green);
      ro.set("vl", LR.fmtF(vlL[sel], 4), C.orange);
    }
    render();
  };
})();
