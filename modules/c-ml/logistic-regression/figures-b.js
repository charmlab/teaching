/* ══════════════════════════════════════════════════════════════
   figures-b.js — Logistic Regression lesson figures, sections 4–6
   Fig 4.1 trainer · Fig 4.2 codeex · Fig 4.3 threshold
   Fig 5.1 reg · Fig 6.1 softmax
   Uses the shared logistic core (LR.logit) from figures-a.js.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const LG = LR.logit;

  const CLS_COLOR = { 0: C.green, 1: C.purple };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — the gradient-descent trainer (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.trainer = function (mount) {
    LR.header(
      mount,
      "Watch it learn (and watch it stall)",
      "Full-batch gradient descent, one epoch per step. The red-ringed point is mislabelled on purpose. Compare cross-entropy against sigmoid+squared, especially after Sabotage."
    );

    // deterministic 2-D data + one deliberately mislabelled point (the
    // "confident mistake" of the lecture, once the model gets confident)
    const rand = LR.rng(29);
    const pts = [];
    for (let i = 0; i < 12; i++) pts.push({ x: 2.0 + LR.gauss(rand) * 0.65, y: 2.2 + LR.gauss(rand) * 0.65, t: 0 });
    for (let i = 0; i < 12; i++) pts.push({ x: 4.4 + LR.gauss(rand) * 0.65, y: 4.1 + LR.gauss(rand) * 0.65, t: 1 });
    const flagged = { x: 4.6, y: 4.4, t: 0, flag: true }; // a class-0 label deep in class-1 land
    pts.push(flagged);
    const aug = pts.map((p) => ({ f: [1, p.x, p.y], t: p.t, src: p }));
    const flaggedAug = aug[aug.length - 1];

    // a good CE fit, used only to build the sabotage init (confidently wrong)
    const wGood = LG.fit(aug, { lr: 0.5, iters: 2500, lam: 0.01 });

    let lossKind = "ce";
    let alpha = 0.3;
    let w, epoch, hist;
    const MAXEP = 400;

    function reset(sab) {
      w = sab ? wGood.map((v) => -2.5 * v) : [0, 0, 0];
      epoch = 0;
      hist = [curLoss()];
      heatDirty = true;
    }
    function curLoss() {
      return lossKind === "ce" ? LG.lossCE(w, aug, 0) : LG.lossSQ(w, aug);
    }
    function stepOnce() {
      if (epoch >= MAXEP) return false;
      const g = LG.grad(w, aug, 0, lossKind);
      for (let j = 0; j < 3; j++) w[j] -= alpha * g[j];
      epoch++;
      hist.push(curLoss());
      heatDirty = true;
      return true;
    }

    // ---- controls ----
    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bCE = LR.button(group, "cross-entropy", function () { lossKind = "ce"; bCE.classList.add("on"); bSQ.classList.remove("on"); stopPlay(); reset(false); drawAll(); }, "small on");
    const bSQ = LR.button(group, "σ + squared", function () { lossKind = "sq"; bSQ.classList.add("on"); bCE.classList.remove("on"); stopPlay(); reset(false); drawAll(); }, "small");
    LR.slider(bar, "α (learning rate)", 0.05, 4, 0.05, alpha, function (v) { alpha = v; }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Step ▸", function () { stopPlay(); stepOnce(); drawAll(); }, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(function () {
        let alive = true;
        const per = LR.reducedMotion ? 1 : 2;
        for (let i = 0; i < per; i++) alive = stepOnce() && alive;
        drawAll();
        if (!alive) stopPlay();
      }, LR.reducedMotion ? 500 : 70);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); beatOff(); reset(false); msg.hide(); drawAll(); });
    LR.button(bar, "Sabotage init 💥", function () {
      stopPlay(); beatOff(); reset(true); drawAll();
      msg.show(
        "Weights set confidently WRONG (every prediction near 0 or 1, on the wrong side). Now press Play. Cross-entropy climbs out fast; sigmoid+squared barely moves, because y(1−y) ≈ 0 mutes every gradient. Switch the loss and compare epochs-to-recover.",
        "info"
      );
    });
    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    // ---- layout ----
    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "data space: boundary + probability shading"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "training curve: loss per epoch"));
    pane.appendChild(lb); pane.appendChild(rb);
    const L = LR.canvas(lb, 400, 340, { aria: "Scatter with the decision boundary and probability shading updating as gradient descent runs" });
    const R = LR.canvas(rb, 400, 340, { aria: "Loss versus epoch curve for the current training run" });

    const ro = LR.readout(mount, [
      { k: "ep", label: "epoch" },
      { k: "w", label: "w" },
      { k: "J", label: "loss J" },
      { k: "fg", label: "flagged point: y and its gradient pull" },
    ]);

    function fmtP(v) { return v > 0.9995 || v < 0.0005 ? LR.fmtF(v, 5) : LR.fmtF(v, 3); }

    // ---- "correct the visual" beat ----
    const beatBox = LR.el("div", "beat-box");
    mount.appendChild(beatBox);
    const msg = LR.msg(mount);
    let beatState = null;
    LR.button(bar, "Pause & predict 🤔", function () {
      stopPlay();
      // pick the worst-classified point right now
      let worst = aug[0], we = -1;
      aug.forEach(function (p) {
        const y = LG.sig(LG.dot(w, p.f));
        const e = Math.abs(y - p.t);
        if (e > we) { we = e; worst = p; }
      });
      beatState = { p: worst, y: LG.sig(LG.dot(w, worst.f)) };
      beatBox.innerHTML = "";
      const qEl = LR.el("div", "beat-q",
        "The highlighted point has t = " + worst.t + " and the model currently says y = " + fmtP(beatState.y) +
        ". After the next gradient step, which way does its predicted y move?");
      beatBox.appendChild(qEl);
      const btnRow = LR.el("div", "beat-btns");
      beatBox.appendChild(btnRow);
      const answer = function (guessUp) {
        // simulate one real step without committing it
        const g = LG.grad(w, aug, 0, lossKind);
        const w2 = w.map((v, j) => v - alpha * g[j]);
        const yNew = LG.sig(LG.dot(w2, beatState.p.f));
        const up = yNew > beatState.y;
        if (guessUp === up) {
          msg.show("✓ Right: y goes " + (up ? "up" : "down") + " (" + fmtP(beatState.y) + " → " + fmtP(yNew) + "). The (y − t) residual on every point votes on the step, and the boundary moves to serve the majority pull.", "good");
        } else {
          msg.show("Not this time: y actually goes " + (up ? "up" : "down") + " (" + fmtP(beatState.y) + " → " + fmtP(yNew) + "). Remember the step aggregates ALL points' (y − t)x pulls, so one point can lose the vote, especially the mislabelled one.", "bad");
        }
        beatOff(); drawAll();
      };
      LR.button(btnRow, "y moves up, toward 1", function () { answer(true); }, "small");
      LR.button(btnRow, "y moves down, toward 0", function () { answer(false); }, "small");
      drawAll();
    });
    function beatOff() { beatState = null; beatBox.innerHTML = ""; }

    // ---- rendering ----
    const P = { x0: 46, y0: 14, w: L.W - 62, h: L.H - 58, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };
    let heatDirty = true;
    const off = document.createElement("canvas");
    off.width = L.W; off.height = L.H;
    const offCtx = off.getContext("2d");
    function renderHeat() {
      offCtx.clearRect(0, 0, L.W, L.H);
      const BLK = 7;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const p = LG.sig(w[0] + w[1] * wx + w[2] * wy);
          offCtx.fillStyle = "rgba(150,150,150," + (p * 0.4).toFixed(3) + ")";
          offCtx.fillRect(px, py, BLK, BLK);
        }
      }
      heatDirty = false;
    }

    function drawLeft() {
      if (heatDirty) renderHeat();
      const ctx = L.ctx;
      ctx.clearRect(0, 0, L.W, L.H);
      ctx.drawImage(off, 0, 0, L.W, L.H);
      const { sx, sy } = LR.plot(ctx, P);

      // boundary line w0 + w1 x + w2 y = 0
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      if (Math.hypot(w[1], w[2]) > 1e-9) {
        const s2 = w[1] * w[1] + w[2] * w[2];
        const cx = -w[0] * w[1] / s2, cy = -w[0] * w[2] / s2;
        const dir = { x: -w[2], y: w[1] };
        const n = Math.hypot(dir.x, dir.y);
        ctx.strokeStyle = C.orange; ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(sx(cx - 40 * dir.x / n), sy(cy - 40 * dir.y / n));
        ctx.lineTo(sx(cx + 40 * dir.x / n), sy(cy + 40 * dir.y / n));
        ctx.stroke();
      }
      ctx.restore();

      pts.forEach(function (p) {
        LR.dot(ctx, sx(p.x), sy(p.y), p.flag ? 7 : 5, CLS_COLOR[p.t], p.flag ? C.red : "#ffffff");
      });
      if (beatState) {
        const p = beatState.p.src;
        LR.dot(ctx, sx(p.x), sy(p.y), 11, "rgba(0,0,0,0)", C.amber);
      }
      const ctx2 = L.ctx;
      ctx2.font = "600 11px Inter, sans-serif"; ctx2.textAlign = "left";
      ctx2.fillStyle = C.red;
      ctx2.fillText("red ring = mislabelled on purpose", P.x0 + 8, P.y0 + 14);
    }

    function drawRight() {
      const ctx = R.ctx;
      ctx.clearRect(0, 0, R.W, R.H);
      const xmax = Math.max(50, hist.length - 1);
      const ymax = Math.max(0.5, Math.max.apply(null, hist.filter(isFinite)) * 1.06);
      const PR = { x0: 50, y0: 14, w: R.W - 66, h: R.H - 58, xmin: 0, xmax: xmax, ymin: 0, ymax: ymax, xlabel: "epoch", ylabel: "J (" + (lossKind === "ce" ? "cross-entropy" : "σ+squared") + ")" };
      const { sx, sy } = LR.plot(ctx, PR);
      ctx.save();
      ctx.beginPath(); ctx.rect(PR.x0, PR.y0, PR.w, PR.h); ctx.clip();
      ctx.strokeStyle = lossKind === "ce" ? C.orange : C.purple;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      hist.forEach(function (v, i) {
        if (!isFinite(v)) return;
        i === 0 ? ctx.moveTo(sx(i), sy(Math.min(v, ymax))) : ctx.lineTo(sx(i), sy(Math.min(v, ymax)));
      });
      ctx.stroke();
      ctx.restore();
      LR.dot(ctx, sx(hist.length - 1), sy(Math.min(hist[hist.length - 1], ymax)), 4.5, C.text);
    }

    function drawAll() {
      drawLeft(); drawRight();
      const yF = LG.sig(LG.dot(w, flaggedAug.f));
      let rF = yF - flaggedAug.t;
      if (lossKind === "sq") rF *= yF * (1 - yF);
      const pull = Math.abs(rF) * Math.hypot(1, flagged.x, flagged.y);
      ro.set("ep", String(epoch));
      ro.set("w", "[" + w.map((v) => LR.fmtF(v, 2)).join(", ") + "]");
      ro.set("J", LR.fmtF(hist[hist.length - 1], 4), C.orange);
      ro.set("fg", "y = " + LR.fmtF(yF, 3) + ", ‖pull‖ = " + LR.fmtF(pull, 3) + (lossKind === "sq" && yF > 0.9 ? " ← muted by y(1−y)" : ""),
        lossKind === "sq" && yF > 0.9 ? C.red : undefined);
    }

    reset(false);
    drawAll();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.2 — the training loop as a live code exercise
     ════════════════════════════════════════════════════════════ */
  LR.figs.codeex = function (mount) {
    LR.header(
      mount,
      "Run the training loop yourself",
      "This Python is mirrored exactly by the code that executes on Run: the exam dataset, the (y − t)x gradient, your α, λ, and epochs."
    );

    // fixed dataset: the exam example from Fig 3.2
    const hours = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5];
    const pass  = [0,   0,    0,   0,    0,   0,    1,    0,   1,    0,   1,    0,   1,    0,   1,   1,    1,   1,    1,   1];
    const aug = hours.map((h, i) => ({ f: [1, h], t: pass[i] }));

    // the converged optimum, for honest convergence messaging: GD on
    // standardized hours converges tightly, weights mapped back to raw
    const muH = hours.reduce((a, b) => a + b, 0) / hours.length;
    const sdH = Math.sqrt(hours.reduce((a, b) => a + (b - muH) ** 2, 0) / hours.length);
    const augSt = hours.map((h, i) => ({ f: [1, (h - muH) / sdH], t: pass[i] }));
    const wSt = LG.fit(augSt, { lr: 1.0, iters: 20000 });
    const Jstar = LG.lossCE([wSt[0] - (wSt[1] * muH) / sdH, wSt[1] / sdH], aug, 0);

    const CODE =
      "hours = np.array([0.5, 0.75, 1.0, ..., 5.0, 5.5])   # 20 students\n" +
      "t     = np.array([0, 0, 0, ..., 1, 1])              # fail / pass\n" +
      "X = np.c_[np.ones(20), hours]                       # bias column + feature\n" +
      "\n" +
      "def fit(X, t, alpha, lam, epochs):\n" +
      "    w = np.zeros(2)\n" +
      "    for _ in range(epochs):\n" +
      "        y = sigmoid(X @ w)                    # forward\n" +
      "        grad = X.T @ (y - t) / len(t)         # (y - t) x, averaged\n" +
      "        w -= alpha * (grad + 2 * lam * w)     # descend (+ L2 pull)\n" +
      "    return w";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);

    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);
    let alpha = 0.3, lam = 0, epochs = 10000;

    const aLab = LR.el("label", "", "alpha = ");
    const aSel = document.createElement("select");
    [0.01, 0.1, 0.3, 1.0, 3.0].forEach((v) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      if (v === 0.3) o.selected = true;
      aSel.appendChild(o);
    });
    aSel.style.cssText = "font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid #cccccc";
    aSel.setAttribute("aria-label", "learning rate alpha");
    aSel.addEventListener("change", () => { alpha = parseFloat(aSel.value); });
    aLab.appendChild(aSel);
    controls.appendChild(aLab);

    const eLab = LR.el("label", "", "epochs = ");
    const eSel = document.createElement("select");
    [100, 1000, 10000, 50000].forEach((v) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      if (v === 10000) o.selected = true;
      eSel.appendChild(o);
    });
    eSel.style.cssText = aSel.style.cssText;
    eSel.setAttribute("aria-label", "number of epochs");
    eSel.addEventListener("change", () => { epochs = parseInt(eSel.value, 10); });
    eLab.appendChild(eSel);
    controls.appendChild(eLab);

    LR.slider(controls, "lam (λ)", 0, 0.3, 0.01, lam, (v) => { lam = v; }, (v) => LR.fmtF(v, 2));
    LR.button(controls, "Run ▸", run, "primary small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; fit(X, t, alpha=0.3, lam=0.0, epochs=10000)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    function run() {
      // exact JS mirror of the Python above
      let w = [0, 0];
      const N = aug.length;
      const marks = [0, Math.floor(epochs / 4), Math.floor(epochs / 2), epochs];
      const traj = [];
      for (let k = 0; k <= epochs; k++) {
        if (marks.indexOf(k) !== -1) traj.push({ k, J: LG.lossCE(w, aug, lam) });
        if (k === epochs) break;
        const g = LG.grad(w, aug, lam, "ce");
        w[0] -= alpha * g[0];
        w[1] -= alpha * g[1];
        if (!isFinite(w[0]) || !isFinite(w[1])) { traj.push({ k: k + 1, J: NaN }); break; }
      }
      const diverged = !isFinite(w[0]) || !isFinite(w[1]) || !isFinite(LG.lossCE(w, aug, 0));
      let acc = 0;
      aug.forEach((p) => { if ((LG.sig(LG.dot(w, p.f)) >= 0.5 ? 1 : 0) === p.t) acc++; });

      let s = "&gt;&gt;&gt; fit(X, t, alpha=" + alpha + ", lam=" + LR.fmtF(lam, 2) + ", epochs=" + epochs + ")\n";
      s += traj.map((m) => "epoch " + String(m.k).padStart(6) + "   J_CE = " + (isFinite(m.J) ? LR.fmtF(m.J, 4) : "nan (diverged!)")).join("\n") + "\n";
      if (diverged) {
        s += "w = [nan, nan]\n<b>diverged: each step overshoots the valley. Lower alpha.</b>";
      } else {
        s += "w = [" + LR.fmtF(w[0], 4) + ", " + LR.fmtF(w[1], 4) + "]" +
          (lam === 0 ? "   # full convergence → [-4.0777, 1.5046]" : "   # λ &gt; 0 pulls the weights smaller") + "\n";
        s += "train accuracy at τ=0.5: " + acc + "/" + N + " (" + LR.fmtF((acc / N) * 100, 0) + "%)";
        const Jq = traj.length > 1 ? traj[1].J : traj[0].J;
        const Jf = traj[traj.length - 1].J;
        if (lam === 0 && Jf > Jstar + 0.05 && Jq - Jf < 1e-3) {
          s += "\n<b>alpha too big for this valley: the loss stopped falling and is bouncing between the walls. Lower alpha.</b>";
        } else if (lam === 0 && Jf > Jstar + 0.05) {
          s += "\nstill above the optimum J* = " + LR.fmtF(Jstar, 4) + ": crawling. More epochs, or a bigger alpha.";
        } else if (lam === 0) {
          s += "\nconverged near the optimum J* = " + LR.fmtF(Jstar, 4) + ".";
        }
      }
      out.innerHTML = s;
    }
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.3 — threshold τ, confusion matrix, precision/recall/F1
     (flagged addition: these metrics come from the lesson spec,
      not from lecture-06-source-content.md)
     ════════════════════════════════════════════════════════════ */
  LR.figs.threshold = function (mount) {
    LR.header(
      mount,
      "Same model, your decision",
      "Every point is placed at the probability the fixed trained model assigns it. Drag τ (or use the slider) and watch the four cells trade against each other."
    );

    // overlapping 2-D data, fit once; only τ moves afterwards
    const rand = LR.rng(41);
    const pts = [];
    for (let i = 0; i < 30; i++) pts.push({ x: 2.3 + LR.gauss(rand) * 0.95, y: 2.5 + LR.gauss(rand) * 0.95, t: 0 });
    for (let i = 0; i < 30; i++) pts.push({ x: 4.0 + LR.gauss(rand) * 0.95, y: 3.8 + LR.gauss(rand) * 0.95, t: 1 });
    const aug = pts.map((p) => ({ f: [1, p.x, p.y], t: p.t }));
    const w = LG.fit(aug, { lr: 0.5, iters: 3000, lam: 0.01 });
    const scored = aug.map(function (p, i) {
      return { p: LG.sig(LG.dot(w, p.f)), t: p.t, jit: rand() };
    });

    let tau = 0.5;

    const bar = LR.controls(mount);
    const tS = LR.slider(bar, "threshold τ", 0.02, 0.98, 0.01, tau, function (v) { tau = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 300, {
      aria: "Training points placed at their predicted probabilities with a draggable decision threshold",
    });

    const grid = LR.el("div", "cm-grid");
    mount.appendChild(grid);
    const cmBox = LR.el("div");
    grid.appendChild(cmBox);
    const metricsBox = LR.el("div");
    grid.appendChild(metricsBox);

    const P = { x0: 58, y0: 22, w: W - 84, h: H - 78, xmin: 0, xmax: 1, ymin: 0, ymax: 1, xlabel: "predicted probability y = p(t = 1 | x)", ylabel: "", yticks: [] };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => SC && Math.abs(p.x - SC.sx(tau)) < 22,
      move: function (p) {
        tau = Math.max(0.02, Math.min(0.98, SC.inv(p.x, p.y).x));
        tS.set(tau);
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // predicted-positive shading
      ctx.fillStyle = "rgba(26,26,26,0.07)";
      ctx.fillRect(sx(tau), P.y0, P.x0 + P.w - sx(tau), P.h);

      // τ line
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(sx(tau), P.y0); ctx.lineTo(sx(tau), P.y0 + P.h); ctx.stroke();
      ctx.font = "700 12px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.fillStyle = C.orange;
      ctx.fillText("τ = " + LR.fmtF(tau, 2), sx(tau), P.y0 - 7);

      // band labels
      ctx.font = "600 11.5px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = CLS_COLOR[1]; ctx.fillText("true positives (t = 1)", P.x0 + 8, sy(0.86));
      ctx.fillStyle = CLS_COLOR[0]; ctx.fillText("true negatives (t = 0)", P.x0 + 8, sy(0.14));
      ctx.fillStyle = C.muted; ctx.textAlign = "right";
      ctx.fillText("→ predicted class 1", P.x0 + P.w - 8, P.y0 + 14);

      // points: class-1 in an upper band, class-0 in a lower band
      let TP = 0, FP = 0, FN = 0, TN = 0;
      scored.forEach(function (s) {
        const yy = s.t === 1 ? 0.58 + s.jit * 0.24 : 0.18 + s.jit * 0.24;
        const pos = s.p >= tau;
        if (s.t === 1 && pos) TP++;
        else if (s.t === 0 && pos) FP++;
        else if (s.t === 1 && !pos) FN++;
        else TN++;
        const wrong = (s.t === 1) !== pos;
        LR.dot(ctx, sx(s.p), sy(yy), 5, CLS_COLOR[s.t], wrong ? C.red : "#ffffff");
      });

      // confusion matrix + metrics (HTML for accessibility)
      cmBox.innerHTML =
        "<table class='cm-table' aria-label='Confusion matrix'>" +
        "<tr><th></th><th>predicted 1</th><th>predicted 0</th></tr>" +
        "<tr><th>actual 1</th><td class='cm-good'>TP = " + TP + "</td><td class='cm-bad'>FN = " + FN + "</td></tr>" +
        "<tr><th>actual 0</th><td class='cm-bad'>FP = " + FP + "</td><td class='cm-good'>TN = " + TN + "</td></tr>" +
        "</table>";
      const prec = TP + FP > 0 ? TP / (TP + FP) : NaN;
      const rec = TP + FN > 0 ? TP / (TP + FN) : NaN;
      const f1 = isFinite(prec) && isFinite(rec) && prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : NaN;
      const acc = (TP + TN) / scored.length;
      metricsBox.innerHTML =
        "<div class='fig-readout' style='margin-top:0'>" +
        "<span class='ro'><span class='ro-label'>precision TP/(TP+FP)</span><span class='ro-val'>" + (isFinite(prec) ? LR.fmtF(prec, 3) : "–") + "</span></span>" +
        "<span class='ro'><span class='ro-label'>recall TP/(TP+FN)</span><span class='ro-val'>" + (isFinite(rec) ? LR.fmtF(rec, 3) : "–") + "</span></span>" +
        "<span class='ro'><span class='ro-label'>F1</span><span class='ro-val'>" + (isFinite(f1) ? LR.fmtF(f1, 3) : "–") + "</span></span>" +
        "<span class='ro'><span class='ro-label'>accuracy</span><span class='ro-val'>" + LR.fmtF(acc, 3) + "</span></span>" +
        "</div>";
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — the regularization dial (λ slider, live refits)
     ════════════════════════════════════════════════════════════ */
  LR.figs.reg = function (mount) {
    LR.header(
      mount,
      "λ vs the noise",
      "Quadratic features let the boundary bend; three flipped labels tempt it to. Nine real fits, one per λ. Watch the weights shrink and the validation loss dip."
    );

    // quadratic feature map on centred/scaled coordinates
    // (a feature map keeps the model linear in w; the boundary bends in x-space)
    function phi(x, y) {
      const u = (x - 3.25) / 1.6, v = (y - 3.25) / 1.6;
      return [1, u, v, u * u, v * v, u * v];
    }
    const FEATS = ["bias", "u", "v", "u²", "v²", "uv"];

    const rand = LR.rng(53);
    function cloud(n, cx, cy, t, arr) {
      for (let i = 0; i < n; i++) {
        const x = cx + LR.gauss(rand) * 0.85, y = cy + LR.gauss(rand) * 0.85;
        arr.push({ x, y, t, f: phi(x, y) });
      }
    }
    const train = [], val = [];
    cloud(10, 2.2, 2.6, 0, train); cloud(10, 4.3, 3.9, 1, train);
    cloud(12, 2.2, 2.6, 0, val);   cloud(12, 4.3, 3.9, 1, val);
    // deliberate label noise in the training set only
    [1, 12, 17].forEach(function (i) { train[i].t = 1 - train[i].t; train[i].flipped = true; });

    const LAMS = [0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
    const fits = LAMS.map((lam) => LG.fit(train, { lr: 0.8, iters: 6000, lam: lam }));
    const trCE = fits.map((w) => LG.lossCE(w, train, 0));
    const vlCE = fits.map((w) => LG.lossCE(w, val, 0));
    const bestI = vlCE.indexOf(Math.min.apply(null, vlCE));

    let li = 0;

    const bar = LR.controls(mount);
    LR.slider(bar, "λ (penalty strength)", 0, LAMS.length - 1, 1, li, function (v) {
      li = Math.round(v); draw();
    }, (v) => String(LAMS[Math.round(v)]));

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "training data + fitted boundary"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "train vs validation loss · weight magnitudes"));
    pane.appendChild(lb); pane.appendChild(rb);
    const L = LR.canvas(lb, 400, 360, { aria: "Training scatter with the fitted, possibly curved, decision boundary at the current lambda" });
    const R = LR.canvas(rb, 400, 360, { aria: "Cross-entropy on train and validation versus lambda, and a bar chart of weight magnitudes" });

    const ro = LR.readout(mount, [
      { k: "lam", label: "λ" },
      { k: "tr", label: "train CE" },
      { k: "vl", label: "validation CE" },
      { k: "nw", label: "‖w‖₂ (without bias)" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 46, y0: 14, w: L.W - 62, h: L.H - 58, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };

    function drawLeft(w) {
      const ctx = L.ctx;
      ctx.clearRect(0, 0, L.W, L.H);
      // probability shading + implicit boundary (p crosses 0.5 between blocks)
      const BLK = 6;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const p = LG.sig(LG.dot(w, phi(wx, wy)));
          ctx.fillStyle = "rgba(150,150,150," + (p * 0.42).toFixed(3) + ")";
          ctx.fillRect(px, py, BLK, BLK);
          if (Math.abs(p - 0.5) < 0.045) {
            ctx.fillStyle = C.orange;
            ctx.fillRect(px, py, BLK, BLK);
          }
        }
      }
      const { sx, sy } = LR.plot(ctx, P);
      train.forEach(function (p) {
        LR.dot(ctx, sx(p.x), sy(p.y), p.flipped ? 6.5 : 5, CLS_COLOR[p.t], p.flipped ? C.red : "#ffffff");
      });
      ctx.font = "600 11px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.red;
      ctx.fillText("red ring = flipped label (noise)", P.x0 + 8, P.y0 + 14);
    }

    function drawRight(w) {
      const ctx = R.ctx;
      ctx.clearRect(0, 0, R.W, R.H);
      // top: loss curves vs λ index
      const ymax = Math.max.apply(null, trCE.concat(vlCE)) * 1.15;
      const PT = { x0: 50, y0: 14, w: R.W - 66, h: 150, xmin: 0, xmax: LAMS.length - 1, ymin: 0, ymax: ymax, xlabel: "", ylabel: "cross-entropy", xticks: LAMS.map((_, i) => i) };
      const SC = LR.plot(ctx, PT);
      // custom tick labels: the λ values
      ctx.font = "9.5px Inter, sans-serif"; ctx.fillStyle = "#ffffff";
      ctx.fillRect(PT.x0 - 8, PT.y0 + PT.h + 5, PT.w + 16, 12);
      ctx.fillStyle = C.faint; ctx.textAlign = "center";
      LAMS.forEach((l, i) => ctx.fillText(String(l), SC.sx(i), PT.y0 + PT.h + 14));
      const curve = function (arr, color) {
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath();
        arr.forEach((v, i) => (i === 0 ? ctx.moveTo(SC.sx(i), SC.sy(v)) : ctx.lineTo(SC.sx(i), SC.sy(v))));
        ctx.stroke();
        arr.forEach((v, i) => LR.dot(ctx, SC.sx(i), SC.sy(v), 2.6, color));
      };
      curve(trCE, C.green);
      curve(vlCE, C.orange);
      // marker at current λ
      ctx.setLineDash([3, 3]); ctx.strokeStyle = C.text; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(SC.sx(li), SC.sy(0)); ctx.lineTo(SC.sx(li), SC.sy(ymax)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.green; ctx.fillText("— train", PT.x0 + PT.w - 92, PT.y0 + 14);
      ctx.fillStyle = C.orange; ctx.fillText("— validation", PT.x0 + PT.w - 92, PT.y0 + 28);

      // bottom: |w_j| bars
      const by0 = 218, bh = 100;
      ctx.font = "600 11px Inter, sans-serif"; ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("weight magnitudes |wⱼ| at this λ", 50, by0 - 8);
      const maxW = Math.max.apply(null, fits.map((f) => Math.max.apply(null, f.map(Math.abs)))) || 1;
      const bw = (R.W - 66) / FEATS.length;
      FEATS.forEach(function (name, j) {
        const v = Math.abs(w[j]);
        const h = Math.max(1.5, (v / maxW) * bh);
        ctx.fillStyle = j === 0 ? "#c6c6c6" : C.purple;
        ctx.fillRect(50 + j * bw + 6, by0 + bh - h, bw - 12, h);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText(name, 50 + j * bw + bw / 2, by0 + bh + 14);
        ctx.fillStyle = C.text;
        ctx.fillText(LR.fmtF(v, 1), 50 + j * bw + bw / 2, by0 + bh - h - 4);
      });
    }

    function draw() {
      const w = fits[li];
      drawLeft(w); drawRight(w);
      let nw = 0;
      for (let j = 1; j < w.length; j++) nw += w[j] * w[j];
      ro.set("lam", String(LAMS[li]));
      ro.set("tr", LR.fmtF(trCE[li], 3), C.green);
      ro.set("vl", LR.fmtF(vlCE[li], 3), C.orange);
      ro.set("nw", LR.fmtF(Math.sqrt(nw), 2));
      if (li === bestI) {
        msg.show("✓ λ = " + LAMS[li] + " is the validation optimum on this data (val CE " + LR.fmtF(vlCE[li], 3) + "). Less than this and the boundary chases the flipped labels; more and it underfits. The MAP reading: this λ is the width of the Gaussian prior that best matches how noisy the data really is.", "good");
      } else if (li === 0) {
        msg.show("λ = 0: unregularized. The boundary bends around the flipped labels and the weights are at their largest. Train loss is at its minimum, validation is not, which is the whole overfitting story in two numbers.", "bad");
      } else if (li === LAMS.length - 1) {
        msg.show("λ = " + LAMS[li] + ": the penalty dominates, the weights are crushed toward zero, and the boundary is too rigid to serve even the clean labels. Underfitting.", "bad");
      } else {
        msg.hide();
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — the softmax explorer (three classes)
     ════════════════════════════════════════════════════════════ */
  LR.figs.softmax = function (mount) {
    LR.header(
      mount,
      "One score per class",
      "Three linear scores feed a softmax. Drag the query: the bars are a real probability distribution, and they always sum to 1."
    );

    const K = 3;
    const NAMES = ["A", "B", "C"];
    const KCOL = [C.green, C.purple, C.amber];
    const KSOFT = ["rgba(102,102,102,0.16)", "rgba(150,150,150,0.14)", "rgba(173,173,173,0.18)"];

    const rand = LR.rng(67);
    const pts = [];
    const centres = [[1.8, 4.7], [4.7, 4.6], [3.25, 1.7]];
    for (let k = 0; k < K; k++) {
      for (let i = 0; i < 12; i++) {
        pts.push({ x: centres[k][0] + LR.gauss(rand) * 0.7, y: centres[k][1] + LR.gauss(rand) * 0.7, t: k });
      }
    }

    // multi-class logistic regression: W[k] = [w0, w1, w2] per class,
    // trained by gradient descent on the multi-class cross-entropy
    const Wk = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    (function fitSoftmax() {
      const lr = 0.8, iters = 1500, N = pts.length;
      for (let it = 0; it < iters; it++) {
        const g = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (const p of pts) {
          const f = [1, p.x, p.y];
          const zs = Wk.map((w) => w[0] * f[0] + w[1] * f[1] + w[2] * f[2]);
          const pr = LG.softmax(zs);
          for (let k = 0; k < K; k++) {
            const r = pr[k] - (p.t === k ? 1 : 0);
            for (let j = 0; j < 3; j++) g[k][j] += r * f[j];
          }
        }
        for (let k = 0; k < K; k++)
          for (let j = 0; j < 3; j++) Wk[k][j] -= (lr * g[k][j]) / N + lr * 0.002 * (j > 0 ? Wk[k][j] : 0);
      }
    })();

    function scoresAt(x, y) {
      return Wk.map((w) => w[0] + w[1] * x + w[2] * y);
    }

    let q = { x: 3.3, y: 3.5 };

    const bar = LR.controls(mount);
    const qxS = LR.slider(bar, "query x₁", 0.2, 6.3, 0.05, q.x, function (v) { q.x = v; draw(); }, (v) => LR.fmtF(v, 2));
    const qyS = LR.slider(bar, "query x₂", 0.2, 6.3, 0.05, q.y, function (v) { q.y = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 440, {
      aria: "Three-class scatter with argmax decision regions, a draggable query point, and live softmax probability bars that sum to one",
    });
    const ro = LR.readout(mount, [
      { k: "z", label: "scores z = (z_A, z_B, z_C)" },
      { k: "p", label: "softmax p" },
      { k: "sum", label: "Σ p" },
      { k: "pred", label: "argmax prediction" },
    ]);

    const P = { x0: 56, y0: 20, w: 480, h: H - 76, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };
    const BX = 590, BW = 200; // bars panel
    let SC = null;

    // cached argmax regions (the fit is fixed)
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const offCtx = off.getContext("2d");
    (function renderRegions() {
      const BLK = 6;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const zs = scoresAt(wx, wy);
          let bi = 0;
          for (let k = 1; k < K; k++) if (zs[k] > zs[bi]) bi = k;
          offCtx.fillStyle = KSOFT[bi];
          offCtx.fillRect(px, py, BLK, BLK);
        }
      }
    })();

    LR.drag(cv, W, H, {
      hit: (p) => SC && (p.x - SC.sx(q.x)) ** 2 + (p.y - SC.sy(q.y)) ** 2 < 900,
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        q.x = Math.max(0.2, Math.min(6.3, d.x));
        q.y = Math.max(0.2, Math.min(6.3, d.y));
        qxS.set(q.x); qyS.set(q.y);
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(off, 0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      pts.forEach((p) => LR.dot(ctx, sx(p.x), sy(p.y), 5, KCOL[p.t], "#ffffff"));

      const zs = scoresAt(q.x, q.y);
      const pr = LG.softmax(zs);
      let pred = 0;
      for (let k = 1; k < K; k++) if (pr[k] > pr[pred]) pred = k;

      // query
      LR.dot(ctx, sx(q.x), sy(q.y), 9.5, KCOL[pred], C.orange);
      ctx.font = "800 12px Inter, sans-serif"; ctx.fillStyle = "#ffffff"; ctx.textAlign = "center";
      ctx.fillText("?", sx(q.x), sy(q.y) + 4);

      // ---- probability bars panel ----
      ctx.font = "700 12.5px Inter, sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = C.text;
      ctx.fillText("scores → softmax → distribution", BX - 24, P.y0 + 8);
      for (let k = 0; k < K; k++) {
        const by = P.y0 + 40 + k * 86;
        ctx.font = "700 12px Inter, sans-serif"; ctx.fillStyle = KCOL[k];
        ctx.fillText("class " + NAMES[k], BX - 24, by);
        ctx.font = "500 11.5px 'JetBrains Mono', Menlo, monospace"; ctx.fillStyle = C.muted;
        ctx.fillText("z = " + LR.fmtF(zs[k], 2), BX - 24, by + 16);
        // bar
        ctx.fillStyle = "#efefef";
        ctx.fillRect(BX - 24, by + 24, BW, 18);
        ctx.fillStyle = KCOL[k];
        ctx.fillRect(BX - 24, by + 24, BW * pr[k], 18);
        ctx.fillStyle = C.text;
        ctx.font = "700 11.5px 'JetBrains Mono', Menlo, monospace";
        ctx.fillText(LR.fmtF(pr[k], 3), BX - 24 + BW + 8, by + 38);
      }
      const sum = pr.reduce((a, b) => a + b, 0);
      ctx.font = "700 12px Inter, sans-serif"; ctx.fillStyle = C.text;
      ctx.fillText("sum = " + LR.fmtF(sum, 3) + " (always)", BX - 24, P.y0 + 40 + 3 * 86);

      ro.set("z", "(" + zs.map((z) => LR.fmtF(z, 2)).join(", ") + ")");
      ro.set("p", "(" + pr.map((p) => LR.fmtF(p, 3)).join(", ") + ")");
      ro.set("sum", LR.fmtF(sum, 4), C.green);
      ro.set("pred", "class " + NAMES[pred], KCOL[pred]);
    }
    draw();
  };
})();
