/* ══════════════════════════════════════════════════════════════
   figures-b.js — Linear Classification lesson figures, sections 5–6
   Fig 5.1 perceptron · Fig 5.2 xor · Fig 5.3 codeex
   Fig 6.1 multiclass
   Uses the shared linear-classifier core (LR.lc) from figures-a.js.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const LC = LR.lc;

  const POS = C.green, NEG = C.purple;

  /* perceptron mechanics with the intercept folded in:
     x̃ = (1, x, y), w = [w0, w1, w2].
     A tie (signed score exactly 0) counts as a mistake so a zero
     start can move; this refines the source's strict t·wᵀx < 0. */
  function pScore(w, p) { return w[0] + w[1] * p.x + w[2] * p.y; }
  function pMistake(w, p) { return p.t * pScore(w, p) <= 0; }
  function pUpdate(w, p) { return [w[0] + p.t, w[1] + p.t * p.x, w[2] + p.t * p.y]; }

  // deterministic separable two-class blob generator with a margin
  // guard: every point satisfies t·(x + y) > gap, so the line
  // x + y = 0 separates the classes and convergence is guaranteed.
  function separableData(seed, nPer, gap, cx, cy, sig) {
    const rand = LR.rng(seed);
    const pts = [];
    const mk = function (px, py, t) {
      let placed = 0, guard = 0;
      while (placed < nPer && guard < 800) {
        guard++;
        const x = px + LR.gauss(rand) * sig;
        const y = py + LR.gauss(rand) * sig;
        if (t * (x + y) < gap) continue;
        if (Math.abs(x) > 4 || Math.abs(y) > 4) continue;
        pts.push({ x, y, t });
        placed++;
      }
    };
    mk(cx, cy, 1);
    mk(-cx, -cy, -1);
    // deterministic interleave so the visit order mixes the classes
    const order = [];
    for (let i = 0; i < nPer; i++) { order.push(pts[i]); order.push(pts[nPer + i]); }
    return order;
  }

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — the perceptron, learning live (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.perceptron = function (mount) {
    LR.header(
      mount,
      "Learning by correction",
      "Step through the data. Correct points are left alone; every mistake adds t·x̃ to the weights and the boundary rotates. Turn on “predict the fix” to be quizzed before each reveal."
    );

    // tuned so the zero start needs 5 updates over 3 passes: long
    // enough to watch the boundary swing, short enough to sit through
    const pts = separableData(87, 10, 0.25, 1.1, 0.9, 1.05);
    const STARTS = {
      zeros: { w: [0, 0, 0], label: "start: w = (0, 0, 0)" },
      tilted: { w: [0.5, -1.0, 0.5], label: "start: w = (0.5, −1, 0.5)" },
      flipped: { w: [0, -1, -1], label: "start: w = (0, −1, −1), pointing the wrong way" },
    };
    let startKey = "zeros";

    let st;
    function resetState() {
      st = {
        w: STARTS[startKey].w.slice(),
        i: 0, pass: 1, updates: 0, passMistakes: 0,
        converged: false, ghosts: [], lastUp: null, pending: false,
      };
    }
    resetState();

    let guessMode = false, playing = null;

    const bar = LR.controls(mount);
    const stepBtn = LR.button(bar, "Step ▸", step, "primary");
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, tickMs());
    });
    LR.button(bar, "Reset ⟲", function () {
      stopPlay(); clearGuess(); resetState();
      msg.show("Reset. " + STARTS[startKey].label + ". Press Step.", "info");
      draw(); sync();
    });
    let speed = 2;
    const spd = LR.slider(bar, "speed (steps/s)", 1, 6, 1, speed, function (v) {
      speed = Math.round(v);
      if (playing) { clearInterval(playing); playing = setInterval(step, tickMs()); }
    }, (v) => String(Math.round(v)));
    function tickMs() { return LR.reducedMotion ? 1100 : Math.round(1000 / speed); }
    function stopPlay() { if (playing) clearInterval(playing); playing = null; playBtn.textContent = "Play ▸▸"; }

    const bar2 = LR.controls(mount);
    const startGroup = LR.el("div", "toggle-group");
    bar2.appendChild(startGroup);
    const startBtns = {};
    Object.keys(STARTS).forEach(function (k, idx) {
      startBtns[k] = LR.button(startGroup, STARTS[k].label, function () {
        startKey = k;
        for (const k2 in startBtns) startBtns[k2].classList.toggle("on", k2 === startKey);
        stopPlay(); clearGuess(); resetState();
        msg.show("New start. Press Step or Play.", "info");
        draw(); sync();
      }, "small" + (idx === 0 ? " on" : ""));
    });
    const guessBtn = LR.button(bar2, "Predict the fix: off", function () {
      guessMode = !guessMode;
      guessBtn.textContent = "Predict the fix: " + (guessMode ? "on" : "off");
      guessBtn.classList.toggle("on", guessMode);
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Animated perceptron: a two-class scatter whose decision boundary rotates as misclassified examples are added into the weight vector",
    });
    const ro = LR.readout(mount, [
      { k: "pass", label: "pass" },
      { k: "upd", label: "updates so far" },
      { k: "w", label: "w = (w₀, w₁, w₂)" },
      { k: "sc", label: "signed score t·wᵀx̃ (current point)" },
    ]);
    const msg = LR.msg(mount);

    // guess buttons (revealed only during the "predict the fix" beat)
    const guessBox = LR.el("div", "fig-controls");
    guessBox.style.display = "none";
    mount.appendChild(guessBox);
    const gAdd = LR.button(guessBox, "Add x̃ into w (pull the normal toward it)", () => answerGuess(1), "small");
    const gSub = LR.button(guessBox, "Subtract x̃ from w (push the normal away)", () => answerGuess(-1), "small");
    function clearGuess() { st && (st.pending = false); guessBox.style.display = "none"; }

    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: -4.4, xmax: 4.4, ymin: -4, ymax: 4, xlabel: "x₁", ylabel: "x₂" };

    msg.show("Press Step. The perceptron starts at w = (0, 0, 0): every point scores 0, so the very first visit is a mistake.", "info");

    function current() { return pts[st.i]; }

    function applyUpdate(p) {
      st.ghosts.push(st.w.slice());
      if (st.ghosts.length > 8) st.ghosts.shift();
      const wOld = st.w.slice();
      st.w = pUpdate(st.w, p);
      st.updates++; st.passMistakes++;
      st.lastUp = { p, wOld };
    }

    function advance() {
      st.i++;
      if (st.i >= pts.length) {
        if (st.passMistakes === 0) {
          st.converged = true;
          stopPlay();
          msg.show("A full pass with zero mistakes: converged. Final w = (" + st.w.map((v) => LR.fmt(v, 2)).join(", ") + ") after " + st.updates + " updates and " + st.pass + " passes. On separable data this was guaranteed.", "good");
        } else {
          st.pass++; st.passMistakes = 0; st.i = 0;
        }
      }
    }

    function answerGuess(dir) {
      const p = current();
      const correctDir = p.t; // +1 → add, −1 → subtract
      if (dir === correctDir) {
        msg.show("Right. t = " + (p.t > 0 ? "+1, so w ← w + x̃" : "−1, so w ← w − x̃") + ": the normal swings so this point's signed score rises by ‖x̃‖².", "good");
      } else {
        msg.show("Other way. The rule is w ← w + t·x̃ with t = " + (p.t > 0 ? "+1" : "−1") + ": " + (p.t > 0 ? "a positive point on the wrong side gets added, pulling w toward it." : "a negative point on the wrong side gets subtracted, pushing w away."), "bad");
      }
      clearGuess();
      applyUpdate(p);
      advance();
      draw(); sync();
    }

    function step() {
      if (st.converged || st.pending) return;
      const p = current();
      const s = pScore(st.w, p);
      st.lastUp = null;
      if (!pMistake(st.w, p)) {
        msg.show("Point (" + LR.fmtF(p.x, 1) + ", " + LR.fmtF(p.y, 1) + "), t = " + (p.t > 0 ? "+1" : "−1") + ": signed score " + LR.fmtF(p.t * s, 2) + " > 0, correct. No update; the perceptron only reacts to mistakes.", "info");
        advance();
        if (st.converged) { draw(); sync(); return; }
      } else {
        if (guessMode) {
          stopPlay();
          st.pending = true;
          guessBox.style.display = "";
          msg.show("Mistake: signed score " + LR.fmtF(p.t * s, 2) + " ≤ 0 on the highlighted point (t = " + (p.t > 0 ? "+1" : "−1") + "). Which way should w move?", "bad");
          draw(); sync();
          return;
        }
        applyUpdate(p);
        msg.show("Mistake (signed score " + LR.fmtF(p.t * s, 2) + " ≤ 0). Update: w ← w + t·x̃ = (" + st.w.map((v) => LR.fmt(v, 2)).join(", ") + "). Its signed score just rose by ‖x̃‖² = " + LR.fmtF(1 + p.x * p.x + p.y * p.y, 2) + ".", "bad");
        advance();
      }
      draw(); sync();
    }

    function sync() {
      ro.set("pass", String(st.pass) + (st.converged ? " (done)" : ""));
      ro.set("upd", String(st.updates), C.orange);
      ro.set("w", "(" + st.w.map((v) => LR.fmt(v, 2)).join(", ") + ")");
      if (st.converged) { ro.set("sc", "all points correct", POS); return; }
      const p = current();
      const ts = p.t * pScore(st.w, p);
      ro.set("sc", LR.fmtF(ts, 2) + (ts <= 0 ? "  (mistake)" : "  (correct)"), ts <= 0 ? C.red : POS);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const hasLine = Math.abs(st.w[1]) > 1e-9 || Math.abs(st.w[2]) > 1e-9;
      if (hasLine) LC.shadeHalfspaces(ctx, P, null, null, st.w);
      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // ghosts of previous boundaries
      st.ghosts.forEach(function (g, i) {
        const alpha = 0.10 + (0.22 * (i + 1)) / st.ghosts.length;
        LC.drawBoundary(ctx, P, sx, sy, g, "rgba(26,26,26," + alpha.toFixed(2) + ")", 1.6, [5, 5]);
      });
      if (hasLine) LC.drawBoundary(ctx, P, sx, sy, st.w, C.orange, 2.6);

      // normal vector from the boundary's closest point to the origin
      if (hasLine) {
        const n2 = st.w[1] ** 2 + st.w[2] ** 2;
        const k = -st.w[0] / n2, nl = Math.sqrt(n2);
        const ax = k * st.w[1], ay = k * st.w[2];
        LR.arrow(ctx, sx(ax), sy(ay), sx(ax + (st.w[1] / nl) * 1.3), sy(ay + (st.w[2] / nl) * 1.3), C.orange, 2.2);
      }

      // points; current one highlighted, mistakes ringed red
      pts.forEach(function (p, idx) {
        const isCur = !st.converged && idx === st.i;
        const wrong = pMistake(st.w, p);
        LR.dot(ctx, sx(p.x), sy(p.y), isCur ? 9 : 5.5, p.t === 1 ? POS : NEG, isCur ? C.orange : wrong ? C.red : "#ffffff");
      });

      // weight-space inset showing the last update: w_old + t·x̃ = w_new
      if (st.lastUp) {
        const IN = { x: P.x0 + P.w - 152, y: P.y0 + 10, w: 142, h: 132 };
        ctx.fillStyle = "rgba(255,255,255,0.93)";
        ctx.fillRect(IN.x, IN.y, IN.w, IN.h);
        ctx.strokeStyle = "#cccccc"; ctx.lineWidth = 1;
        ctx.strokeRect(IN.x, IN.y, IN.w, IN.h);
        const cxI = IN.x + IN.w / 2, cyI = IN.y + IN.h / 2 + 8;
        const all = [st.lastUp.wOld, st.w];
        const mx = Math.max(1, ...all.map((v) => Math.max(Math.abs(v[1]), Math.abs(v[2]))));
        const sc = 48 / mx;
        const wo = st.lastUp.wOld, p = st.lastUp.p;
        ctx.font = "600 10.5px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("weight space (w₁, w₂)", IN.x + 8, IN.y + 14);
        if (Math.abs(wo[1]) + Math.abs(wo[2]) > 1e-9) {
          LR.arrow(ctx, cxI, cyI, cxI + wo[1] * sc, cyI - wo[2] * sc, "#999999", 1.8);
        }
        LR.arrow(ctx, cxI + wo[1] * sc, cyI - wo[2] * sc, cxI + (wo[1] + p.t * p.x) * sc, cyI - (wo[2] + p.t * p.y) * sc, p.t === 1 ? POS : NEG, 1.8);
        LR.arrow(ctx, cxI, cyI, cxI + st.w[1] * sc, cyI - st.w[2] * sc, C.orange, 2.2);
        ctx.fillStyle = C.orange;
        ctx.fillText("new w", cxI + st.w[1] * sc + 4, cyI - st.w[2] * sc);
        ctx.fillStyle = p.t === 1 ? POS : NEG;
        ctx.fillText("+ t·x̃", IN.x + 8, IN.y + IN.h - 8);
      }

      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = POS; ctx.fillText("● t = +1", P.x0 + 8, P.y0 + 16);
      ctx.fillStyle = NEG; ctx.fillText("● t = −1", P.x0 + 8, P.y0 + 33);
    }

    draw(); sync();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — the XOR wall
     ════════════════════════════════════════════════════════════ */
  LR.figs.xor = function (mount) {
    LR.header(
      mount,
      "Four points, no line",
      "Drag the two square handles and try to put both green points on one side and both purple points on the other. Then let the perceptron try."
    );

    const GATES = {
      AND: (a, b) => (a === 1 && b === 1 ? 1 : -1),
      OR: (a, b) => (a === 1 || b === 1 ? 1 : -1),
      XOR: (a, b) => (a !== b ? 1 : -1),
    };
    let gate = "XOR";
    function gatePts() {
      return [0, 1].flatMap((a) => [0, 1].map((b) => ({ x: a, y: b, t: GATES[gate](a, b) })));
    }

    // the reader's line: two draggable handles + a side flip
    let h1 = { x: -0.35, y: 0.15 }, h2 = { x: 1.4, y: 0.9 };
    let sideSign = 1;
    let auto = null, autoW = null, autoN = 0, autoMsg = "";

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const gbtns = {};
    ["AND", "OR", "XOR"].forEach(function (g) {
      gbtns[g] = LR.button(group, g, function () {
        gate = g;
        for (const k in gbtns) gbtns[k].classList.toggle("on", k === gate);
        stopAuto(); draw();
      }, "small" + (g === "XOR" ? " on" : ""));
    });
    LR.button(bar, "Swap sides ⇄", function () { sideSign = -sideSign; stopAuto(); draw(); });
    const tryBtn = LR.button(bar, "Let the perceptron try ▸", runPerceptron, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 460, {
      aria: "The four XOR points with a draggable separating line that can never classify all four correctly; AND and OR toggles show separable cases",
    });
    const ro = LR.readout(mount, [
      { k: "gate", label: "gate" },
      { k: "correct", label: "correctly classified" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: -0.7, xmax: 1.7, ymin: -0.55, ymax: 1.55, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!SC) return false;
        for (const h of [h1, h2]) {
          if ((p.x - SC.sx(h.x)) ** 2 + (p.y - SC.sy(h.y)) ** 2 < 900) { dragging = h; return true; }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        dragging.x = Math.max(P.xmin + 0.05, Math.min(P.xmax - 0.05, d.x));
        dragging.y = Math.max(P.ymin + 0.05, Math.min(P.ymax - 0.05, d.y));
        stopAuto();
        draw();
      },
      up: function () { dragging = null; },
    });

    // the reader's line as w = [w0, w1, w2]: the normal to (h2 − h1),
    // with sideSign choosing which half-plane counts as positive
    function lineW() {
      const dx = h2.x - h1.x, dy = h2.y - h1.y;
      const n1 = -dy, n2 = dx;
      return [sideSign * -(n1 * h1.x + n2 * h1.y), sideSign * n1, sideSign * n2];
    }

    function classify(w, p) { return LC.sign(w[0] + w[1] * p.x + w[2] * p.y); }

    function runPerceptron() {
      stopAuto();
      autoW = [0, 0, 0]; autoN = 0;
      const pts = gatePts();
      let i = 0, sinceMistake = 0;
      tryBtn.disabled = true;
      const tick = function () {
        // one visit per tick
        const p = pts[i % 4];
        i++;
        if (pMistake(autoW, p)) {
          autoW = pUpdate(autoW, p);
          autoN++;
          sinceMistake = 0;
        } else {
          sinceMistake++;
        }
        if (sinceMistake >= 4) {
          autoMsg = gate + " is linearly separable: the perceptron converged after " + autoN + " updates. Final w = (" + autoW.join(", ") + ").";
          msg.show(autoMsg, "good");
          stopAutoTimer();
          return;
        }
        if (autoN >= 60) {
          autoMsg = "60 updates and still making mistakes. On XOR there is no separating line, so the guarantee is void and the perceptron thrashes forever. This is Minsky and Papert's 1969 point.";
          msg.show(autoMsg, "bad");
          stopAutoTimer();
          return;
        }
        draw();
      };
      auto = setInterval(tick, LR.reducedMotion ? 600 : 120);
      function stopAutoTimer() { if (auto) clearInterval(auto); auto = null; tryBtn.disabled = false; draw(); }
    }
    function stopAuto() {
      if (auto) clearInterval(auto);
      auto = null; autoW = null; tryBtn.disabled = false;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const pts = gatePts();
      const w = autoW && (Math.abs(autoW[1]) > 1e-9 || Math.abs(autoW[2]) > 1e-9) ? autoW : lineW();
      const usingAuto = autoW !== null;

      if (Math.abs(w[1]) > 1e-9 || Math.abs(w[2]) > 1e-9) LC.shadeHalfspaces(ctx, P, null, null, w);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // the line
      LC.drawBoundary(ctx, P, sx, sy, w, usingAuto ? NEG : C.orange, 2.6);

      // handles (only meaningful in manual mode)
      if (!usingAuto) {
        [h1, h2].forEach(function (h) {
          ctx.fillStyle = "#ffffff"; ctx.strokeStyle = C.orange; ctx.lineWidth = 2;
          ctx.fillRect(sx(h.x) - 6, sy(h.y) - 6, 12, 12);
          ctx.strokeRect(sx(h.x) - 6, sy(h.y) - 6, 12, 12);
        });
      }

      // the four gate points
      let ok = 0;
      pts.forEach(function (p) {
        const right = classify(w, p) === p.t;
        if (right) ok++;
        LR.dot(ctx, sx(p.x), sy(p.y), 11, p.t === 1 ? POS : NEG, right ? "#ffffff" : C.red);
        ctx.font = "700 11px Inter, sans-serif";
        ctx.fillStyle = "#ffffff"; ctx.textAlign = "center";
        ctx.fillText("(" + p.x + "," + p.y + ")", sx(p.x), sy(p.y) + 3.5);
      });

      ro.set("gate", gate + "  (green = +1, purple = −1)");
      ro.set("correct", ok + " / 4", ok === 4 ? POS : C.red);

      if (!usingAuto) {
        if (ok === 4) msg.show("4 of 4. " + gate + " is linearly separable, one line does it.", "good");
        else if (gate === "XOR") msg.show(ok + " of 4. Keep trying, but 3 is the ceiling: the positives sit on one diagonal and the negatives on the other, and no line splits two diagonals.", "info");
        else msg.show(ok + " of 4. A separating line exists for " + gate + ", keep adjusting (or press Swap sides).", "info");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.3 — interactive perceptron_fit code exercise
     ════════════════════════════════════════════════════════════ */
  LR.figs.codeex = function (mount) {
    LR.header(
      mount,
      "Run the perceptron yourself",
      "The loop below really executes. The default zero start on the tiny dataset reproduces the worked example: two updates, final w = (0, 1, −3)."
    );

    // the tiny dataset from the worked example (x̃ = (1, x1, x2))
    const TINY = [
      { x: 1, y: 2, t: -1 },
      { x: 2, y: -1, t: 1 },
      { x: -1, y: 1, t: -1 },
    ];
    // tuned so the zero start needs 8 updates over 5 passes,
    // a meatier trace than the 2-update tiny set
    const BIG = separableData(89, 8, 0.3, 1.2, 1.0, 1.0);
    let dataset = "tiny";

    const CODE =
      "X = np.array([[1,  1,  2],\n" +
      "              [1,  2, -1],\n" +
      "              [1, -1,  1]])      # leading 1 folds the intercept into w\n" +
      "t = np.array([-1, +1, -1])\n" +
      "\n" +
      "def perceptron_fit(X, t, w, epochs):\n" +
      "    for epoch in range(epochs):\n" +
      "        mistakes = 0\n" +
      "        for n in range(len(t)):\n" +
      "            if t[n] * (w @ X[n]) <= 0:   # mistake (ties count, so w = 0 can move)\n" +
      "                w = w + t[n] * X[n]      # the perceptron update\n" +
      "                mistakes += 1\n" +
      "        if mistakes == 0:                # a clean pass: converged\n" +
      "            break\n" +
      "    return w";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);

    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);
    let w0 = 0, w1 = 0, w2 = 0, epochs = 10;
    LR.slider(controls, "start w₀", -3, 3, 0.5, w0, (v) => { w0 = v; }, (v) => LR.fmtF(v, 1));
    LR.slider(controls, "start w₁", -3, 3, 0.5, w1, (v) => { w1 = v; }, (v) => LR.fmtF(v, 1));
    LR.slider(controls, "start w₂", -3, 3, 0.5, w2, (v) => { w2 = v; }, (v) => LR.fmtF(v, 1));
    const epLab = LR.el("label", "", "epochs = ");
    const epSel = document.createElement("select");
    [1, 2, 5, 10, 20].forEach((v) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      if (v === 10) o.selected = true;
      epSel.appendChild(o);
    });
    epSel.style.cssText = "font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid #cccccc";
    epSel.addEventListener("change", () => { epochs = parseInt(epSel.value, 10); });
    epLab.appendChild(epSel);
    controls.appendChild(epLab);
    const dsLab = LR.el("label", "", "dataset = ");
    const dsSel = document.createElement("select");
    [["tiny", "tiny (3 points, worked example)"], ["big", "larger (16 points)"]].forEach((d) => {
      const o = document.createElement("option");
      o.value = d[0]; o.textContent = d[1];
      dsSel.appendChild(o);
    });
    dsSel.style.cssText = epSel.style.cssText;
    dsSel.addEventListener("change", () => { dataset = dsSel.value; });
    dsLab.appendChild(dsSel);
    controls.appendChild(dsLab);
    LR.button(controls, "Run ▸", run, "primary small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; perceptron_fit(X, t, w=np.array([0., 0., 0.]), epochs=10)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    const { cv, ctx, W, H } = LR.canvas(mount, 520, 300, {
      aria: "The exercise dataset with the boundary learned by the perceptron run",
    });
    cv.style.marginTop = "14px";
    const P = { x0: 50, y0: 16, w: W - 68, h: H - 60, xmin: -4.4, xmax: 4.4, ymin: -4, ymax: 4, xlabel: "x₁", ylabel: "x₂" };

    function run() {
      const pts = dataset === "tiny" ? TINY : BIG;
      let w = [w0, w1, w2];
      const trace = [];
      let converged = false, passesUsed = 0, totalUpdates = 0;
      for (let ep = 0; ep < epochs && !converged; ep++) {
        passesUsed = ep + 1;
        let mistakes = 0;
        for (let n = 0; n < pts.length; n++) {
          const p = pts[n];
          const s = w[0] + w[1] * p.x + w[2] * p.y;
          if (p.t * s <= 0) {
            w = pUpdate(w, p);
            mistakes++; totalUpdates++;
            trace.push(
              "pass " + (ep + 1) + ", x(" + (n + 1) + ")=(" + LR.fmt(p.x, 1) + "," + LR.fmt(p.y, 1) + "), t=" + (p.t > 0 ? "+1" : "-1") +
              ": signed score " + LR.fmtF(p.t * s, 2) + " ≤ 0 → w = (" + w.map((v) => LR.fmt(v, 2)).join(", ") + ")"
            );
          }
        }
        if (mistakes === 0) converged = true;
      }
      // final mistake count (real check on the returned w)
      let finalErrs = 0;
      pts.forEach((p) => { if (p.t * (w[0] + w[1] * p.x + w[2] * p.y) <= 0) finalErrs++; });

      const shown = trace.slice(0, 12);
      let txt =
        "&gt;&gt;&gt; perceptron_fit(X, t, w=np.array([" + [w0, w1, w2].map((v) => LR.fmtF(v, 1)).join(", ") + "]), epochs=" + epochs + ")\n";
      txt += shown.join("\n") + (trace.length > 12 ? "\n… (" + (trace.length - 12) + " more updates)" : "");
      txt += (trace.length ? "\n" : "") +
        (converged ? "clean pass on pass " + passesUsed + ": converged, " + totalUpdates + " update" + (totalUpdates === 1 ? "" : "s") + "\n"
                   : "epoch cap reached with " + finalErrs + " point(s) still misclassified\n");
      txt += "final w = <b>(" + w.map((v) => LR.fmt(v, 2)).join(", ") + ")</b>" +
        (Math.abs(w[1]) > 1e-9 || Math.abs(w[2]) > 1e-9
          ? "   boundary: " + LR.fmt(w[0], 2) + " + " + LR.fmt(w[1], 2) + "·x₁ + " + LR.fmt(w[2], 2) + "·x₂ = 0"
          : "");
      out.innerHTML = txt;
      draw(pts, w);
    }

    function draw(pts, w) {
      ctx.clearRect(0, 0, W, H);
      if (w && (Math.abs(w[1]) > 1e-9 || Math.abs(w[2]) > 1e-9)) LC.shadeHalfspaces(ctx, P, null, null, w);
      const SC = LR.plot(ctx, P);
      if (w) LC.drawBoundary(ctx, P, SC.sx, SC.sy, w, C.orange, 2.4);
      (pts || TINY).forEach(function (p) {
        const wrong = w ? p.t * (w[0] + w[1] * p.x + w[2] * p.y) <= 0 : false;
        LR.dot(ctx, SC.sx(p.x), SC.sy(p.y), 6, p.t === 1 ? POS : NEG, wrong ? C.red : "#ffffff");
      });
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("green = +1, purple = −1, red ring = misclassified", P.x0 + 8, P.y0 + 14);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — multi-class strategies explorer
     ════════════════════════════════════════════════════════════ */
  LR.figs.multiclass = function (mount) {
    LR.header(
      mount,
      "Three ways to referee K classes",
      "The regions come from real least-squares classifiers trained on this data. Drag the query point; the readout shows exactly how each strategy decides."
    );

    const NAMES = ["A", "B", "C"];
    const COLS = { A: C.green, B: C.purple, C: C.amber };
    const SOFT = { A: "rgba(102,102,102,0.20)", B: "rgba(150,150,150,0.17)", C: "rgba(173,173,173,0.22)" };
    const CONFLICT = "rgba(141,141,141,0.16)";

    // three blobs (deterministic)
    const rand = LR.rng(71);
    const pts = [];
    const mk = (cx, cy, cls) => {
      for (let i = 0; i < 10; i++) pts.push({ x: cx + LR.gauss(rand) * 0.8, y: cy + LR.gauss(rand) * 0.8, cls });
    };
    mk(-2.2, 2.0, "A"); mk(2.4, 1.9, "B"); mk(0.1, -2.4, "C");

    // least-squares fit of a linear score to arbitrary targets on (1, x, y)
    function lsFit(targets) {
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      const b = [0, 0, 0];
      pts.forEach(function (p, i) {
        const phi = [1, p.x, p.y];
        for (let j = 0; j < 3; j++) {
          b[j] += phi[j] * targets[i];
          for (let k = 0; k < 3; k++) A[j][k] += phi[j] * phi[k];
        }
      });
      return LR.solve(A, b);
    }

    // one-hot least squares: one scorer per class, targets 1 / 0
    const wOneHot = NAMES.map((c) => lsFit(pts.map((p) => (p.cls === c ? 1 : 0))));
    // one-vs-rest: targets +1 / −1
    const wOvr = NAMES.map((c) => lsFit(pts.map((p) => (p.cls === c ? 1 : -1))));
    // one-vs-one: one classifier per pair (+1 = first class of the pair);
    // fit only on the pair's own points (least squares on the subset)
    const PAIRS = [["A", "B"], ["A", "C"], ["B", "C"]];
    const wOvo = PAIRS.map(function (pair) {
      const sub = pts.filter((p) => p.cls === pair[0] || p.cls === pair[1]);
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      const b = [0, 0, 0];
      sub.forEach(function (p) {
        const phi = [1, p.x, p.y], t = p.cls === pair[0] ? 1 : -1;
        for (let j = 0; j < 3; j++) {
          b[j] += phi[j] * t;
          for (let k = 0; k < 3; k++) A[j][k] += phi[j] * phi[k];
        }
      });
      return LR.solve(A, b);
    });

    const sc = (w, x, y) => w[0] + w[1] * x + w[2] * y;

    function decide(method, x, y) {
      if (method === "onehot") {
        const scores = wOneHot.map((w) => sc(w, x, y));
        let win = 0;
        scores.forEach((s, i) => { if (s > scores[win]) win = i; });
        return { scores, win: NAMES[win], conflict: false };
      }
      if (method === "ovr") {
        const scores = wOvr.map((w) => sc(w, x, y));
        const yes = scores.filter((s) => s >= 0).length;
        let win = 0;
        scores.forEach((s, i) => { if (s > scores[win]) win = i; });
        return { scores, yes, win: NAMES[win], conflict: yes !== 1 };
      }
      // ovo: each pairwise classifier votes
      const votes = { A: 0, B: 0, C: 0 };
      const picks = PAIRS.map(function (pair, i) {
        const s = sc(wOvo[i], x, y);
        const pick = s >= 0 ? pair[0] : pair[1];
        votes[pick]++;
        return pick;
      });
      let win = "A", tie = false;
      NAMES.forEach((c) => { if (votes[c] > votes[win]) win = c; });
      tie = NAMES.filter((c) => votes[c] === votes[win]).length > 1;
      return { votes, picks, win, conflict: tie };
    }

    let method = "onehot";
    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const mbtns = {};
    [["onehot", "one-hot least squares"], ["ovr", "one-vs-rest"], ["ovo", "one-vs-one"]].forEach(function (m) {
      mbtns[m[0]] = LR.button(group, m[1], function () {
        method = m[0];
        for (const k in mbtns) mbtns[k].classList.toggle("on", k === method);
        draw();
      }, "small" + (m[0] === "onehot" ? " on" : ""));
    });

    let q = { x: 0.2, y: 0.4 };

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Three-class scatter with decision regions for one-hot least squares, one-vs-rest, and one-vs-one, plus a draggable query point whose scores and votes are shown",
    });
    const verdict = LR.el("div", "fig-msg show info");
    mount.appendChild(verdict);

    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: -5, xmax: 5, ymin: -4.6, ymax: 4.6, xlabel: "x₁", ylabel: "x₂" };
    let SC = null;

    // cached region layers, one per method
    const layers = {};
    function regionLayer(m) {
      if (layers[m]) return layers[m];
      const off = document.createElement("canvas");
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      off.width = W * dpr; off.height = H * dpr;
      const octx = off.getContext("2d");
      octx.scale(dpr, dpr);
      const BLK = 5;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const d = decide(m, wx, wy);
          octx.fillStyle = d.conflict ? CONFLICT : SOFT[d.win];
          octx.fillRect(px, py, BLK, BLK);
        }
      }
      layers[m] = off;
      return off;
    }

    LR.drag(cv, W, H, {
      hit: (p) => SC && (p.x - SC.sx(q.x)) ** 2 + (p.y - SC.sy(q.y)) ** 2 < 2200,
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        q.x = Math.max(P.xmin + 0.15, Math.min(P.xmax - 0.15, d.x));
        q.y = Math.max(P.ymin + 0.15, Math.min(P.ymax - 0.15, d.y));
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(regionLayer(method), 0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // pairwise / per-class boundary lines
      const ws = method === "onehot" ? null : method === "ovr" ? wOvr : wOvo;
      if (ws) ws.forEach((w, i) => LC.drawBoundary(ctx, P, sx, sy, w, "rgba(17,17,17,0.35)", 1.3, [5, 4]));

      pts.forEach((p) => LR.dot(ctx, sx(p.x), sy(p.y), 5.5, COLS[p.cls], "#ffffff"));

      const d = decide(method, q.x, q.y);
      LR.dot(ctx, sx(q.x), sy(q.y), 10, d.conflict ? "#b4b4b4" : COLS[d.win], C.orange);
      ctx.font = "800 12px Inter, sans-serif";
      ctx.fillStyle = "#ffffff"; ctx.textAlign = "center";
      ctx.fillText("?", sx(q.x), sy(q.y) + 4);

      // legend
      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      NAMES.forEach(function (c, i) {
        ctx.fillStyle = COLS[c];
        ctx.fillText("● class " + c, P.x0 + 8, P.y0 + 16 + i * 17);
      });
      ctx.fillStyle = "#8d8d8d";
      ctx.fillText("▨ conflict zone", P.x0 + 8, P.y0 + 16 + 3 * 17);

      // verdict text
      if (method === "onehot") {
        const s = d.scores.map((v, i) => "y" + (i + 1) + "(" + NAMES[i] + ") = " + LR.fmtF(v, 2));
        verdict.innerHTML = "<b>One-hot least squares.</b> " + s.join(", &nbsp;") +
          " → argmax picks <b style='color:" + COLS[d.win] + "'>" + d.win + "</b>. " +
          "Note the scores are not probabilities: they can be negative and do not sum to 1 (sum = " +
          LR.fmtF(d.scores[0] + d.scores[1] + d.scores[2], 2) + ").";
        verdict.className = "fig-msg show info";
      } else if (method === "ovr") {
        const s = d.scores.map((v, i) => NAMES[i] + "-vs-rest: " + LR.fmtF(v, 2) + (v >= 0 ? " (yes)" : " (no)"));
        verdict.innerHTML = "<b>One-vs-rest.</b> " + s.join(", &nbsp;") + " → " +
          (d.yes === 1
            ? "exactly one classifier says yes: predict <b style='color:" + COLS[d.win] + "'>" + d.win + "</b>."
            : d.yes === 0
              ? "<b>nobody says yes.</b> Fall back to argmax (the least negative score): <b style='color:" + COLS[d.win] + "'>" + d.win + "</b>, an arbitrary tie-break, since scores from different classifiers are not directly comparable."
              : "<b>" + d.yes + " classifiers say yes at once.</b> Fall back to argmax: <b style='color:" + COLS[d.win] + "'>" + d.win + "</b>, again comparing scores that were trained on different problems.");
        verdict.className = "fig-msg show " + (d.conflict ? "bad" : "info");
      } else {
        const v = PAIRS.map((pair, i) => pair.join(" vs ") + " → " + d.picks[i]);
        const tally = NAMES.map((c) => c + ": " + d.votes[c]).join(", ");
        verdict.innerHTML = "<b>One-vs-one.</b> " + v.join("; &nbsp;") + ". Votes: " + tally + " → " +
          (d.conflict
            ? "<b>a tied vote (a cycle).</b> Pairwise preferences went around in a circle, so majority voting has no answer and a tie-break rule must decide."
            : "predict <b style='color:" + COLS[d.win] + "'>" + d.win + "</b> with " + d.votes[d.win] + " of 3 votes.");
        verdict.className = "fig-msg show " + (d.conflict ? "bad" : "info");
      }
    }
    draw();
  };
})();
