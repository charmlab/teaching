/* ══════════════════════════════════════════════════════════════
   figures-a.js — interactive figures for sections 1–6
   Fig 1.1 regclass · Fig 3.1 realtor · Fig 4.1 playground
   Fig 5.1 race · Fig 6.1 residuals
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — regression vs classification, one dataset, a morph
     ════════════════════════════════════════════════════════════ */
  LR.figs.regclass = function (mount) {
    LR.header(
      mount,
      "One dataset, two problems",
      "Regression reads the vertical axis as a continuous target. Classification reads it as a second feature and separates two classes."
    );

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bReg = LR.button(group, "Regression", () => setMode(0), "on");
    const bCls = LR.button(group, "Classification", () => setMode(1));
    const hint = LR.el("span", "", "");
    hint.style.cssText = "font-weight:400;color:#555555;";
    bar.appendChild(hint);

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Scatter plot that morphs between a regression fit line and a classification boundary",
    });

    // data + the two "answers", both really computed
    const pts = LR.houseData(26, 5, 0.8);
    const fit = LR.fit1d(pts); // regression answer
    // classes assigned by a ground-truth boundary with a margin, so the boundary is exact
    const B = { w0: 7.4, w1: -1.35 }; // t = 7.4 - 1.35 x
    pts.forEach((p) => (p.cls = p.t > B.w0 + B.w1 * p.x ? 1 : 0));

    let mode = 0; // 0 regression, 1 classification
    let m = 0; // morph parameter 0..1
    let anim = null;

    function setMode(next) {
      mode = next;
      bReg.classList.toggle("on", mode === 0);
      bCls.classList.toggle("on", mode === 1);
      hint.textContent =
        mode === 0
          ? "The line predicts a number: t̂ for any x."
          : "The boundary predicts a label: which side are you on?";
      if (anim) cancelAnimationFrame(anim);
      if (LR.reducedMotion) { m = mode; draw(); return; }
      const tick = function () {
        m += (mode - m) * 0.14;
        if (Math.abs(mode - m) < 0.004) { m = mode; draw(); return; }
        draw();
        anim = requestAnimationFrame(tick);
      };
      tick();
    }

    function lerp(a, b, u) { return a + (b - a) * u; }
    function mixColor(u, c0, c1) {
      // c0/c1 as [r,g,b]
      const r = Math.round(lerp(c0[0], c1[0], u));
      const g = Math.round(lerp(c0[1], c1[1], u));
      const b = Math.round(lerp(c0[2], c1[2], u));
      return "rgb(" + r + "," + g + "," + b + ")";
    }
    const BLACK = [17, 17, 17], GREEN = [102, 102, 102], PURPLE = [150, 150, 150];

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 60, y0: 22, w: W - 84, h: H - 74,
        xmin: 0, xmax: 5, ymin: -1, ymax: 10,
        xlabel: m < 0.5 ? "x (input feature)" : "x₁ (feature one)",
        ylabel: m < 0.5 ? "t (continuous target)" : "x₂ (feature two)",
      };
      const { sx, sy } = LR.plot(ctx, P);

      // the morphing line: fit → boundary
      const w0 = lerp(fit.w0, B.w0, m), w1 = lerp(fit.w1, B.w1, m);
      // shaded class regions fade in with m
      if (m > 0.02) {
        ctx.save();
        ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
        ctx.globalAlpha = 0.10 * m;
        // above the line
        ctx.fillStyle = "rgb(" + GREEN.join(",") + ")";
        ctx.beginPath();
        ctx.moveTo(sx(0), sy(w0)); ctx.lineTo(sx(5), sy(w0 + 5 * w1));
        ctx.lineTo(sx(5), sy(10)); ctx.lineTo(sx(0), sy(10));
        ctx.closePath(); ctx.fill();
        ctx.fillStyle = "rgb(" + PURPLE.join(",") + ")";
        ctx.beginPath();
        ctx.moveTo(sx(0), sy(w0)); ctx.lineTo(sx(5), sy(w0 + 5 * w1));
        ctx.lineTo(sx(5), sy(-1)); ctx.lineTo(sx(0), sy(-1));
        ctx.closePath(); ctx.fill();
        ctx.restore();
      }

      // line
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      ctx.strokeStyle = m < 0.5 ? C.orange : C.text;
      ctx.lineWidth = 2.6;
      if (m >= 0.5) ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(sx(-0.2), sy(w0 - 0.2 * w1));
      ctx.lineTo(sx(5.2), sy(w0 + 5.2 * w1));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // points
      pts.forEach(function (p) {
        const target = p.cls === 1 ? GREEN : PURPLE;
        LR.dot(ctx, sx(p.x), sy(p.t), 5, mixColor(m, BLACK, target));
      });

      // labels on the line
      ctx.font = "700 12.5px Inter, sans-serif";
      if (m < 0.5) {
        ctx.fillStyle = C.orange;
        ctx.textAlign = "left";
        ctx.fillText("t̂ = " + LR.fmtF(fit.w0, 2) + " + " + LR.fmtF(fit.w1, 2) + "·x   (least-squares fit)", P.x0 + 10, P.y0 + 18);
      } else {
        ctx.fillStyle = C.text;
        ctx.textAlign = "left";
        ctx.fillText("decision boundary: class = side of the line", P.x0 + 10, P.y0 + 18);
      }
    }

    setMode(0);
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the realtor animation (use case, kept narrative)
     ════════════════════════════════════════════════════════════ */
  LR.figs.realtor = function (mount) {
    LR.header(
      mount,
      "A realtor learns the market",
      "Each sale is a data point. The asking line is the running least-squares fit; it settles as evidence accumulates."
    );

    const bar = LR.controls(mount);
    const replay = LR.button(bar, "Replay ▸", start, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Animation of house sales appearing on a size versus price plot while a trend line settles, ending with a predicted price for a new listing",
    });

    const sales = LR.houseData(14, 11, 0.5); // t in $100k, x in 1000 sqft
    const NEWX = 3.8;
    let shown = 0, lineW = null, phase = "idle", timer = null, anim = null;

    function runningFit(k) {
      if (k < 2) return null;
      return LR.fit1d(sales.slice(0, k));
    }

    function start() {
      if (timer) clearInterval(timer);
      if (anim) cancelAnimationFrame(anim);
      shown = 0; lineW = null; phase = "selling";
      const stepMs = LR.reducedMotion ? 260 : 420;
      timer = setInterval(function () {
        shown += 1;
        const f = runningFit(shown);
        if (f) lineW = f;
        draw();
        if (shown >= sales.length) {
          clearInterval(timer); timer = null;
          phase = "listing";
          setTimeout(function () { phase = "priced"; draw(); }, LR.reducedMotion ? 200 : 700);
          draw();
        }
      }, stepMs);
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 64, y0: 22, w: W - 92, h: H - 74,
        xmin: 0, xmax: 5, ymin: 0, ymax: 9,
        xlabel: "size (1,000 sq ft)", ylabel: "sale price ($100k)",
      };
      const { sx, sy } = LR.plot(ctx, P);

      // settled line
      if (lineW) {
        ctx.save();
        ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
        ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(sx(0), sy(lineW.w0));
        ctx.lineTo(sx(5), sy(lineW.w0 + 5 * lineW.w1));
        ctx.stroke();
        ctx.restore();
        ctx.font = "700 12.5px Inter, sans-serif";
        ctx.fillStyle = C.orange; ctx.textAlign = "left";
        ctx.fillText("asking line (fit to " + Math.min(shown, sales.length) + " sales)", P.x0 + 10, P.y0 + 18);
      }

      // sold houses
      for (let i = 0; i < Math.min(shown, sales.length); i++) {
        const s = sales[i];
        drawHouse(sx(s.x), sy(s.t), C.text, false);
      }

      // the new listing
      if ((phase === "listing" || phase === "priced") && lineW) {
        const py = lineW.w0 + lineW.w1 * NEWX;
        // vertical guide
        ctx.strokeStyle = C.green; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(sx(NEWX), sy(0)); ctx.lineTo(sx(NEWX), sy(py)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(sx(NEWX), sy(py)); ctx.lineTo(sx(0), sy(py)); ctx.stroke();
        ctx.setLineDash([]);
        drawHouse(sx(NEWX), sy(py), C.green, true);

        if (phase === "priced") {
          const price = Math.round(py * 100) * 1000;
          const label = "predicted: $" + price.toLocaleString("en-CA");
          ctx.font = "700 13px Inter, sans-serif";
          const tw = ctx.measureText(label).width;
          const bx = sx(NEWX) - tw - 26, by = sy(py) - 44;
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = C.green; ctx.lineWidth = 1.6;
          roundRect(ctx, bx - 10, by - 17, tw + 20, 28, 8);
          ctx.fill(); ctx.stroke();
          ctx.fillStyle = C.green; ctx.textAlign = "left";
          ctx.fillText(label, bx, by + 3);
        }
      }

      if (phase === "idle") {
        ctx.font = "600 14px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("Press Replay to watch the market teach the line", P.x0 + P.w / 2, P.y0 + P.h / 2);
      }
    }

    function drawHouse(px, py, color, isNew) {
      ctx.save();
      ctx.translate(px, py);
      ctx.fillStyle = isNew ? "#ffffff" : color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      // body
      ctx.beginPath(); ctx.rect(-5, -3, 10, 8);
      ctx.fill(); ctx.stroke();
      // roof
      ctx.beginPath(); ctx.moveTo(-7, -3); ctx.lineTo(0, -10); ctx.lineTo(7, -3); ctx.closePath();
      ctx.fill(); ctx.stroke();
      ctx.restore();
    }

    function roundRect(c, x, y, w, h, r) {
      c.beginPath();
      c.moveTo(x + r, y);
      c.arcTo(x + w, y, x + w, y + h, r);
      c.arcTo(x + w, y + h, x, y + h, r);
      c.arcTo(x, y + h, x, y, r);
      c.arcTo(x, y, x + w, y, r);
      c.closePath();
    }

    draw();
    // auto-play once when scrolled into view
    if (!LR.reducedMotion) setTimeout(start, 350);
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — the y = w0 + w1 x playground
     ════════════════════════════════════════════════════════════ */
  LR.figs.playground = function (mount) {
    LR.header(
      mount,
      "You are the optimizer",
      "Two knobs, one line. Try to make the loss readout as small as you can, by hand."
    );

    const state = { w0: 4.5, w1: -0.6, grid: true };
    const pts = LR.houseData(14, 23, 0.55);

    const bar = LR.controls(mount);
    const s0 = LR.slider(bar, "w₀ (intercept)", -2, 6, 0.05, state.w0, (v) => { state.w0 = v; draw(); });
    const s1 = LR.slider(bar, "w₁ (slope)", -2, 3, 0.05, state.w1, (v) => { state.w1 = v; draw(); });
    const gridBtn = LR.button(bar, "Grid: on", function () {
      state.grid = !state.grid;
      gridBtn.textContent = "Grid: " + (state.grid ? "on" : "off");
      draw();
    }, "small");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Scatter plot with a line controlled by intercept and slope sliders",
    });

    const ro = LR.readout(mount, [
      { k: "eq", label: "model" },
      { k: "J", label: "loss J(w)" },
      { k: "best", label: "best possible J" },
    ]);
    const best = LR.fit1d(pts);
    const bestJ = LR.loss1d(pts, best.w0, best.w1);

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 60, y0: 20, w: W - 84, h: H - 72,
        xmin: 0, xmax: 5, ymin: -2, ymax: 10,
        xlabel: "x", ylabel: "y",
      };
      if (!state.grid) { P.xticks = []; P.yticks = []; }
      const { sx, sy } = LR.plot(ctx, P);

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // intercept marker
      LR.dot(ctx, sx(0), sy(state.w0), 5.5, "#ffffff", C.purple);
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillStyle = C.purple; ctx.textAlign = "left";
      ctx.fillText("w₀", sx(0) + 9, sy(state.w0) - 8);

      // the line
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(sx(-0.3), sy(state.w0 - 0.3 * state.w1));
      ctx.lineTo(sx(5.3), sy(state.w0 + 5.3 * state.w1));
      ctx.stroke();

      // slope triangle: run 1, rise w1 at x=2.5
      const tx = 2.5;
      ctx.strokeStyle = C.green; ctx.lineWidth = 1.8;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(tx), sy(state.w0 + tx * state.w1));
      ctx.lineTo(sx(tx + 1), sy(state.w0 + tx * state.w1));
      ctx.lineTo(sx(tx + 1), sy(state.w0 + (tx + 1) * state.w1));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.green;
      ctx.fillText("1", sx(tx + 0.5) - 3, sy(state.w0 + tx * state.w1) + 14);
      ctx.fillText("w₁", sx(tx + 1) + 6, sy(state.w0 + (tx + 0.5) * state.w1) + 4);

      ctx.restore();

      // data
      pts.forEach((p) => LR.dot(ctx, sx(p.x), sy(p.t), 4.5, C.text));

      const J = LR.loss1d(pts, state.w0, state.w1);
      ro.set("eq", "y = " + LR.fmtF(state.w0, 2) + " + " + LR.fmtF(state.w1, 2) + "·x");
      const close = J < bestJ * 1.05;
      ro.set("J", LR.fmtF(J, 3) + (close ? "  ← nearly optimal!" : ""), close ? C.green : C.orange);
      ro.set("best", LR.fmtF(bestJ, 3));
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — loops vs vectorization: the race
     ════════════════════════════════════════════════════════════ */
  LR.figs.race = function (mount) {
    LR.header(
      mount,
      "The same dot product, two speeds",
      "Left: the Python loop, one multiply-add per interpreter tick. Right: one vectorized call. Both really compute w·x."
    );

    let D = 64;
    const bar = LR.controls(mount);
    const dSlider = LR.slider(bar, "D (features)", 8, 512, 8, D, function (v) { D = v; reset(); }, (v) => String(v));
    const go = LR.button(bar, "Race ▸", race, "primary");
    LR.button(bar, "Reset ⟲", reset);

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 300, {
      aria: "Animated comparison of a slow element-by-element loop versus an instant vectorized dot product",
    });

    // deterministic vectors; the dot product is genuinely computed
    let w, x, truth, k, done, raf;
    function buildData() {
      const rand = LR.rng(100 + D);
      w = Array.from({ length: D }, () => Math.round((rand() * 2 - 1) * 100) / 100);
      x = Array.from({ length: D }, () => Math.round(rand() * 100) / 100);
      truth = 0;
      for (let i = 0; i < D; i++) truth += w[i] * x[i];
    }

    function reset() {
      if (raf) cancelAnimationFrame(raf);
      buildData();
      k = 0; done = false;
      draw();
    }

    function race() {
      if (raf) cancelAnimationFrame(raf);
      k = 0; done = true;
      const perFrame = LR.reducedMotion ? Math.ceil(D / 12) : Math.max(1, Math.round(D / 240));
      const tick = function () {
        k = Math.min(D, k + perFrame);
        draw();
        if (k < D) raf = requestAnimationFrame(tick);
      };
      tick();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const colW = (W - 60) / 2;
      const LX = 24, RX = 36 + colW;

      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "left";
      ctx.fillText("for j in range(D): y += w[j]*x[j]", LX, 28);
      ctx.fillText("y = np.dot(w, x)", RX, 28);

      // ---- loop side: cell strip filling one at a time ----
      const stripY = 46, stripH = 34;
      const cells = Math.min(D, 64);
      const per = Math.ceil(D / cells);
      const cw = colW / cells;
      for (let c = 0; c < cells; c++) {
        const doneCells = Math.floor(k / per);
        ctx.fillStyle = c < doneCells ? C.orange : "#f1f1f1";
        ctx.strokeStyle = "#dddddd";
        ctx.fillRect(LX + c * cw, stripY, cw - 1, stripH);
      }
      // partial sum, really accumulated
      let part = 0;
      for (let i = 0; i < k; i++) part += w[i] * x[i];
      ctx.font = "600 12.5px 'JetBrains Mono', Menlo, monospace";
      ctx.fillStyle = C.muted;
      ctx.fillText("ticks: " + k + " / " + D, LX, stripY + stripH + 22);
      ctx.fillStyle = k >= D && done ? C.green : C.text;
      ctx.fillText("y = " + (k > 0 ? LR.fmtF(part, 4) : "0.0") + (k < D ? "  (still adding…)" : "  ✓"), LX, stripY + stripH + 42);

      // interpreter overhead note
      ctx.font = "500 11.5px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.fillText("each tick: fetch, interpret, index, multiply, add", LX, stripY + stripH + 62);

      // ---- vector side: fills instantly when race starts ----
      const vDone = done;
      ctx.fillStyle = vDone ? C.green : "#f1f1f1";
      ctx.fillRect(RX, stripY, colW, stripH);
      ctx.strokeStyle = "#dddddd"; ctx.strokeRect(RX, stripY, colW, stripH);
      if (vDone) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "700 13px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("one fused, parallel step", RX + colW / 2, stripY + 22);
        ctx.textAlign = "left";
      }
      ctx.font = "600 12.5px 'JetBrains Mono', Menlo, monospace";
      ctx.fillStyle = C.muted;
      ctx.fillText("ticks: " + (vDone ? 1 : 0) + " / 1", RX, stripY + stripH + 22);
      ctx.fillStyle = vDone ? C.green : C.text;
      ctx.fillText("y = " + (vDone ? LR.fmtF(truth, 4) + "  ✓" : "–"), RX, stripY + stripH + 42);
      ctx.font = "500 11.5px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.fillText("compiled BLAS, SIMD lanes, cache-aware", RX, stripY + stripH + 62);

      // ---- verdict bar ----
      if (done && k >= D) {
        ctx.font = "800 15px Inter, sans-serif";
        ctx.fillStyle = C.text;
        ctx.textAlign = "center";
        ctx.fillText(
          "same answer, " + D + "× fewer interpreter ticks, and the gap grows with D",
          W / 2, H - 26
        );
        ctx.textAlign = "left";
      }

      // divider
      ctx.strokeStyle = "#e6e6e6";
      ctx.beginPath(); ctx.moveTo(W / 2, 14); ctx.lineTo(W / 2, H - 54); ctx.stroke();
    }

    reset();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — residuals & the loss, derived on screen
     ════════════════════════════════════════════════════════════ */
  LR.figs.residuals = function (mount) {
    LR.header(
      mount,
      "Deriving the loss",
      "Drag the two handles to move the line. Show the residuals, square them, and watch J(w) respond. Then try to beat Solve."
    );

    const pts = LR.houseData(11, 31, 0.6);
    const best = LR.fit1d(pts);
    const bestJ = LR.loss1d(pts, best.w0, best.w1);

    // line defined by two draggable handles at fixed x positions
    const HX = [0.8, 4.2];
    let hy = [6.5, 3.2]; // handle heights (deliberately bad start)
    let showRes = false, showSq = false, anim = null;

    const bar = LR.controls(mount);
    const bRes = LR.button(bar, "Show residuals", function () {
      showRes = !showRes;
      bRes.classList.toggle("on", showRes);
      if (!showRes) { showSq = false; bSq.classList.remove("on"); }
      draw();
    });
    const bSq = LR.button(bar, "Square them", function () {
      showSq = !showSq;
      bSq.classList.toggle("on", showSq);
      if (showSq && !showRes) { showRes = true; bRes.classList.add("on"); }
      draw();
    });
    LR.button(bar, "Best fit (Solve)", function () {
      if (anim) cancelAnimationFrame(anim);
      const target = [best.w0 + best.w1 * HX[0], best.w0 + best.w1 * HX[1]];
      if (LR.reducedMotion) { hy = target; draw(); return; }
      const tick = function () {
        hy[0] += (target[0] - hy[0]) * 0.18;
        hy[1] += (target[1] - hy[1]) * 0.18;
        draw();
        if (Math.abs(target[0] - hy[0]) + Math.abs(target[1] - hy[1]) > 0.004) anim = requestAnimationFrame(tick);
        else { hy = target.slice(); draw(); }
      };
      tick();
    }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Scatter plot with a draggable line, optional residual segments and squared-error squares, and a live loss readout",
    });

    const ro = LR.readout(mount, [
      { k: "eq", label: "line" },
      { k: "J", label: "J(w) = mean ½(y−t)²" },
      { k: "best", label: "optimum J" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 5, ymin: -1, ymax: 10, xlabel: "x", ylabel: "t" };
    let SC = null; // scale fns

    function lineW() {
      const w1 = (hy[1] - hy[0]) / (HX[1] - HX[0]);
      return { w0: hy[0] - w1 * HX[0], w1 };
    }

    let dragging = -1;
    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!SC) return false;
        for (let i = 0; i < 2; i++) {
          const px = SC.sx(HX[i]), py = SC.sy(hy[i]);
          if ((p.x - px) ** 2 + (p.y - py) ** 2 < 900) { dragging = i; return true; }
        }
        // grab the line body: translate both handles
        const w = lineW();
        const yAt = SC.sy(w.w0 + w.w1 * SC.inv(p.x, p.y).x);
        if (Math.abs(p.y - yAt) < 18) { dragging = 2; return true; }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        const y = Math.max(P.ymin, Math.min(P.ymax, d.y));
        if (dragging === 0) hy[0] = y;
        else if (dragging === 1) hy[1] = y;
        else if (dragging === 2) {
          const w = lineW();
          const mid = w.w0 + w.w1 * d.x;
          const dy = y - mid;
          hy[0] += dy; hy[1] += dy;
        }
        draw();
      },
      up: function () { dragging = -1; },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;
      const w = lineW();

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // squared-error boxes (area = r², drawn as true squares in pixel space)
      if (showSq) {
        const yPerPx = P.h / (P.ymax - P.ymin);
        pts.forEach(function (p, i) {
          const yhat = w.w0 + w.w1 * p.x;
          const side = Math.abs(sy(p.t) - sy(yhat));
          const top = Math.min(sy(p.t), sy(yhat));
          const dir = i % 2 === 0 ? 1 : -1; // alternate sides to cut overlap
          ctx.fillStyle = "rgba(26,26,26, 0.16)";
          ctx.strokeStyle = "rgba(26,26,26, 0.55)";
          ctx.lineWidth = 1;
          const xx = dir > 0 ? sx(p.x) : sx(p.x) - side;
          ctx.fillRect(xx, top, side, side);
          ctx.strokeRect(xx, top, side, side);
        });
      }

      // residual segments
      if (showRes) {
        ctx.strokeStyle = C.red; ctx.lineWidth = 1.8; ctx.setLineDash([4, 3]);
        pts.forEach(function (p) {
          const yhat = w.w0 + w.w1 * p.x;
          ctx.beginPath(); ctx.moveTo(sx(p.x), sy(p.t)); ctx.lineTo(sx(p.x), sy(yhat)); ctx.stroke();
        });
        ctx.setLineDash([]);
      }

      // the line
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(sx(-0.3), sy(w.w0 - 0.3 * w.w1));
      ctx.lineTo(sx(5.3), sy(w.w0 + 5.3 * w.w1));
      ctx.stroke();
      ctx.restore();

      // data
      pts.forEach((p) => LR.dot(ctx, sx(p.x), sy(p.t), 4.5, C.text));

      // handles
      for (let i = 0; i < 2; i++) {
        LR.dot(ctx, sx(HX[i]), sy(hy[i]), 8, "#ffffff", C.orange);
        LR.dot(ctx, sx(HX[i]), sy(hy[i]), 3.2, C.orange);
      }

      const J = LR.loss1d(pts, w.w0, w.w1);
      ro.set("eq", "y = " + LR.fmtF(w.w0, 2) + " + " + LR.fmtF(w.w1, 2) + "·x");
      const close = J < bestJ * 1.03;
      ro.set("J", LR.fmtF(J, 4) + (close ? "  ← essentially optimal" : ""), close ? C.green : C.red);
      ro.set("best", LR.fmtF(bestJ, 4));
    }
    draw();
  };
})();
