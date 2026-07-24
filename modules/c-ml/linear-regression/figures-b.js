/* ══════════════════════════════════════════════════════════════
   figures-b.js — interactive figures for sections 7–12
   Fig 7.1 spaces · Fig 8.1 critical · Fig 8.2 gradfield
   Fig 9.2 bowl · Fig 9.3 fixit · Fig 10.1 poly · Fig 10.2 ridge
   Fig 11.1 losses
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ════════════════════════════════════════════════════════════
     Fig 7.1 — data space vs weight space (the signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.spaces = function (mount) {
    LR.header(
      mount,
      "One model, two spaces",
      "Left: a line in data space. Right: the same model as a single point on the loss contours in weight space. Drag either side."
    );

    const pts = LR.houseData(12, 3, 0.6);
    const opt = LR.fit1d(pts);
    const optJ = LR.loss1d(pts, opt.w0, opt.w1);

    // weight-space window
    const WSP = { w0min: -2, w0max: 6, w1min: -1.2, w1max: 3.2 };

    const PRESETS = [
      { name: "red", color: C.red, w0: 5.2, w1: -0.55 },
      { name: "green", color: C.green, w0: 1.1, w1: 1.28 },
      { name: "yellow", color: C.yellow, w0: -1.0, w1: 2.6 },
    ];

    let cur = { w0: 4.5, w1: -0.3 };
    let anim = null;

    const bar = LR.controls(mount);
    const lbl = LR.el("span", "", "Candidates:");
    lbl.style.fontWeight = "600";
    bar.appendChild(lbl);
    PRESETS.forEach(function (p) {
      const b = LR.button(bar, p.name, function () { flyTo(p.w0, p.w1); }, "small");
      b.style.borderColor = p.color;
      b.style.color = p.color;
      b.addEventListener("mouseenter", () => (b.style.background = p.color, b.style.color = "#ffffff"));
      b.addEventListener("mouseleave", () => (b.style.background = "#ffffff", b.style.color = p.color));
    });
    LR.button(bar, "Solve  (normal equation)", function () { flyTo(opt.w0, opt.w1); }, "primary");

    function flyTo(w0, w1) {
      if (anim) cancelAnimationFrame(anim);
      if (LR.reducedMotion) { cur = { w0, w1 }; drawAll(); return; }
      const tick = function () {
        cur.w0 += (w0 - cur.w0) * 0.16;
        cur.w1 += (w1 - cur.w1) * 0.16;
        drawAll();
        if (Math.abs(w0 - cur.w0) + Math.abs(w1 - cur.w1) > 0.003) anim = requestAnimationFrame(tick);
        else { cur = { w0, w1 }; drawAll(); }
      };
      tick();
    }

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const leftBox = LR.el("div");
    leftBox.appendChild(LR.el("div", "pane-label", "Data space — a model is a line"));
    const rightBox = LR.el("div");
    rightBox.appendChild(LR.el("div", "pane-label", "Weight space — a model is a point"));
    pane.appendChild(leftBox); pane.appendChild(rightBox);

    const L = LR.canvas(leftBox, 400, 350, { aria: "Data space: scatter with draggable regression line" });
    const R = LR.canvas(rightBox, 400, 350, { aria: "Weight space: loss contours with draggable weight point" });

    const ro = LR.readout(mount, [
      { k: "w", label: "(w₀, w₁)" },
      { k: "J", label: "J(w)" },
      { k: "opt", label: "optimum" },
    ]);

    // ---- precompute contour bands over weight space ----
    const GN = 110, GM = 96;
    const bands = [];
    const levels = [optJ + 0.03, optJ + 0.12, optJ + 0.35, optJ + 0.8, optJ + 1.6, optJ + 3, optJ + 6, optJ + 11, optJ + 18, optJ + 28];
    (function () {
      for (let i = 0; i < GN; i++) {
        bands.push(new Array(GM));
        for (let j = 0; j < GM; j++) {
          const w0 = WSP.w0min + ((WSP.w0max - WSP.w0min) * i) / (GN - 1);
          const w1 = WSP.w1min + ((WSP.w1max - WSP.w1min) * j) / (GM - 1);
          const J = LR.loss1d(pts, w0, w1);
          let b = 0;
          while (b < levels.length && J > levels[b]) b++;
          bands[i][j] = b;
        }
      }
    })();

    const PL = { x0: 46, y0: 16, w: L.W - 62, h: L.H - 62, xmin: 0, xmax: 5, ymin: -1, ymax: 9, xlabel: "x", ylabel: "t" };
    const PR = { x0: 50, y0: 16, w: R.W - 66, h: R.H - 62, xmin: WSP.w0min, xmax: WSP.w0max, ymin: WSP.w1min, ymax: WSP.w1max, xlabel: "w₀ (intercept)", ylabel: "w₁ (slope)" };
    let SL = null, SR = null;

    // dragging: left = line body; right = the weight point
    LR.drag(L.cv, L.W, L.H, {
      hit: function (p) {
        if (!SL) return false;
        const d = SL.inv(p.x, p.y);
        const yAt = SL.sy(cur.w0 + cur.w1 * d.x);
        return Math.abs(p.y - yAt) < 22;
      },
      move: function (p) {
        if (anim) cancelAnimationFrame(anim);
        const d = SL.inv(p.x, p.y);
        // translate + tilt: move intercept so line passes through drag point,
        // keep slope unless dragging near edges (pivot around centre x=2.5)
        const pivot = 2.5;
        if (d.x < 1.4 || d.x > 3.6) {
          // tilt around the pivot point
          const yPivot = cur.w0 + cur.w1 * pivot;
          const w1 = (d.y - yPivot) / (d.x - pivot);
          cur.w1 = Math.max(WSP.w1min, Math.min(WSP.w1max, w1));
          cur.w0 = yPivot - cur.w1 * pivot;
        } else {
          cur.w0 = Math.max(WSP.w0min, Math.min(WSP.w0max, d.y - cur.w1 * d.x));
        }
        drawAll();
      },
    });

    LR.drag(R.cv, R.W, R.H, {
      hit: function (p) {
        return p.x > PR.x0 && p.x < PR.x0 + PR.w && p.y > PR.y0 && p.y < PR.y0 + PR.h;
      },
      move: function (p) {
        if (anim) cancelAnimationFrame(anim);
        const d = SR.inv(p.x, p.y);
        cur.w0 = Math.max(WSP.w0min, Math.min(WSP.w0max, d.x));
        cur.w1 = Math.max(WSP.w1min, Math.min(WSP.w1max, d.y));
        drawAll();
      },
    });

    const bandColor = function (b) {
      // deep purple centre → pale outside
      const t = b / levels.length;
      const light = 32 + t * 64; // 32%..96%
      return "hsl(0, 0%, " + light + "%)";
    };

    function drawAll() {
      /* ---- left: data space ---- */
      L.ctx.clearRect(0, 0, L.W, L.H);
      SL = LR.plot(L.ctx, PL);
      const lc = L.ctx;
      lc.save();
      lc.beginPath(); lc.rect(PL.x0, PL.y0, PL.w, PL.h); lc.clip();
      // faint preset lines
      PRESETS.forEach(function (p) {
        lc.strokeStyle = p.color; lc.globalAlpha = 0.35; lc.lineWidth = 1.6;
        lc.beginPath();
        lc.moveTo(SL.sx(-0.3), SL.sy(p.w0 - 0.3 * p.w1));
        lc.lineTo(SL.sx(5.3), SL.sy(p.w0 + 5.3 * p.w1));
        lc.stroke();
      });
      lc.globalAlpha = 1;
      // current line
      lc.strokeStyle = C.orange; lc.lineWidth = 3;
      lc.beginPath();
      lc.moveTo(SL.sx(-0.3), SL.sy(cur.w0 - 0.3 * cur.w1));
      lc.lineTo(SL.sx(5.3), SL.sy(cur.w0 + 5.3 * cur.w1));
      lc.stroke();
      lc.restore();
      pts.forEach((p) => LR.dot(lc, SL.sx(p.x), SL.sy(p.t), 4, C.text));

      /* ---- right: weight space ---- */
      R.ctx.clearRect(0, 0, R.W, R.H);
      const rc = R.ctx;
      // contour bands first (under the axes)
      const cw = PR.w / (GN - 1), ch = PR.h / (GM - 1);
      for (let i = 0; i < GN - 1; i++) {
        for (let j = 0; j < GM - 1; j++) {
          rc.fillStyle = bandColor(bands[i][j]);
          const px = PR.x0 + i * cw;
          const py = PR.y0 + PR.h - (j + 1) * ch;
          rc.fillRect(px, py, cw + 0.6, ch + 0.6);
        }
      }
      SR = LR.plot(rc, PR);
      // presets as dots
      PRESETS.forEach(function (p) {
        LR.dot(rc, SR.sx(p.w0), SR.sy(p.w1), 6, p.color, "#ffffff");
      });
      // optimum star
      drawStar(rc, SR.sx(opt.w0), SR.sy(opt.w1), 8, "#ffffff", C.text);
      // current point
      LR.dot(rc, SR.sx(cur.w0), SR.sy(cur.w1), 8, C.orange, "#ffffff");

      rc.font = "600 11px Inter, sans-serif";
      rc.fillStyle = "#ffffff";
      rc.textAlign = "center";

      const J = LR.loss1d(pts, cur.w0, cur.w1);
      ro.set("w", "(" + LR.fmtF(cur.w0, 2) + ", " + LR.fmtF(cur.w1, 2) + ")");
      const close = J < optJ * 1.02 + 0.01;
      ro.set("J", LR.fmtF(J, 4) + (close ? "  ← at the bottom of the bowl" : ""), close ? C.green : C.orange);
      ro.set("opt", "w★ = (" + LR.fmtF(opt.w0, 2) + ", " + LR.fmtF(opt.w1, 2) + "), J = " + LR.fmtF(optJ, 4));
    }

    function drawStar(ctx, x, y, r, fill, stroke) {
      ctx.save();
      ctx.beginPath();
      for (let k = 0; k < 10; k++) {
        const rr = k % 2 === 0 ? r : r * 0.45;
        const a = (Math.PI * k) / 5 - Math.PI / 2;
        const px = x + rr * Math.cos(a), py = y + rr * Math.sin(a);
        k === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = fill; ctx.fill();
      ctx.strokeStyle = stroke; ctx.lineWidth = 1.6; ctx.stroke();
      ctx.restore();
    }

    drawAll();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 8.1 — critical points on f(x) = x⁴/4 − x
     ════════════════════════════════════════════════════════════ */
  LR.figs.critical = function (mount) {
    LR.header(
      mount,
      "Where the ground is flat",
      "Drag the point along the curve. The tangent's slope is f′(x) = x³ − 1; the minimum is exactly where it hits zero."
    );

    const f = (x) => Math.pow(x, 4) / 4 - x;
    const fp = (x) => Math.pow(x, 3) - 1;

    let px = -1.4;

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Curve of x to the fourth over four minus x with a draggable point and its tangent line",
    });
    const ro = LR.readout(mount, [
      { k: "x", label: "x" },
      { k: "f", label: "f(x)" },
      { k: "fp", label: "f′(x) = x³ − 1" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -2.2, xmax: 2.4, ymin: -1.6, ymax: 3.4, xlabel: "x", ylabel: "f(x)" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!SC) return false;
        const hx = SC.sx(px), hy = SC.sy(f(px));
        return (p.x - hx) ** 2 + (p.y - hy) ** 2 < 1200;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        px = Math.max(P.xmin + 0.05, Math.min(P.xmax - 0.05, d.x));
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // the curve
      ctx.strokeStyle = C.text; ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let X = P.xmin; X <= P.xmax; X += 0.01) {
        const px2 = sx(X), py2 = sy(f(X));
        X === P.xmin ? ctx.moveTo(px2, py2) : ctx.lineTo(px2, py2);
      }
      ctx.stroke();

      // minimum marker at x = 1
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = C.green; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(sx(1), sy(P.ymin)); ctx.lineTo(sx(1), sy(f(1))); ctx.stroke();
      ctx.setLineDash([]);
      LR.dot(ctx, sx(1), sy(f(1)), 5, "#ffffff", C.green);
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "left";
      ctx.fillText("x★ = 1, f = −0.75, f′ = 0", sx(1) + 10, sy(f(1)) + 16);

      // tangent at px
      const slope = fp(px);
      const y0t = f(px) - slope * 0.9;
      const y1t = f(px) + slope * 0.9;
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(sx(px - 0.9), sy(y0t));
      ctx.lineTo(sx(px + 0.9), sy(y1t));
      ctx.stroke();

      // downhill arrow (sign of the derivative points the way)
      if (Math.abs(slope) > 0.03) {
        const dir = slope > 0 ? -1 : 1; // move opposite the slope
        LR.arrow(ctx, sx(px), sy(f(px)) - 26, sx(px) + dir * 42, sy(f(px)) - 26, C.purple, 2.4);
        ctx.fillStyle = C.purple;
        ctx.font = "600 11.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("downhill", sx(px) + dir * 21, sy(f(px)) - 34);
      }
      ctx.restore();

      // the draggable point
      LR.dot(ctx, sx(px), sy(f(px)), 8, C.orange, "#ffffff");

      const flat = Math.abs(slope) < 0.05;
      ro.set("x", LR.fmtF(px, 2));
      ro.set("f", LR.fmtF(f(px), 3));
      ro.set("fp", LR.fmtF(slope, 3) + (flat ? "  ← flat: critical point" : slope > 0 ? "  (uphill to the right)" : "  (downhill to the right)"), flat ? C.green : C.orange);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 8.2 — the 2D gradient vanishes at the optimum
     ════════════════════════════════════════════════════════════ */
  LR.figs.gradfield = function (mount) {
    LR.header(
      mount,
      "Gradient = 0, in two dimensions",
      "Drag the probe. The arrow is −∇J, the steepest downhill direction. It shrinks to nothing exactly at w★ = (1, 2)."
    );

    const J = (a, b) => (a - 1) ** 2 + (b - 2) ** 2;
    const grad = (a, b) => [2 * (a - 1), 2 * (b - 2)];

    let w = [-1.2, 4.3];

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Circular contours of a quadratic bowl with a draggable probe showing the negative gradient arrow",
    });
    const ro = LR.readout(mount, [
      { k: "w", label: "(w₀, w₁)" },
      { k: "g", label: "∇J = (2(w₀−1), 2(w₁−2))" },
      { k: "mag", label: "‖∇J‖" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -2, xmax: 4.4, ymin: -0.8, ymax: 5, xlabel: "w₀", ylabel: "w₁" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => p.x > P.x0 && p.x < P.x0 + P.w && p.y > P.y0 && p.y < P.y0 + P.h,
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        w = [Math.max(P.xmin, Math.min(P.xmax, d.x)), Math.max(P.ymin, Math.min(P.ymax, d.y))];
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // circular contours around (1,2)
      const rPix = Math.abs(sx(1) - sx(0)); // pixels per unit (x)
      const ry = Math.abs(sy(1) - sy(0));
      for (let r = 0.5; r <= 5; r += 0.5) {
        ctx.strokeStyle = "hsl(0, 0%, " + Math.min(92, 40 + r * 11) + "%)";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.ellipse(sx(1), sy(2), r * rPix, r * ry, 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      // optimum
      LR.dot(ctx, sx(1), sy(2), 6, C.purple, "#ffffff");
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.purple; ctx.textAlign = "left";
      ctx.fillText("w★ = (1, 2)", sx(1) + 11, sy(2) - 9);

      // negative gradient arrow at the probe
      const g = grad(w[0], w[1]);
      const mag = Math.hypot(g[0], g[1]);
      if (mag > 0.04) {
        const s = 0.28; // display scale
        LR.arrow(ctx, sx(w[0]), sy(w[1]), sx(w[0] - g[0] * s), sy(w[1] - g[1] * s), C.orange, 3);
      }
      ctx.restore();

      LR.dot(ctx, sx(w[0]), sy(w[1]), 8, C.orange, "#ffffff");

      const flat = mag < 0.12;
      ro.set("w", "(" + LR.fmtF(w[0], 2) + ", " + LR.fmtF(w[1], 2) + ")");
      ro.set("g", "(" + LR.fmtF(g[0], 2) + ", " + LR.fmtF(g[1], 2) + ")");
      ro.set("mag", LR.fmtF(mag, 3) + (flat ? "  ← the gradient has vanished: this is the minimum" : ""), flat ? C.green : C.orange);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 9.2 — gradient descent on the 3D loss bowl
     ════════════════════════════════════════════════════════════ */
  LR.figs.bowl = function (mount) {
    LR.header(
      mount,
      "The loss bowl",
      "The real surface J(w₀, w₁) for the dataset below. Drag to rotate. Drop the ball, pick α, and watch it roll — or overshoot."
    );

    const pts = LR.houseData(12, 3, 0.6);
    const N = pts.length;
    const opt = LR.fit1d(pts);
    const optJ = LR.loss1d(pts, opt.w0, opt.w1);

    const DOM = { w0min: -2, w0max: 6, w1min: -1.2, w1max: 3.6 };
    const G = 26; // surface grid
    const Jmax = (function () {
      let m = 0;
      for (let i = 0; i <= G; i++)
        for (let j = 0; j <= G; j++)
          m = Math.max(m, Jat(gw0(i), gw1(j)));
      return m;
    })();

    function gw0(i) { return DOM.w0min + ((DOM.w0max - DOM.w0min) * i) / G; }
    function gw1(j) { return DOM.w1min + ((DOM.w1max - DOM.w1min) * j) / G; }
    function Jat(a, b) { return LR.loss1d(pts, a, b); }
    function gradAt(a, b) {
      let g0 = 0, g1 = 0;
      for (const p of pts) { const r = a + b * p.x - p.t; g0 += r; g1 += r * p.x; }
      return [g0 / N, g1 / N];
    }
    function gradStoch(a, b, i) {
      const p = pts[i];
      const r = a + b * p.x - p.t;
      return [r, r * p.x];
    }

    // ---- controls ----
    let alpha = 0.08, stochastic = false;
    const bar = LR.controls(mount);
    const aS = LR.slider(bar, "α (learning rate)", 0.005, 0.4, 0.005, alpha, (v) => { alpha = v; }, (v) => LR.fmtF(v, 3));
    const modeGroup = LR.el("div", "toggle-group");
    bar.appendChild(modeGroup);
    const bBatch = LR.button(modeGroup, "Batch", function () { stochastic = false; bBatch.classList.add("on"); bStoch.classList.remove("on"); }, "on small");
    const bStoch = LR.button(modeGroup, "Stochastic", function () { stochastic = true; bStoch.classList.add("on"); bBatch.classList.remove("on"); }, "small");
    LR.button(bar, "Step ▸", step, "small");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, LR.reducedMotion ? 500 : 160);
    }, "small primary");
    LR.button(bar, "Reset ⟲", reset, "small");

    // ---- state ----
    let ball, path, diverged, sIdx = 0;
    function reset() {
      stopPlay();
      ball = [5.4, 3.1];
      path = [ball.slice()];
      diverged = false;
      msg.hide();
      draw();
    }
    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    function step() {
      if (diverged) { stopPlay(); return; }
      const g = stochastic ? gradStoch(ball[0], ball[1], (sIdx = (sIdx + 7) % N)) : gradAt(ball[0], ball[1]);
      ball = [ball[0] - alpha * g[0], ball[1] - alpha * g[1]];
      path.push(ball.slice());
      if (path.length > 400) path.shift();
      if (Math.abs(ball[0]) > 40 || Math.abs(ball[1]) > 40 || !isFinite(ball[0])) {
        diverged = true;
        stopPlay();
        msg.show("Diverged! Each step overshot the valley and landed higher. Lower α and press Reset.", "bad");
      } else if (LR.loss1d(pts, ball[0], ball[1]) < optJ * 1.01 + 0.005) {
        msg.show("Converged: w ≈ (" + LR.fmtF(ball[0], 2) + ", " + LR.fmtF(ball[1], 2) + "), matching the normal equation's w★ = (" + LR.fmtF(opt.w0, 2) + ", " + LR.fmtF(opt.w1, 2) + ").", "good");
      }
      draw();
    }

    // ---- 3D view ----
    let yaw = 0.72, pitch = 0.52;
    const { cv, ctx, W, H } = LR.canvas(mount, 820, 460, {
      aria: "Rotatable 3D surface of the loss bowl with a ball performing gradient descent steps",
    });
    const msg = LR.msg(mount);
    const ro = LR.readout(mount, [
      { k: "w", label: "ball (w₀, w₁)" },
      { k: "J", label: "J(w)" },
      { k: "steps", label: "steps" },
    ]);

    LR.drag(cv, W, H, {
      hit: () => true,
      down: function (p) { this._last = p; },
      move: function (p) {
        if (!this._last) { this._last = p; return; }
        yaw += (p.x - this._last.x) * 0.008;
        pitch = Math.max(0.15, Math.min(1.25, pitch + (p.y - this._last.y) * 0.006));
        this._last = p;
        draw();
      },
      up: function () { this._last = null; },
    });

    const CX = W / 2, CY = H / 2 + 30, SCALE = 46;
    function proj(a, b, J) {
      // normalize to centred coords
      const X = (a - (DOM.w0min + DOM.w0max) / 2) / ((DOM.w0max - DOM.w0min) / 2) * 2.6;
      const Y = (b - (DOM.w1min + DOM.w1max) / 2) / ((DOM.w1max - DOM.w1min) / 2) * 2.6;
      const Z = (Math.min(J, Jmax * 1.05) / Jmax) * 3.4;
      const cx = Math.cos(yaw), sx = Math.sin(yaw);
      const x1 = X * cx - Y * sx;
      const y1 = X * sx + Y * cx;
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const z2 = Z * cp - y1 * sp;      // screen height component
      const depth = y1 * cp + Z * sp;   // distance from viewer
      return { x: CX + x1 * SCALE, y: CY - z2 * SCALE, d: depth };
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // surface quads, painter's algorithm
      const quads = [];
      for (let i = 0; i < G; i++) {
        for (let j = 0; j < G; j++) {
          const c = [
            proj(gw0(i), gw1(j), Jat(gw0(i), gw1(j))),
            proj(gw0(i + 1), gw1(j), Jat(gw0(i + 1), gw1(j))),
            proj(gw0(i + 1), gw1(j + 1), Jat(gw0(i + 1), gw1(j + 1))),
            proj(gw0(i), gw1(j + 1), Jat(gw0(i), gw1(j + 1))),
          ];
          const h = Jat(gw0(i) + (gw0(1) - gw0(0)) / 2, gw1(j) + (gw1(1) - gw1(0)) / 2);
          quads.push({ c, d: (c[0].d + c[1].d + c[2].d + c[3].d) / 4, h });
        }
      }
      quads.sort((a, b) => b.d - a.d);
      quads.forEach(function (q) {
        const t = Math.min(1, q.h / Jmax);
        // low = warm orange, high = pale grey
        const hgt = Math.round(96 - t * 62);
        ctx.fillStyle = "hsla(0, 0%, " + hgt + "%, 0.94)";
        ctx.strokeStyle = "rgba(95,95,95, 0.25)";
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        ctx.moveTo(q.c[0].x, q.c[0].y);
        for (let k = 1; k < 4; k++) ctx.lineTo(q.c[k].x, q.c[k].y);
        ctx.closePath();
        ctx.fill(); ctx.stroke();
      });

      // optimum marker
      const po = proj(opt.w0, opt.w1, optJ);
      LR.dot(ctx, po.x, po.y, 4.5, C.green, "#ffffff");

      // path
      if (path.length > 1) {
        ctx.strokeStyle = C.purple; ctx.lineWidth = 2;
        ctx.beginPath();
        path.forEach(function (p, i) {
          const q = proj(p[0], p[1], Math.min(Jat(p[0], p[1]), Jmax * 1.05));
          i === 0 ? ctx.moveTo(q.x, q.y) : ctx.lineTo(q.x, q.y);
        });
        ctx.stroke();
        // step dots
        path.forEach(function (p, i) {
          if (i % 2) return;
          const q = proj(p[0], p[1], Math.min(Jat(p[0], p[1]), Jmax * 1.05));
          LR.dot(ctx, q.x, q.y, 2.2, C.purple);
        });
      }

      // ball
      const bJ = Math.min(Jat(ball[0], ball[1]), Jmax * 1.05);
      const pb = proj(ball[0], ball[1], bJ);
      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.35)"; ctx.shadowBlur = 6; ctx.shadowOffsetY = 2;
      LR.dot(ctx, pb.x, pb.y, 8, C.red, "#ffffff");
      ctx.restore();

      // axis labels
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      const a0 = proj(DOM.w0max + 0.4, DOM.w1min, 0);
      const a1 = proj(DOM.w0min, DOM.w1max + 0.4, 0);
      ctx.textAlign = "center";
      ctx.fillText("w₀", a0.x, a0.y);
      ctx.fillText("w₁", a1.x, a1.y);
      ctx.fillText("J(w₀, w₁) ↑", 70, 26);
      ctx.fillText("drag to rotate", W - 80, 26);

      // ---- inset: 1D sign-of-gradient intuition ----
      drawInset();

      const g = gradAt(ball[0], ball[1]);
      ro.set("w", diverged ? "💥" : "(" + LR.fmtF(ball[0], 2) + ", " + LR.fmtF(ball[1], 2) + ")");
      ro.set("J", diverged ? "exploded" : LR.fmtF(Jat(ball[0], ball[1]), 4), diverged ? C.red : C.orange);
      ro.set("steps", String(path.length - 1));
    }

    function drawInset() {
      const IX = 18, IY = H - 130, IW = 190, IH = 112;
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.94)";
      ctx.strokeStyle = "#dddddd";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(IX, IY, IW, IH, 8);
      ctx.fill(); ctx.stroke();

      ctx.font = "700 10.5px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("1D cross-section: sign of ∇J points the way", IX + 8, IY + 15);

      // parabola J(w1) through current w0 (cross-section)
      const w1min = opt.w1 - 2.2, w1max = opt.w1 + 2.2;
      const Js = [];
      let jmin = Infinity, jmax = 0;
      for (let k = 0; k <= 40; k++) {
        const b = w1min + ((w1max - w1min) * k) / 40;
        const v = Jat(diverged ? opt.w0 : ball[0], b);
        Js.push(v);
        jmin = Math.min(jmin, v); jmax = Math.max(jmax, v);
      }
      const mx = (b) => IX + 12 + ((b - w1min) / (w1max - w1min)) * (IW - 24);
      const my = (v) => IY + IH - 12 - ((v - jmin) / Math.max(1e-9, jmax - jmin)) * (IH - 44);
      ctx.strokeStyle = C.text; ctx.lineWidth = 1.6;
      ctx.beginPath();
      Js.forEach(function (v, k) {
        const b = w1min + ((w1max - w1min) * k) / 40;
        k === 0 ? ctx.moveTo(mx(b), my(v)) : ctx.lineTo(mx(b), my(v));
      });
      ctx.stroke();

      if (!diverged) {
        const b = Math.max(w1min, Math.min(w1max, ball[1]));
        const v = Jat(ball[0], b);
        LR.dot(ctx, mx(b), my(Math.min(v, jmax)), 4, C.red, "#ffffff");
        const g1 = gradAt(ball[0], b)[1];
        if (Math.abs(g1) > 0.02) {
          const dir = g1 > 0 ? -1 : 1;
          LR.arrow(ctx, mx(b), my(Math.min(v, jmax)) - 10, mx(b) + dir * 26, my(Math.min(v, jmax)) - 10, C.purple, 2);
        }
      }
      ctx.restore();
    }

    reset();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 9.3 — correct the visual (two rescue missions)
     ════════════════════════════════════════════════════════════ */
  LR.figs.fixit = function (mount) {
    LR.header(
      mount,
      "Your turn: two rescue missions",
      "Mission 1: the fit is wrong, drag the line to the least-squares optimum. Mission 2: the ball is stranded, walk it to the bottom."
    );

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);

    /* ---- Mission 1: fix the line ---- */
    const box1 = LR.el("div");
    box1.appendChild(LR.el("div", "pane-label", "Mission 1 — fix the fit"));
    pane.appendChild(box1);

    const pts = LR.houseData(10, 77, 0.5);
    const best = LR.fit1d(pts);
    const bestJ = LR.loss1d(pts, best.w0, best.w1);
    const HX = [0.8, 4.2];
    let hy = [7.6, 1.0]; // clearly wrong start

    const M1 = LR.canvas(box1, 400, 320, { aria: "Scatter with a badly fit draggable line to correct" });
    const bar1 = LR.controls(box1);
    LR.button(bar1, "Check my line", check1, "primary small");
    LR.button(bar1, "Reset", function () { hy = [7.6, 1.0]; msg1.hide(); draw1(); }, "small");
    const msg1 = LR.msg(box1);

    const P1 = { x0: 46, y0: 14, w: M1.W - 62, h: M1.H - 58, xmin: 0, xmax: 5, ymin: -1, ymax: 9, xlabel: "x", ylabel: "t" };
    let S1 = null, drag1 = -1;

    function lw() {
      const w1 = (hy[1] - hy[0]) / (HX[1] - HX[0]);
      return { w0: hy[0] - w1 * HX[0], w1 };
    }

    LR.drag(M1.cv, M1.W, M1.H, {
      hit: function (p) {
        for (let i = 0; i < 2; i++) {
          if ((p.x - S1.sx(HX[i])) ** 2 + (p.y - S1.sy(hy[i])) ** 2 < 900) { drag1 = i; return true; }
        }
        const w = lw();
        if (Math.abs(p.y - S1.sy(w.w0 + w.w1 * S1.inv(p.x, p.y).x)) < 18) { drag1 = 2; return true; }
        return false;
      },
      move: function (p) {
        const d = S1.inv(p.x, p.y);
        const y = Math.max(-1, Math.min(9, d.y));
        if (drag1 === 0) hy[0] = y;
        else if (drag1 === 1) hy[1] = y;
        else if (drag1 === 2) {
          const w = lw();
          const dy = y - (w.w0 + w.w1 * d.x);
          hy[0] += dy; hy[1] += dy;
        }
        msg1.hide();
        draw1();
      },
      up: function () { drag1 = -1; },
    });

    function check1() {
      const w = lw();
      const J = LR.loss1d(pts, w.w0, w.w1);
      if (J <= bestJ * 1.06) {
        msg1.show("✓ Nailed it. Your J = " + LR.fmtF(J, 4) + " vs the optimum " + LR.fmtF(bestJ, 4) + " — within 6%. The normal equation agrees with your eyes.", "good");
      } else if (J <= bestJ * 1.5) {
        msg1.show("Close: J = " + LR.fmtF(J, 4) + ", optimum is " + LR.fmtF(bestJ, 4) + ". Tilt or shift a little more.", "info");
      } else {
        msg1.show("Not yet: J = " + LR.fmtF(J, 4) + ", the optimum is " + LR.fmtF(bestJ, 4) + ". Look at where the residuals are biggest.", "bad");
      }
    }

    function draw1() {
      const ctx = M1.ctx;
      ctx.clearRect(0, 0, M1.W, M1.H);
      S1 = LR.plot(ctx, P1);
      const w = lw();
      ctx.save();
      ctx.beginPath(); ctx.rect(P1.x0, P1.y0, P1.w, P1.h); ctx.clip();
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(S1.sx(-0.3), S1.sy(w.w0 - 0.3 * w.w1));
      ctx.lineTo(S1.sx(5.3), S1.sy(w.w0 + 5.3 * w.w1));
      ctx.stroke();
      ctx.restore();
      pts.forEach((p) => LR.dot(ctx, S1.sx(p.x), S1.sy(p.t), 4, C.text));
      for (let i = 0; i < 2; i++) {
        LR.dot(ctx, S1.sx(HX[i]), S1.sy(hy[i]), 7.5, "#ffffff", C.orange);
        LR.dot(ctx, S1.sx(HX[i]), S1.sy(hy[i]), 3, C.orange);
      }
    }

    /* ---- Mission 2: walk the ball down ---- */
    const box2 = LR.el("div");
    box2.appendChild(LR.el("div", "pane-label", "Mission 2 — walk the ball to the minimum"));
    pane.appendChild(box2);

    // 1D cross-section of the same loss: J(w1) with w0 held at optimum
    const Jw1 = (b) => LR.loss1d(pts, best.w0, b);
    const w1opt = best.w1;
    let bw = w1opt + 1.7; // stranded on the wall

    const M2 = LR.canvas(box2, 400, 320, { aria: "One-dimensional loss curve with a draggable ball to move to the minimum" });
    const bar2 = LR.controls(box2);
    LR.button(bar2, "◀ step", function () { bw -= 0.12; msg2.hide(); draw2(); }, "small");
    LR.button(bar2, "step ▶", function () { bw += 0.12; msg2.hide(); draw2(); }, "small");
    LR.button(bar2, "Check position", check2, "primary small");
    LR.button(bar2, "Reset", function () { bw = w1opt + 1.7; msg2.hide(); draw2(); }, "small");
    const msg2 = LR.msg(box2);

    const w1lo = w1opt - 2.4, w1hi = w1opt + 2.4;
    const Jlo = Jw1(w1opt), Jhi = Math.max(Jw1(w1lo), Jw1(w1hi));
    const P2 = { x0: 50, y0: 14, w: M2.W - 66, h: M2.H - 58, xmin: w1lo, xmax: w1hi, ymin: 0, ymax: Jhi * 1.08, xlabel: "w₁ (slope)", ylabel: "J(w₁)" };
    let S2 = null;

    LR.drag(M2.cv, M2.W, M2.H, {
      hit: function (p) {
        return (p.x - S2.sx(bw)) ** 2 + (p.y - S2.sy(Jw1(bw))) ** 2 < 1400;
      },
      move: function (p) {
        bw = Math.max(w1lo, Math.min(w1hi, S2.inv(p.x, p.y).x));
        msg2.hide();
        draw2();
      },
    });

    function check2() {
      if (Math.abs(bw - w1opt) < 0.1) {
        msg2.show("✓ Bottom of the bowl: w₁ = " + LR.fmtF(bw, 2) + ", true minimum w₁★ = " + LR.fmtF(w1opt, 2) + ". The slope here is zero, so gradient descent would stop too.", "good");
      } else {
        const g = (Jw1(bw + 1e-4) - Jw1(bw - 1e-4)) / 2e-4;
        msg2.show("Not the minimum: the slope here is " + LR.fmtF(g, 2) + ", so downhill is to the " + (g > 0 ? "left" : "right") + ".", "bad");
      }
    }

    function draw2() {
      const ctx = M2.ctx;
      ctx.clearRect(0, 0, M2.W, M2.H);
      S2 = LR.plot(ctx, P2);
      ctx.save();
      ctx.beginPath(); ctx.rect(P2.x0, P2.y0, P2.w, P2.h); ctx.clip();
      ctx.strokeStyle = C.text; ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let k = 0; k <= 100; k++) {
        const b = w1lo + ((w1hi - w1lo) * k) / 100;
        k === 0 ? ctx.moveTo(S2.sx(b), S2.sy(Jw1(b))) : ctx.lineTo(S2.sx(b), S2.sy(Jw1(b)));
      }
      ctx.stroke();
      // target marker (subtle)
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = "#cccccc";
      ctx.beginPath(); ctx.moveTo(S2.sx(w1opt), S2.sy(0)); ctx.lineTo(S2.sx(w1opt), S2.sy(Jhi)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
      LR.dot(ctx, S2.sx(bw), S2.sy(Jw1(bw)), 9, C.red, "#ffffff");
    }

    draw1(); draw2();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 10.1 — polynomial basis functions & overfitting
     ════════════════════════════════════════════════════════════ */
  LR.figs.poly = function (mount) {
    LR.header(
      mount,
      "Capacity vs truth",
      "Data from t = sin(2πx) + noise. Raise the degree M and watch train loss fall while test loss turns against you."
    );

    const TRUE = (x) => Math.sin(2 * Math.PI * x);
    const NOISE = 0.25;

    function makeSet(n, seed) {
      const rand = LR.rng(seed);
      const xs = [], ts = [];
      for (let i = 0; i < n; i++) {
        const x = n <= 1 ? 0.5 : i / (n - 1);
        xs.push(x);
        ts.push(TRUE(x) + LR.gauss(rand) * NOISE);
      }
      return { xs, ts };
    }

    let M = 3, N = 10;
    let train = makeSet(N, 55);
    const test = makeSet(40, 999); // fixed held-out set

    const bar = LR.controls(mount);
    const mS = LR.slider(bar, "M (degree)", 0, 9, 1, M, function (v) { M = Math.round(v); refit(); }, (v) => String(Math.round(v)));
    const nS = LR.slider(bar, "N (train points)", 10, 100, 5, N, function (v) {
      N = Math.round(v);
      train = makeSet(N, 55);
      refit();
    }, (v) => String(Math.round(v)));
    let showW = false;
    const wBtn = LR.button(bar, "Weights table", function () {
      showW = !showW;
      wBtn.classList.toggle("on", showW);
      wtableWrap.style.display = showW ? "block" : "none";
      renderTable();
    }, "small");

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "The fit in data space"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Train vs test RMS error, by degree"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 330, { aria: "Polynomial fit of chosen degree to noisy samples of a sine curve" });
    const RC = LR.canvas(rb, 400, 330, { aria: "Train and test error curves as a function of polynomial degree" });

    const ro = LR.readout(mount, [
      { k: "trn", label: "train RMS" },
      { k: "tst", label: "test RMS" },
      { k: "verdict", label: "verdict" },
    ]);

    const wtableWrap = LR.el("div", "wtable-wrap");
    wtableWrap.style.display = "none";
    mount.appendChild(wtableWrap);

    let w = null, curves = null;

    function rms(w, set) { return Math.sqrt(2 * LR.polyLoss(w, set.xs, set.ts)); }

    function refit() {
      w = LR.polyfit(train.xs, train.ts, M, 0);
      curves = { trn: [], tst: [] };
      for (let m = 0; m <= 9; m++) {
        const wm = LR.polyfit(train.xs, train.ts, m, 0);
        curves.trn.push(rms(wm, train));
        curves.tst.push(rms(wm, test));
      }
      drawL(); drawR(); renderTable();

      const tr = rms(w, train), te = rms(w, test);
      ro.set("trn", LR.fmtF(tr, 3), C.green);
      ro.set("tst", LR.fmtF(te, 3), C.red);
      let verdict;
      if (M <= 1) verdict = "underfitting: not enough capacity for a sine";
      else if (te < tr * 1.8) verdict = "healthy: train and test agree";
      else verdict = "overfitting: memorizing noise (" + N + " points, " + (M + 1) + " coefficients)";
      ro.set("verdict", verdict, te < tr * 1.8 && M > 1 ? C.green : C.red);
    }

    function drawL() {
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      const P = { x0: 46, y0: 14, w: LC.W - 62, h: LC.H - 58, xmin: -0.02, xmax: 1.02, ymin: -1.8, ymax: 1.8, xlabel: "x", ylabel: "t" };
      const { sx, sy } = LR.plot(ctx, P);
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // true curve
      ctx.strokeStyle = C.green; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let k = 0; k <= 140; k++) {
        const x = k / 140;
        k === 0 ? ctx.moveTo(sx(x), sy(TRUE(x))) : ctx.lineTo(sx(x), sy(TRUE(x)));
      }
      ctx.stroke(); ctx.setLineDash([]);

      // fitted polynomial
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let k = 0; k <= 240; k++) {
        const x = k / 240;
        const y = Math.max(-8, Math.min(8, LR.polyval(w, x)));
        k === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
      }
      ctx.stroke();
      ctx.restore();

      // train points
      for (let i = 0; i < train.xs.length; i++) LR.dot(ctx, sx(train.xs[i]), sy(train.ts[i]), 3.6, C.text);

      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "left";
      ctx.fillText("- - true curve", P.x0 + 8, P.y0 + 14);
      ctx.fillStyle = C.orange;
      ctx.fillText("— degree-" + M + " fit", P.x0 + 8, P.y0 + 28);
    }

    function drawR() {
      const ctx = RC.ctx;
      ctx.clearRect(0, 0, RC.W, RC.H);
      const ymax = Math.min(2, Math.max(0.6, Math.max.apply(null, curves.tst.filter(isFinite)) * 1.1));
      const P = { x0: 50, y0: 14, w: RC.W - 66, h: RC.H - 58, xmin: 0, xmax: 9, ymin: 0, ymax: ymax, xlabel: "M (polynomial degree)", ylabel: "RMS error", xticks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] };
      const { sx, sy } = LR.plot(ctx, P);

      const plotCurve = function (arr, color) {
        ctx.strokeStyle = color; ctx.lineWidth = 2.2;
        ctx.beginPath();
        arr.forEach(function (v, m) {
          const y = Math.min(ymax, v);
          m === 0 ? ctx.moveTo(sx(m), sy(y)) : ctx.lineTo(sx(m), sy(y));
        });
        ctx.stroke();
        arr.forEach(function (v, m) {
          LR.dot(ctx, sx(m), sy(Math.min(ymax, v)), m === M ? 5.5 : 3.4, color, m === M ? "#ffffff" : null);
        });
      };
      plotCurve(curves.trn, C.green);
      plotCurve(curves.tst, C.red);

      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "left";
      ctx.fillText("train", sx(8.2), sy(curves.trn[9]) - 10);
      ctx.fillStyle = C.red;
      ctx.fillText("test", sx(8.2), sy(Math.min(ymax, curves.tst[9])) - 10);
    }

    function renderTable() {
      if (!showW) return;
      let html = "<table class='wtable'><tr><th></th>";
      for (let j = 0; j <= M; j++) html += "<th>w" + j + "</th>";
      html += "</tr><tr><td style='font-family:Inter,sans-serif'>fit</td>";
      for (let j = 0; j <= M; j++) {
        const v = w[j];
        const big = Math.abs(v) > 100;
        html += "<td class='" + (big ? "w-big" : "") + "'>" + (Math.abs(v) > 9999 ? v.toExponential(1) : LR.fmtF(v, 2)) + "</td>";
      }
      html += "</tr></table>";
      wtableWrap.innerHTML = html;
    }

    refit();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 10.2 — ridge regression tames the degree-9 monster
     ════════════════════════════════════════════════════════════ */
  LR.figs.ridge = function (mount) {
    LR.header(
      mount,
      "The λ dial",
      "Degree 9, ten points, guaranteed overfit — until the penalty λ·wᵀw enters. Slide ln λ and watch the weights shrink."
    );

    const TRUE = (x) => Math.sin(2 * Math.PI * x);
    function makeSet(n, seed) {
      const rand = LR.rng(seed);
      const xs = [], ts = [];
      for (let i = 0; i < n; i++) {
        const x = i / (n - 1);
        xs.push(x); ts.push(TRUE(x) + LR.gauss(rand) * 0.25);
      }
      return { xs, ts };
    }
    const train = makeSet(10, 55);
    const test = makeSet(40, 999);
    const M = 9;

    let lnl = -35; // ≈ no regularization

    const bar = LR.controls(mount);
    LR.slider(bar, "ln λ", -35, 3, 1, lnl, function (v) { lnl = v; refit(); }, function (v) {
      return v <= -35 ? "−∞ (off)" : String(Math.round(v));
    });

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Degree-9 fit under the penalty"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Weight magnitudes |wⱼ| (log scale)"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 330, { aria: "Degree nine polynomial fit relaxing as lambda increases" });
    const RC = LR.canvas(rb, 400, 330, { aria: "Bar chart of weight magnitudes shrinking with regularization" });

    const ro = LR.readout(mount, [
      { k: "lam", label: "λ" },
      { k: "trn", label: "train RMS" },
      { k: "tst", label: "test RMS" },
      { k: "wmax", label: "max |wⱼ|" },
    ]);

    let w = null;
    function refit() {
      const lam = lnl <= -35 ? 0 : Math.exp(lnl);
      w = LR.polyfit(train.xs, train.ts, M, lam);
      drawL(); drawR();
      const tr = Math.sqrt(2 * LR.polyLoss(w, train.xs, train.ts));
      const te = Math.sqrt(2 * LR.polyLoss(w, test.xs, test.ts));
      const wmax = Math.max.apply(null, w.map(Math.abs));
      ro.set("lam", lam === 0 ? "0 (pure least squares)" : "e^" + lnl + " ≈ " + lam.toExponential(1));
      ro.set("trn", LR.fmtF(tr, 3), C.green);
      let note = "";
      if (lnl <= -35) note = "  ← overfit";
      else if (lnl >= 0) note = "  ← now underfitting";
      else if (te < 0.35) note = "  ← the sweet spot";
      ro.set("tst", LR.fmtF(te, 3) + note, te < 0.35 ? C.green : C.red);
      ro.set("wmax", Math.abs(wmax) > 9999 ? wmax.toExponential(1) : LR.fmtF(wmax, 1), Math.abs(wmax) > 100 ? C.red : C.green);
    }

    function drawL() {
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      const P = { x0: 46, y0: 14, w: LC.W - 62, h: LC.H - 58, xmin: -0.02, xmax: 1.02, ymin: -1.8, ymax: 1.8, xlabel: "x", ylabel: "t" };
      const { sx, sy } = LR.plot(ctx, P);
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      ctx.strokeStyle = C.green; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let k = 0; k <= 140; k++) {
        const x = k / 140;
        k === 0 ? ctx.moveTo(sx(x), sy(TRUE(x))) : ctx.lineTo(sx(x), sy(TRUE(x)));
      }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = C.purple; ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let k = 0; k <= 240; k++) {
        const x = k / 240;
        const y = Math.max(-8, Math.min(8, LR.polyval(w, x)));
        k === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
      }
      ctx.stroke();
      ctx.restore();
      for (let i = 0; i < train.xs.length; i++) LR.dot(ctx, sx(train.xs[i]), sy(train.ts[i]), 3.8, C.text);
    }

    function drawR() {
      const ctx = RC.ctx;
      ctx.clearRect(0, 0, RC.W, RC.H);
      const P = { x0: 50, y0: 14, w: RC.W - 66, h: RC.H - 58 };
      // log-scale bars: |w| from 1e-2 to 1e5
      const lo = -2, hi = 5;
      const bw = P.w / 10;
      ctx.font = "600 10.5px Inter, sans-serif";
      // gridlines
      for (let e = lo; e <= hi; e++) {
        const y = P.y0 + P.h - ((e - lo) / (hi - lo)) * P.h;
        ctx.strokeStyle = "#eeeeee";
        ctx.beginPath(); ctx.moveTo(P.x0, y); ctx.lineTo(P.x0 + P.w, y); ctx.stroke();
        ctx.fillStyle = C.faint; ctx.textAlign = "right";
        ctx.fillText("1e" + e, P.x0 - 5, y + 3);
      }
      for (let j = 0; j <= 9; j++) {
        const v = Math.abs(w[j]);
        const lv = Math.max(lo, Math.min(hi, Math.log10(Math.max(v, 1e-9))));
        const h = ((lv - lo) / (hi - lo)) * P.h;
        const big = v > 100;
        ctx.fillStyle = big ? C.red : C.purple;
        ctx.fillRect(P.x0 + j * bw + 5, P.y0 + P.h - h, bw - 10, h);
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("w" + j, P.x0 + j * bw + bw / 2, P.y0 + P.h + 14);
      }
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1.4;
      ctx.strokeRect(P.x0, P.y0, P.w, P.h);
    }

    refit();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 11.1 — loss shapes compared
     ════════════════════════════════════════════════════════════ */
  LR.figs.losses = function (mount) {
    LR.header(
      mount,
      "Four candidate losses",
      "Each curve is the penalty as a function of the residual r = y − t. Toggle them and read off the failure modes."
    );

    const CURVES = [
      { key: "raw", label: "r  (raw residual)", color: "#9a9a9a", on: true, f: (r) => r },
      { key: "abs", label: "|r|", color: C.green, on: true, f: (r) => Math.abs(r) },
      { key: "sq", label: "r²", color: C.orange, on: true, f: (r) => r * r },
      { key: "cube", label: "r³", color: C.purple, on: false, f: (r) => r * r * r },
      { key: "pow", label: "r^(2n)", color: C.red, on: false, f: null }, // uses n slider
    ];
    let n = 3; // r^6 by default

    const bar = LR.controls(mount);
    CURVES.forEach(function (c) {
      const b = LR.button(bar, c.label, function () {
        c.on = !c.on;
        b.classList.toggle("on", c.on);
        draw();
      }, "small" + (c.on ? " on" : ""));
      b.style.borderColor = c.color;
    });
    LR.slider(bar, "n (for r^2n)", 2, 8, 1, n, function (v) { n = Math.round(v); draw(); }, (v) => "2n = " + 2 * Math.round(v));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 420, {
      aria: "Comparison of loss functions plotted against the residual",
    });

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -3, xmax: 3, ymin: -4, ymax: 8, xlabel: "residual r = y − t", ylabel: "loss L(r)" };

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const { sx, sy } = LR.plot(ctx, P);

      // zero lines
      ctx.strokeStyle = "#cfcfcf"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(P.xmin), sy(0)); ctx.lineTo(sx(P.xmax), sy(0)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(0), sy(P.ymin)); ctx.lineTo(sx(0), sy(P.ymax)); ctx.stroke();

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      CURVES.forEach(function (c) {
        if (!c.on) return;
        const f = c.key === "pow" ? (r) => Math.pow(r, 2 * n) : c.f;
        ctx.strokeStyle = c.color; ctx.lineWidth = 2.4;
        ctx.beginPath();
        let started = false;
        for (let k = 0; k <= 400; k++) {
          const r = P.xmin + ((P.xmax - P.xmin) * k) / 400;
          const v = f(r);
          if (!isFinite(v) || v > P.ymax * 3 || v < P.ymin * 3) { started = false; continue; }
          const X = sx(r), Y = sy(v);
          if (!started) { ctx.moveTo(X, Y); started = true; }
          else ctx.lineTo(X, Y);
        }
        ctx.stroke();
      });

      // annotations
      ctx.font = "600 11.5px Inter, sans-serif";
      const rawOn = CURVES[0].on, absOn = CURVES[1].on, powOn = CURVES[4].on;
      if (rawOn) {
        ctx.fillStyle = "#9a9a9a";
        ctx.textAlign = "left";
        ctx.fillText("negative for r < 0: cancellation →", sx(-2.9), sy(-2.4));
      }
      if (absOn) {
        ctx.fillStyle = C.green;
        ctx.textAlign = "center";
        ctx.fillText("kink: no derivative at 0", sx(0), sy(0) + 26);
      }
      if (powOn) {
        ctx.fillStyle = C.red;
        ctx.textAlign = "left";
        ctx.fillText("flat for |r| < 1: tiny errors ignored; explodes on outliers", sx(-1.05), sy(P.ymax) + 16);
      }
      ctx.restore();
    }
    draw();
  };
})();
