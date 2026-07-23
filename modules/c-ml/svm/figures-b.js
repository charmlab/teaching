/* ══════════════════════════════════════════════════════════════
   figures-b.js — SVM lesson figures, section 5 (kernels)
   Fig 5.1 lift · Fig 5.2 kerncalc · Fig 5.3 kernexplore
   Fig 5.4 xormap · Fig 5.5 codeex
   Uses the shared SVM core (LR.svm) from figures-a.js.
   All kernel values, plane heights, and boundaries computed live.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const S = LR.svm;

  const T_COLOR = { "1": C.green, "-1": C.purple };

  /* ── tiny 3-D orbit projection (no library) ─────────────── */
  // yaw about the vertical axis, then pitch (elevation). pitch = π/2 is a
  // straight-down top view, i.e. an ordinary 2-D plot.
  function makeProj(yaw, pitch, scale, cx, cy) {
    const ca = Math.cos(yaw), sa = Math.sin(yaw);
    const ce = Math.cos(pitch), se = Math.sin(pitch);
    return function (x, y, z) {
      const x1 = x * ca - y * sa;
      const y1 = x * sa + y * ca;
      return {
        px: cx + x1 * scale,
        py: cy - (y1 * se + z * ce) * scale,
        depth: y1 * ce - z * se,
      };
    };
  }
  const lerp = (a, b, u) => a + (b - a) * u;
  const ease = (u) => u * u * (3 - 2 * u);

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — the feature-map lift (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.lift = function (mount) {
    LR.header(
      mount,
      "Lift the circles until a plane can cut them",
      "No line separates a disk from the ring around it. Add one feature, the squared radius, and a flat plane does the job. Drag to orbit once lifted."
    );

    const rand = LR.rng(59);
    const pts = [];
    for (let i = 0; i < 24; i++) {
      const r = Math.sqrt(rand()) * 0.95, ang = rand() * Math.PI * 2;
      pts.push({ x: r * Math.cos(ang), y: r * Math.sin(ang), t: 1 });
    }
    for (let i = 0; i < 28; i++) {
      const r = 1.75 + rand() * 0.65, ang = rand() * Math.PI * 2;
      pts.push({ x: r * Math.cos(ang), y: r * Math.sin(ang), t: -1 });
    }
    pts.forEach((p) => (p.r2 = p.x * p.x + p.y * p.y));

    // separating plane height: halfway between the largest inner r² and the
    // smallest outer r², computed from the actual data
    let maxIn = 0, minOut = Infinity;
    pts.forEach(function (p) {
      if (p.t === 1) maxIn = Math.max(maxIn, p.r2);
      else minOut = Math.min(minOut, p.r2);
    });
    const cPlane = (maxIn + minOut) / 2;
    const ZS = 0.55; // display scale for the lifted axis

    let liftT = 0, yaw = 0, pitch = Math.PI / 2;
    let phase = "2d"; // 2d | lifting | 3d | dropping
    let dropped = false;
    let anim = null;

    const bar = LR.controls(mount);
    const liftBtn = LR.button(bar, "Lift into 3-D ▸", function () {
      if (phase === "2d" || phase === "dropping") animateTo(1);
      else if (phase === "3d" || phase === "lifting") animateTo(0);
    }, "primary");
    LR.button(bar, "Reset view ⟲", function () {
      if (phase === "3d") { yaw = 0.65; pitch = 0.42; draw(); }
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 460, {
      aria: "Concentric-circle dataset lifted into three dimensions by the squared-radius feature, where a flat plane separates the classes; draggable orbit view",
    });
    const ro = LR.readout(mount, [
      { k: "phi", label: "feature map" },
      { k: "plane", label: "separating plane (computed)" },
      { k: "circle", label: "boundary back in 2-D" },
    ]);
    const msg = LR.msg(mount);
    ro.set("phi", "φ(x) = (x₁, x₂, x₁²+x₂²)");
    ro.set("plane", "z = " + LR.fmtF(cPlane, 2));
    ro.set("circle", "circle of radius √" + LR.fmtF(cPlane, 2) + " = " + LR.fmtF(Math.sqrt(cPlane), 2));
    msg.show("In 2-D this is hopeless for a line: the green class is surrounded. Press Lift.", "info");

    const SCL = 74, CX = W / 2, CY = H / 2 + 26;

    function animateTo(target) {
      if (anim) cancelAnimationFrame(anim);
      phase = target === 1 ? "lifting" : "dropping";
      liftBtn.textContent = target === 1 ? "Drop back to 2-D ▾" : "Lift into 3-D ▸";
      const t0 = liftT;
      const yaw0 = yaw, pitch0 = pitch;
      const yaw1 = target === 1 ? 0.65 : 0;
      const pitch1 = target === 1 ? 0.42 : Math.PI / 2;
      if (LR.reducedMotion) {
        liftT = target; yaw = yaw1; pitch = pitch1;
        finish(); draw(); return;
      }
      let u = 0;
      const tick = function () {
        u = Math.min(1, u + 0.022);
        const e = ease(u);
        liftT = lerp(t0, target, e);
        yaw = lerp(yaw0, yaw1, e);
        pitch = lerp(pitch0, pitch1, e);
        draw();
        if (u < 1) anim = requestAnimationFrame(tick);
        else finish();
      };
      tick();
      function finish() {
        phase = target === 1 ? "3d" : "2d";
        if (phase === "3d") {
          msg.show("Linearly separable after all: the plane z = " + LR.fmtF(cPlane, 2) + " has every green point below it and every purple point above it. Drag to orbit, then drop it back.", "good");
        } else {
          dropped = true;
          msg.show("The flat cut z = " + LR.fmtF(cPlane, 2) + " lands in the input space as the circle x₁² + x₂² = " + LR.fmtF(cPlane, 2) + ": a nonlinear boundary drawn by a linear classifier.", "good");
          draw();
        }
      }
    }

    let last = null;
    LR.drag(cv, W, H, {
      hit: () => liftT > 0.4,
      down: function (p) { last = p; },
      move: function (p) {
        if (!last) { last = p; return; }
        yaw += (p.x - last.x) * 0.008;
        pitch = Math.max(0.15, Math.min(1.35, pitch + (p.y - last.y) * 0.006));
        last = p;
        draw();
      },
      up: function () { last = null; },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const proj = makeProj(yaw, pitch, SCL, CX, CY);

      // ground grid: axes and reference rings at r = 1 and 2
      ctx.strokeStyle = "#e2e2e2";
      ctx.lineWidth = 1;
      [1, 2].forEach(function (r) {
        ctx.beginPath();
        for (let k = 0; k <= 64; k++) {
          const a = (k / 64) * Math.PI * 2;
          const q = proj(r * Math.cos(a), r * Math.sin(a), 0);
          k === 0 ? ctx.moveTo(q.px, q.py) : ctx.lineTo(q.px, q.py);
        }
        ctx.stroke();
      });
      const ax1 = proj(-2.8, 0, 0), ax2 = proj(2.8, 0, 0);
      const ay1 = proj(0, -2.8, 0), ay2 = proj(0, 2.8, 0);
      ctx.strokeStyle = "#cfcfcf";
      ctx.beginPath(); ctx.moveTo(ax1.px, ax1.py); ctx.lineTo(ax2.px, ax2.py); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ay1.px, ay1.py); ctx.lineTo(ay2.px, ay2.py); ctx.stroke();
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.faint; ctx.textAlign = "left";
      ctx.fillText("x₁", ax2.px + 6, ax2.py + 4);
      ctx.fillText("x₂", ay2.px + 6, ay2.py + 4);

      // vertical feature axis once lifting
      if (liftT > 0.02) {
        const z0 = proj(0, 0, 0), z1 = proj(0, 0, 3.4 * ZS);
        ctx.strokeStyle = "#cfcfcf";
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(z0.px, z0.py); ctx.lineTo(z1.px, z1.py); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.faint;
        ctx.fillText("x₁² + x₂²", z1.px + 6, z1.py);
      }

      // dropped-back boundary circle in the input plane
      if (dropped && liftT < 0.02) {
        const rc = Math.sqrt(cPlane);
        ctx.strokeStyle = C.orange;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        for (let k = 0; k <= 72; k++) {
          const a = (k / 72) * Math.PI * 2;
          const q = proj(rc * Math.cos(a), rc * Math.sin(a), 0);
          k === 0 ? ctx.moveTo(q.px, q.py) : ctx.lineTo(q.px, q.py);
        }
        ctx.stroke();
        ctx.fillStyle = C.orange;
        ctx.font = "700 12px Inter, sans-serif";
        const lab = proj(Math.sqrt(cPlane) * 0.72, Math.sqrt(cPlane) * 0.72, 0);
        ctx.fillText("x₁²+x₂² = " + LR.fmtF(cPlane, 2), lab.px + 6, lab.py - 6);
      }

      // points below the plane, then the plane, then points above
      const zc = cPlane * ZS;
      const drawPt = function (p) {
        const q = proj(p.x, p.y, liftT * p.r2 * ZS);
        // vertical drop line while lifted (helps read height)
        if (liftT > 0.05) {
          const g = proj(p.x, p.y, 0);
          ctx.strokeStyle = "rgba(0,0,0,0.07)";
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(g.px, g.py); ctx.lineTo(q.px, q.py); ctx.stroke();
        }
        LR.dot(ctx, q.px, q.py, 5, T_COLOR[p.t], "#ffffff");
      };
      const below = pts.filter((p) => liftT * p.r2 * ZS < zc);
      const above = pts.filter((p) => liftT * p.r2 * ZS >= zc);
      const bySort = (arr) =>
        arr
          .map((p) => ({ p, d: proj(p.x, p.y, liftT * p.r2 * ZS).depth }))
          .sort((a, b) => b.d - a.d)
          .map((o) => o.p);

      bySort(below).forEach(drawPt);

      // the separating plane appears near the top of the lift
      if (liftT > 0.85) {
        const alpha = (liftT - 0.85) / 0.15;
        const E = 2.7;
        const corners = [
          proj(-E, -E, zc), proj(E, -E, zc), proj(E, E, zc), proj(-E, E, zc),
        ];
        ctx.fillStyle = "rgba(26,26,26," + 0.13 * alpha + ")";
        ctx.strokeStyle = "rgba(26,26,26," + 0.75 * alpha + ")";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        corners.forEach((q, i) => (i === 0 ? ctx.moveTo(q.px, q.py) : ctx.lineTo(q.px, q.py)));
        ctx.closePath();
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = "rgba(26,26,26," + alpha + ")";
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillText("z = " + LR.fmtF(cPlane, 2), corners[1].px - 66, corners[1].py - 8);
      }

      bySort(above).forEach(drawPt);

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = T_COLOR["1"]; ctx.fillText("● inner class", 24, 26);
      ctx.fillStyle = T_COLOR["-1"]; ctx.fillText("● outer ring", 24, 43);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — the kernel-trick calculator
     ════════════════════════════════════════════════════════════ */
  LR.figs.kerncalc = function (mount) {
    LR.header(
      mount,
      "Two routes to the same number",
      "Drag x and y. The short route squares one dot product; the long route builds both 6-D feature vectors. They always agree."
    );

    let X = { x: 1, y: 2 }, Y = { x: 2, y: 1 };

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const left = LR.el("div");
    left.appendChild(LR.el("div", "pane-label", "Input space (drag the points)"));
    const right = LR.el("div");
    right.appendChild(LR.el("div", "pane-label", "The computation, both routes"));
    pane.appendChild(left); pane.appendChild(right);

    const { cv, ctx, W, H } = LR.canvas(left, 400, 340, {
      aria: "Two draggable points whose degree-two polynomial kernel is computed directly and via the explicit feature map",
    });
    const tableBox = LR.el("div");
    right.appendChild(tableBox);

    const P = { x0: 44, y0: 14, w: W - 60, h: H - 56, xmin: -3, xmax: 3, ymin: -3, ymax: 3, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null;

    const phi = (p) => [1, p.x * p.x, p.y * p.y, Math.SQRT2 * p.x, Math.SQRT2 * p.y, Math.SQRT2 * p.x * p.y];

    LR.drag(cv, W, H, {
      hit: function (p) {
        for (const pt of [X, Y]) {
          if ((p.x - SC.sx(pt.x)) ** 2 + (p.y - SC.sy(pt.y)) ** 2 < 500) { dragging = pt; return true; }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        dragging.x = Math.round(Math.max(-2.8, Math.min(2.8, d.x)) * 20) / 20;
        dragging.y = Math.round(Math.max(-2.8, Math.min(2.8, d.y)) * 20) / 20;
        draw();
      },
      up: function () { dragging = null; },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      LR.dot(ctx, sx(X.x), sy(X.y), 8, C.orange, "#ffffff");
      LR.dot(ctx, sx(Y.x), sy(Y.y), 8, C.purple, "#ffffff");
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "left";
      ctx.fillText("x = (" + LR.fmt(X.x, 2) + ", " + LR.fmt(X.y, 2) + ")", sx(X.x) + 11, sy(X.y) - 8);
      ctx.fillStyle = C.purple;
      ctx.fillText("y = (" + LR.fmt(Y.x, 2) + ", " + LR.fmt(Y.y, 2) + ")", sx(Y.x) + 11, sy(Y.y) - 8);

      const dot = X.x * Y.x + X.y * Y.y;
      const direct = Math.pow(1 + dot, 2);
      const px = phi(X), py = phi(Y);
      let long = 0;
      for (let i = 0; i < 6; i++) long += px[i] * py[i];
      const diff = Math.abs(direct - long);

      const fv = (v) => "(" + v.map((n) => LR.fmtF(n, 2)).join(", ") + ")";
      tableBox.innerHTML =
        "<table class='kern-table'>" +
        "<tr><th>step</th><th>value</th></tr>" +
        "<tr><td>xᵀy</td><td>" + LR.fmtF(X.x, 2) + "·" + LR.fmtF(Y.x, 2) + " + " + LR.fmtF(X.y, 2) + "·" + LR.fmtF(Y.y, 2) + " = " + LR.fmtF(dot, 2) + "</td></tr>" +
        "<tr><td><b>short route</b> (1 + xᵀy)²</td><td>(1 + " + LR.fmtF(dot, 2) + ")² = <b>" + LR.fmtF(direct, 4) + "</b></td></tr>" +
        "<tr><td>φ(x) ∈ ℝ⁶</td><td>" + fv(px) + "</td></tr>" +
        "<tr><td>φ(y) ∈ ℝ⁶</td><td>" + fv(py) + "</td></tr>" +
        "<tr><td><b>long route</b> φ(x)ᵀφ(y)</td><td><b>" + LR.fmtF(long, 4) + "</b></td></tr>" +
        "<tr><td>verdict</td><td class='kern-match'>identical ✓ (difference " + (diff < 1e-9 ? "0" : diff.toExponential(1)) + ")</td></tr>" +
        "</table>";
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.3 — the kernel boundary explorer
     ════════════════════════════════════════════════════════════ */
  LR.figs.kernexplore = function (mount) {
    LR.header(
      mount,
      "One solver, many geometries",
      "A real kernelized soft-margin SVM (simplified SMO) retrained on every change. Ringed points are the support vectors; the pale band is the margin region."
    );

    // interleaved two-moon data (deterministic, centred so kernel values
    // stay moderate for the polynomial kernel)
    const rand = LR.rng(31);
    const pts = [];
    for (let i = 0; i < 18; i++) {
      const th = (Math.PI * i) / 17;
      pts.push({
        x: -0.4 + 1.7 * Math.cos(th) + LR.gauss(rand) * 0.17,
        y: -0.5 + 1.5 * Math.sin(th) + LR.gauss(rand) * 0.17,
        t: 1,
      });
    }
    for (let i = 0; i < 18; i++) {
      const th = (Math.PI * i) / 17;
      pts.push({
        x: 0.4 - 1.7 * Math.cos(th) + LR.gauss(rand) * 0.17,
        y: 0.4 - 1.5 * Math.sin(th) + LR.gauss(rand) * 0.17,
        t: -1,
      });
    }

    const Gs = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 80];
    const CCs = [0.1, 0.3, 1, 3, 10, 100];
    let kernel = "rbf", deg = 3, gi = 4, ci = 2;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const kbtns = {};
    [["linear", "Linear"], ["poly", "Polynomial"], ["rbf", "RBF"]].forEach(function (m) {
      kbtns[m[0]] = LR.button(group, m[1], function () {
        kernel = m[0];
        for (const k in kbtns) kbtns[k].classList.toggle("on", k === kernel);
        schedule();
      }, "small" + (m[0] === "rbf" ? " on" : ""));
    });
    LR.slider(bar, "d (polynomial)", 2, 5, 1, deg, function (v) { deg = Math.round(v); if (kernel === "poly") schedule(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "γ (RBF)", 0, Gs.length - 1, 1, gi, function (v) { gi = Math.round(v); if (kernel === "rbf") schedule(); }, (v) => String(Gs[Math.round(v)]));
    LR.slider(bar, "C", 0, CCs.length - 1, 1, ci, function (v) { ci = Math.round(v); schedule(); }, (v) => String(CCs[Math.round(v)]));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Kernel SVM decision boundary explorer on two-moon data with kernel toggle and gamma and C sliders; support vectors ringed",
    });
    const ro = LR.readout(mount, [
      { k: "kern", label: "kernel" },
      { k: "sv", label: "support vectors" },
      { k: "acc", label: "training accuracy" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -3, xmax: 3, ymin: -2, ymax: 2, xlabel: "x₁", ylabel: "x₂" };

    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const offCtx = off.getContext("2d");

    let fit = null, timer = null;

    function kernFn() {
      if (kernel === "linear") return S.kLinear;
      if (kernel === "poly") return S.kPoly(deg);
      return S.kRBF(Gs[gi]);
    }
    function kernName() {
      if (kernel === "linear") return "linear  xᵀy";
      if (kernel === "poly") return "(1 + xᵀy)^" + deg;
      return "RBF, γ = " + Gs[gi];
    }

    function schedule() {
      // debounce: retraining an SVM on every slider tick is wasteful
      if (timer) clearTimeout(timer);
      timer = setTimeout(retrain, 120);
    }

    function retrain() {
      fit = S.smo(pts, CCs[ci], kernFn(), { seed: 9, maxSweeps: 160 });
      renderBoundary();
      draw();
    }

    function renderBoundary() {
      offCtx.clearRect(0, 0, W, H);
      const BLK = 7;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const f = fit.decide({ x: wx, y: wy });
          const inMargin = Math.abs(f) < 1;
          offCtx.fillStyle =
            f >= 0
              ? "rgba(102,102,102," + (inMargin ? 0.08 : 0.18) + ")"
              : "rgba(150,150,150," + (inMargin ? 0.07 : 0.16) + ")";
          offCtx.fillRect(px, py, BLK, BLK);
        }
      }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(off, 0, 0, W, H);
      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      const svSet = new Set(fit.sv);
      let correct = 0;
      pts.forEach(function (p, i) {
        const f = fit.decide(p);
        if (f * p.t > 0) correct++;
        const isSv = svSet.has(i);
        LR.dot(ctx, sx(p.x), sy(p.y), isSv ? 6.5 : 5, T_COLOR[p.t], isSv ? C.orange : "#ffffff");
      });
      const acc = correct / pts.length;

      ro.set("kern", kernName());
      ro.set("sv", fit.sv.length + " of " + pts.length);
      ro.set("acc", LR.fmtF(acc * 100, 1) + "%", acc === 1 ? C.green : acc < 0.9 ? C.red : undefined);

      if (kernel === "linear") {
        msg.show("A line cannot separate interleaved moons; accuracy caps out at " + LR.fmtF(acc * 100, 0) + "% no matter what C does. This is the wrong-shape failure that slack cannot fix.", "info");
      } else if (kernel === "rbf" && gi <= 1) {
        msg.show("γ = " + Gs[gi] + " is tiny: every pair of points looks similar, the geometry flattens, and the boundary underfits toward a nearly straight cut.", "info");
      } else if (kernel === "rbf" && gi >= 8) {
        msg.show("γ = " + Gs[gi] + " is huge: similarity dies within a tiny radius, so the model grows islands around individual points. Perfect on training data, and it would be terrible on new data: overfitting.", "bad");
      } else {
        msg.hide();
      }
    }
    retrain();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.4 — XOR, defeated by one feature
     Note: the source content poses XOR abstractly; placing the four
     corners at (±1, ±1) is a presentation choice that makes the map
     φ(x) = (x₁, x₂, x₁x₂) maximally clean.
     ════════════════════════════════════════════════════════════ */
  LR.figs.xormap = function (mount) {
    LR.header(
      mount,
      "Four points, no line, one multiplication",
      "Left: try to separate XOR with a line; the correct-count is computed from your line. Right: apply the feature map and the plane x₃ = 0 does it."
    );

    const XOR = [
      { x: 1, y: 1, t: 1 }, { x: -1, y: -1, t: 1 },
      { x: 1, y: -1, t: -1 }, { x: -1, y: 1, t: -1 },
    ];

    let e1 = { x: 0.2, y: 1.7 }, e2 = { x: 1.7, y: 0.2 };
    let applied = false;
    let yaw = 0.6, pitch = 0.5;

    const bar = LR.controls(mount);
    const applyBtn = LR.button(bar, "Apply φ(x) = (x₁, x₂, x₁x₂) ▸", function () {
      applied = !applied;
      applyBtn.textContent = applied ? "Back to raw inputs ▾" : "Apply φ(x) = (x₁, x₂, x₁x₂) ▸";
      if (applied) {
        msg.show("Lifted. The third coordinate x₁x₂ is +1 for the green diagonal and −1 for the purple one, so the plane x₃ = 0 separates all four with margin 1. Induced kernel: K(u,v) = u₁v₁ + u₂v₂ + u₁u₂v₁v₂.", "good");
      } else {
        msg.hide();
      }
      drawRight();
    }, "primary");

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const left = LR.el("div");
    left.appendChild(LR.el("div", "pane-label", "Input space: no line works"));
    const right = LR.el("div");
    right.appendChild(LR.el("div", "pane-label", "Feature space: a plane works (drag to orbit)"));
    pane.appendChild(left); pane.appendChild(right);

    const LC = LR.canvas(left, 400, 330, {
      aria: "The four XOR points with a draggable candidate line and a live count of correctly classified points",
    });
    const RC = LR.canvas(right, 400, 330, {
      aria: "The XOR points lifted into three dimensions by the product feature, separated by the horizontal plane",
    });

    const ro = LR.readout(mount, [
      { k: "correct", label: "your line classifies" },
      { k: "kern", label: "induced kernel" },
    ]);
    const msg = LR.msg(mount);
    ro.set("kern", "K(u,v) = u₁v₁ + u₂v₂ + u₁u₂v₁v₂");

    const P = { x0: 44, y0: 14, w: LC.W - 60, h: LC.H - 56, xmin: -1.9, xmax: 1.9, ymin: -1.9, ymax: 1.9, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null;

    LR.drag(LC.cv, LC.W, LC.H, {
      hit: function (p) {
        for (const e of [e1, e2]) {
          if ((p.x - SC.sx(e.x)) ** 2 + (p.y - SC.sy(e.y)) ** 2 < 500) { dragging = e; return true; }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        dragging.x = Math.max(-1.8, Math.min(1.8, d.x));
        dragging.y = Math.max(-1.8, Math.min(1.8, d.y));
        drawLeft();
      },
      up: function () { dragging = null; },
    });

    let last = null;
    LR.drag(RC.cv, RC.W, RC.H, {
      hit: () => applied,
      down: function (p) { last = p; },
      move: function (p) {
        if (!last) { last = p; return; }
        yaw += (p.x - last.x) * 0.009;
        pitch = Math.max(0.15, Math.min(1.35, pitch + (p.y - last.y) * 0.007));
        last = p;
        drawRight();
      },
      up: function () { last = null; },
    });

    function drawLeft() {
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // candidate line, extended
      const ux = e2.x - e1.x, uy = e2.y - e1.y;
      const ul = Math.hypot(ux, uy) || 1e-9;
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(sx(e1.x - (ux / ul) * 10), sy(e1.y - (uy / ul) * 10));
      ctx.lineTo(sx(e1.x + (ux / ul) * 10), sy(e1.y + (uy / ul) * 10));
      ctx.stroke();
      ctx.restore();

      // correct count under the better of the two orientations
      const n = { x: -uy / ul, y: ux / ul };
      const b = -(n.x * e1.x + n.y * e1.y);
      let c1 = 0, c2 = 0;
      XOR.forEach(function (p) {
        const z = n.x * p.x + n.y * p.y + b;
        if (p.t * z > 0) c1++; else if (p.t * z < 0) c2++;
      });
      const correct = Math.max(c1, c2);

      XOR.forEach(function (p) {
        LR.dot(ctx, sx(p.x), sy(p.y), 9, T_COLOR[p.t], "#ffffff");
        ctx.font = "600 11px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("(" + p.x + "," + p.y + ")", sx(p.x), sy(p.y) - 14);
      });
      [e1, e2].forEach((e) => LR.dot(ctx, sx(e.x), sy(e.y), 6, "#ffffff", C.orange));

      ro.set(
        "correct",
        correct + " of 4" + (correct <= 3 ? "  (3 is the ceiling for any line)" : ""),
        correct <= 2 ? C.red : C.amber
      );
    }

    function drawRight() {
      const ctx = RC.ctx;
      ctx.clearRect(0, 0, RC.W, RC.H);
      if (!applied) {
        ctx.font = "600 13px Inter, sans-serif";
        ctx.fillStyle = C.faint;
        ctx.textAlign = "center";
        ctx.fillText("press “Apply φ” to lift the four corners", RC.W / 2, RC.H / 2);
        return;
      }
      const proj = makeProj(yaw, pitch, 78, RC.W / 2, RC.H / 2);
      const ZS = 0.85;

      // plane x₃ = 0 as a translucent quad with a light grid
      const E = 1.6;
      const corners = [proj(-E, -E, 0), proj(E, -E, 0), proj(E, E, 0), proj(-E, E, 0)];
      ctx.fillStyle = "rgba(26,26,26,0.10)";
      ctx.strokeStyle = "rgba(26,26,26,0.7)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      corners.forEach((q, i) => (i === 0 ? ctx.moveTo(q.px, q.py) : ctx.lineTo(q.px, q.py)));
      ctx.closePath();
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.orange;
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("x₃ = 0", corners[2].px - 44, corners[2].py + 14);

      // lifted points with stems, drawn far-to-near
      XOR.map(function (p) {
        const z = p.x * p.y * ZS;
        return { p, z, q: proj(p.x, p.y, z), g: proj(p.x, p.y, 0) };
      })
        .sort((a, b) => b.q.depth - a.q.depth)
        .forEach(function (o) {
          ctx.strokeStyle = "rgba(0,0,0,0.18)";
          ctx.lineWidth = 1.2;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(o.g.px, o.g.py); ctx.lineTo(o.q.px, o.q.py); ctx.stroke();
          ctx.setLineDash([]);
          LR.dot(ctx, o.q.px, o.q.py, 8.5, T_COLOR[o.p.t], "#ffffff");
          ctx.font = "600 11px Inter, sans-serif";
          ctx.fillStyle = C.muted; ctx.textAlign = "left";
          ctx.fillText("x₁x₂ = " + (o.p.x * o.p.y > 0 ? "+1" : "−1"), o.q.px + 11, o.q.py + 4);
        });
    }

    drawLeft();
    drawRight();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.5 — complete-the-kernel code exercise
     ════════════════════════════════════════════════════════════ */
  LR.figs.codeex = function (mount) {
    LR.header(
      mount,
      "Complete the kernel, then let the numbers judge you",
      "Choose the return line and press Run. The code really executes on the lesson's fixed points and is checked against the hand-worked arithmetic."
    );

    const POLY_OPTS = [
      { code: "(x @ y) ** d", fn: (dot, d) => Math.pow(dot, d) },
      { code: "(1 + x @ y) ** d", fn: (dot, d) => Math.pow(1 + dot, d) },
      { code: "1 + (x @ y) ** d", fn: (dot, d) => 1 + Math.pow(dot, d) },
    ];
    const RBF_OPTS = [
      { code: "np.exp(-gamma * np.sum((x - y) ** 2))", fn: (d2, g) => Math.exp(-g * d2) },
      { code: "np.exp(-gamma * np.sqrt(np.sum((x - y) ** 2)))", fn: (d2, g) => Math.exp(-g * Math.sqrt(d2)) },
      { code: "-gamma * np.sum((x - y) ** 2)", fn: (d2, g) => -g * d2 },
    ];

    let which = "poly", polyPick = 0, rbfPick = 0, gamma = 0.5;

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);
    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);

    const mkSelect = function (labelText, options, onchange) {
      const lab = LR.el("label", "", labelText + " ");
      const sel = document.createElement("select");
      options.forEach(function (o, i) {
        const opt = document.createElement("option");
        opt.value = i; opt.textContent = o;
        sel.appendChild(opt);
      });
      sel.style.cssText = "font-family:var(--mono);font-size:12.5px;padding:2px 6px;border-radius:6px;border:1px solid #cccccc;max-width:340px";
      sel.addEventListener("change", () => onchange(parseInt(sel.value, 10)));
      lab.appendChild(sel);
      controls.appendChild(lab);
      return sel;
    };

    mkSelect("function:", ["polynomial_kernel (d = 2)", "rbf_kernel"], function (i) {
      which = i === 0 ? "poly" : "rbf";
      rebuildReturnSelect();
      renderCode();
    });
    let retSel = null;
    function rebuildReturnSelect() {
      if (retSel) retSel.parentElement.remove();
      retSel = mkSelect("return:", (which === "poly" ? POLY_OPTS : RBF_OPTS).map((o) => o.code), function (i) {
        if (which === "poly") polyPick = i; else rbfPick = i;
        renderCode();
      });
      retSel.value = String(which === "poly" ? polyPick : rbfPick);
      controls.appendChild(runBtnLabel);
    }
    const gS = LR.slider(controls, "γ (rbf only)", 0.1, 2, 0.05, gamma, function (v) { gamma = v; }, (v) => LR.fmtF(v, 2));
    const runBtnLabel = LR.el("span");
    controls.appendChild(runBtnLabel);
    const runBtn = LR.button(runBtnLabel, "Run ▸", run, "primary small");
    rebuildReturnSelect();

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    function renderCode() {
      let src;
      if (which === "poly") {
        src =
          "x = np.array([1, 2]); y = np.array([2, 1]); d = 2\n" +
          "\n" +
          "def polynomial_kernel(x, y, d):\n" +
          "    return " + POLY_OPTS[polyPick].code + "\n" +
          "\n" +
          "# the long route, for checking: build phi explicitly\n" +
          "def phi(v):\n" +
          "    return np.array([1, v[0]**2, v[1]**2,\n" +
          "                     np.sqrt(2)*v[0], np.sqrt(2)*v[1],\n" +
          "                     np.sqrt(2)*v[0]*v[1]])";
      } else {
        src =
          "u = np.array([1.0, 1.0])\n" +
          "v_near = np.array([1.2, 0.9])   # ||u - v||^2 = 0.05\n" +
          "v_far  = np.array([4.0, 3.0])   # ||u - v||^2 = 13\n" +
          "\n" +
          "def rbf_kernel(x, y, gamma):\n" +
          "    return " + RBF_OPTS[rbfPick].code;
      }
      codeEl.innerHTML = LR.highlight(src);
    }

    function run() {
      if (which === "poly") {
        const dot = 1 * 2 + 2 * 1; // x = (1,2), y = (2,1)
        const v = POLY_OPTS[polyPick].fn(dot, 2);
        // explicit phi dot product (the long route)
        const phi = (a, b2) => [1, a * a, b2 * b2, Math.SQRT2 * a, Math.SQRT2 * b2, Math.SQRT2 * a * b2];
        const px = phi(1, 2), py = phi(2, 1);
        let long = 0;
        for (let i = 0; i < 6; i++) long += px[i] * py[i];
        const ok = Math.abs(v - long) < 1e-9;
        out.innerHTML =
          "&gt;&gt;&gt; polynomial_kernel(x, y, d=2)\n" + LR.fmtF(v, 4) + "\n" +
          "&gt;&gt;&gt; phi(x) @ phi(y)          # the long route\n" + LR.fmtF(long, 4) + "\n" +
          (ok
            ? "<b>✓ " + LR.fmtF(v, 1) + " == " + LR.fmtF(long, 1) + "</b>: the kernel-trick identity holds. Same number, no 6-D vectors needed."
            : "<b>✗ mismatch</b>: " + LR.fmtF(v, 1) + " ≠ " + LR.fmtF(long, 1) + ". This line is not (1 + xᵀy)², so it is not the kernel of that feature map. (Hint: the constant 1 carries the linear and constant features.)");
      } else {
        const g = gamma;
        const self = RBF_OPTS[rbfPick].fn(0, g);
        const near = RBF_OPTS[rbfPick].fn(0.05, g);
        const far = RBF_OPTS[rbfPick].fn(13, g);
        const refNear = Math.exp(-g * 0.05), refFar = Math.exp(-g * 13);
        const ok1 = Math.abs(self - 1) < 1e-9;
        const ok2 = Math.abs(near - refNear) < 1e-9 && Math.abs(far - refFar) < 1e-9;
        out.innerHTML =
          "&gt;&gt;&gt; rbf_kernel(u, u, gamma=" + LR.fmtF(g, 2) + ")\n" + LR.fmtF(self, 4) + (ok1 ? "   ✓ K(u,u) = 1" : "   ✗ should be exactly 1: a point is perfectly similar to itself") + "\n" +
          "&gt;&gt;&gt; rbf_kernel(u, v_near, gamma=" + LR.fmtF(g, 2) + ")\n" + LR.fmtF(near, 4) + "   (hand arithmetic: exp(−γ·0.05) = " + LR.fmtF(refNear, 4) + ")\n" +
          "&gt;&gt;&gt; rbf_kernel(u, v_far, gamma=" + LR.fmtF(g, 2) + ")\n" + (far < 0.0001 && far >= 0 ? far.toExponential(2) : LR.fmtF(far, 4)) + "   (hand arithmetic: exp(−γ·13) = " + refFar.toExponential(2) + ")\n" +
          (ok1 && ok2
            ? "<b>✓ matches the worked derivation</b>: near pairs ≈ 1, far pairs ≈ 0, and at γ = 0.5 the near pair gives the lesson's 0.9753."
            : "<b>✗ does not match the derivation</b> K(u,v) = exp(−γ‖u−v‖²). " + (self < 0 || far < 0 ? "A negative similarity can never be a dot product: K(u,u) = ‖φ(u)‖² ≥ 0." : "Check whether you squared the distance."));
      }
    }
    renderCode();
  };
})();
