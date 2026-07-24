/* ══════════════════════════════════════════════════════════════
   figures-a.js — Neural Networks lesson figures, sections 1–3
   Fig 1.1 andxor · Fig 2.1 lift · Fig 3.1 neuron · Fig 3.2 layer
   Also defines the shared network core LR.nn (neuron, layer,
   forward pass, activations) used by figures-b.js and intended
   for reuse by the backpropagation lesson.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared network core (used by figures-b.js too) ─────── */
  const NN = (LR.nn = {
    sigmoid: (z) => 1 / (1 + Math.exp(-z)),
    tanh: (z) => Math.tanh(z),
    relu: (z) => Math.max(0, z),
    identity: (z) => z,
    step: (z) => (z > 0 ? 1 : 0),
    // affine map: (W x + b), W is rows of weight vectors
    affine: function (x, W, b) {
      return W.map(function (row, i) {
        let s = b[i];
        for (let j = 0; j < row.length; j++) s += row[j] * x[j];
        return s;
      });
    },
    // one layer: affine then element-wise activation
    layer: function (x, W, b, act) {
      return NN.affine(x, W, b).map(act);
    },
    // full forward pass; layers: [{W, b, act}]; returns every
    // intermediate activation (acts[0] = input) and pre-activations
    forward: function (x, layers) {
      const acts = [x.slice()];
      const zs = [];
      let h = x.slice();
      for (const L of layers) {
        const z = NN.affine(h, L.W, L.b);
        zs.push(z);
        h = z.map(L.act);
        acts.push(h);
      }
      return { y: h, acts, zs };
    },
    // the lecture's hand-built XOR network (hard thresholds)
    XOR: {
      layers: function () {
        return [
          { W: [[1, 1], [1, 1]], b: [-0.5, -1.5], act: NN.step },
          { W: [[1, -1]], b: [-0.5], act: NN.step },
        ];
      },
    },
  });

  // logic-gate dataset shared by several figures
  const GATE_PTS = [
    { x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 0 }, { x: 1, y: 1 },
  ];
  const GATES = {
    AND: (p) => (p.x === 1 && p.y === 1 ? 1 : 0),
    XOR: (p) => (p.x !== p.y ? 1 : 0),
  };
  const POS = C.green, NEG = C.purple;
  const POS_SOFT = "rgba(102,102,102,0.14)", NEG_SOFT = "rgba(150,150,150,0.12)";

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — AND vs XOR line-drawing (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.andxor = function (mount) {
    LR.header(
      mount,
      "Try to draw the line",
      "Drag either handle to steer the boundary. Green points are t = 1, purple are t = 0. AND yields in seconds; XOR never does."
    );

    let gate = "AND";
    let showHulls = false;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const gbtns = {};
    ["AND", "XOR"].forEach(function (g) {
      gbtns[g] = LR.button(group, g, function () {
        gate = g;
        for (const k in gbtns) gbtns[k].classList.toggle("on", k === gate);
        msg.hide();
        solvedShown = false;
        draw();
      }, "small" + (g === "AND" ? " on" : ""));
    });
    const hullBtn = LR.button(bar, "Show the convexity picture", function () {
      showHulls = !showHulls;
      hullBtn.classList.toggle("on", showHulls);
      hullBtn.textContent = showHulls ? "Hide the convexity picture" : "Show the convexity picture";
      draw();
    });

    // two draggable handles define the line (starts at y = 0.5, not separating AND)
    const A = { x: -0.45, y: 0.5 }, B = { x: 1.45, y: 0.5 };

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 460, {
      aria: "Four logic-gate points with a draggable separating line; toggle between AND and XOR and an overlay showing the convexity argument",
    });
    const msg = LR.msg(mount);
    const ro = LR.readout(mount, [
      { k: "gate", label: "gate" },
      { k: "score", label: "correctly classified" },
      { k: "verdict", label: "verdict" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -0.6, xmax: 1.6, ymin: -0.6, ymax: 1.6, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null, solvedShown = false;

    LR.drag(cv, W, H, {
      hit: function (p) {
        for (const h of [A, B]) {
          if ((p.x - SC.sx(h.x)) ** 2 + (p.y - SC.sy(h.y)) ** 2 < 900) { dragging = h; return true; }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        dragging.x = Math.max(P.xmin + 0.05, Math.min(P.xmax - 0.05, d.x));
        dragging.y = Math.max(P.ymin + 0.05, Math.min(P.ymax - 0.05, d.y));
        draw();
      },
      up: function () { dragging = null; },
    });

    // signed side of the line through A,B for a point p
    function side(p) {
      return (B.x - A.x) * (p.y - A.y) - (B.y - A.y) * (p.x - A.x);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // score both orientations, keep the better one
      const t = GATE_PTS.map(GATES[gate]);
      let best = -1, orient = 1;
      for (const o of [1, -1]) {
        let ok = 0;
        GATE_PTS.forEach(function (p, i) {
          const pred = o * side(p) > 0 ? 1 : 0;
          if (pred === t[i]) ok++;
        });
        if (ok > best) { best = ok; orient = o; }
      }
      const degenerate = Math.hypot(B.x - A.x, B.y - A.y) < 1e-6;

      // shade the two half-planes by predicted class (block shading)
      const BLK = 7;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const pred = orient * side({ x: wx, y: wy }) > 0;
          ctx.fillStyle = pred ? POS_SOFT : NEG_SOFT;
          ctx.fillRect(px, py, BLK, BLK);
        }
      }

      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // the line, extended across the panel
      if (!degenerate) {
        const dx = B.x - A.x, dy = B.y - A.y;
        const L = 10;
        ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(sx(A.x - dx * L), sy(A.y - dy * L));
        ctx.lineTo(sx(A.x + dx * L), sy(A.y + dy * L));
        ctx.stroke();
      }

      // convexity overlay
      if (showHulls) {
        const pos = GATE_PTS.filter((p, i) => t[i] === 1);
        const neg = GATE_PTS.filter((p, i) => t[i] === 0);
        ctx.lineWidth = 3;
        if (gate === "XOR") {
          // two crossing segments
          ctx.strokeStyle = POS;
          ctx.beginPath(); ctx.moveTo(sx(pos[0].x), sy(pos[0].y)); ctx.lineTo(sx(pos[1].x), sy(pos[1].y)); ctx.stroke();
          ctx.strokeStyle = NEG;
          ctx.beginPath(); ctx.moveTo(sx(neg[0].x), sy(neg[0].y)); ctx.lineTo(sx(neg[1].x), sy(neg[1].y)); ctx.stroke();
          LR.dot(ctx, sx(0.5), sy(0.5), 8, "#ffffff", C.red);
          ctx.font = "700 12.5px Inter, sans-serif";
          ctx.fillStyle = C.red; ctx.textAlign = "left";
          ctx.fillText("(0.5, 0.5): both segments claim it", sx(0.5) + 14, sy(0.5) - 8);
        } else {
          // AND: negative hull is a triangle, positive hull a single point; disjoint
          ctx.fillStyle = "rgba(150,150,150,0.16)";
          ctx.strokeStyle = NEG;
          ctx.beginPath();
          ctx.moveTo(sx(0), sy(0)); ctx.lineTo(sx(0), sy(1)); ctx.lineTo(sx(1), sy(0));
          ctx.closePath(); ctx.fill(); ctx.stroke();
          ctx.strokeStyle = POS;
          ctx.beginPath(); ctx.arc(sx(1), sy(1), 12, 0, Math.PI * 2); ctx.stroke();
          ctx.font = "700 12.5px Inter, sans-serif";
          ctx.fillStyle = C.muted; ctx.textAlign = "left";
          ctx.fillText("the two hulls never touch: a line fits between", sx(0.15), sy(1.35));
        }
      }
      ctx.restore();

      // handles
      [A, B].forEach(function (h) {
        LR.dot(ctx, sx(h.x), sy(h.y), 8, "#ffffff", C.orange);
        LR.dot(ctx, sx(h.x), sy(h.y), 3, C.orange);
      });

      // the four points
      GATE_PTS.forEach(function (p, i) {
        LR.dot(ctx, sx(p.x), sy(p.y), 9, t[i] ? POS : NEG, "#ffffff");
        ctx.font = "600 11.5px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("(" + p.x + "," + p.y + ")", sx(p.x), sy(p.y) + 24);
      });

      ro.set("gate", gate + "  (t = 1 on " + (gate === "AND" ? "(1,1)" : "(0,1), (1,0)") + ")");
      ro.set("score", best + " / 4", best === 4 ? C.green : best === 3 ? C.amber : C.red);
      if (best === 4) {
        ro.set("verdict", "separated ✓", C.green);
        if (!solvedShown) {
          solvedShown = true;
          msg.show("Separated. AND's feasible weight region is nonempty, and you just found a member of it. Now switch to XOR and try again.", "good");
        }
      } else {
        ro.set("verdict", gate === "XOR" && best === 3 ? "3/4 is the ceiling for any line" : "not separated", gate === "XOR" ? C.red : C.muted);
        if (gate === "XOR" && solvedShown) msg.hide();
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — the feature-map lift for XOR
     ════════════════════════════════════════════════════════════ */
  LR.figs.lift = function (mount) {
    LR.header(
      mount,
      "Add the feature a line could not see",
      "Left: the four XOR points rise to height z = x₁x₂. Only (1,1) moves, and that is enough for a plane. Right: the same plane, dropped back into the input plane, is a curve."
    );

    let t = 0, yaw = 0.6, showPlane = false;

    const bar = LR.controls(mount);
    LR.slider(bar, "lift: z = t · x₁x₂", 0, 1, 0.01, 0, function (v) { t = v; drawAll(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar, "view angle", 0.15, 1.3, 0.01, yaw, function (v) { yaw = v; drawAll(); }, (v) => LR.fmt((v * 180) / Math.PI, 0) + "°");
    const planeBtn = LR.button(bar, "Show the separating plane", function () {
      showPlane = !showPlane;
      planeBtn.classList.toggle("on", showPlane);
      planeBtn.textContent = showPlane ? "Hide the separating plane" : "Show the separating plane";
      drawAll();
    });

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Feature space (x₁, x₂, x₁x₂)"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Input space: the boundary, dropped back"));
    pane.appendChild(lb); pane.appendChild(rb);
    const L3 = LR.canvas(lb, 400, 340, { aria: "Three-dimensional view of the four XOR points lifting to height x1 times x2 with a separating plane" });
    const R2 = LR.canvas(rb, 400, 340, { aria: "Two-dimensional input plane shaded by the decision rule induced by the lifted plane" });

    const ro = LR.readout(mount, [
      { k: "phi", label: "φ(1,1) right now" },
      { k: "score", label: "plane classifies correctly" },
    ]);
    const msg = LR.msg(mount);

    // decision score in feature space: x1 + x2 - 2 z - 0.5 (z = lifted third coord)
    function score(p) { return p.x + p.y - 2 * (t * p.x * p.y) - 0.5; }
    const t01 = GATE_PTS.map(GATES.XOR);

    // isometric-ish projection for the left pane
    function proj(x, y, z) {
      const cx = 0.5, cy = 0.5;
      const u = (x - cx) * Math.cos(yaw) + (y - cy) * Math.sin(yaw);
      const v = -(x - cx) * Math.sin(yaw) + (y - cy) * Math.cos(yaw);
      const S = 150;
      return { x: 200 + u * S, y: 215 + v * S * 0.42 - z * S * 0.62 };
    }

    function draw3d() {
      const ctx = L3.ctx;
      ctx.clearRect(0, 0, L3.W, L3.H);

      // base grid at z=0
      ctx.strokeStyle = "#e4e4e4"; ctx.lineWidth = 1;
      for (let g = -0.25; g <= 1.25 + 1e-9; g += 0.25) {
        let p1 = proj(g, -0.25, 0), p2 = proj(g, 1.25, 0);
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        p1 = proj(-0.25, g, 0); p2 = proj(1.25, g, 0);
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      }
      // axes labels
      ctx.font = "600 11px Inter, sans-serif"; ctx.fillStyle = C.muted;
      const ax = proj(1.4, -0.25, 0), ay = proj(-0.25, 1.4, 0), az = proj(-0.25, -0.25, 1.15);
      ctx.fillText("x₁", ax.x, ax.y);
      ctx.fillText("x₂", ay.x, ay.y);
      ctx.fillText("x₁x₂", az.x - 10, az.y);
      // z axis line
      const z0 = proj(-0.25, -0.25, 0), z1 = proj(-0.25, -0.25, 1.05);
      ctx.strokeStyle = "#cfcfcf";
      ctx.beginPath(); ctx.moveTo(z0.x, z0.y); ctx.lineTo(z1.x, z1.y); ctx.stroke();

      // the separating plane z = (x1 + x2 - 0.5) / 2, drawn over the unit square
      if (showPlane) {
        const corners = [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1]]
          .map(([x, y]) => proj(x, y, (x + y - 0.5) / 2));
        ctx.beginPath();
        corners.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
        ctx.closePath();
        ctx.fillStyle = "rgba(26,26,26,0.13)";
        ctx.strokeStyle = C.orange; ctx.lineWidth = 1.6;
        ctx.fill(); ctx.stroke();
        ctx.font = "600 11px Inter, sans-serif";
        ctx.fillStyle = C.orange;
        const lbl = proj(1.1, 1.1, (2.2 - 0.5) / 2);
        ctx.fillText("x₁ + x₂ − 2z = 0.5", lbl.x - 105, lbl.y - 6);
      }

      // stems + points at lifted height
      GATE_PTS.forEach(function (p, i) {
        const z = t * p.x * p.y;
        const base = proj(p.x, p.y, 0), top = proj(p.x, p.y, z);
        if (z > 1e-3) {
          ctx.strokeStyle = "#bbbbbb"; ctx.lineWidth = 1.4;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(base.x, base.y); ctx.lineTo(top.x, top.y); ctx.stroke();
          ctx.setLineDash([]);
        }
        const ok = (score(p) > 0 ? 1 : 0) === t01[i];
        LR.dot(ctx, top.x, top.y, 8, t01[i] ? POS : NEG, showPlane ? (ok ? "#ffffff" : C.red) : "#ffffff");
        ctx.font = "600 10.5px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("(" + p.x + "," + p.y + ")", top.x, top.y - 13);
      });
    }

    function draw2d() {
      const ctx = R2.ctx;
      ctx.clearRect(0, 0, R2.W, R2.H);
      const P = { x0: 46, y0: 14, w: R2.W - 62, h: R2.H - 56, xmin: -0.6, xmax: 1.6, ymin: -0.6, ymax: 1.6, xlabel: "x₁", ylabel: "x₂" };

      // shade by the decision rule dropped back to input space (at current lift)
      const BLK = 6;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const s = wx + wy - 2 * (t * wx * wy) - 0.5;
          ctx.fillStyle = s > 0 ? POS_SOFT : NEG_SOFT;
          ctx.fillRect(px, py, BLK, BLK);
        }
      }
      const SC = LR.plot(ctx, P);
      GATE_PTS.forEach(function (p, i) {
        LR.dot(ctx, SC.sx(p.x), SC.sy(p.y), 7, t01[i] ? POS : NEG, "#ffffff");
      });
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "left";
      ctx.fillText(t < 0.05 ? "t = 0: plain line, XOR fails" : "boundary: x₁ + x₂ − 2t·x₁x₂ = 0.5", P.x0 + 8, P.y0 + 16);
    }

    function drawAll() {
      draw3d(); draw2d();
      const z11 = t * 1 * 1;
      ro.set("phi", "(1, 1, " + LR.fmtF(z11, 2) + ")");
      let ok = 0;
      GATE_PTS.forEach(function (p, i) { if ((score(p) > 0 ? 1 : 0) === t01[i]) ok++; });
      ro.set("score", ok + " / 4", ok === 4 ? C.green : C.red);
      if (ok === 4 && t > 0.99) {
        msg.show("Fully lifted: (1,1) climbed to height 1, crossed the plane, and all four points classify correctly. A plane in feature space, a curve in input space.", "good");
      } else if (t < 0.05) {
        msg.show("No lift yet: the rule x₁ + x₂ − 0.5 > 0 is a plain line and gets (1,1) wrong. Slide the lift.", "info");
      } else {
        msg.hide();
      }
    }
    drawAll();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — single neuron playground
     ════════════════════════════════════════════════════════════ */
  LR.figs.neuron = function (mount) {
    LR.header(
      mount,
      "One unit, one boundary",
      "The plane is shaded by σ(w₁x₁ + w₂x₂ + b), computed at every block. Drag the probe to see the arithmetic; switch the activation to change the squashing, never the line."
    );

    let w1 = 1.5, w2 = 1.5, b = -1.5, actName = "sigmoid";
    const probe = { x: 0.8, y: 0.8 };

    const bar = LR.controls(mount);
    LR.slider(bar, "w₁", -4, 4, 0.1, w1, function (v) { w1 = v; draw(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "w₂", -4, 4, 0.1, w2, function (v) { w2 = v; draw(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "b", -4, 4, 0.1, b, function (v) { b = v; draw(); }, (v) => LR.fmtF(v, 1));
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const abtns = {};
    [["sigmoid", "sigmoid"], ["tanh", "tanh"], ["relu", "ReLU"]].forEach(function (a) {
      abtns[a[0]] = LR.button(group, a[1], function () {
        actName = a[0];
        for (const k in abtns) abtns[k].classList.toggle("on", k === actName);
        draw();
      }, "small" + (a[0] === "sigmoid" ? " on" : ""));
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 440, {
      aria: "Plane shaded by a single neuron's output with weight and bias sliders, an activation toggle, and a draggable probe point",
    });
    const ro = LR.readout(mount, [
      { k: "z", label: "pre-activation z = w·x + b" },
      { k: "a", label: "output σ(z)" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -2, xmax: 2, ymin: -2, ymax: 2, xlabel: "x₁", ylabel: "x₂" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => SC && (p.x - SC.sx(probe.x)) ** 2 + (p.y - SC.sy(probe.y)) ** 2 < 1600,
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        probe.x = Math.max(P.xmin + 0.05, Math.min(P.xmax - 0.05, d.x));
        probe.y = Math.max(P.ymin + 0.05, Math.min(P.ymax - 0.05, d.y));
        draw();
      },
    });

    // map an activation value to a fill colour (purple low → white → orange high)
    function heat(v01) {
      if (v01 >= 0.5) {
        const a = (v01 - 0.5) * 2;
        return "rgba(26,26,26," + (0.04 + 0.42 * a).toFixed(3) + ")";
      }
      const a = (0.5 - v01) * 2;
      return "rgba(150,150,150," + (0.04 + 0.34 * a).toFixed(3) + ")";
    }
    function normalize(a) {
      if (actName === "sigmoid") return a;
      if (actName === "tanh") return (a + 1) / 2;
      return Math.min(1, 0.5 + a / 4); // relu: 0 maps to mid, grows toward orange
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const act = NN[actName];

      const BLK = 7;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          ctx.fillStyle = heat(normalize(act(w1 * wx + w2 * wy + b)));
          ctx.fillRect(px, py, BLK, BLK);
        }
      }

      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // the boundary z = 0 (for ReLU: the hinge where the flat region ends)
      if (Math.abs(w1) > 1e-9 || Math.abs(w2) > 1e-9) {
        ctx.strokeStyle = C.orange; ctx.lineWidth = 2.4;
        ctx.beginPath();
        if (Math.abs(w2) > Math.abs(w1)) {
          for (let i = 0; i <= 60; i++) {
            const x = P.xmin + ((P.xmax - P.xmin) * i) / 60;
            const y = -(w1 * x + b) / w2;
            i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
          }
        } else {
          for (let i = 0; i <= 60; i++) {
            const y = P.ymin + ((P.ymax - P.ymin) * i) / 60;
            const x = -(w2 * y + b) / w1;
            i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
          }
        }
        ctx.stroke();

        // weight vector arrow from a point on the line
        const n = Math.hypot(w1, w2);
        const ox = -b * w1 / (n * n), oy = -b * w2 / (n * n);
        LR.arrow(ctx, sx(ox), sy(oy), sx(ox + (w1 / n) * 0.7), sy(oy + (w2 / n) * 0.7), C.text, 2);
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillStyle = C.text; ctx.textAlign = "left";
        ctx.fillText("w", sx(ox + (w1 / n) * 0.85), sy(oy + (w2 / n) * 0.85));
      }
      ctx.restore();

      // probe
      LR.dot(ctx, sx(probe.x), sy(probe.y), 8, "#ffffff", C.orange);
      LR.dot(ctx, sx(probe.x), sy(probe.y), 3, C.orange);

      const z = w1 * probe.x + w2 * probe.y + b;
      const a = act(z);
      ro.set("z", LR.fmtF(w1, 1) + "·" + LR.fmtF(probe.x, 2) + " + " + LR.fmtF(w2, 1) + "·" + LR.fmtF(probe.y, 2) + " + " + LR.fmtF(b, 1) + " = " + LR.fmtF(z, 3));
      ro.set("a", actName + "(" + LR.fmtF(z, 3) + ") = " + LR.fmtF(a, 3), C.orange);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.2 — a layer of neurons carves the plane
     ════════════════════════════════════════════════════════════ */
  LR.figs.layer = function (mount) {
    LR.header(
      mount,
      "Three units, three boundaries, many regions",
      "Each unit draws its own line; darker shading means more units are active. The region count is measured by sampling the real on/off patterns."
    );

    // units parameterized by boundary angle and offset; unit 3 is fixed
    const units = [
      { th: 0.35, b: -0.2, color: C.green, name: "unit 1" },
      { th: 1.75, b: 0.3, color: C.purple, name: "unit 2" },
      { th: 4.1, b: 0.5, color: C.amber, name: "unit 3 (fixed)" },
    ];

    const bar = LR.controls(mount);
    LR.slider(bar, "unit 1 angle", 0, 6.28, 0.02, units[0].th, function (v) { units[0].th = v; draw(); }, (v) => LR.fmt((v * 180) / Math.PI, 0) + "°");
    LR.slider(bar, "unit 1 offset", -1.5, 1.5, 0.05, units[0].b, function (v) { units[0].b = v; draw(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar, "unit 2 angle", 0, 6.28, 0.02, units[1].th, function (v) { units[1].th = v; draw(); }, (v) => LR.fmt((v * 180) / Math.PI, 0) + "°");
    LR.slider(bar, "unit 2 offset", -1.5, 1.5, 0.05, units[1].b, function (v) { units[1].b = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 440, {
      aria: "Plane carved by three neuron boundaries with sliders steering two of them and a draggable probe reading the activation pattern",
    });
    const ro = LR.readout(mount, [
      { k: "pat", label: "probe pattern (z₁, z₂, z₃ > 0)" },
      { k: "h", label: "probe h = σ(Wx + b)" },
      { k: "regions", label: "distinct regions carved" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: -2, xmax: 2, ymin: -2, ymax: 2, xlabel: "x₁", ylabel: "x₂" };
    let SC = null;
    const probe = { x: 0.6, y: 0.9 };

    LR.drag(cv, W, H, {
      hit: (p) => SC && (p.x - SC.sx(probe.x)) ** 2 + (p.y - SC.sy(probe.y)) ** 2 < 1600,
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        probe.x = Math.max(P.xmin + 0.05, Math.min(P.xmax - 0.05, d.x));
        probe.y = Math.max(P.ymin + 0.05, Math.min(P.ymax - 0.05, d.y));
        draw();
      },
    });

    function zs(x, y) {
      return units.map((u) => Math.cos(u.th) * x + Math.sin(u.th) * y + u.b);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // shading by number of active units + region count by pattern sampling
      const BLK = 7;
      const seen = new Set();
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const z = zs(wx, wy);
          const bits = z.map((v) => (v > 0 ? 1 : 0));
          seen.add(bits.join(""));
          const on = bits[0] + bits[1] + bits[2];
          ctx.fillStyle = "rgba(26,26,26," + (0.03 + 0.13 * on).toFixed(3) + ")";
          ctx.fillRect(px, py, BLK, BLK);
        }
      }

      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // each unit's boundary line + normal arrow
      units.forEach(function (u) {
        const c = Math.cos(u.th), s = Math.sin(u.th);
        ctx.strokeStyle = u.color; ctx.lineWidth = 2.2;
        ctx.beginPath();
        // param the line c*x + s*y + b = 0 by its direction (-s, c)
        const ox = -u.b * c, oy = -u.b * s;
        ctx.moveTo(sx(ox - -s * 8), sy(oy - c * 8));
        ctx.lineTo(sx(ox + -s * 8), sy(oy + c * 8));
        ctx.stroke();
        LR.arrow(ctx, sx(ox), sy(oy), sx(ox + c * 0.45), sy(oy + s * 0.45), u.color, 1.8);
      });
      ctx.restore();

      // legend
      ctx.font = "600 11.5px Inter, sans-serif"; ctx.textAlign = "left";
      units.forEach(function (u, i) {
        ctx.fillStyle = u.color;
        ctx.fillText("— " + u.name, P.x0 + 10, P.y0 + 16 + i * 16);
      });

      // probe
      LR.dot(ctx, sx(probe.x), sy(probe.y), 8, "#ffffff", C.orange);
      LR.dot(ctx, sx(probe.x), sy(probe.y), 3, C.orange);

      const z = zs(probe.x, probe.y);
      const bits = z.map((v) => (v > 0 ? 1 : 0));
      ro.set("pat", "(" + bits.join(", ") + ")");
      ro.set("h", "(" + z.map((v) => LR.fmtF(NN.sigmoid(v), 2)).join(", ") + ")", C.orange);
      ro.set("regions", String(seen.size) + " of at most 7", C.text);
    }
    draw();
  };
})();
