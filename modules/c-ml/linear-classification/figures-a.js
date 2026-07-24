/* ══════════════════════════════════════════════════════════════
   figures-a.js — Linear Classification lesson figures, sections 0–4
   Fig 0.1 pipeline · Fig 1.1 regthresh · Fig 2.1 boundary
   Fig 3.1 costdemo · Fig 4.1 cmatrix · Fig 4.2 prcurve
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  const POS = C.green;                       // class +1
  const NEG = C.purple;                      // class -1
  const POS_SOFT = "rgba(102,102,102,0.13)";
  const NEG_SOFT = "rgba(150,150,150,0.11)";

  /* ── shared linear-classifier core (also used by figures-b.js) ── */
  const LC = (LR.lc = {
    // score of a 2-D point under w = [w0, w1, w2]
    score2: (w, p) => w[0] + w[1] * p.x + w[2] * p.y,
    sign: (s) => (s >= 0 ? 1 : -1),

    // confusion counts for scored items [{s, t}] at threshold thr
    // (predict +1 when s >= thr)
    confusion: function (items, thr) {
      let tp = 0, fp = 0, tn = 0, fn = 0;
      for (const it of items) {
        const pos = it.s >= thr;
        if (it.t === 1) { pos ? tp++ : fn++; }
        else { pos ? fp++ : tn++; }
      }
      return { tp, fp, tn, fn };
    },
    metrics: function (cm) {
      const { tp, fp, tn, fn } = cm;
      const P = tp + fp > 0 ? tp / (tp + fp) : NaN;
      const R = tp + fn > 0 ? tp / (tp + fn) : NaN;
      const F1 = isFinite(P) && isFinite(R) && P + R > 0 ? (2 * P * R) / (P + R) : NaN;
      const acc = (tp + tn) / (tp + fp + tn + fn);
      return { P, R, F1, acc };
    },

    /* the shared screening dataset: 60 patients, 15 diseased (t=+1),
       45 healthy (t=-1), each with a deterministic 1-D risk score.
       Used by Figs 3.1, 4.1, 4.2. */
    screening: (function () {
      const rand = LR.rng(11);
      const items = [];
      for (let i = 0; i < 45; i++) items.push({ s: -1.0 + LR.gauss(rand) * 0.75, t: -1, jit: rand() });
      for (let i = 0; i < 15; i++) items.push({ s: 0.9 + LR.gauss(rand) * 0.85, t: 1, jit: rand() });
      return items;
    })(),
    /* a deliberately weaker scorer on the SAME patients: the class
       means sit closer and the noise is larger, so every threshold
       buys less separation. */
    screeningWeak: (function () {
      const rand = LR.rng(12);
      const items = [];
      for (let i = 0; i < 45; i++) items.push({ s: -0.35 + LR.gauss(rand) * 1.05, t: -1 });
      for (let i = 0; i < 15; i++) items.push({ s: 0.4 + LR.gauss(rand) * 1.05, t: 1 });
      return items;
    })(),

    // shared threshold linking Fig 4.1 and Fig 4.2
    thr: 0.0,
    _subs: [],
    onThr: function (fn) { LC._subs.push(fn); },
    setThr: function (v) {
      LC.thr = v;
      LC._subs.forEach((fn) => fn(v));
    },

    // sweep the threshold over every gap between sorted scores;
    // returns [{thr, P, R}] ordered from strictest to loosest
    prSweep: function (items) {
      const ss = items.map((it) => it.s).sort((a, b) => b - a);
      const thrs = [ss[0] + 0.5];
      for (let i = 0; i < ss.length - 1; i++) thrs.push((ss[i] + ss[i + 1]) / 2);
      thrs.push(ss[ss.length - 1] - 0.5);
      return thrs.map(function (thr) {
        const m = LC.metrics(LC.confusion(items, thr));
        return { thr, P: m.P, R: m.R };
      });
    },

    // shade the two half-spaces of w0 + w1 x + w2 y = 0 inside a plot
    shadeHalfspaces: function (ctx, P, sx, sy, w) {
      const BLK = 7;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const s = w[0] + w[1] * wx + w[2] * wy;
          ctx.fillStyle = s >= 0 ? POS_SOFT : NEG_SOFT;
          ctx.fillRect(px, py, Math.min(BLK, P.x0 + P.w - px), Math.min(BLK, P.y0 + P.h - py));
        }
      }
    },

    // draw the boundary line w0 + w1 x + w2 y = 0 clipped to the plot
    drawBoundary: function (ctx, P, sx, sy, w, color, width, dash) {
      if (Math.abs(w[1]) < 1e-9 && Math.abs(w[2]) < 1e-9) return;
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      ctx.strokeStyle = color; ctx.lineWidth = width || 2.4;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      if (Math.abs(w[2]) > Math.abs(w[1])) {
        // y as a function of x
        const y = (x) => -(w[0] + w[1] * x) / w[2];
        ctx.moveTo(sx(P.xmin), sy(y(P.xmin)));
        ctx.lineTo(sx(P.xmax), sy(y(P.xmax)));
      } else {
        const x = (y) => -(w[0] + w[2] * y) / w[1];
        ctx.moveTo(sx(x(P.ymin)), sy(P.ymin));
        ctx.lineTo(sx(x(P.ymax)), sy(P.ymax));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    },
  });

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — the score → threshold → label pipeline
     ════════════════════════════════════════════════════════════ */
  LR.figs.pipeline = function (mount) {
    LR.header(
      mount,
      "Score → threshold → label",
      "One applicant, one feature. Drag the point: the linear score is computed for real, and the label flips the instant the score crosses the threshold."
    );

    // the worked example's weights: s(x) = -1 + 2x, boundary at x = 0.5
    const w0 = -1, w1 = 2;
    let px1 = 1.6; // the draggable applicant's feature value

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Draggable point on a feature axis with its linear score, the zero threshold, and the resulting class label",
    });
    const ro = LR.readout(mount, [
      { k: "score", label: "score s(x) = w₀ + w₁x₁" },
      { k: "test", label: "threshold test s ≥ 0 ?" },
      { k: "label", label: "label y(x)" },
    ]);

    const P = { x0: 62, y0: 24, w: W - 90, h: H - 80, xmin: -1, xmax: 3, ymin: -3.5, ymax: 5.5, xlabel: "x₁ (the feature)", ylabel: "score s(x₁)" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!SC) return false;
        const s = w0 + w1 * px1;
        return (p.x - SC.sx(px1)) ** 2 + (p.y - SC.sy(s)) ** 2 < 2600;
      },
      move: function (p) {
        px1 = Math.max(P.xmin + 0.08, Math.min(P.xmax - 0.08, SC.inv(p.x, p.y).x));
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const xb = -w0 / w1; // 0.5

      // region shading: left of the boundary predicts -1, right predicts +1
      const preW = ((xb - P.xmin) / (P.xmax - P.xmin)) * P.w;
      ctx.fillStyle = NEG_SOFT;
      ctx.fillRect(P.x0, P.y0, preW, P.h);
      ctx.fillStyle = POS_SOFT;
      ctx.fillRect(P.x0 + preW, P.y0, P.w - preW, P.h);

      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;
      const s = w0 + w1 * px1;
      const lab = s >= 0 ? 1 : -1;

      // threshold line s = 0
      ctx.strokeStyle = C.text; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(sx(P.xmin), sy(0)); ctx.lineTo(sx(P.xmax), sy(0)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "left";
      ctx.fillText("threshold: s = 0", sx(P.xmin) + 8, sy(0) - 7);

      // decision boundary marker on the feature axis
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.6; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx(xb), P.y0); ctx.lineTo(sx(xb), P.y0 + P.h); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.orange; ctx.textAlign = "center";
      ctx.fillText("boundary x₁ = " + LR.fmt(xb, 2), sx(xb), P.y0 + P.h - 8);

      // the score line s = w0 + w1 x
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(sx(P.xmin), sy(w0 + w1 * P.xmin));
      ctx.lineTo(sx(P.xmax), sy(w0 + w1 * P.xmax));
      ctx.stroke();

      // projection guides: down to the axis, across to the score
      ctx.strokeStyle = "rgba(0,0,0,0.3)"; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx(px1), sy(s)); ctx.lineTo(sx(px1), P.y0 + P.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(px1), sy(s)); ctx.lineTo(P.x0, sy(s)); ctx.stroke();
      ctx.setLineDash([]);

      // the applicant
      LR.dot(ctx, sx(px1), sy(s), 10, lab === 1 ? POS : NEG, C.orange);
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillStyle = lab === 1 ? POS : NEG;
      ctx.textAlign = "left";
      ctx.fillText(lab === 1 ? "y = +1 (approve)" : "y = −1 (reject)", sx(px1) + 14, sy(s) - 10);

      // corner legend with the fixed weights
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("fixed weights: w₀ = −1, w₁ = 2", P.x0 + 8, P.y0 + 16);

      ro.set("score", "−1 + 2 × " + LR.fmtF(px1, 2) + " = " + LR.fmtF(s, 2), C.orange);
      ro.set("test", s >= 0 ? LR.fmtF(s, 2) + " ≥ 0 ✓" : LR.fmtF(s, 2) + " < 0 ✗");
      ro.set("label", lab === 1 ? "+1 (approve)" : "−1 (reject)", lab === 1 ? POS : NEG);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — a regression line, thresholded into a classifier
     ════════════════════════════════════════════════════════════ */
  LR.figs.regthresh = function (mount) {
    LR.header(
      mount,
      "Least squares, then chop",
      "Targets are −1 and +1. The line is the real least-squares fit; where it crosses zero becomes the boundary. Then add far-away positives and watch the fit betray it."
    );

    // base data: two 1-D clusters (deterministic)
    const rand = LR.rng(33);
    const base = [];
    for (let i = 0; i < 9; i++) base.push({ x: 0.6 + i * 0.42 + rand() * 0.3, t: -1 });
    for (let i = 0; i < 9; i++) base.push({ x: 5.6 + i * 0.42 + rand() * 0.3, t: 1 });
    // the "perfectly valid but distant" positives
    const distant = [{ x: 17.5, t: 1 }, { x: 18.6, t: 1 }, { x: 19.8, t: 1 }, { x: 21.0, t: 1 }];

    let withDistant = false;

    const bar = LR.controls(mount);
    const btn = LR.button(bar, "Add 4 distant positives ▸", function () {
      withDistant = !withDistant;
      btn.textContent = withDistant ? "Remove the distant positives ⟲" : "Add 4 distant positives ▸";
      draw();
    }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "One-dimensional two-class data with a least-squares line, its zero crossing as decision boundary, and a button that adds distant positive points",
    });
    const ro = LR.readout(mount, [
      { k: "fit", label: "least-squares fit" },
      { k: "xb", label: "boundary x₁ = −w₀/w₁" },
      { k: "err", label: "misclassified" },
    ]);
    const msg = LR.msg(mount);

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const pts = withDistant ? base.concat(distant) : base;
      const fit = LR.fit1d(pts);              // real least squares on (x, t)
      const xb = -fit.w0 / fit.w1;

      const xmax = withDistant ? 22 : 11;
      const P = { x0: 62, y0: 24, w: W - 90, h: H - 80, xmin: 0, xmax, ymin: -2.2, ymax: 2.2, xlabel: "x₁ (the feature)", ylabel: "target t and fitted score" };

      // predicted regions from the thresholded fit
      const bx = Math.max(P.xmin, Math.min(P.xmax, xb));
      const preW = ((bx - P.xmin) / (P.xmax - P.xmin)) * P.w;
      const rightPos = fit.w1 > 0;
      ctx.fillStyle = rightPos ? NEG_SOFT : POS_SOFT;
      ctx.fillRect(P.x0, P.y0, preW, P.h);
      ctx.fillStyle = rightPos ? POS_SOFT : NEG_SOFT;
      ctx.fillRect(P.x0 + preW, P.y0, P.w - preW, P.h);

      const { sx, sy } = LR.plot(ctx, P);

      // zero line + boundary
      ctx.strokeStyle = C.text; ctx.lineWidth = 1.4; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(sx(P.xmin), sy(0)); ctx.lineTo(sx(P.xmax), sy(0)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.6; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx(bx), P.y0); ctx.lineTo(sx(bx), P.y0 + P.h); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "center";
      ctx.fillText("boundary " + LR.fmtF(xb, 2), sx(bx), P.y0 + 14);

      // the fitted line, clipped
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.moveTo(sx(P.xmin), sy(fit.w0 + fit.w1 * P.xmin));
      ctx.lineTo(sx(P.xmax), sy(fit.w0 + fit.w1 * P.xmax));
      ctx.stroke();
      ctx.restore();

      // points at their target heights; errors ringed red
      let errs = 0;
      pts.forEach(function (p) {
        const pred = LC.sign(fit.w0 + fit.w1 * p.x);
        const wrong = pred !== p.t;
        if (wrong) errs++;
        LR.dot(ctx, sx(p.x), sy(p.t), wrong ? 7 : 5.5, p.t === 1 ? POS : NEG, wrong ? C.red : "#ffffff");
      });

      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = POS; ctx.fillText("● t = +1", P.x0 + 8, P.y0 + 16);
      ctx.fillStyle = NEG; ctx.fillText("● t = −1", P.x0 + 8, P.y0 + 33);

      ro.set("fit", "s(x) = " + LR.fmtF(fit.w0, 2) + " + " + LR.fmtF(fit.w1, 3) + "·x₁");
      ro.set("xb", LR.fmtF(xb, 2), C.orange);
      ro.set("err", errs + " of " + pts.length + (errs ? "  (red rings)" : ""), errs ? C.red : POS);

      if (withDistant) {
        msg.show(
          "The 4 new points are positives, deep in positive territory, the easiest cases imaginable. But their fitted scores sit far above +1, so squared loss calls them errors and tilts the line down to appease them. The crossing slides right and " + errs + " genuinely borderline " + (errs === 1 ? "point now sits" : "points now sit") + " on the wrong side.",
          "bad"
        );
      } else {
        msg.show("With compact clusters the chop works: the fitted line crosses zero between the classes and every point is classified correctly.", "good");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — the decision boundary in 1-D, 2-D, 3-D (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.boundary = function (mount) {
    LR.header(
      mount,
      "One equation, three shapes",
      "The boundary is always w₀ + wᵀx = 0. In 2-D, drag the arrowhead to rotate w, drag elsewhere to slide the line, or drive the sliders directly."
    );

    let mode = "2d";
    let w = [-0.5, 1.0, 0.8]; // [w0, w1, w2]

    // two 2-D classes (deterministic)
    const rand = LR.rng(47);
    const pts = [];
    for (let i = 0; i < 12; i++) pts.push({ x: 2.0 + LR.gauss(rand) * 0.95, y: 1.8 + LR.gauss(rand) * 0.9, t: 1 });
    for (let i = 0; i < 12; i++) pts.push({ x: -1.9 + LR.gauss(rand) * 0.95, y: -1.5 + LR.gauss(rand) * 0.9, t: -1 });
    // 1-D projection data for the 1-D tab (feature x₁ only)
    const pts1d = [
      { x: -1.4, t: -1 }, { x: -0.7, t: -1 }, { x: -0.2, t: -1 }, { x: 0.2, t: -1 },
      { x: 0.9, t: 1 }, { x: 1.4, t: 1 }, { x: 2.1, t: 1 }, { x: 2.7, t: 1 },
    ];

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const tabs = {};
    [["1d", "1-D: a point"], ["2d", "2-D: a line"], ["3d", "3-D: a plane"]].forEach(function (m) {
      tabs[m[0]] = LR.button(group, m[1], function () {
        mode = m[0];
        for (const k in tabs) tabs[k].classList.toggle("on", k === mode);
        draw();
      }, "small" + (m[0] === "2d" ? " on" : ""));
    });

    const bar2 = LR.controls(mount);
    const s0 = LR.slider(bar2, "w₀", -4, 4, 0.1, w[0], function (v) { w[0] = v; draw(); }, (v) => LR.fmtF(v, 1));
    const s1 = LR.slider(bar2, "w₁", -2, 2, 0.1, w[1], function (v) { w[1] = v; draw(); }, (v) => LR.fmtF(v, 1));
    const s2 = LR.slider(bar2, "w₂", -2, 2, 0.1, w[2], function (v) { w[2] = v; draw(); }, (v) => LR.fmtF(v, 1));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Two-class scatter with an adjustable linear decision boundary, its normal vector, and shaded half-spaces; tabs show the 1-D and 3-D versions",
    });
    const ro = LR.readout(mount, [
      { k: "eq", label: "boundary equation" },
      { k: "side", label: "positive half-space" },
      { k: "acc", label: "correctly classified" },
    ]);

    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: -5, xmax: 5, ymin: -4.2, ymax: 4.2, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragKind = null;

    // point on the boundary closest to the origin, and the arrowhead
    function anchor() {
      const n2 = w[1] * w[1] + w[2] * w[2];
      if (n2 < 1e-9) return null;
      const k = -w[0] / n2;
      const ax = k * w[1], ay = k * w[2];
      const nl = Math.sqrt(n2);
      return { ax, ay, hx: ax + (w[1] / nl) * 1.6, hy: ay + (w[2] / nl) * 1.6, nl };
    }

    LR.drag(cv, W, H, {
      hit: function (p) {
        if (mode !== "2d" || !SC) return false;
        const a = anchor();
        if (a && (p.x - SC.sx(a.hx)) ** 2 + (p.y - SC.sy(a.hy)) ** 2 < 700) { dragKind = "normal"; return true; }
        if (p.x > P.x0 && p.x < P.x0 + P.w && p.y > P.y0 && p.y < P.y0 + P.h) { dragKind = "line"; return true; }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        if (dragKind === "normal") {
          const a = anchor();
          if (!a) return;
          const dx = d.x - a.ax, dy = d.y - a.ay;
          const len = Math.hypot(dx, dy);
          if (len < 0.15) return;
          // keep |w| fixed, rotate toward the pointer; keep the same anchor point
          const nl = a.nl;
          const nw1 = (dx / len) * nl, nw2 = (dy / len) * nl;
          w = [-(nw1 * a.ax + nw2 * a.ay), nw1, nw2];
        } else {
          // slide the boundary so it passes through the pointer
          w[0] = -(w[1] * d.x + w[2] * d.y);
        }
        w[0] = Math.max(-4, Math.min(4, w[0]));
        s0.set(w[0]); s1.set(w[1]); s2.set(w[2]);
        draw();
      },
      up: function () { dragKind = null; },
    });

    function draw2d() {
      LC.shadeHalfspaces(ctx, P, null, null, w);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      LC.drawBoundary(ctx, P, sx, sy, w, C.orange, 2.6);

      // the normal vector
      const a = anchor();
      if (a) {
        LR.arrow(ctx, sx(a.ax), sy(a.ay), sx(a.hx), sy(a.hy), C.orange, 2.4);
        LR.dot(ctx, sx(a.hx), sy(a.hy), 6, "#ffffff", C.orange);
        ctx.font = "700 12.5px Inter, sans-serif";
        ctx.fillStyle = C.orange; ctx.textAlign = "left";
        ctx.fillText("w", sx(a.hx) + 10, sy(a.hy) - 6);
      }

      // points, errors ringed red
      let ok = 0;
      pts.forEach(function (p) {
        const right = LC.sign(LC.score2(w, p)) === p.t;
        if (right) ok++;
        LR.dot(ctx, sx(p.x), sy(p.y), right ? 5.5 : 7, p.t === 1 ? POS : NEG, right ? "#ffffff" : C.red);
      });

      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = POS; ctx.fillText("● t = +1  (green half-space should hold these)", P.x0 + 8, P.y0 + 16);
      ctx.fillStyle = NEG; ctx.fillText("● t = −1", P.x0 + 8, P.y0 + 33);

      ro.set("eq", LR.fmtF(w[0], 1) + " + " + LR.fmtF(w[1], 1) + "·x₁ + " + LR.fmtF(w[2], 1) + "·x₂ = 0");
      ro.set("side", "the side w points into", POS);
      ro.set("acc", ok + " / " + pts.length, ok === pts.length ? POS : C.red);
    }

    function draw1d() {
      // a number line with the boundary point at x = -w0/w1
      const axisY = P.y0 + P.h / 2;
      const sx = (x) => P.x0 + ((x - P.xmin) / (P.xmax - P.xmin)) * P.w;
      const hasB = Math.abs(w[1]) > 1e-9;
      const xb = hasB ? -w[0] / w[1] : null;

      // half-line shading
      if (hasB) {
        const bx = Math.max(P.xmin, Math.min(P.xmax, xb));
        const preW = sx(bx) - P.x0;
        const rightPos = w[1] > 0;
        ctx.fillStyle = rightPos ? NEG_SOFT : POS_SOFT;
        ctx.fillRect(P.x0, axisY - 46, preW, 92);
        ctx.fillStyle = rightPos ? POS_SOFT : NEG_SOFT;
        ctx.fillRect(P.x0 + preW, axisY - 46, P.w - preW, 92);
      }

      // the axis with ticks
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(P.x0, axisY); ctx.lineTo(P.x0 + P.w, axisY); ctx.stroke();
      ctx.font = "11px Inter, sans-serif"; ctx.fillStyle = C.faint; ctx.textAlign = "center";
      for (let t = Math.ceil(P.xmin); t <= P.xmax; t++) {
        ctx.beginPath(); ctx.moveTo(sx(t), axisY - 4); ctx.lineTo(sx(t), axisY + 4);
        ctx.strokeStyle = C.axis; ctx.lineWidth = 1; ctx.stroke();
        ctx.fillText(String(t), sx(t), axisY + 20);
      }
      ctx.font = "600 12px Inter, sans-serif"; ctx.fillStyle = C.text;
      ctx.fillText("x₁", P.x0 + P.w / 2, axisY + 42);

      // data points on the line, classified by the current w0, w1
      pts1d.forEach(function (p) {
        const right = LC.sign(w[0] + w[1] * p.x) === p.t;
        LR.dot(ctx, sx(p.x), axisY, right ? 6 : 7.5, p.t === 1 ? POS : NEG, right ? "#ffffff" : C.red);
      });

      if (hasB && xb >= P.xmin && xb <= P.xmax) {
        ctx.strokeStyle = C.orange; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(sx(xb), axisY - 52); ctx.lineTo(sx(xb), axisY + 52); ctx.stroke();
        LR.dot(ctx, sx(xb), axisY, 5.5, C.orange, "#ffffff");
        ctx.font = "700 12.5px Inter, sans-serif"; ctx.fillStyle = C.orange; ctx.textAlign = "center";
        ctx.fillText("boundary: a point, x₁ = −w₀/w₁ = " + LR.fmtF(xb, 2), sx(xb), axisY - 62);
      }

      ctx.font = "600 12px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.muted;
      ctx.fillText("1-D uses only w₀ and w₁ (set w₀ = −1, w₁ = 2 to reproduce the worked example: x₁ = 0.5)", P.x0, P.y0 + 12);

      let ok = 0;
      pts1d.forEach((p) => { if (LC.sign(w[0] + w[1] * p.x) === p.t) ok++; });
      ro.set("eq", LR.fmtF(w[0], 1) + " + " + LR.fmtF(w[1], 1) + "·x₁ = 0");
      ro.set("side", !hasB ? "undefined (w₁ = 0)" : (w[1] > 0 ? "right of the point" : "left of the point"), POS);
      ro.set("acc", ok + " / " + pts1d.length, ok === pts1d.length ? POS : C.red);
    }

    function draw3d() {
      // static oblique-projection illustration: the boundary as a plane
      const cx = P.x0 + P.w / 2, cy = P.y0 + P.h / 2 + 20;
      const ex = { x: 42, y: 10 }, ey = { x: 22, y: -17 }, ez = { x: 0, y: -44 };
      const proj = (x, y, z) => ({ px: cx + x * ex.x + y * ey.x + z * ez.x, py: cy + x * ex.y + y * ey.y + z * ez.y });

      // axes
      ctx.strokeStyle = C.axis; ctx.lineWidth = 1.3;
      ctx.font = "600 12px Inter, sans-serif"; ctx.fillStyle = C.muted;
      [[3.4, 0, 0, "x₁"], [0, 3.0, 0, "x₂"], [0, 0, 2.4, "x₃"]].forEach(function (ax) {
        const o = proj(0, 0, 0), e = proj(ax[0], ax[1], ax[2]);
        LR.arrow(ctx, o.px, o.py, e.px, e.py, C.axis, 1.3);
        ctx.textAlign = "center";
        ctx.fillText(ax[3], e.px + 12, e.py);
      });

      // the plane z = 0 patch (the boundary), drawn as a parallelogram
      const corners = [proj(-2.6, -2.2, 0), proj(2.6, -2.2, 0), proj(2.6, 2.2, 0), proj(-2.6, 2.2, 0)];
      ctx.beginPath();
      corners.forEach((c, i) => (i === 0 ? ctx.moveTo(c.px, c.py) : ctx.lineTo(c.px, c.py)));
      ctx.closePath();
      ctx.fillStyle = "rgba(26,26,26,0.10)";
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2;
      ctx.fill(); ctx.stroke();

      // normal vector
      const o = proj(0, 0, 0), n = proj(0, 0, 1.6);
      LR.arrow(ctx, o.px, o.py, n.px, n.py, C.orange, 2.4);
      ctx.fillStyle = C.orange; ctx.textAlign = "left";
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillText("w (the normal)", n.px + 10, n.py + 2);

      // a few points above (+1) and below (−1) the plane
      const rand3 = LR.rng(5);
      for (let i = 0; i < 7; i++) {
        const a = proj(-2 + rand3() * 4, -1.6 + rand3() * 3.2, 0.5 + rand3() * 1.4);
        LR.dot(ctx, a.px, a.py, 5.5, POS, "#ffffff");
      }
      for (let i = 0; i < 7; i++) {
        const b = proj(-2 + rand3() * 4, -1.6 + rand3() * 3.2, -0.5 - rand3() * 1.4);
        LR.dot(ctx, b.px, b.py, 5.5, NEG, "#ffffff");
      }

      ctx.font = "600 12.5px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("in 3-D the same equation w₀ + wᵀx = 0 is a plane;", P.x0 + 8, P.y0 + 18);
      ctx.fillText("above it (along w) the score is positive, below it negative", P.x0 + 8, P.y0 + 36);

      ro.set("eq", "w₀ + w₁x₁ + w₂x₂ + w₃x₃ = 0");
      ro.set("side", "the side the normal w points into", POS);
      ro.set("acc", "illustration (drag in the 2-D tab)");
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      if (mode === "2d") draw2d();
      else if (mode === "1d") draw1d();
      else draw3d();
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — asymmetric loss: the cost of a threshold
     ════════════════════════════════════════════════════════════ */
  LR.figs.costdemo = function (mount) {
    LR.header(
      mount,
      "Which mistakes are you willing to make?",
      "60 screened patients with real risk scores. Drag the threshold; the cost curve below is α·FP + β·FN computed at every threshold. Then make misses expensive."
    );

    const items = LC.screening;
    let alpha = 1, beta = 1;
    let tau = 0.0;

    const bar = LR.controls(mount);
    LR.slider(bar, "α (cost of a false positive)", 1, 10, 1, alpha, function (v) { alpha = Math.round(v); draw(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "β (cost of a false negative)", 1, 10, 1, beta, function (v) { beta = Math.round(v); draw(); }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 520, {
      aria: "Patient risk scores with a draggable decision threshold, plus the total asymmetric cost as a function of the threshold with its minimum marked",
    });
    const ro = LR.readout(mount, [
      { k: "fp", label: "false positives (healthy, flagged)" },
      { k: "fn", label: "false negatives (diseased, missed)" },
      { k: "cost", label: "total cost α·FP + β·FN" },
      { k: "best", label: "cost-minimizing threshold" },
    ]);

    const XMIN = -3.4, XMAX = 3.4;
    const TOP = { x0: 62, y0: 26, w: W - 90, h: 170 };
    const BOT = { x0: 62, y0: 250, w: W - 90, h: H - 250 - 56, xmin: XMIN, xmax: XMAX, ymin: 0, ymax: 1, xlabel: "threshold τ (flag when score ≥ τ)", ylabel: "total cost" };

    const sxTop = (v) => TOP.x0 + ((v - XMIN) / (XMAX - XMIN)) * TOP.w;

    function costAt(t) {
      const cm = LC.confusion(items, t);
      return alpha * cm.fp + beta * cm.fn;
    }

    LR.drag(cv, W, H, {
      hit: (p) => p.x > TOP.x0 - 10 && p.x < TOP.x0 + TOP.w + 10,
      down: move, move,
    });
    function move(p) {
      tau = Math.max(XMIN, Math.min(XMAX, XMIN + ((p.x - TOP.x0) / TOP.w) * (XMAX - XMIN)));
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // ── top: the score strip ──
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("each dot is one patient's risk score", TOP.x0, TOP.y0 - 8);

      // region tint
      const tx = sxTop(tau);
      ctx.fillStyle = NEG_SOFT; ctx.fillRect(TOP.x0, TOP.y0, tx - TOP.x0, TOP.h);
      ctx.fillStyle = POS_SOFT; ctx.fillRect(tx, TOP.y0, TOP.x0 + TOP.w - tx, TOP.h);
      ctx.strokeStyle = "#dcdcdc"; ctx.lineWidth = 1;
      ctx.strokeRect(TOP.x0, TOP.y0, TOP.w, TOP.h);

      const cm = LC.confusion(items, tau);

      items.forEach(function (it) {
        const rowY = it.t === 1 ? TOP.y0 + 34 : TOP.y0 + 108;
        const y = rowY + it.jit * 34;
        const flagged = it.s >= tau;
        const wrong = (it.t === 1) !== flagged;
        LR.dot(ctx, sxTop(it.s), y, wrong ? 6 : 4.5, it.t === 1 ? POS : NEG, wrong ? C.red : "#ffffff");
      });
      ctx.font = "600 11.5px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = POS; ctx.fillText("diseased (t = +1)", TOP.x0 + 6, TOP.y0 + 22);
      ctx.fillStyle = NEG; ctx.fillText("healthy (t = −1)", TOP.x0 + 6, TOP.y0 + 96);

      // threshold handle
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(tx, TOP.y0 - 4); ctx.lineTo(tx, TOP.y0 + TOP.h + 4); ctx.stroke();
      ctx.fillStyle = C.orange;
      ctx.beginPath();
      ctx.moveTo(tx - 7, TOP.y0 - 4); ctx.lineTo(tx + 7, TOP.y0 - 4); ctx.lineTo(tx, TOP.y0 + 6);
      ctx.closePath(); ctx.fill();
      ctx.font = "700 12px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("τ = " + LR.fmtF(tau, 2), tx, TOP.y0 + TOP.h + 18);

      // ── bottom: the cost curve ──
      const GRID = 240;
      let best = { t: XMIN, c: Infinity };
      const curve = [];
      for (let i = 0; i <= GRID; i++) {
        const t = XMIN + ((XMAX - XMIN) * i) / GRID;
        const c = costAt(t);
        curve.push({ t, c });
        if (c < best.c) best = { t, c };
      }
      const ymax = Math.max(4, Math.max.apply(null, curve.map((p) => p.c))) * 1.08;
      BOT.ymax = ymax;
      const SC = LR.plot(ctx, BOT);

      ctx.save();
      ctx.beginPath(); ctx.rect(BOT.x0, BOT.y0, BOT.w, BOT.h); ctx.clip();
      ctx.strokeStyle = C.purple; ctx.lineWidth = 2.4;
      ctx.beginPath();
      curve.forEach((p, i) => (i === 0 ? ctx.moveTo(SC.sx(p.t), SC.sy(p.c)) : ctx.lineTo(SC.sx(p.t), SC.sy(p.c))));
      ctx.stroke();
      ctx.restore();

      // minimum marker
      LR.dot(ctx, SC.sx(best.t), SC.sy(best.c), 7, "#ffffff", POS);
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = POS; ctx.textAlign = "center";
      ctx.fillText("best τ ≈ " + LR.fmtF(best.t, 2) + " (cost " + best.c + ")", SC.sx(best.t), SC.sy(best.c) - 12);

      // current threshold marker on the curve
      const cNow = alpha * cm.fp + beta * cm.fn;
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(SC.sx(tau), BOT.y0); ctx.lineTo(SC.sx(tau), BOT.y0 + BOT.h); ctx.stroke();
      ctx.setLineDash([]);
      LR.dot(ctx, SC.sx(tau), SC.sy(cNow), 6, C.orange, "#ffffff");

      ro.set("fp", String(cm.fp), NEG);
      ro.set("fn", String(cm.fn), C.red);
      ro.set("cost", alpha + "·" + cm.fp + " + " + beta + "·" + cm.fn + " = " + cNow, C.orange);
      ro.set("best", "τ ≈ " + LR.fmtF(best.t, 2) + (beta > alpha ? "  (misses cost more → flag earlier)" : beta < alpha ? "  (alarms cost more → flag later)" : "  (equal costs: this is zero/one loss)"), POS);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — confusion matrix and metrics explorer
     ════════════════════════════════════════════════════════════ */
  LR.figs.cmatrix = function (mount) {
    LR.header(
      mount,
      "Four bins, four metrics",
      "The same 60 patients. Drag the threshold: every patient falls into TP, FP, TN, or FN, and the metrics are recomputed from the live counts."
    );

    const items = LC.screening;

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 300, {
      aria: "Patient scores with a draggable classification threshold that re-bins every point into the confusion matrix",
    });

    // confusion matrix as an HTML table
    const wrap = LR.el("div", "wtable-wrap");
    const tbl = LR.el("table", "wtable");
    tbl.innerHTML =
      "<tr><th></th><th>predicted +1 (flag)</th><th>predicted −1 (clear)</th></tr>" +
      "<tr><th>truth +1 (diseased)</th><td id='lc-tp'>–</td><td id='lc-fn'>–</td></tr>" +
      "<tr><th>truth −1 (healthy)</th><td id='lc-fp'>–</td><td id='lc-tn'>–</td></tr>";
    wrap.appendChild(tbl);
    mount.appendChild(wrap);

    const ro = LR.readout(mount, [
      { k: "p", label: "precision TP/(TP+FP)" },
      { k: "r", label: "recall TP/(TP+FN)" },
      { k: "f1", label: "F₁" },
      { k: "acc", label: "accuracy" },
    ]);

    const XMIN = -3.4, XMAX = 3.4;
    const P = { x0: 62, y0: 26, w: W - 90, h: H - 82 };
    const sx = (v) => P.x0 + ((v - XMIN) / (XMAX - XMIN)) * P.w;

    LR.drag(cv, W, H, {
      hit: (p) => p.x > P.x0 - 10 && p.x < P.x0 + P.w + 10,
      down: move, move,
    });
    function move(p) {
      LC.setThr(Math.max(XMIN, Math.min(XMAX, XMIN + ((p.x - P.x0) / P.w) * (XMAX - XMIN))));
    }
    LC.onThr(draw);

    function cell(id, v, hot) {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = v;
      el.style.fontWeight = "700";
      el.style.color = hot || C.text;
    }

    function draw() {
      const thr = LC.thr;
      ctx.clearRect(0, 0, W, H);
      const tx = sx(thr);

      ctx.fillStyle = NEG_SOFT; ctx.fillRect(P.x0, P.y0, tx - P.x0, P.h);
      ctx.fillStyle = POS_SOFT; ctx.fillRect(tx, P.y0, P.x0 + P.w - tx, P.h);
      ctx.strokeStyle = "#dcdcdc"; ctx.lineWidth = 1;
      ctx.strokeRect(P.x0, P.y0, P.w, P.h);

      const cm = LC.confusion(items, thr);
      const m = LC.metrics(cm);

      items.forEach(function (it) {
        const rowY = it.t === 1 ? P.y0 + 30 : P.y0 + 122;
        const y = rowY + it.jit * 52;
        const flagged = it.s >= thr;
        const wrong = (it.t === 1) !== flagged;
        LR.dot(ctx, sx(it.s), y, wrong ? 6 : 4.5, it.t === 1 ? POS : NEG, wrong ? C.red : "#ffffff");
      });

      ctx.font = "600 11.5px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = POS; ctx.fillText("diseased", P.x0 + 6, P.y0 + 18);
      ctx.fillStyle = NEG; ctx.fillText("healthy", P.x0 + 6, P.y0 + 112);
      ctx.fillStyle = C.muted;
      ctx.textAlign = "right";
      ctx.fillText("cleared ←", tx - 8, P.y0 + P.h - 8);
      ctx.textAlign = "left";
      ctx.fillText("→ flagged", tx + 8, P.y0 + P.h - 8);

      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath(); ctx.moveTo(tx, P.y0 - 4); ctx.lineTo(tx, P.y0 + P.h + 4); ctx.stroke();
      ctx.fillStyle = C.orange;
      ctx.beginPath();
      ctx.moveTo(tx - 7, P.y0 - 4); ctx.lineTo(tx + 7, P.y0 - 4); ctx.lineTo(tx, P.y0 + 6);
      ctx.closePath(); ctx.fill();
      ctx.font = "700 12px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("τ = " + LR.fmtF(thr, 2), tx, P.y0 + P.h + 22);

      cell("lc-tp", "TP = " + cm.tp, POS);
      cell("lc-fn", "FN = " + cm.fn, C.red);
      cell("lc-fp", "FP = " + cm.fp, C.red);
      cell("lc-tn", "TN = " + cm.tn, NEG);

      const f = (v, d) => (isFinite(v) ? LR.fmtF(v, d === undefined ? 2 : d) : "undefined");
      ro.set("p", f(m.P) + (cm.tp + cm.fp === 0 ? " (nobody flagged)" : ""), C.orange);
      ro.set("r", f(m.R), C.orange);
      ro.set("f1", f(m.F1));
      ro.set("acc", f(m.acc) + "  (all-clear baseline: " + LR.fmtF(45 / 60, 2) + ")");
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.2 — the precision-recall curve, linked to Fig 4.1
     ════════════════════════════════════════════════════════════ */
  LR.figs.prcurve = function (mount) {
    LR.header(
      mount,
      "Every threshold at once",
      "Each dot is one threshold setting on the Fig 4.1 data. The orange marker is the threshold currently set there; drag here to move it. Toggle the weaker model."
    );

    const items = LC.screening;
    const sweep = LC.prSweep(items).filter((p) => isFinite(p.P) && isFinite(p.R));
    const sweepWeak = LC.prSweep(LC.screeningWeak).filter((p) => isFinite(p.P) && isFinite(p.R));

    let showWeak = false;
    const bar = LR.controls(mount);
    const wb = LR.button(bar, "Show the weaker model: off", function () {
      showWeak = !showWeak;
      wb.textContent = "Show the weaker model: " + (showWeak ? "on" : "off");
      wb.classList.toggle("on", showWeak);
      draw();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 440, {
      aria: "Precision-recall curve traced by sweeping the classification threshold, with a marker at the currently selected threshold and an optional weaker model's curve",
    });
    const ro = LR.readout(mount, [
      { k: "thr", label: "threshold τ" },
      { k: "pr", label: "operating point (R, P)" },
      { k: "f1", label: "F₁ at this τ" },
    ]);

    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: 0, xmax: 1.02, ymin: 0, ymax: 1.05, xlabel: "recall R (of all diseased, fraction caught)", ylabel: "precision P" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => SC && p.x > P.x0 - 15 && p.x < P.x0 + P.w + 15 && p.y > P.y0 - 15 && p.y < P.y0 + P.h + 15,
      down: pick, move: pick,
    });
    function pick(p) {
      // snap the shared threshold to the sweep point nearest the pointer
      let best = null, bd = Infinity;
      sweep.forEach(function (q) {
        const d = (p.x - SC.sx(q.R)) ** 2 + (p.y - SC.sy(q.P)) ** 2;
        if (d < bd) { bd = d; best = q; }
      });
      if (best) LC.setThr(best.thr);
    }
    LC.onThr(draw);

    function drawCurve(sw, color, dash) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.2;
      if (dash) ctx.setLineDash([6, 4]);
      ctx.beginPath();
      sw.forEach((q, i) => (i === 0 ? ctx.moveTo(SC.sx(q.R), SC.sy(q.P)) : ctx.lineTo(SC.sx(q.R), SC.sy(q.P))));
      ctx.stroke();
      ctx.setLineDash([]);
      sw.forEach((q) => LR.dot(ctx, SC.sx(q.R), SC.sy(q.P), 2.6, color));
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);

      if (showWeak) drawCurve(sweepWeak, NEG, true);
      drawCurve(sweep, C.orange);

      // legend
      ctx.font = "600 11.5px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = C.orange; ctx.fillText("— the Fig 4.1 scorer", P.x0 + 12, P.y0 + 18);
      if (showWeak) { ctx.fillStyle = NEG; ctx.fillText("- - weaker scorer, same patients: lower everywhere", P.x0 + 12, P.y0 + 36); }

      // marker at the shared threshold
      const m = LC.metrics(LC.confusion(items, LC.thr));
      if (isFinite(m.P) && isFinite(m.R)) {
        LR.dot(ctx, SC.sx(m.R), SC.sy(m.P), 8, "#ffffff", C.orange);
        LR.dot(ctx, SC.sx(m.R), SC.sy(m.P), 3.5, C.orange);
        ctx.fillStyle = C.orange;
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillText("you are here", SC.sx(m.R) + 12, SC.sy(m.P) - 10);
        ro.set("pr", "(" + LR.fmtF(m.R, 2) + ", " + LR.fmtF(m.P, 2) + ")", C.orange);
        ro.set("f1", LR.fmtF(m.F1, 2));
      } else {
        ro.set("pr", "no flags at this τ: precision undefined");
        ro.set("f1", "–");
      }
      ro.set("thr", LR.fmtF(LC.thr, 2), C.orange);
    }
    draw();
  };
})();
