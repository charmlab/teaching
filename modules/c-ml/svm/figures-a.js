/* ══════════════════════════════════════════════════════════════
   figures-a.js — SVM lesson figures, sections 0–4
   Fig 0.1 whichline · Fig 1.1 marginwidth · Fig 2.1 maxmargin
   Fig 4.1 softc
   Shared SVM core (LR.svm): scores, geometric margins, an exact
   2-D hard-margin solver, kernels, and a simplified-SMO dual
   solver. All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared SVM core (also used by figures-b.js) ────────── */
  const S = (LR.svm = {
    score: (w, b, p) => w.x * p.x + w.y * p.y + b,
    norm: (w) => Math.hypot(w.x, w.y),
    marginWidth: (w) => 2 / Math.hypot(w.x, w.y),
    slack: (w, b, p) => Math.max(0, 1 - p.t * (w.x * p.x + w.y * p.y + b)),

    // geometric margin of an arbitrary line vs labelled points:
    // min_i t·z / ||w||  (negative when the line misclassifies someone)
    geoMargin: function (w, b, pts) {
      const nw = S.norm(w);
      let m = Infinity;
      for (const p of pts) m = Math.min(m, (p.t * S.score(w, b, p)) / nw);
      return m;
    },

    /* Exact 2-D hard-margin SVM by candidate enumeration.
       The optimal active set is a pair (one point per class) or a
       triple (two of one class, one of the other); enumerate all,
       keep the feasible candidates, return the widest margin.   */
    hardMargin: function (pts) {
      const n = pts.length;
      const eps = 1e-9;
      const feasible = (w, b) =>
        pts.every((p) => p.t * (w.x * p.x + w.y * p.y + b) >= 1 - 1e-6);
      let best = null;
      const consider = function (w, b) {
        if (!feasible(w, b)) return;
        const M = S.marginWidth(w);
        if (!best || M > best.M + eps) best = { w, b, M };
      };
      // pairs: one +1 point, one -1 point; w = 2(x⁺ - x⁻)/||x⁺ - x⁻||²
      for (let i = 0; i < n; i++) {
        if (pts[i].t !== 1) continue;
        for (let j = 0; j < n; j++) {
          if (pts[j].t !== -1) continue;
          const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 1e-12) continue;
          const w = { x: (2 * dx) / d2, y: (2 * dy) / d2 };
          const b = 1 - (w.x * pts[i].x + w.y * pts[i].y);
          consider(w, b);
        }
      }
      // triples: i, j same class define one margin line, k the other
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (pts[i].t !== pts[j].t) continue;
          const ux = pts[j].x - pts[i].x, uy = pts[j].y - pts[i].y;
          const ul = Math.hypot(ux, uy);
          if (ul < 1e-9) continue;
          const nx = -uy / ul, ny = ux / ul; // unit normal to the i–j line
          for (let k = 0; k < n; k++) {
            if (pts[k].t === pts[i].t) continue;
            const gap = nx * (pts[k].x - pts[i].x) + ny * (pts[k].y - pts[i].y);
            if (Math.abs(gap) < 1e-9) continue;
            const s = (pts[k].t - pts[i].t) / gap; // scale so scores hit ±1
            const w = { x: s * nx, y: s * ny };
            const b = pts[i].t - (w.x * pts[i].x + w.y * pts[i].y);
            consider(w, b);
          }
        }
      }
      if (best) {
        best.sv = [];
        pts.forEach(function (p, idx) {
          if (Math.abs(p.t * S.score(best.w, best.b, p) - 1) < 1e-4) best.sv.push(idx);
        });
      }
      return best;
    },

    /* kernels (u, v are {x, y}) */
    kLinear: (u, v) => u.x * v.x + u.y * v.y,
    kPoly: (d) => (u, v) => Math.pow(1 + u.x * v.x + u.y * v.y, d),
    kRBF: (g) => (u, v) =>
      Math.exp(-g * ((u.x - v.x) * (u.x - v.x) + (u.y - v.y) * (u.y - v.y))),

    /* Simplified SMO (Platt) for the soft-margin dual:
       max Σα − ½ ΣΣ αᵢαⱼ tᵢtⱼ K(xᵢ,xⱼ)  s.t.  0 ≤ α ≤ C, Σαt = 0.
       Deterministic via a seeded RNG. Returns α, b, the support set,
       and a decision function f(x) = Σ_{i∈S} αᵢ tᵢ K(xᵢ, x) + b.   */
    smo: function (pts, Cval, kern, opts) {
      opts = opts || {};
      const n = pts.length;
      const K = [];
      for (let i = 0; i < n; i++) {
        K.push(new Array(n));
        for (let j = 0; j < n; j++) K[i][j] = kern(pts[i], pts[j]);
      }
      const t = pts.map((p) => p.t);
      const a = new Array(n).fill(0);
      let b = 0;
      const tol = 1e-3;
      const rand = LR.rng(opts.seed || 12345);
      const fi = function (i) {
        let s = b;
        for (let j = 0; j < n; j++) if (a[j] !== 0) s += a[j] * t[j] * K[i][j];
        return s;
      };
      let quiet = 0, sweeps = 0;
      const maxSweeps = opts.maxSweeps || 200;
      while (quiet < 5 && sweeps < maxSweeps) {
        let changed = 0;
        sweeps++;
        for (let i = 0; i < n; i++) {
          const Ei = fi(i) - t[i];
          if (!((t[i] * Ei < -tol && a[i] < Cval) || (t[i] * Ei > tol && a[i] > 0))) continue;
          let j = Math.floor(rand() * (n - 1));
          if (j >= i) j++;
          const Ej = fi(j) - t[j];
          const ai0 = a[i], aj0 = a[j];
          let L, H;
          if (t[i] !== t[j]) { L = Math.max(0, aj0 - ai0); H = Math.min(Cval, Cval + aj0 - ai0); }
          else { L = Math.max(0, ai0 + aj0 - Cval); H = Math.min(Cval, ai0 + aj0); }
          if (L >= H) continue;
          const eta = 2 * K[i][j] - K[i][i] - K[j][j];
          if (eta >= 0) continue;
          let aj = aj0 - (t[j] * (Ei - Ej)) / eta;
          aj = Math.max(L, Math.min(H, aj));
          if (Math.abs(aj - aj0) < 1e-5) continue;
          const ai = ai0 + t[i] * t[j] * (aj0 - aj);
          a[i] = ai; a[j] = aj;
          const b1 = b - Ei - t[i] * (ai - ai0) * K[i][i] - t[j] * (aj - aj0) * K[i][j];
          const b2 = b - Ej - t[i] * (ai - ai0) * K[i][j] - t[j] * (aj - aj0) * K[j][j];
          b = ai > 0 && ai < Cval ? b1 : aj > 0 && aj < Cval ? b2 : (b1 + b2) / 2;
          changed++;
        }
        quiet = changed === 0 ? quiet + 1 : 0;
      }
      const sv = [];
      for (let i = 0; i < n; i++) if (a[i] > 1e-6) sv.push(i);
      return {
        a, b, sv,
        decide: function (q) {
          let s = b;
          for (const i of sv) s += a[i] * t[i] * kern(q, pts[i]);
          return s;
        },
      };
    },

    /* soft-margin linear fit via SMO, with w recovered as Σ α t x */
    softLinear: function (pts, Cval, opts) {
      const fit = S.smo(pts, Cval, S.kLinear, opts);
      let wx = 0, wy = 0;
      for (const i of fit.sv) { wx += fit.a[i] * pts[i].t * pts[i].x; wy += fit.a[i] * pts[i].t * pts[i].y; }
      return { w: { x: wx, y: wy }, b: fit.b, fit };
    },
  });

  /* ── drawing helpers for lines w·x + b = c ──────────────── */
  // endpoints (world coords) of the level line across the plot window
  function levelPts(w, b, c, P) {
    if (Math.abs(w.y) >= Math.abs(w.x)) {
      return [
        { x: P.xmin, y: (c - b - w.x * P.xmin) / w.y },
        { x: P.xmax, y: (c - b - w.x * P.xmax) / w.y },
      ];
    }
    return [
      { x: (c - b - w.y * P.ymin) / w.x, y: P.ymin },
      { x: (c - b - w.y * P.ymax) / w.x, y: P.ymax },
    ];
  }
  function strokeLevel(ctx, SC, w, b, c, P, color, width, dash) {
    const [p1, p2] = levelPts(w, b, c, P);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(SC.sx(p1.x), SC.sy(p1.y));
    ctx.lineTo(SC.sx(p2.x), SC.sy(p2.y));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  function shadeBand(ctx, SC, w, b, c1, c2, P, fill) {
    const [a1, a2] = levelPts(w, b, c1, P);
    const [b1, b2] = levelPts(w, b, c2, P);
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.moveTo(SC.sx(a1.x), SC.sy(a1.y));
    ctx.lineTo(SC.sx(a2.x), SC.sy(a2.y));
    ctx.lineTo(SC.sx(b2.x), SC.sy(b2.y));
    ctx.lineTo(SC.sx(b1.x), SC.sy(b1.y));
    ctx.closePath();
    ctx.fill();
  }
  LR.svmDraw = { levelPts, strokeLevel, shadeBand };

  const T_COLOR = { "1": C.green, "-1": C.purple };

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — which line is best?
     ════════════════════════════════════════════════════════════ */
  LR.figs.whichline = function (mount) {
    LR.header(
      mount,
      "Three perfect separators, one worth trusting",
      "All three lines classify every training point correctly. Drag any endpoint; each line's margin, the distance to its nearest point, is computed live."
    );

    const rand = LR.rng(29);
    const pts = [];
    for (let i = 0; i < 8; i++) pts.push({ x: 0.8 + rand() * 2.0, y: 1.0 + rand() * 2.0, t: -1 });
    for (let i = 0; i < 8; i++) pts.push({ x: 3.5 + rand() * 2.0, y: 3.4 + rand() * 2.2, t: 1 });

    // three candidate boundaries, each stored as two draggable endpoints
    const LINES = [
      { name: "A", color: C.orange, p1: { x: 3.3, y: 0.4 }, p2: { x: 2.9, y: 5.8 } },
      { name: "B", color: C.green, p1: { x: 0.4, y: 5.2 }, p2: { x: 5.7, y: 1.2 } },
      { name: "C", color: C.purple, p1: { x: 0.3, y: 3.4 }, p2: { x: 5.8, y: 3.1 } },
    ];

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 450, {
      aria: "Separable two-class scatter with three draggable candidate boundaries, each scored by its margin to the nearest point",
    });
    const ro = LR.readout(mount, [
      { k: "A", label: "margin of A" },
      { k: "B", label: "margin of B" },
      { k: "C", label: "margin of C" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 6.2, ymin: 0, ymax: 6.2, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null;

    // margin of a candidate line: unit normal n, offset b; orientation is
    // chosen to favour the data, and the min of t·z over the points is the
    // true signed distance to the nearest point (negative = misclassifies).
    function lineMargin(L) {
      const ux = L.p2.x - L.p1.x, uy = L.p2.y - L.p1.y;
      const ul = Math.hypot(ux, uy) || 1e-9;
      let n = { x: -uy / ul, y: ux / ul };
      let b = -(n.x * L.p1.x + n.y * L.p1.y);
      const m1 = S.geoMargin(n, b, pts);
      const m2 = S.geoMargin({ x: -n.x, y: -n.y }, -b, pts);
      if (m2 > m1) { n = { x: -n.x, y: -n.y }; b = -b; }
      const m = Math.max(m1, m2);
      let nearest = null, bestTz = Infinity;
      for (const p of pts) {
        const tz = p.t * S.score(n, b, p);
        if (tz < bestTz) { bestTz = tz; nearest = p; }
      }
      return { n, b, m, nearest };
    }

    LR.drag(cv, W, H, {
      hit: function (p) {
        for (const L of LINES) {
          for (const e of [L.p1, L.p2]) {
            if ((p.x - SC.sx(e.x)) ** 2 + (p.y - SC.sy(e.y)) ** 2 < 500) { dragging = e; return true; }
          }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        dragging.x = Math.max(0.1, Math.min(6.1, d.x));
        dragging.y = Math.max(0.1, Math.min(6.1, d.y));
        draw();
      },
      up: function () { dragging = null; },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      const infos = LINES.map((L) => ({ L, info: lineMargin(L) }));
      let winner = null;
      infos.forEach(function (o) {
        if (o.info.m > 0 && (!winner || o.info.m > winner.info.m)) winner = o;
      });

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      infos.forEach(function (o) {
        const { L, info } = o;
        // the line itself, drawn long
        const ux = L.p2.x - L.p1.x, uy = L.p2.y - L.p1.y;
        const ul = Math.hypot(ux, uy) || 1e-9;
        const ex1 = { x: L.p1.x - (ux / ul) * 20, y: L.p1.y - (uy / ul) * 20 };
        const ex2 = { x: L.p1.x + (ux / ul) * 20, y: L.p1.y + (uy / ul) * 20 };
        ctx.strokeStyle = L.color;
        ctx.lineWidth = o === winner ? 3.4 : 2;
        ctx.beginPath();
        ctx.moveTo(sx(ex1.x), sy(ex1.y));
        ctx.lineTo(sx(ex2.x), sy(ex2.y));
        ctx.stroke();
        // dashed tick to the nearest (margin-limiting) point
        if (info.nearest) {
          const p = info.nearest;
          const z = S.score(info.n, info.b, p); // signed distance (unit normal)
          const foot = { x: p.x - z * info.n.x, y: p.y - z * info.n.y };
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = L.color;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(sx(p.x), sy(p.y));
          ctx.lineTo(sx(foot.x), sy(foot.y));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
      ctx.restore();

      // points
      pts.forEach(function (p) {
        LR.dot(ctx, sx(p.x), sy(p.y), 5.5, T_COLOR[p.t], "#ffffff");
      });

      // endpoint handles + labels
      infos.forEach(function (o) {
        const L = o.L;
        [L.p1, L.p2].forEach(function (e) {
          LR.dot(ctx, sx(e.x), sy(e.y), 6, "#ffffff", L.color);
        });
        ctx.font = "800 13px Inter, sans-serif";
        ctx.fillStyle = L.color;
        ctx.textAlign = "left";
        ctx.fillText(L.name, sx(L.p2.x) + 9, sy(L.p2.y) - 8);
      });

      infos.forEach(function (o) {
        const bad = o.info.m <= 0;
        ro.set(
          o.L.name,
          bad ? "misclassifies a point" : LR.fmtF(o.info.m, 2) + (o === winner ? "  ← widest buffer" : ""),
          bad ? C.red : o === winner ? o.L.color : undefined
        );
      });
      if (winner) {
        msg.show(
          "Line " + winner.L.name + " keeps the biggest buffer: " + LR.fmtF(winner.info.m, 2) +
          " units to its nearest point. All three are perfect on the training data; only the buffer tells them apart.",
          "info"
        );
      } else {
        msg.show("Every line currently misclassifies at least one point. Drag the handles until the lines separate the classes again.", "bad");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — where M = 2/||w|| comes from
     ════════════════════════════════════════════════════════════ */
  LR.figs.marginwidth = function (mount) {
    LR.header(
      mount,
      "The margin ruler",
      "The boundary is fixed. The slider rescales ||w||, which moves the ±1 level sets: the score climbs by ||w|| per unit step, so the ±1 lines sit 1/||w|| away on each side."
    );

    const theta = 0.56;             // orientation of w (radians), fixed
    const c0 = { x: 3, y: 3 };      // a point on the boundary, fixed
    let wn = 2.0;                    // ||w||, the slider value

    const bar = LR.controls(mount);
    LR.slider(bar, "‖w‖", 0.4, 3, 0.05, wn, function (v) { wn = v; draw(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Diagram of the decision boundary and the two margin hyperplanes, with the norm of w adjustable to widen or narrow the margin",
    });
    const ro = LR.readout(mount, [
      { k: "wn", label: "‖w‖" },
      { k: "half", label: "each side: 1/‖w‖" },
      { k: "M", label: "margin M = 2/‖w‖" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 6, ymin: 0, ymax: 6, xlabel: "x₁", ylabel: "x₂" };

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      const w = { x: wn * Math.cos(theta), y: wn * Math.sin(theta) };
      const b = -(w.x * c0.x + w.y * c0.y);
      const nhat = { x: Math.cos(theta), y: Math.sin(theta) };
      const half = 1 / wn;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      shadeBand(ctx, SC, w, b, -1, 1, P, "rgba(26,26,26,0.08)");
      strokeLevel(ctx, SC, w, b, 0, P, C.orange, 2.8);
      strokeLevel(ctx, SC, w, b, 1, P, C.green, 2, [7, 5]);
      strokeLevel(ctx, SC, w, b, -1, P, C.purple, 2, [7, 5]);
      ctx.restore();

      // w vector arrow from the boundary point, length ||w|| in world units
      LR.arrow(ctx, sx(c0.x), sy(c0.y), sx(c0.x + w.x), sy(c0.y + w.y), C.text, 2.2);
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillStyle = C.text;
      ctx.textAlign = "left";
      ctx.fillText("w  (‖w‖ = " + LR.fmtF(wn, 2) + ")", sx(c0.x + w.x) + 8, sy(c0.y + w.y) - 6);

      // double-headed arrow across the band, offset along the line so it
      // does not sit on top of the w arrow
      const along = { x: -Math.sin(theta), y: Math.cos(theta) };
      const mid = { x: c0.x + along.x * 1.5, y: c0.y + along.y * 1.5 };
      const e1 = { x: mid.x - nhat.x * half, y: mid.y - nhat.y * half };
      const e2 = { x: mid.x + nhat.x * half, y: mid.y + nhat.y * half };
      LR.arrow(ctx, sx(mid.x), sy(mid.y), sx(e2.x), sy(e2.y), C.red, 2);
      LR.arrow(ctx, sx(mid.x), sy(mid.y), sx(e1.x), sy(e1.y), C.red, 2);
      ctx.fillStyle = C.red;
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillText("M = " + LR.fmtF(2 / wn, 2), sx(e2.x) + 8, sy(e2.y) + 14);

      // level-set labels
      ctx.font = "600 12px Inter, sans-serif";
      const lbl = function (cval, color, txt) {
        const [p1] = levelPts(w, b, cval, P);
        // pick a label spot inside the plot along the line
        const [q1, q2] = levelPts(w, b, cval, P);
        const lx = q1.x + (q2.x - q1.x) * 0.06, ly = q1.y + (q2.y - q1.y) * 0.06;
        ctx.fillStyle = color;
        ctx.fillText(txt, sx(lx) + 4, sy(ly) - 6);
      };
      lbl(1, C.green, "wᵀx + b = +1");
      lbl(0, C.orange, "wᵀx + b = 0");
      lbl(-1, C.purple, "wᵀx + b = −1");

      ro.set("wn", LR.fmtF(wn, 2));
      ro.set("half", LR.fmtF(half, 2));
      ro.set("M", LR.fmtF(2 / wn, 2), C.red);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — the max-margin explorer (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.maxmargin = function (mount) {
    LR.header(
      mount,
      "Find the widest buffer, then meet the points that own it",
      "Drag the square handle to slide your boundary and the round handle to rotate it. Press Solve for the exact max-margin answer, then drag data points and watch which ones matter."
    );

    const rand = LR.rng(47);
    const initPts = [];
    for (let i = 0; i < 7; i++) initPts.push({ x: 3.6 + rand() * 1.9, y: 3.5 + rand() * 2.0, t: 1 });
    for (let i = 0; i < 7; i++) initPts.push({ x: 0.7 + rand() * 1.9, y: 0.8 + rand() * 2.1, t: -1 });
    let pts = initPts.map((p) => ({ x: p.x, y: p.y, t: p.t }));

    // manual line state: a centre point and an orientation
    let cp = { x: 3.2, y: 3.1 }, theta = -0.75;
    let mode = "manual"; // "manual" | "solved"
    let fit = null, lastFit = null;
    let fitAtDragStart = null, draggedPt = null;

    const bar = LR.controls(mount);
    LR.button(bar, "Solve the max margin ▸", function () {
      solve();
      if (fit) msg.show("Solved. The widest buffer is M = 2/‖w‖ = " + LR.fmtF(fit.M, 3) + ", and only the ringed points hold it up. Try dragging them, and try dragging the others.", "good");
      draw();
    }, "primary");
    LR.button(bar, "Reset ⟲", function () {
      pts = initPts.map((p) => ({ x: p.x, y: p.y, t: p.t }));
      cp = { x: 3.2, y: 3.1 }; theta = -0.75;
      mode = "manual"; fit = null; lastFit = null;
      msg.hide(); draw();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 480, {
      aria: "Max-margin explorer with a draggable boundary, draggable training points, shaded margin, ringed support vectors, and live margin width",
    });
    const ro = LR.readout(mount, [
      { k: "margin", label: "margin width" },
      { k: "norm", label: "‖w‖" },
      { k: "sv", label: "support vectors" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };
    let SC = null;
    const ROT_R = 1.7; // world-unit distance of the rotation handle from cp

    function solve() {
      const f = S.hardMargin(pts);
      if (f) { fit = f; lastFit = f; mode = "solved"; }
      else {
        fit = null; mode = "solved";
        msg.show("The classes are no longer linearly separable, so no hard-margin solution exists. Drag the offending point back out (the greyed line is the last valid solution).", "bad");
      }
    }

    // manual line as (unit normal, offset), oriented toward the +1 class
    function manualLine() {
      let n = { x: -Math.sin(theta), y: Math.cos(theta) };
      let b = -(n.x * cp.x + n.y * cp.y);
      const m1 = S.geoMargin(n, b, pts);
      const m2 = S.geoMargin({ x: -n.x, y: -n.y }, -b, pts);
      if (m2 > m1) { n = { x: -n.x, y: -n.y }; b = -b; }
      return { n, b, m: Math.max(m1, m2) };
    }

    function rotHandle() {
      return { x: cp.x + ROT_R * Math.cos(theta), y: cp.y + ROT_R * Math.sin(theta) };
    }

    let dragKind = null; // "centre" | "rotate" | "point"
    LR.drag(cv, W, H, {
      hit: function (p) {
        if (mode === "manual") {
          const rh = rotHandle();
          if ((p.x - SC.sx(rh.x)) ** 2 + (p.y - SC.sy(rh.y)) ** 2 < 500) { dragKind = "rotate"; return true; }
          if ((p.x - SC.sx(cp.x)) ** 2 + (p.y - SC.sy(cp.y)) ** 2 < 500) { dragKind = "centre"; return true; }
        }
        for (const pt of pts) {
          if ((p.x - SC.sx(pt.x)) ** 2 + (p.y - SC.sy(pt.y)) ** 2 < 420) {
            dragKind = "point"; draggedPt = pt;
            fitAtDragStart = mode === "solved" && fit ? { w: { x: fit.w.x, y: fit.w.y }, b: fit.b } : null;
            return true;
          }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        if (dragKind === "centre") {
          cp.x = Math.max(0.3, Math.min(6.2, d.x));
          cp.y = Math.max(0.3, Math.min(6.2, d.y));
        } else if (dragKind === "rotate") {
          theta = Math.atan2(d.y - cp.y, d.x - cp.x);
        } else if (dragKind === "point") {
          draggedPt.x = Math.max(0.15, Math.min(6.35, d.x));
          draggedPt.y = Math.max(0.15, Math.min(6.35, d.y));
          if (mode === "solved") solve();
        }
        draw();
      },
      up: function () {
        if (dragKind === "point" && fitAtDragStart) {
          if (fit) {
            const moved =
              Math.abs(fit.w.x - fitAtDragStart.w.x) + Math.abs(fit.w.y - fitAtDragStart.w.y) +
              Math.abs(fit.b - fitAtDragStart.b) > 1e-7;
            msg.show(
              moved
                ? "The boundary moved: that point is (or became) a support vector, and the whole solution follows it."
                : "The boundary did not move at all: that point is not a support vector. Its constraint was slack, so the optimum ignores it.",
              moved ? "info" : "good"
            );
          }
        }
        dragKind = null; draggedPt = null; fitAtDragStart = null;
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      let svSet = new Set();

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      if (mode === "manual") {
        const L = manualLine();
        if (L.m > 0) {
          shadeBand(ctx, SC, L.n, L.b, -L.m, L.m, P, "rgba(26,26,26,0.07)");
          strokeLevel(ctx, SC, L.n, L.b, L.m, P, C.faint, 1.6, [6, 5]);
          strokeLevel(ctx, SC, L.n, L.b, -L.m, P, C.faint, 1.6, [6, 5]);
          pts.forEach(function (p, i) {
            if (p.t * S.score(L.n, L.b, p) <= L.m + 1e-6) svSet.add(i);
          });
        }
        strokeLevel(ctx, SC, L.n, L.b, 0, P, C.orange, 2.6);
        ro.set("margin", L.m > 0 ? LR.fmtF(2 * L.m, 3) + "  (your line)" : "line misclassifies a point", L.m > 0 ? C.orange : C.red);
        ro.set("norm", "drag, or press Solve");
        ro.set("sv", L.m > 0 ? "ringed: the point(s) limiting you" : "–");
      } else if (fit) {
        shadeBand(ctx, SC, fit.w, fit.b, -1, 1, P, "rgba(26,26,26,0.08)");
        strokeLevel(ctx, SC, fit.w, fit.b, 1, P, C.green, 1.8, [7, 5]);
        strokeLevel(ctx, SC, fit.w, fit.b, -1, P, C.purple, 1.8, [7, 5]);
        strokeLevel(ctx, SC, fit.w, fit.b, 0, P, C.orange, 2.8);
        fit.sv.forEach((i) => svSet.add(i));
        ro.set("margin", "M = 2/‖w‖ = " + LR.fmtF(fit.M, 3), C.orange);
        ro.set("norm", LR.fmtF(S.norm(fit.w), 3));
        ro.set("sv", fit.sv.length + " of " + pts.length + " points", C.green);
      } else if (lastFit) {
        strokeLevel(ctx, SC, lastFit.w, lastFit.b, 0, P, "#bbbbbb", 2, [4, 4]);
        ro.set("margin", "not separable", C.red);
        ro.set("norm", "–");
        ro.set("sv", "–");
      }
      ctx.restore();

      // data points (rings mark support vectors / limiting points)
      pts.forEach(function (p, i) {
        const ringed = svSet.has(i);
        LR.dot(ctx, sx(p.x), sy(p.y), ringed ? 7.5 : 5.5, T_COLOR[p.t], ringed ? C.orange : "#ffffff");
        if (ringed) {
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 11, 0, Math.PI * 2);
          ctx.strokeStyle = C.orange;
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
      });

      // manual handles
      if (mode === "manual") {
        const rh = rotHandle();
        ctx.fillStyle = C.orange;
        ctx.fillRect(sx(cp.x) - 6, sy(cp.y) - 6, 12, 12);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
        ctx.strokeRect(sx(cp.x) - 6, sy(cp.y) - 6, 12, 12);
        LR.dot(ctx, sx(rh.x), sy(rh.y), 7, "#ffffff", C.orange);
        ctx.font = "600 11.5px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("slide", sx(cp.x) + 10, sy(cp.y) + 4);
        ctx.fillText("rotate", sx(rh.x) + 10, sy(rh.y) + 4);
      }

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = T_COLOR["1"]; ctx.fillText("● t = +1", P.x0 + 10, P.y0 + 16);
      ctx.fillStyle = T_COLOR["-1"]; ctx.fillText("● t = −1", P.x0 + 10, P.y0 + 33);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — the soft-margin C dial (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.softc = function (mount) {
    LR.header(
      mount,
      "Price a violation, buy a margin",
      "Overlapping classes, so no hard margin exists. Each slider position solves the real soft-margin dual at that C. Ring colours are the slack regimes."
    );

    const rand = LR.rng(83);
    const pts = [];
    for (let i = 0; i < 16; i++) pts.push({ x: 3.5 + LR.gauss(rand) * 0.8, y: 3.6 + LR.gauss(rand) * 0.8, t: 1 });
    for (let i = 0; i < 16; i++) pts.push({ x: 2.3 + LR.gauss(rand) * 0.85, y: 2.3 + LR.gauss(rand) * 0.85, t: -1 });
    // clamp into the plot window
    pts.forEach(function (p) {
      p.x = Math.max(0.2, Math.min(6.3, p.x));
      p.y = Math.max(0.2, Math.min(6.3, p.y));
    });

    const Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100];
    let ci = 4;
    const cache = {};

    function getFit(idx) {
      if (!cache[idx]) cache[idx] = S.softLinear(pts, Cs[idx], { seed: 5, maxSweeps: 250 });
      return cache[idx];
    }

    const bar = LR.controls(mount);
    LR.slider(bar, "C (violation price)", 0, Cs.length - 1, 1, ci, function (v) {
      ci = Math.round(v);
      draw();
    }, (v) => String(Cs[Math.round(v)]));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Soft-margin SVM on non-separable data with a C slider; points are coloured by slack regime and the margin width and total slack update live",
    });

    const legend = LR.el("div", "legend-row");
    legend.innerHTML =
      "<span class='legend-chip'><span class='swatch' style='background:#ffffff;border-color:#999999'></span>ξ = 0 (outside, correct)</span>" +
      "<span class='legend-chip'><span class='swatch' style='background:#ffffff;border-color:" + C.orange + "'></span>on the margin</span>" +
      "<span class='legend-chip'><span class='swatch' style='background:#ffffff;border-color:" + C.amber + "'></span>0 &lt; ξ &lt; 1 (inside, correct)</span>" +
      "<span class='legend-chip'><span class='swatch' style='background:#ffffff;border-color:" + C.red + "'></span>ξ &gt; 1 (misclassified)</span>";
    mount.appendChild(legend);

    const ro = LR.readout(mount, [
      { k: "C", label: "C" },
      { k: "M", label: "margin M = 2/‖w‖" },
      { k: "xi", label: "Σ ξᵢ (bounds errors)" },
      { k: "err", label: "actual errors" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };

    function regimeRing(tz, xi) {
      if (Math.abs(tz - 1) <= 0.05) return C.orange;   // on the margin
      if (xi <= 0.02) return "#ffffff";                    // comfortably outside
      if (xi > 1.02) return C.red;                      // misclassified
      if (Math.abs(xi - 1) <= 0.02) return C.text;      // on the boundary itself
      return C.amber;                                   // inside the margin, correct
    }

    function draw() {
      const { w, b } = getFit(ci);
      ctx.clearRect(0, 0, W, H);
      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      shadeBand(ctx, SC, w, b, -1, 1, P, "rgba(26,26,26,0.07)");
      strokeLevel(ctx, SC, w, b, 1, P, C.faint, 1.6, [7, 5]);
      strokeLevel(ctx, SC, w, b, -1, P, C.faint, 1.6, [7, 5]);
      strokeLevel(ctx, SC, w, b, 0, P, C.orange, 2.7);
      ctx.restore();

      let sumXi = 0, errors = 0;
      pts.forEach(function (p) {
        const tz = p.t * S.score(w, b, p);
        const xi = Math.max(0, 1 - tz);
        sumXi += xi;
        if (tz < 0) errors++;
        const ring = regimeRing(tz, xi);
        LR.dot(ctx, sx(p.x), sy(p.y), ring === "#ffffff" ? 5.5 : 6.5, T_COLOR[p.t], ring);
        if (ring !== "#ffffff") {
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 10, 0, Math.PI * 2);
          ctx.strokeStyle = ring;
          ctx.lineWidth = 1.7;
          ctx.stroke();
        }
      });

      ro.set("C", String(Cs[ci]));
      ro.set("M", LR.fmtF(S.marginWidth(w), 2), C.orange);
      ro.set("xi", LR.fmtF(sumXi, 2), C.amber);
      ro.set("err", errors + "  (≤ " + LR.fmtF(sumXi, 2) + " ✓)", errors ? C.red : C.green);

      if (ci === 0) {
        msg.show("C → 0 territory: violations are nearly free, so the solver shrinks ‖w‖ toward zero. The margin balloons (M = " + LR.fmtF(S.marginWidth(w), 1) + "), the buffer swallows the data, and slack piles up.", "info");
      } else if (ci === Cs.length - 1) {
        msg.show("Large C: every unit of slack is expensive, so the margin narrows to M = " + LR.fmtF(S.marginWidth(w), 2) + " and almost nothing is tolerated inside it. On separable data this limit recovers the hard margin.", "info");
      } else {
        msg.hide();
      }
    }
    draw();
  };
})();
