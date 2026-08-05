/* ══════════════════════════════════════════════════════════════
   figures-knn-a.js — kNN lesson figures, sections 0–2
   Fig 0.1 hook · Fig 1.1 sorter · Fig 1.2 vectorize
   Fig 2.1 metrics · Fig 2.2 cosine · Fig 2.3 voronoi
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared kNN core (also used by figures-knn-b.js) ────── */
  const K = (LR.knn = {
    d2: (a, b) => Math.hypot(a.x - b.x, a.y - b.y),
    d1: (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y),
    dInf: (a, b) => Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y)),
    // k nearest under a metric fn; returns [{p, d}] sorted ascending
    nearest: function (pts, q, k, metric) {
      const m = metric || K.d2;
      return pts
        .map((p) => ({ p, d: m(p, q) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, k);
    },
    // majority vote over neighbour list; returns {label, tally}
    vote: function (nbrs) {
      const tally = {};
      nbrs.forEach((n) => (tally[n.p.cls] = (tally[n.p.cls] || 0) + 1));
      let best = null;
      for (const c in tally) if (best === null || tally[c] > tally[best]) best = c;
      return { label: best, tally };
    },
  });

  const CLS_COLOR = { A: C.green, B: C.purple, C: C.amber };
  const CLS_SOFT = { A: "rgba(47,158,68,0.16)", B: "rgba(112,72,232,0.14)", C: "rgba(240,162,2,0.16)" };

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — the hook: similar things, similar labels
     ════════════════════════════════════════════════════════════ */
  LR.figs.hook = function (mount) {
    LR.header(
      mount,
      "Label the mystery fruit",
      "Green fruits are sweet, purple ones are tart. Drag the mystery fruit anywhere; it copies the label of whichever example it most resembles."
    );

    // two fruit clusters (deterministic)
    const rand = LR.rng(21);
    const pts = [];
    for (let i = 0; i < 9; i++) pts.push({ x: 1 + rand() * 1.8, y: 1.2 + rand() * 2.2, cls: "A" });
    for (let i = 0; i < 9; i++) pts.push({ x: 3.1 + rand() * 1.7, y: 4.2 + rand() * 2.4, cls: "B" });

    let q = { x: 2.6, y: 3.4 };

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 420, {
      aria: "Scatter of labelled fruits with a draggable mystery fruit that copies its nearest neighbour's label",
    });
    const ro = LR.readout(mount, [
      { k: "near", label: "most similar fruit" },
      { k: "verdict", label: "verdict" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 5.5, ymin: 0, ymax: 7.5, xlabel: "size", ylabel: "colour (green → purple)" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => SC && (p.x - SC.sx(q.x)) ** 2 + (p.y - SC.sy(q.y)) ** 2 < 2000,
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        q = {
          x: Math.max(P.xmin + 0.15, Math.min(P.xmax - 0.15, d.x)),
          y: Math.max(P.ymin + 0.2, Math.min(P.ymax - 0.2, d.y)),
        };
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      const nn = K.nearest(pts, q, 1)[0];

      // connection line
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(sx(q.x), sy(q.y));
      ctx.lineTo(sx(nn.p.x), sy(nn.p.y));
      ctx.stroke();
      ctx.setLineDash([]);

      // training fruits
      pts.forEach(function (p) {
        const r = p === nn.p ? 9 : 6;
        LR.dot(ctx, sx(p.x), sy(p.y), r, CLS_COLOR[p.cls], p === nn.p ? C.orange : "#fff");
      });

      // mystery fruit: takes on the copied colour, keeps a "?"
      LR.dot(ctx, sx(q.x), sy(q.y), 11, CLS_COLOR[nn.p.cls], C.orange);
      ctx.font = "800 13px Inter, sans-serif";
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.fillText("?", sx(q.x), sy(q.y) + 4.5);

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = CLS_COLOR.A; ctx.fillText("● sweet", P.x0 + 10, P.y0 + 16);
      ctx.fillStyle = CLS_COLOR.B; ctx.fillText("● tart", P.x0 + 10, P.y0 + 33);

      ro.set("near", (nn.p.cls === "A" ? "a sweet one" : "a tart one") + ", distance " + LR.fmtF(nn.d, 2));
      ro.set("verdict", nn.p.cls === "A" ? "sweet 🍈" : "tart 🍋", CLS_COLOR[nn.p.cls]);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — feature-type sorter
     ════════════════════════════════════════════════════════════ */
  LR.figs.sorter = function (mount) {
    LR.header(
      mount,
      "Sort the features",
      "Click an attribute, then click the bucket it belongs in. The four planet attributes come straight from the lecture."
    );

    // First four from the lecture; last two are additions for practice
    // (flagged: not in lecture-02-source-content.md, added for a fuller exercise).
    const ITEMS = [
      { label: "Planet mass (kg)", type: "continuous" },
      { label: "Order from the sun (1st, 2nd, …)", type: "ordinal" },
      { label: "Number of moons", type: "discrete" },
      { label: "Gas giant or rocky", type: "nominal" },
      { label: "Average surface temperature (°C)", type: "continuous" },
      { label: "Has rings (yes / no)", type: "nominal" },
    ];
    const BUCKETS = [
      { key: "continuous", title: "Continuous", sub: "any real value in a range" },
      { key: "ordinal", title: "Ordinal", sub: "ordered, gaps meaningless" },
      { key: "discrete", title: "Discrete", sub: "countable whole numbers" },
      { key: "nominal", title: "Nominal", sub: "named, no order" },
    ];

    const itemsBox = LR.el("div", "sorter-items");
    mount.appendChild(itemsBox);
    const bucketsBox = LR.el("div", "sorter-buckets");
    mount.appendChild(bucketsBox);
    const msg = LR.msg(mount);

    let selected = null;
    let placed = 0;

    const chips = ITEMS.map(function (it) {
      const chip = LR.el("button", "sorter-chip", it.label);
      chip.type = "button";
      chip.addEventListener("click", function () {
        if (chip.classList.contains("placed")) return;
        chips.forEach((c) => c.classList.remove("selected"));
        selected = it;
        chip.classList.add("selected");
        chip.__el = chip;
        buckets.forEach((b) => b.classList.add("armed"));
      });
      it.__chip = chip;
      itemsBox.appendChild(chip);
      return chip;
    });

    const buckets = BUCKETS.map(function (b) {
      const el = LR.el("div", "sorter-bucket");
      el.setAttribute("role", "button");
      el.setAttribute("tabindex", "0");
      el.setAttribute("aria-label", "Bucket: " + b.title);
      el.innerHTML = "<h4>" + b.title + "</h4><div class='bucket-sub'>" + b.sub + "</div>";
      const drop = function () {
        if (!selected) return;
        const right = selected.type === b.key;
        const tag = LR.el("span", "bucket-item " + (right ? "right" : "wrong"), selected.label.split(" (")[0]);
        el.appendChild(tag);
        if (right) {
          selected.__chip.classList.remove("selected");
          selected.__chip.classList.add("placed");
          placed += 1;
          msg.show(
            placed === ITEMS.length
              ? "All " + ITEMS.length + " sorted. You now speak feature-type."
              : "✓ Right: that one is " + b.key + ".",
            "good"
          );
        } else {
          msg.show("Not quite: think about whether order and gaps mean anything for it.", "bad");
          setTimeout(function () { tag.remove(); }, 1400);
          selected.__chip.classList.remove("selected");
        }
        selected = null;
        buckets.forEach((bk) => bk.classList.remove("armed"));
      };
      el.addEventListener("click", drop);
      el.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); drop(); }
      });
      bucketsBox.appendChild(el);
      return el;
    });
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.2 — image → matrix → vector
     ════════════════════════════════════════════════════════════ */
  LR.figs.vectorize = function (mount) {
    LR.header(
      mount,
      "An image is already a vector",
      "An 8×8 digit is 64 pixel intensities. Unroll the rows and it is one point in ℝ⁶⁴, ready for linear algebra."
    );

    // an 8x8 "7", intensities in [0,1]
    const G = [
      [0,0,0,0,0,0,0,0],
      [0,.9,.9,.9,.9,.9,.7,0],
      [0,0,0,0,0,.8,.9,0],
      [0,0,0,0,.7,.9,.2,0],
      [0,0,0,.6,.9,.3,0,0],
      [0,0,.4,.9,.5,0,0,0],
      [0,0,.8,.9,0,0,0,0],
      [0,0,0,0,0,0,0,0],
    ];
    const N = 8;

    const bar = LR.controls(mount);
    const unrollBtn = LR.button(bar, "Unroll ▸", unroll, "primary");
    LR.button(bar, "Reset ⟲", reset);

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 360, {
      aria: "Animation of an eight by eight pixel digit unrolling row by row into a sixty four dimensional vector",
    });

    // layout: grid on the left, vector strip along the bottom
    const CELL = 26, GX = 80, GY = 30;
    const STRIP_Y = 300, STRIP_H = 16;
    const stripW = (W - 60) / 64;

    let t = 0, anim = null, state = "grid"; // 0..1 morph

    function gridPos(r, c) { return { x: GX + c * CELL, y: GY + r * CELL, s: CELL }; }
    function stripPos(i) { return { x: 30 + i * stripW, y: STRIP_Y, s: stripW }; }

    function unroll() {
      if (anim) cancelAnimationFrame(anim);
      state = "unrolling";
      if (LR.reducedMotion) { t = 1; state = "vector"; draw(); return; }
      const tick = function () {
        t = Math.min(1, t + 0.012);
        draw();
        if (t < 1) anim = requestAnimationFrame(tick);
        else { state = "vector"; draw(); }
      };
      tick();
    }
    function reset() {
      if (anim) cancelAnimationFrame(anim);
      t = 0; state = "grid"; draw();
    }

    const ease = (u) => u * u * (3 - 2 * u);

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // each pixel morphs from grid slot to strip slot, staggered by index
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          const i = r * N + c;
          // stagger: pixel i starts moving at i/64 * 0.6 and takes 0.4
          const local = Math.max(0, Math.min(1, (t - (i / 64) * 0.6) / 0.4));
          const u = ease(local);
          const g = gridPos(r, c), s = stripPos(i);
          const x = g.x + (s.x - g.x) * u;
          const y = g.y + (s.y - g.y) * u;
          const sz = g.s + (s.s - g.s) * u;
          const v = G[r][c];
          const shade = Math.round(245 - v * 215);
          ctx.fillStyle = "rgb(" + shade + "," + shade + "," + shade + ")";
          ctx.strokeStyle = "#d8d8d8";
          ctx.lineWidth = 0.8;
          ctx.fillRect(x, y, sz - 1, (u < 1 ? sz : STRIP_H) - 1);
          ctx.strokeRect(x, y, sz - 1, (u < 1 ? sz : STRIP_H) - 1);
        }
      }

      ctx.font = "600 12.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      if (state === "grid") {
        ctx.fillText("image: 8 × 8 matrix of intensities", GX, GY - 10);
      } else if (state === "vector") {
        ctx.fillText("vector: x ∈ ℝ⁶⁴, one long row", 30, STRIP_Y - 12);
        ctx.font = "600 12px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.text;
        const row0 = G.flat().slice(8, 15).map((v) => v.toFixed(1)).join(", ");
        ctx.fillText("x = [0.0, …, " + row0 + ", …]  (64 numbers)", 30, STRIP_Y + 42);
      } else {
        ctx.fillText("unrolling row by row…", GX, GY - 10);
      }

      // side note
      ctx.font = "500 12px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.textAlign = "left";
      ctx.fillText("digitization was already an approximation:", 420, 70);
      ctx.fillText("resolution and bit depth were choices", 420, 88);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — distance-metric explorer with unit balls
     ════════════════════════════════════════════════════════════ */
  LR.figs.metrics = function (mount) {
    LR.header(
      mount,
      "What does “near” mean?",
      "Drag either point. Each metric disagrees about the distance, and the ball through B shows the set of points the metric calls equally near."
    );

    let metric = "L2";
    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = {};
    [["L2", "Euclidean ℓ₂"], ["L1", "Manhattan ℓ₁"], ["Linf", "Chebyshev ℓ∞"]].forEach(function (m) {
      btns[m[0]] = LR.button(group, m[1], function () {
        metric = m[0];
        for (const k2 in btns) btns[k2].classList.toggle("on", k2 === metric);
        draw();
      }, "small" + (m[0] === "L2" ? " on" : ""));
    });

    let A = { x: 1.5, y: 2 }, B = { x: 4, y: 4.5 };

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 440, {
      aria: "Two draggable points with Euclidean, Manhattan, and Chebyshev distances and the unit ball of the selected metric",
    });
    const ro = LR.readout(mount, [
      { k: "l2", label: "ℓ₂ (Euclidean)" },
      { k: "l1", label: "ℓ₁ (Manhattan)" },
      { k: "linf", label: "ℓ∞ (Chebyshev)" },
      { k: "cos", label: "cos similarity (from origin)" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 7, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        for (const pt of [A, B]) {
          if ((p.x - SC.sx(pt.x)) ** 2 + (p.y - SC.sy(pt.y)) ** 2 < 900) { dragging = pt; return true; }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        dragging.x = Math.max(0.2, Math.min(6.8, d.x));
        dragging.y = Math.max(0.2, Math.min(6.3, d.y));
        draw();
      },
      up: function () { dragging = null; },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      const dl2 = K.d2(A, B), dl1 = K.d1(A, B), dli = K.dInf(A, B);
      const r = metric === "L2" ? dl2 : metric === "L1" ? dl1 : dli;

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // the "ball" of the selected metric, centred at A, through B
      ctx.strokeStyle = C.orange;
      ctx.fillStyle = "rgba(232,89,12,0.07)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (metric === "L2") {
        ctx.ellipse(sx(A.x), sy(A.y), Math.abs(sx(A.x + r) - sx(A.x)), Math.abs(sy(A.y + r) - sy(A.y)), 0, 0, Math.PI * 2);
      } else if (metric === "L1") {
        ctx.moveTo(sx(A.x + r), sy(A.y));
        ctx.lineTo(sx(A.x), sy(A.y + r));
        ctx.lineTo(sx(A.x - r), sy(A.y));
        ctx.lineTo(sx(A.x), sy(A.y - r));
        ctx.closePath();
      } else {
        ctx.rect(sx(A.x - r), sy(A.y + r), sx(A.x + r) - sx(A.x - r), sy(A.y - r) - sy(A.y + r));
      }
      ctx.fill(); ctx.stroke();

      // path illustrations: L2 straight, L1 staircase
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = C.green; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(sx(A.x), sy(A.y)); ctx.lineTo(sx(B.x), sy(B.y)); ctx.stroke();
      ctx.strokeStyle = C.purple;
      ctx.beginPath();
      ctx.moveTo(sx(A.x), sy(A.y));
      ctx.lineTo(sx(B.x), sy(A.y));
      ctx.lineTo(sx(B.x), sy(B.y));
      ctx.stroke();
      ctx.setLineDash([]);

      // cosine: faint rays from origin
      ctx.strokeStyle = "rgba(0,0,0,0.18)";
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(0), sy(0)); ctx.lineTo(sx(A.x), sy(A.y)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(0), sy(0)); ctx.lineTo(sx(B.x), sy(B.y)); ctx.stroke();

      ctx.restore();

      // points
      LR.dot(ctx, sx(A.x), sy(A.y), 8, C.orange, "#fff");
      LR.dot(ctx, sx(B.x), sy(B.y), 8, C.text, "#fff");
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "left";
      ctx.fillText("A", sx(A.x) + 11, sy(A.y) - 8);
      ctx.fillStyle = C.text;
      ctx.fillText("B", sx(B.x) + 11, sy(B.y) - 8);

      const cos = (A.x * B.x + A.y * B.y) / (Math.hypot(A.x, A.y) * Math.hypot(B.x, B.y));
      ro.set("l2", LR.fmtF(dl2, 2) + (metric === "L2" ? " ← ball shown" : ""), metric === "L2" ? C.orange : undefined);
      ro.set("l1", LR.fmtF(dl1, 2) + (metric === "L1" ? " ← ball shown" : ""), metric === "L1" ? C.orange : undefined);
      ro.set("linf", LR.fmtF(dli, 2) + (metric === "Linf" ? " ← ball shown" : ""), metric === "Linf" ? C.orange : undefined);
      ro.set("cos", LR.fmtF(cos, 3) + "  (θ = " + LR.fmtF((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI, 1) + "°)");
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.2 — cosine vs Euclidean: the e-commerce example
     ════════════════════════════════════════════════════════════ */
  LR.figs.cosine = function (mount) {
    LR.header(
      mount,
      "Who is most similar to User 1?",
      "Three shopping baskets over five items. The metric decides the answer, and the two metrics disagree."
    );

    const ITEMS = ["eggs", "flour", "sugar", "vodka", "Red Bull"];
    const USERS = [
      { name: "User 1 · the baker", v: [1, 1, 1, 0, 0], color: C.text },
      { name: "User 2 · the bulk baker", v: [100, 100, 100, 0, 0], color: C.green },
      { name: "User 3 · the partier", v: [1, 0, 0, 1, 1], color: C.purple },
    ];

    let metric = "euclid";
    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bE = LR.button(group, "Euclidean distance", function () { metric = "euclid"; bE.classList.add("on"); bC.classList.remove("on"); render(); }, "small on");
    const bC = LR.button(group, "Cosine similarity", function () { metric = "cosine"; bC.classList.add("on"); bE.classList.remove("on"); render(); }, "small");

    const basketBox = LR.el("div");
    mount.appendChild(basketBox);
    const resultBox = LR.el("div");
    mount.appendChild(resultBox);
    const msg = LR.msg(mount);

    // baskets as labelled bars (log scale for the 100s)
    (function drawBaskets() {
      let html = "";
      USERS.forEach(function (u) {
        html += "<div class='basket-row'><span class='basket-label' style='color:" + u.color + "'>" + u.name + "</span><span style='font-family:var(--mono);font-size:12px'>";
        html += u.v.map((n, i) => (n ? n + "× " + ITEMS[i] : null)).filter(Boolean).join(", ");
        html += "</span></div>";
      });
      basketBox.innerHTML = html;
    })();

    function euclid(a, b) {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
      return Math.sqrt(s);
    }
    function cossim(a, b) {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
      return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    function render() {
      const u1 = USERS[0].v;
      const d12 = euclid(u1, USERS[1].v), d13 = euclid(u1, USERS[2].v);
      const c12 = cossim(u1, USERS[1].v), c13 = cossim(u1, USERS[2].v);

      let html = "<div class='fig-readout' style='margin-top:14px'>";
      if (metric === "euclid") {
        const w2 = d12 < d13;
        html += "<span class='ro'><span class='ro-label'>dist(U1, U2)</span><span class='ro-val' style='color:" + (w2 ? C.green : C.muted) + "'>" + LR.fmtF(d12, 2) + "</span></span>";
        html += "<span class='ro'><span class='ro-label'>dist(U1, U3)</span><span class='ro-val' style='color:" + (!w2 ? C.purple : C.muted) + "'>" + LR.fmtF(d13, 2) + "</span></span>";
      } else {
        const w2 = c12 > c13;
        html += "<span class='ro'><span class='ro-label'>cossim(U1, U2)</span><span class='ro-val' style='color:" + (w2 ? C.green : C.muted) + "'>" + LR.fmtF(c12, 3) + "</span></span>";
        html += "<span class='ro'><span class='ro-label'>cossim(U1, U3)</span><span class='ro-val' style='color:" + (!w2 ? C.purple : C.muted) + "'>" + LR.fmtF(c13, 3) + "</span></span>";
      }
      html += "</div>";
      resultBox.innerHTML = html;

      if (metric === "euclid") {
        msg.show(
          "Euclidean says User 3 is closest (distance " + LR.fmtF(d13, 2) + " vs " + LR.fmtF(d12, 2) + "): the raw coordinate gaps to the partier are tiny, while the bulk baker sits ~171 units away. Magnitude dominates.",
          "info"
        );
      } else {
        msg.show(
          "Cosine says User 2 is most similar (similarity " + LR.fmtF(c12, 3) + " vs " + LR.fmtF(c13, 3) + "): identical direction, different magnitude. Same taste, bigger kitchen. For purchase profiles and text, this is usually the right question.",
          "good"
        );
      }
    }
    render();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.3 — 1-NN Voronoi diagram, live, with label corruption
     ════════════════════════════════════════════════════════════ */
  LR.figs.voronoi = function (mount) {
    LR.header(
      mount,
      "1-NN's map of the world",
      "Every training point owns a Voronoi cell. Drag points to reshape the map, drag the white query anywhere, then corrupt a label and watch the damage."
    );

    const rand = LR.rng(63);
    const pts = [];
    for (let i = 0; i < 6; i++) pts.push({ x: 0.6 + rand() * 2.6, y: 0.6 + rand() * 4.6, cls: "A" });
    for (let i = 0; i < 6; i++) pts.push({ x: 3.2 + rand() * 2.8, y: 0.8 + rand() * 4.8, cls: "B" });
    // the point we can corrupt: pick the A point nearest the class border (largest x)
    const corruptIdx = pts.reduce((best, p, i) => (p.cls === "A" && p.x > pts[best].x ? i : best), 0);
    let corrupted = false;

    let q = { x: 3.0, y: 3.0 };

    const bar = LR.controls(mount);
    const corruptBtn = LR.button(bar, "Corrupt one label 💥", function () {
      corrupted = !corrupted;
      pts[corruptIdx].cls = corrupted ? "B" : "A";
      corruptBtn.textContent = corrupted ? "Restore the label ↺" : "Corrupt one label 💥";
      msg.show(
        corrupted
          ? "One training point flipped class. Its entire Voronoi cell now answers wrongly, with full confidence, for every query inside it."
          : "Label restored. The map heals instantly, because the map is the data.",
        corrupted ? "bad" : "good"
      );
      draw();
    }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 460, {
      aria: "Live Voronoi diagram over draggable class-coloured training points with a draggable query point",
    });
    const msg = LR.msg(mount);
    const ro = LR.readout(mount, [
      { k: "cell", label: "query lands in" },
      { k: "label", label: "1-NN prediction" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6, xlabel: "x₁", ylabel: "x₂" };
    let SC = null, dragging = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        if ((p.x - SC.sx(q.x)) ** 2 + (p.y - SC.sy(q.y)) ** 2 < 800) { dragging = q; return true; }
        for (const pt of pts) {
          if ((p.x - SC.sx(pt.x)) ** 2 + (p.y - SC.sy(pt.y)) ** 2 < 700) { dragging = pt; return true; }
        }
        return false;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        dragging.x = Math.max(0.15, Math.min(6.35, d.x));
        dragging.y = Math.max(0.15, Math.min(5.85, d.y));
        draw();
      },
      up: function () { dragging = null; },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // per-block nearest-point shading (the Voronoi cells)
      const BLK = 5;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          let bi = 0, bd = Infinity;
          for (let i = 0; i < pts.length; i++) {
            const dd = (pts[i].x - wx) ** 2 + (pts[i].y - wy) ** 2;
            if (dd < bd) { bd = dd; bi = i; }
          }
          ctx.fillStyle = CLS_SOFT[pts[bi].cls];
          ctx.fillRect(px, py, BLK, BLK);
        }
      }

      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // cell borders: cheap approximation, draw perpendicular bisector edges by
      // sampling: skip (the shading already reads as cells)

      // training points
      pts.forEach(function (p, i) {
        LR.dot(ctx, sx(p.x), sy(p.y), i === corruptIdx && corrupted ? 8 : 6, CLS_COLOR[p.cls], i === corruptIdx && corrupted ? C.red : "#fff");
      });

      // query
      const nn = K.nearest(pts, q, 1)[0];
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(sx(q.x), sy(q.y)); ctx.lineTo(sx(nn.p.x), sy(nn.p.y)); ctx.stroke();
      ctx.setLineDash([]);
      LR.dot(ctx, sx(q.x), sy(q.y), 9, "#fff", C.orange);
      ctx.font = "800 12px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "center";
      ctx.fillText("?", sx(q.x), sy(q.y) + 4);

      ro.set("cell", "the cell of a class-" + nn.p.cls + " point, distance " + LR.fmtF(nn.d, 2));
      ro.set("label", "class " + nn.p.cls, CLS_COLOR[nn.p.cls]);
    }
    draw();
  };
})();
