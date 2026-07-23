/* ══════════════════════════════════════════════════════════════
   figures-a.js — Decision Trees lesson figures, sections 0–4
   Fig 0.1 walk · Fig 1.1 duality · Fig 1.2 splitchooser
   Fig 3.1 entropy · Fig 4.1 igcalc
   Shared tree core lives here as LR.dt (also used by figures-b.js).
   All entropies, gains, and regions are computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared decision-tree core ───────────────────────────────
     Feature keys are "w" (plotted on x) and "h" (plotted on y).
     For abstract datasets in figures-b.js the same keys stand in
     for x₁ and x₂. */
  const T = (LR.dt = {
    log2: (x) => Math.log(x) / Math.LN2,

    // binary entropy of a coin with P(heads) = p, in bits
    entropyP: function (p) {
      if (p <= 0 || p >= 1) return 0;
      return -(p * T.log2(p) + (1 - p) * T.log2(1 - p));
    },
    giniP: (p) => 2 * p * (1 - p),

    counts: function (pts) {
      const c = {};
      pts.forEach((p) => (c[p.cls] = (c[p.cls] || 0) + 1));
      return c;
    },
    entropyOf: function (pts) {
      const c = T.counts(pts);
      const n = pts.length;
      if (!n) return 0;
      let h = 0;
      for (const k in c) {
        const p = c[k] / n;
        if (p > 0) h -= p * T.log2(p);
      }
      return h;
    },
    giniOf: function (pts) {
      const c = T.counts(pts);
      const n = pts.length;
      if (!n) return 0;
      let g = 1;
      for (const k in c) {
        const p = c[k] / n;
        g -= p * p;
      }
      return g;
    },
    majority: function (pts) {
      const c = T.counts(pts);
      let best = null;
      for (const k in c) if (best === null || c[k] > c[best]) best = k;
      return best;
    },

    // one candidate split, fully evaluated (used by the IG calculator & code exercise)
    splitEval: function (pts, f, thr, impurity) {
      const imp = impurity || T.entropyOf;
      const n = pts.length;
      const L = pts.filter((p) => p[f] <= thr);
      const R = pts.filter((p) => p[f] > thr);
      const H = imp(pts), HL = imp(L), HR = imp(R);
      const HW = (L.length / n) * HL + (R.length / n) * HR;
      return { L, R, H, HL, HR, HW, ig: H - HW };
    },

    // exhaustive threshold search: midpoints between consecutive distinct values
    bestSplit: function (pts, feats, minLeaf, impurity) {
      const imp = impurity || T.entropyOf;
      const ml = minLeaf || 1;
      const H = imp(pts);
      const n = pts.length;
      let best = null;
      feats.forEach(function (f) {
        const vals = Array.from(new Set(pts.map((p) => p[f]))).sort((a, b) => a - b);
        for (let i = 0; i + 1 < vals.length; i++) {
          const thr = (vals[i] + vals[i + 1]) / 2;
          const L = pts.filter((p) => p[f] <= thr);
          const R = pts.filter((p) => p[f] > thr);
          if (L.length < ml || R.length < ml) continue;
          const ig = H - (L.length / n) * imp(L) - (R.length / n) * imp(R);
          if (!best || ig > best.ig + 1e-12) best = { feat: f, thr, ig };
        }
      });
      return best;
    },

    // greedy recursive builder (max information gain at every node)
    build: function (pts, opts) {
      opts = opts || {};
      const feats = opts.feats || ["w", "h"];
      const maxDepth = opts.maxDepth === undefined ? Infinity : opts.maxDepth;
      const minLeaf = opts.minLeaf || 1;
      const minGain = opts.minGain === undefined ? 1e-9 : opts.minGain;
      let nid = 0;
      function rec(sub, depth) {
        const node = {
          id: nid++,
          pts: sub,
          n: sub.length,
          counts: T.counts(sub),
          H: T.entropyOf(sub),
          label: T.majority(sub),
          depth,
          leaf: true,
        };
        if (depth >= maxDepth || node.H <= 1e-12 || sub.length < 2 * minLeaf) return node;
        const s = T.bestSplit(sub, feats, minLeaf);
        if (!s || s.ig < minGain) return node;
        node.leaf = false;
        node.feat = s.feat;
        node.thr = s.thr;
        node.ig = s.ig;
        node.left = rec(sub.filter((p) => p[s.feat] <= s.thr), depth + 1);
        node.right = rec(sub.filter((p) => p[s.feat] > s.thr), depth + 1);
        return node;
      }
      return rec(pts, 0);
    },

    // route one example root → leaf; isLeaf lets pruning treat internal nodes as leaves
    path: function (root, p, isLeaf) {
      const isL = isLeaf || ((nd) => nd.leaf);
      const out = [root];
      let nd = root;
      while (!isL(nd)) {
        nd = p[nd.feat] <= nd.thr ? nd.left : nd.right;
        out.push(nd);
      }
      return out;
    },
    predict: function (root, p, isLeaf) {
      const path = T.path(root, p, isLeaf);
      return path[path.length - 1].label;
    },
    accuracy: function (root, pts, isLeaf) {
      let right = 0;
      pts.forEach((p) => { if (T.predict(root, p, isLeaf) === p.cls) right++; });
      return pts.length ? right / pts.length : 0;
    },

    // every leaf owns an axis-aligned rectangle of feature space
    regions: function (root, b, isLeaf) {
      const isL = isLeaf || ((nd) => nd.leaf);
      const out = [];
      (function rec(nd, bb) {
        if (isL(nd)) { out.push({ x0: bb.x0, x1: bb.x1, y0: bb.y0, y1: bb.y1, node: nd }); return; }
        if (nd.feat === "w") {
          rec(nd.left, { x0: bb.x0, x1: nd.thr, y0: bb.y0, y1: bb.y1 });
          rec(nd.right, { x0: nd.thr, x1: bb.x1, y0: bb.y0, y1: bb.y1 });
        } else {
          rec(nd.left, { x0: bb.x0, x1: bb.x1, y0: bb.y0, y1: nd.thr });
          rec(nd.right, { x0: bb.x0, x1: bb.x1, y0: nd.thr, y1: bb.y1 });
        }
      })(root, b);
      return out;
    },

    /* ── canvas tree renderer (walk, duality, pruning) ─────── */
    layout: function (root, isLeaf) {
      const isL = isLeaf || ((nd) => nd.leaf);
      let leafI = 0, maxD = 0;
      (function rec(nd, d) {
        nd.__d = d;
        if (d > maxD) maxD = d;
        if (isL(nd)) { nd.__lx = leafI++; return; }
        rec(nd.left, d + 1);
        rec(nd.right, d + 1);
        nd.__lx = (nd.left.__lx + nd.right.__lx) / 2;
      })(root, 0);
      return { leaves: leafI, depth: maxD };
    },

    rr: function (ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    },

    drawTree: function (ctx, root, o) {
      const isLeaf = o.isLeaf || ((nd) => nd.leaf);
      const info = T.layout(root, isLeaf);
      const colW = o.w / Math.max(1, info.leaves);
      const boxH = o.boxH || 26;
      const levelH = info.depth === 0 ? 0 : (o.h - boxH) / info.depth;
      const hi = o.hi || new Set();
      const px = (nd) => o.x0 + (nd.__lx + 0.5) * colW;
      const py = (nd) => o.y0 + nd.__d * levelH + boxH / 2;
      const font = o.font || "600 12px Inter, sans-serif";
      const edgeFont = o.edgeFont || "600 10.5px Inter, sans-serif";

      ctx.save();

      // edges first
      (function edges(nd) {
        if (isLeaf(nd)) return;
        [nd.left, nd.right].forEach(function (ch, i) {
          const onPath = hi.has(nd) && hi.has(ch);
          ctx.strokeStyle = onPath ? C.orange : "#c9c9c9";
          ctx.lineWidth = onPath ? 2.6 : 1.4;
          ctx.beginPath();
          ctx.moveTo(px(nd), py(nd) + boxH / 2);
          ctx.lineTo(px(ch), py(ch) - boxH / 2);
          ctx.stroke();
          if (o.edgeLabels !== false) {
            ctx.font = edgeFont;
            ctx.fillStyle = onPath ? C.orange : C.faint;
            const mx = px(nd) + (px(ch) - px(nd)) * 0.45;
            const my = py(nd) + boxH / 2 + (py(ch) - boxH / 2 - (py(nd) + boxH / 2)) * 0.45;
            ctx.textAlign = i === 0 ? "right" : "left";
            ctx.fillText(i === 0 ? "no " : " yes", mx, my);
          }
        });
        edges(nd.left);
        edges(nd.right);
      })(root);

      // nodes
      (function nodes(nd) {
        const x = px(nd), y = py(nd);
        const inHi = hi.has(nd);
        const isSel = o.sel === nd;
        ctx.font = font;
        if (isLeaf(nd)) {
          const txt = o.leafText ? o.leafText(nd) : nd.label;
          const tw = ctx.measureText(txt).width;
          const bw = Math.max(o.leafMinW || 30, Math.min(colW - 4, tw + 14));
          const col = o.leafColor ? o.leafColor(nd) : C.green;
          T.rr(ctx, x - bw / 2, y - boxH / 2, bw, boxH, 7);
          ctx.fillStyle = o.leafFill ? o.leafFill(nd) : "#ffffff";
          ctx.fill();
          ctx.strokeStyle = inHi || isSel ? C.orange : col;
          ctx.lineWidth = inHi || isSel ? 2.6 : 1.6;
          ctx.stroke();
          if (bw > tw + 6) {
            ctx.fillStyle = col;
            ctx.textAlign = "center";
            ctx.fillText(txt, x, y + 4);
          }
          nd.__px = x; nd.__py = y; nd.__bw = bw; nd.__bh = boxH;
        } else {
          const txt = o.testText ? o.testText(nd) : nd.feat + " > " + LR.fmt(nd.thr, 2) + "?";
          const tw = ctx.measureText(txt).width;
          const bw = tw + 18;
          T.rr(ctx, x - bw / 2, y - boxH / 2, bw, boxH, 7);
          ctx.fillStyle = inHi ? "#f4f4f4" : "#ffffff";
          ctx.fill();
          ctx.strokeStyle = inHi || isSel ? C.orange : C.axis;
          ctx.lineWidth = inHi || isSel ? 2.6 : 1.5;
          ctx.stroke();
          ctx.fillStyle = C.text;
          ctx.textAlign = "center";
          ctx.fillText(txt, x, y + 4);
          nd.__px = x; nd.__py = y; nd.__bw = bw; nd.__bh = boxH;
          nodes(nd.left);
          nodes(nd.right);
        }
      })(root);

      ctx.restore();
      return info;
    },

    /* ── the citrus sorting dataset (deterministic) ────────── */
    citrus: null,
    CC: { lemon: "#bdbdbd", orange: "#1a1a1a" },
    CCsoft: { lemon: "rgba(189,189,189,0.16)", orange: "rgba(26,26,26,0.11)" },
    CCstrong: { lemon: "rgba(189,189,189,0.38)", orange: "rgba(26,26,26,0.30)" },
    // plot window shared by every citrus figure
    WIN: { wmin: 4.2, wmax: 8.8, hmin: 5.4, hmax: 10.6 },
  });

  // 18 oranges (wide, squat) and 14 lemons (narrow, tall), with honest overlap
  // (cluster spread tuned so no single split is pure: the greedy tree needs depth 2–3)
  T.citrus = (function () {
    const rand = LR.rng(657);
    const pts = [];
    for (let i = 0; i < 18; i++)
      pts.push({ w: 7.15 + LR.gauss(rand) * 0.62, h: 6.9 + LR.gauss(rand) * 0.62, cls: "orange" });
    for (let i = 0; i < 14; i++)
      pts.push({ w: 6.05 + LR.gauss(rand) * 0.62, h: 8.4 + LR.gauss(rand) * 0.775, cls: "lemon" });
    pts.forEach(function (p) {
      p.w = Math.max(4.4, Math.min(8.6, Math.round(p.w * 20) / 20));
      p.h = Math.max(5.6, Math.min(10.4, Math.round(p.h * 20) / 20));
    });
    return pts;
  })();

  function fruitTest(nd) {
    return (nd.feat === "w" ? "width > " : "height > ") + LR.fmtF(nd.thr, 1) + " cm?";
  }
  function fruitLeafText(nd) {
    return nd.label + " (" + (nd.counts[nd.label] || 0) + "/" + nd.n + ")";
  }

  function drawFruits(ctx, SC, pts, opts) {
    opts = opts || {};
    pts.forEach(function (p) {
      LR.dot(ctx, SC.sx(p.w), SC.sy(p.h), opts.r || 5.5, T.CC[p.cls], "#ffffff");
    });
  }

  function fruitLegend(ctx, x, y) {
    ctx.font = "600 12px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = T.CC.orange;
    ctx.fillText("● orange", x, y);
    ctx.fillStyle = T.CC.lemon;
    ctx.fillText("● lemon", x, y + 17);
  }

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — twenty-questions classifier walk
     ════════════════════════════════════════════════════════════ */
  LR.figs.walk = function (mount) {
    LR.header(
      mount,
      "Twenty questions, played by a tree",
      "Set the mystery fruit's measurements, press Classify, and watch it flow from the root to a leaf. Each node asks one question; the answer picks the branch."
    );

    // a real tree, learned greedily from the citrus data (nothing hand-placed)
    const tree = T.build(T.citrus, { maxDepth: 3, minLeaf: 3 });

    let fh = 8.4, fw = 5.6;
    let path = [], step = -1, timer = null;

    const bar = LR.controls(mount);
    LR.slider(bar, "height (cm)", 5.6, 10.4, 0.1, fh, function (v) { fh = v; resetWalk(); }, (v) => LR.fmtF(v, 1));
    LR.slider(bar, "width (cm)", 4.4, 8.6, 0.1, fw, function (v) { fw = v; resetWalk(); }, (v) => LR.fmtF(v, 1));
    LR.button(bar, "Classify ▸", classify, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Decision tree that classifies a fruit by height and width, animating the root to leaf path",
    });
    const ro = LR.readout(mount, [
      { k: "path", label: "questions asked" },
      { k: "pred", label: "prediction" },
    ]);

    function resetWalk() {
      if (timer) clearInterval(timer);
      timer = null; path = []; step = -1;
      ro.set("path", "–");
      ro.set("pred", "press Classify", C.muted);
      draw();
    }

    function classify() {
      if (timer) clearInterval(timer);
      path = T.path(tree, { h: fh, w: fw });
      if (LR.reducedMotion) { step = path.length - 1; finish(); draw(); return; }
      step = 0;
      draw(); narrate();
      timer = setInterval(function () {
        step += 1;
        if (step >= path.length - 1) { clearInterval(timer); timer = null; step = path.length - 1; finish(); }
        else narrate();
        draw();
      }, 700);
    }

    function narrate() {
      const parts = [];
      for (let i = 0; i < step; i++) {
        const nd = path[i];
        const v = nd.feat === "w" ? fw : fh;
        parts.push(fruitTest(nd).replace(" cm?", "") + "? " + (v > nd.thr ? "yes" : "no"));
      }
      ro.set("path", parts.length ? parts.join("  ·  ") : "at the root…");
      ro.set("pred", "walking…", C.muted);
    }

    function finish() {
      const leaf = path[path.length - 1];
      const parts = [];
      for (let i = 0; i < path.length - 1; i++) {
        const nd = path[i];
        const v = nd.feat === "w" ? fw : fh;
        parts.push(fruitTest(nd).replace(" cm?", "") + "? " + (v > nd.thr ? "yes" : "no"));
      }
      ro.set("path", parts.join("  ·  "));
      ro.set("pred", leaf.label + "  (leaf holds " + (leaf.counts.lemon || 0) + " lemons, " + (leaf.counts.orange || 0) + " oranges)", T.CC[leaf.label]);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const hi = new Set(path.slice(0, Math.max(0, step + 1)));
      T.drawTree(ctx, tree, {
        x0: 14, y0: 34, w: W - 28, h: H - 78,
        boxH: 30,
        font: "600 13px Inter, sans-serif",
        testText: fruitTest,
        leafText: fruitLeafText,
        leafColor: (nd) => T.CC[nd.label],
        leafFill: (nd) => T.CCsoft[nd.label],
        hi,
      });
      ctx.font = "600 12.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("mystery fruit: height " + LR.fmtF(fh, 1) + " cm, width " + LR.fmtF(fw, 1) + " cm", 14, 18);
      if (step >= 0 && step < path.length - 1) {
        const nd = path[step];
        const v = nd.feat === "w" ? fw : fh;
        ctx.fillStyle = C.orange;
        ctx.textAlign = "right";
        ctx.fillText(
          (nd.feat === "w" ? "width = " + LR.fmtF(fw, 1) : "height = " + LR.fmtF(fh, 1)) +
          " → " + (v > nd.thr ? "yes, go right" : "no, go left"),
          W - 14, 18
        );
      }
    }

    resetWalk();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — tree ↔ decision-region duality
     ════════════════════════════════════════════════════════════ */
  LR.figs.duality = function (mount) {
    LR.header(
      mount,
      "One tree, two pictures",
      "The same model drawn twice. Click a leaf on the left, or a rectangle on the right: each leaf owns exactly one axis-aligned decision region."
    );

    const tree = T.build(T.citrus, { maxDepth: 3, minLeaf: 4 });
    const B = { x0: T.WIN.wmin, x1: T.WIN.wmax, y0: T.WIN.hmin, y1: T.WIN.hmax };
    const regs = T.regions(tree, B);

    let sel = null;

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Decision tree beside the two dimensional feature space it partitions into rectangles; clicking a leaf highlights its rectangle",
    });
    const msg = LR.msg(mount);

    const TB = { x0: 10, y0: 40, w: 420, h: 340 };
    const P = { x0: 510, y0: 34, w: W - 540, h: 340, xmin: B.x0, xmax: B.x1, ymin: B.y0, ymax: B.y1, xlabel: "width (cm)", ylabel: "height (cm)" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: () => true,
      down: function (p) {
        // leaf boxes on the tree side
        let hitNode = null;
        regs.forEach(function (r) {
          const nd = r.node;
          if (nd.__px !== undefined &&
              Math.abs(p.x - nd.__px) < nd.__bw / 2 + 4 &&
              Math.abs(p.y - nd.__py) < nd.__bh / 2 + 4) hitNode = nd;
        });
        // rectangles on the region side
        if (!hitNode && SC) {
          const d = SC.inv(p.x, p.y);
          if (p.x >= P.x0 && p.x <= P.x0 + P.w && p.y >= P.y0 && p.y <= P.y0 + P.h) {
            regs.forEach(function (r) {
              if (d.x >= r.x0 && d.x <= r.x1 && d.y >= r.y0 && d.y <= r.y1) hitNode = r.node;
            });
          }
        }
        sel = hitNode === sel ? null : hitNode;
        if (sel) {
          const r = regs.find((r2) => r2.node === sel);
          msg.show(
            "Leaf ↔ region: this leaf predicts " + sel.label + " for every fruit with " +
            LR.fmtF(r.x0, 1) + " < width ≤ " + LR.fmtF(r.x1, 1) + " and " +
            LR.fmtF(r.y0, 1) + " < height ≤ " + LR.fmtF(r.y1, 1) +
            " cm. It holds " + (sel.counts.lemon || 0) + " lemons and " + (sel.counts.orange || 0) + " oranges from training.",
            "info"
          );
        } else msg.hide();
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);

      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("THE RULES (tree)", TB.x0 + 6, 22);
      ctx.fillText("THE MAP (decision regions)", P.x0, 22);

      T.drawTree(ctx, tree, {
        x0: TB.x0, y0: TB.y0, w: TB.w, h: TB.h,
        boxH: 26,
        font: "600 11.5px Inter, sans-serif",
        testText: fruitTest,
        leafText: fruitLeafText,
        leafColor: (nd) => T.CC[nd.label],
        leafFill: (nd) => T.CCsoft[nd.label],
        sel,
      });

      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      regs.forEach(function (r) {
        const x = sx(r.x0), y = sy(r.y1);
        const w2 = sx(r.x1) - sx(r.x0), h2 = sy(r.y0) - sy(r.y1);
        ctx.fillStyle = r.node === sel ? T.CCstrong[r.node.label] : T.CCsoft[r.node.label];
        ctx.fillRect(x, y, w2, h2);
        ctx.strokeStyle = r.node === sel ? C.orange : "#d5d5d5";
        ctx.lineWidth = r.node === sel ? 2.6 : 1;
        ctx.strokeRect(x, y, w2, h2);
      });

      drawFruits(ctx, SC, T.citrus, { r: 4.5 });
      fruitLegend(ctx, P.x0 + 8, P.y0 + 18);
    }

    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 1.2 — split chooser on the citrus scatter
     ════════════════════════════════════════════════════════════ */
  LR.figs.splitchooser = function (mount) {
    LR.header(
      mount,
      "One split, two groups",
      "Pick a feature, drag the threshold (on the plot or with the slider), and watch the class mix of the two groups. Which split leaves the groups purest?"
    );

    const RANGE = { w: [T.WIN.wmin, T.WIN.wmax], h: [T.WIN.hmin, T.WIN.hmax] };
    let feat = "w", frac = 0.5;
    const thr = () => RANGE[feat][0] + frac * (RANGE[feat][1] - RANGE[feat][0]);

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bW = LR.button(group, "split on width", function () { feat = "w"; bW.classList.add("on"); bH.classList.remove("on"); sl.set(frac); draw(); }, "small on");
    const bH = LR.button(group, "split on height", function () { feat = "h"; bH.classList.add("on"); bW.classList.remove("on"); sl.set(frac); draw(); }, "small");
    const sl = LR.slider(bar, "threshold", 0, 1, 0.005, frac, function (v) { frac = v; draw(); },
      (v) => LR.fmtF(RANGE[feat][0] + v * (RANGE[feat][1] - RANGE[feat][0]), 2) + " cm");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 420, {
      aria: "Citrus scatter plot with a draggable split threshold showing the class mix of the two resulting groups",
    });
    const mixBox = LR.el("div", "vote-tally");
    mount.appendChild(mixBox);

    const P = { x0: 60, y0: 22, w: W - 88, h: H - 76, xmin: RANGE.w[0], xmax: RANGE.w[1], ymin: RANGE.h[0], ymax: RANGE.h[1], xlabel: "width (cm)", ylabel: "height (cm)" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!SC) return false;
        const t = thr();
        return feat === "w" ? Math.abs(p.x - SC.sx(t)) < 16 : Math.abs(p.y - SC.sy(t)) < 16;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        const v = feat === "w" ? d.x : d.y;
        frac = Math.max(0, Math.min(1, (v - RANGE[feat][0]) / (RANGE[feat][1] - RANGE[feat][0])));
        sl.set(frac);
        draw();
      },
    });

    function mixPill(title, pts) {
      const c = T.counts(pts);
      const n = pts.length;
      const l = c.lemon || 0, o = c.orange || 0;
      const purity = n ? Math.max(l, o) / n : 0;
      let barHtml = "";
      if (n) {
        barHtml =
          "<span style='display:inline-flex;width:90px;height:10px;border-radius:5px;overflow:hidden;border:1px solid #dddddd'>" +
          "<span style='width:" + (100 * o) / n + "%;background:" + T.CC.orange + "'></span>" +
          "<span style='width:" + (100 * l) / n + "%;background:" + T.CC.lemon + "'></span></span>";
      }
      return (
        "<span class='vote-pill'><b>" + title + "</b> " + barHtml + " " +
        o + " orange · " + l + " lemon" +
        (n ? " · " + Math.round(purity * 100) + "% pure" : " · empty") + "</span>"
      );
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;
      const t = thr();
      const s = T.splitEval(T.citrus, feat, t);

      // shade the two half-planes very lightly by their majority class
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      const majL = T.majority(s.L), majR = T.majority(s.R);
      if (feat === "w") {
        if (majL) { ctx.fillStyle = T.CCsoft[majL]; ctx.fillRect(P.x0, P.y0, sx(t) - P.x0, P.h); }
        if (majR) { ctx.fillStyle = T.CCsoft[majR]; ctx.fillRect(sx(t), P.y0, P.x0 + P.w - sx(t), P.h); }
      } else {
        if (majL) { ctx.fillStyle = T.CCsoft[majL]; ctx.fillRect(P.x0, sy(t), P.w, P.y0 + P.h - sy(t)); }
        if (majR) { ctx.fillStyle = T.CCsoft[majR]; ctx.fillRect(P.x0, P.y0, P.w, sy(t) - P.y0); }
      }
      // split line
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 2.4;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      if (feat === "w") { ctx.moveTo(sx(t), P.y0); ctx.lineTo(sx(t), P.y0 + P.h); }
      else { ctx.moveTo(P.x0, sy(t)); ctx.lineTo(P.x0 + P.w, sy(t)); }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      drawFruits(ctx, SC, T.citrus);
      fruitLegend(ctx, P.x0 + 8, P.y0 + 18);

      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.text;
      if (feat === "w") {
        ctx.textAlign = "center";
        ctx.fillText("width > " + LR.fmtF(t, 2) + "?", sx(t), P.y0 - 7);
      } else {
        ctx.textAlign = "left";
        ctx.fillText("height > " + LR.fmtF(t, 2) + "?", P.x0 + P.w - 150, sy(t) - 7);
      }

      const name = feat === "w" ? "width" : "height";
      mixBox.innerHTML =
        mixPill(name + " ≤ " + LR.fmtF(t, 2), s.L) +
        mixPill(name + " > " + LR.fmtF(t, 2), s.R);
    }

    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the binary entropy curve, with two draggable coins
     ════════════════════════════════════════════════════════════ */
  LR.figs.entropy = function (mount) {
    LR.header(
      mount,
      "How uncertain is a coin?",
      "Two coins on the entropy curve. Drag them along the p axis. The more loaded the coin, the closer its entropy sits to zero."
    );

    const coins = [
      { name: "coin A", p: 0.5, color: C.orange },
      { name: "coin B", p: 0.9, color: C.purple },
    ];

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Binary entropy curve with two draggable markers representing coins of different bias",
    });
    const ro = LR.readout(mount, [
      { k: "a", label: "coin A" },
      { k: "b", label: "coin B" },
      { k: "verdict", label: "which is more loaded?" },
    ]);

    const P = { x0: 66, y0: 24, w: W - 100, h: H - 86, xmin: 0, xmax: 1, ymin: 0, ymax: 1.04, xlabel: "p = P(heads)", ylabel: "entropy H(p) in bits" };
    let SC = null, dragging = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!SC) return false;
        for (const c of coins) {
          const cx = SC.sx(c.p), cy = SC.sy(T.entropyP(c.p));
          if ((p.x - cx) ** 2 + (p.y - cy) ** 2 < 900) { dragging = c; return true; }
        }
        return false;
      },
      move: function (p) {
        if (!dragging) return;
        const d = SC.inv(p.x, p.y);
        dragging.p = Math.max(0, Math.min(1, d.x));
        draw();
      },
      up: function () { dragging = null; },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // the curve
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const p = i / 300;
        const y = T.entropyP(p);
        i === 0 ? ctx.moveTo(sx(p), sy(y)) : ctx.lineTo(sx(p), sy(y));
      }
      ctx.stroke();

      // peak annotation
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = C.faint;
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(0.5), sy(0)); ctx.lineTo(sx(0.5), sy(1)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.textAlign = "center";
      ctx.fillText("fair coin: 1 bit, maximum uncertainty", sx(0.5), sy(1) - 10);

      // coins
      coins.forEach(function (c) {
        const hp = T.entropyP(c.p);
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = c.color;
        ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(sx(c.p), sy(0)); ctx.lineTo(sx(c.p), sy(hp)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(sx(0), sy(hp)); ctx.lineTo(sx(c.p), sy(hp)); ctx.stroke();
        ctx.setLineDash([]);
        LR.dot(ctx, sx(c.p), sy(hp), 9, c.color, "#ffffff");
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillStyle = c.color;
        ctx.textAlign = "center";
        ctx.fillText(c.name, sx(c.p), sy(hp) - 14);
      });

      const hA = T.entropyP(coins[0].p), hB = T.entropyP(coins[1].p);
      ro.set("a", "p = " + LR.fmtF(coins[0].p, 2) + " → H = " + LR.fmtF(hA, 3) + " bits", coins[0].color);
      ro.set("b", "p = " + LR.fmtF(coins[1].p, 2) + " → H = " + LR.fmtF(hB, 3) + " bits", coins[1].color);
      const diff = Math.abs(hA - hB);
      if (diff < 0.005) ro.set("verdict", "equally uncertain", C.muted);
      else {
        const loaded = hA < hB ? coins[0] : coins[1];
        ro.set("verdict", loaded.name + " (lower entropy = more predictable)", loaded.color);
      }
    }

    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — the information-gain calculator
     ════════════════════════════════════════════════════════════ */
  LR.figs.igcalc = function (mount) {
    LR.header(
      mount,
      "Hunt for the maximum-gain split",
      "Same scatter, now with the full computation overlaid. Every number updates as you drag. Find the split a greedy learner would choose."
    );

    const RANGE = { w: [T.WIN.wmin, T.WIN.wmax], h: [T.WIN.hmin, T.WIN.hmax] };
    let feat = "w", frac = 0.5;
    const thr = () => RANGE[feat][0] + frac * (RANGE[feat][1] - RANGE[feat][0]);

    // the true optimum over both features (what greedy would pick)
    const OPT = T.bestSplit(T.citrus, ["w", "h"]);
    let bestFound = 0;

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bW = LR.button(group, "split on width", function () { feat = "w"; bW.classList.add("on"); bH.classList.remove("on"); sl.set(frac); draw(); }, "small on");
    const bH = LR.button(group, "split on height", function () { feat = "h"; bH.classList.add("on"); bW.classList.remove("on"); sl.set(frac); draw(); }, "small");
    const sl = LR.slider(bar, "threshold", 0, 1, 0.005, frac, function (v) { frac = v; draw(); },
      (v) => LR.fmtF(RANGE[feat][0] + v * (RANGE[feat][1] - RANGE[feat][0]), 2) + " cm");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 460, {
      aria: "Citrus scatter with a draggable split and a live information gain computation, including an information gain meter",
    });
    const ro = LR.readout(mount, [
      { k: "hy", label: "H(Y) before" },
      { k: "hl", label: "H(left)" },
      { k: "hr", label: "H(right)" },
      { k: "hw", label: "H(Y | split)" },
      { k: "ig", label: "IG" },
    ]);
    const msg = LR.msg(mount);

    const P = { x0: 60, y0: 22, w: W - 88, h: H - 128, xmin: RANGE.w[0], xmax: RANGE.w[1], ymin: RANGE.h[0], ymax: RANGE.h[1], xlabel: "width (cm)", ylabel: "height (cm)" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: function (p) {
        if (!SC) return false;
        const t = thr();
        return feat === "w" ? Math.abs(p.x - SC.sx(t)) < 16 : Math.abs(p.y - SC.sy(t)) < 16;
      },
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        const v = feat === "w" ? d.x : d.y;
        frac = Math.max(0, Math.min(1, (v - RANGE[feat][0]) / (RANGE[feat][1] - RANGE[feat][0])));
        sl.set(frac);
        draw();
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;
      const t = thr();
      const s = T.splitEval(T.citrus, feat, t);

      // split line
      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 2.4;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      if (feat === "w") { ctx.moveTo(sx(t), P.y0); ctx.lineTo(sx(t), P.y0 + P.h); }
      else { ctx.moveTo(P.x0, sy(t)); ctx.lineTo(P.x0 + P.w, sy(t)); }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      drawFruits(ctx, SC, T.citrus);
      fruitLegend(ctx, P.x0 + 8, P.y0 + 18);

      // per-side entropy tags
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillStyle = C.text;
      if (feat === "w") {
        ctx.textAlign = "right";
        ctx.fillText("H = " + LR.fmtF(s.HL, 3) + " (" + s.L.length + " fruits)", Math.max(P.x0 + 120, sx(t) - 8), P.y0 + P.h - 10);
        ctx.textAlign = "left";
        ctx.fillText("H = " + LR.fmtF(s.HR, 3) + " (" + s.R.length + " fruits)", Math.min(P.x0 + P.w - 130, sx(t) + 8), P.y0 + P.h - 10);
      } else {
        ctx.textAlign = "left";
        ctx.fillText("H = " + LR.fmtF(s.HR, 3) + " (" + s.R.length + " fruits)", P.x0 + P.w - 150, Math.max(P.y0 + 46, sy(t) - 10));
        ctx.fillText("H = " + LR.fmtF(s.HL, 3) + " (" + s.L.length + " fruits)", P.x0 + P.w - 150, Math.min(P.y0 + P.h - 8, sy(t) + 18));
      }

      // IG meter along the bottom, scaled to the global optimum
      const mx = P.x0, my = P.y0 + P.h + 52, mw = P.w, mh = 14;
      ctx.fillStyle = "#ececec";
      ctx.fillRect(mx, my, mw, mh);
      const fracIG = Math.max(0, Math.min(1, s.ig / OPT.ig));
      ctx.fillStyle = fracIG > 0.985 ? C.green : C.orange;
      ctx.fillRect(mx, my, mw * fracIG, mh);
      ctx.strokeStyle = C.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(mx, my, mw, mh);
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("IG meter (full bar = the best split anywhere: IG = " + LR.fmtF(OPT.ig, 3) + " bits)", mx, my - 6);

      if (s.ig > bestFound) bestFound = s.ig;
      ro.set("hy", LR.fmtF(s.H, 4) + " bits");
      ro.set("hl", LR.fmtF(s.HL, 4) + "  (n=" + s.L.length + ")");
      ro.set("hr", LR.fmtF(s.HR, 4) + "  (n=" + s.R.length + ")");
      ro.set("hw", LR.fmtF(s.HW, 4) + " bits");
      ro.set("ig", LR.fmtF(s.ig, 4) + " bits", C.orange);

      if (s.ig >= OPT.ig - 0.004) {
        msg.show(
          "✓ Found it. " + (feat === "w" ? "width" : "height") + " > " + LR.fmtF(t, 2) +
          " is (within rounding) the maximum-gain split: IG = " + LR.fmtF(s.ig, 3) +
          " bits. This is exactly the split a greedy learner takes first.",
          "good"
        );
      } else {
        msg.show(
          "Best you have found so far: IG = " + LR.fmtF(bestFound, 3) +
          " bits. The maximum over both features is " + LR.fmtF(OPT.ig, 3) +
          " bits. Keep dragging, and try both features.",
          "info"
        );
      }
    }

    draw();
  };
})();
