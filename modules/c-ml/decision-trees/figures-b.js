/* ══════════════════════════════════════════════════════════════
   figures-b.js — Decision Trees lesson figures, sections 5–6
   Fig 5.1 builder · Fig 5.2 codeex · Fig 6.1 depth
   Fig 6.2 pruning · Fig 6.3 gini · Fig 6.4 knnvstree
   Uses the shared tree core (LR.dt) from figures-a.js.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const T = LR.dt;

  const CLS_COLOR = { A: C.green, B: C.purple };
  const CLS_SOFT = { A: "rgba(47,158,68,0.18)", B: "rgba(112,72,232,0.15)" };

  /* ── the restaurant "will we wait?" dataset ──────────────────
     The slides use the classic 12-example restaurant table and
     quote IG(Type) = 0 and IG(Patrons) ≈ 0.541. We keep four of
     the classic attributes (Patrons and Type appear in the
     lecture; Hungry and Fri/Sat are retained from the classic
     table as extra split candidates so the recursion can finish;
     flagged: the full row-level table is the standard one behind
     the slide numbers, not printed in the source file). */
  const REST = [
    { i: 1,  Patrons: "Some", Type: "French",  Hungry: "Yes", Fri: "No",  cls: "Yes" },
    { i: 2,  Patrons: "Full", Type: "Thai",    Hungry: "Yes", Fri: "No",  cls: "No"  },
    { i: 3,  Patrons: "Some", Type: "Burger",  Hungry: "No",  Fri: "No",  cls: "Yes" },
    { i: 4,  Patrons: "Full", Type: "Thai",    Hungry: "Yes", Fri: "Yes", cls: "Yes" },
    { i: 5,  Patrons: "Full", Type: "French",  Hungry: "No",  Fri: "Yes", cls: "No"  },
    { i: 6,  Patrons: "Some", Type: "Italian", Hungry: "Yes", Fri: "No",  cls: "Yes" },
    { i: 7,  Patrons: "None", Type: "Burger",  Hungry: "No",  Fri: "No",  cls: "No"  },
    { i: 8,  Patrons: "Some", Type: "Thai",    Hungry: "Yes", Fri: "No",  cls: "Yes" },
    { i: 9,  Patrons: "Full", Type: "Burger",  Hungry: "No",  Fri: "Yes", cls: "No"  },
    { i: 10, Patrons: "Full", Type: "Italian", Hungry: "Yes", Fri: "Yes", cls: "No"  },
    { i: 11, Patrons: "None", Type: "Thai",    Hungry: "No",  Fri: "No",  cls: "No"  },
    { i: 12, Patrons: "Full", Type: "Burger",  Hungry: "Yes", Fri: "Yes", cls: "Yes" },
  ];
  const RFEATS = {
    Patrons: ["None", "Some", "Full"],
    Type: ["French", "Italian", "Thai", "Burger"],
    Hungry: ["Yes", "No"],
    Fri: ["Yes", "No"],
  };
  const RNAME = { Patrons: "Patrons", Type: "Type", Hungry: "Hungry", Fri: "Fri/Sat" };
  const YN_COLOR = { Yes: C.green, No: C.red };
  const YN_SOFT = { Yes: "rgba(47,158,68,0.14)", No: "rgba(224,49,49,0.10)" };

  // expected conditional entropy of a categorical split, on real counts
  function catIG(rows, feat) {
    const H = T.entropyOf(rows);
    const groups = {};
    rows.forEach(function (r) { (groups[r[feat]] = groups[r[feat]] || []).push(r); });
    let hw = 0;
    for (const v in groups) hw += (groups[v].length / rows.length) * T.entropyOf(groups[v]);
    return { ig: H - hw, hw, groups, H };
  }

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — the greedy tree builder (restaurant example)
     ════════════════════════════════════════════════════════════ */
  LR.figs.builder = function (mount) {
    LR.header(
      mount,
      "Build the tree yourself, one split at a time",
      "The orange node is the one being decided. Every candidate feature shows its real information gain on that node's examples. Pick one, or let greedy pick."
    );

    let root, queue;

    const bar = LR.controls(mount);
    const featBar = LR.el("div", "fig-controls");
    const autoBtn = LR.button(bar, "Auto-greedy step ▸", autoStep, "primary");
    const allBtn = LR.button(bar, "Auto-build all ▸▸", function () { let guard = 0; while (queue.length && guard++ < 20) autoStep(); });
    LR.button(bar, "Reset ⟲", reset);
    mount.appendChild(featBar);

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Growing decision tree over the twelve restaurant examples, with information gain shown for each candidate split",
    });
    const ro = LR.readout(mount, [
      { k: "node", label: "deciding node" },
      { k: "mix", label: "examples there" },
      { k: "H", label: "entropy there" },
    ]);
    const msg = LR.msg(mount);

    function mkNode(rows, used, edge) {
      return {
        rows, used, edge: edge || "root",
        n: rows.length,
        counts: T.counts(rows),
        H: T.entropyOf(rows),
        label: T.majority(rows),
        leaf: false, feat: null, kids: null,
      };
    }

    function reset() {
      root = mkNode(REST, []);
      queue = [root];
      msg.hide();
      update();
    }

    function unusedFeats(node) {
      return Object.keys(RFEATS).filter((f) => node.used.indexOf(f) === -1);
    }

    function splitCurrent(feat) {
      const node = queue.shift();
      const { ig, groups } = catIG(node.rows, feat);
      node.feat = feat;
      node.kids = [];
      RFEATS[feat].forEach(function (v) {
        if (!groups[v]) return;
        const ch = mkNode(groups[v], node.used.concat([feat]), RNAME[feat] + " = " + v);
        node.kids.push({ val: v, node: ch });
        if (ch.H <= 1e-12 || unusedFeats(ch).length === 0) ch.leaf = true;
        else queue.push(ch);
      });
      if (feat === "Type" && node === root) {
        msg.show(
          "IG(Type) = 0.000 bits: French, Italian, Thai, and Burger are each still a 50/50 mix, exactly as uncertain as the parent. A legal split, but a useless one.",
          "bad"
        );
      } else {
        msg.show(
          "Split on " + RNAME[feat] + ": IG = " + LR.fmtF(ig, 3) + " bits. " +
          (queue.length ? "Next impure node is highlighted." : "Every leaf is pure. The tree is finished."),
          ig > 0.2 ? "good" : "info"
        );
      }
      update();
    }

    function autoStep() {
      if (!queue.length) return;
      const node = queue[0];
      let best = null;
      unusedFeats(node).forEach(function (f) {
        const g = catIG(node.rows, f);
        if (!best || g.ig > best.ig + 1e-12) best = { f, ig: g.ig };
      });
      if (best) splitCurrent(best.f);
    }

    function update() {
      // candidate feature buttons for the current node
      featBar.innerHTML = "";
      const cur = queue[0] || null;
      if (cur) {
        const lab = LR.el("span", "", "<b>split this node on:</b>");
        lab.style.cssText = "font-family:var(--sans);font-size:13px";
        featBar.appendChild(lab);
        unusedFeats(cur).forEach(function (f) {
          const g = catIG(cur.rows, f);
          LR.button(featBar, RNAME[f] + " · IG " + LR.fmtF(g.ig, 3), function () { splitCurrent(f); }, "small");
        });
        ro.set("node", cur.edge, C.orange);
        ro.set("mix", (cur.counts.Yes || 0) + " Yes / " + (cur.counts.No || 0) + " No");
        ro.set("H", LR.fmtF(cur.H, 3) + " bits");
      } else {
        const lab = LR.el("span", "", "<b>done:</b> every leaf is pure, no node left to split.");
        lab.style.cssText = "font-family:var(--sans);font-size:13px";
        featBar.appendChild(lab);
        ro.set("node", "none (finished)", C.green);
        ro.set("mix", "–");
        ro.set("H", "–");
      }
      autoBtn.disabled = !cur;
      allBtn.disabled = !cur;
      draw();
    }

    // multiway tree layout + draw (categorical, so not the shared binary renderer)
    function draw() {
      ctx.clearRect(0, 0, W, H);
      const cur = queue[0] || null;

      // layout: leaves get columns
      let leafI = 0, maxD = 0;
      (function lay(nd, d) {
        nd.__d = d;
        if (d > maxD) maxD = d;
        if (!nd.kids) { nd.__lx = leafI++; return; }
        nd.kids.forEach((k) => lay(k.node, d + 1));
        nd.__lx = (nd.kids[0].node.__lx + nd.kids[nd.kids.length - 1].node.__lx) / 2;
      })(root, 0);

      const colW = (W - 20) / Math.max(1, leafI);
      const boxH = 30;
      const levelH = maxD === 0 ? 0 : (H - 40 - boxH) / maxD;
      const px = (nd) => 10 + (nd.__lx + 0.5) * colW;
      const py = (nd) => 22 + nd.__d * levelH + boxH / 2;

      // edges
      (function edges(nd) {
        if (!nd.kids) return;
        nd.kids.forEach(function (k) {
          ctx.strokeStyle = "#c9c9c9";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(px(nd), py(nd) + boxH / 2);
          ctx.lineTo(px(k.node), py(k.node) - boxH / 2);
          ctx.stroke();
          const mx = px(nd) + (px(k.node) - px(nd)) * 0.5;
          const my = py(nd) + boxH / 2 + (py(k.node) - boxH / 2 - (py(nd) + boxH / 2)) * 0.55;
          ctx.font = "600 10.5px Inter, sans-serif";
          ctx.fillStyle = C.muted;
          ctx.textAlign = "center";
          ctx.fillText(k.val, mx, my);
          edges(k.node);
        });
      })(root);

      // nodes
      (function nodes(nd) {
        const x = px(nd), y = py(nd);
        const isCur = nd === cur;
        ctx.font = "600 12px Inter, sans-serif";
        let txt, fill, stroke, txtCol, dashed = false;
        if (nd.kids) {
          txt = RNAME[nd.feat] + "?";
          fill = "#fff"; stroke = C.axis; txtCol = C.text;
        } else if (nd.leaf) {
          txt = nd.label + " (" + nd.n + ")";
          fill = YN_SOFT[nd.label]; stroke = YN_COLOR[nd.label]; txtCol = YN_COLOR[nd.label];
        } else {
          txt = (nd.counts.Yes || 0) + " Y / " + (nd.counts.No || 0) + " N";
          fill = "#fff"; stroke = "#aaaaaa"; txtCol = C.muted; dashed = true;
        }
        const tw = ctx.measureText(txt).width;
        const bw = Math.max(40, tw + 16);
        if (dashed) ctx.setLineDash([4, 3]);
        T.rr(ctx, x - bw / 2, y - boxH / 2, bw, boxH, 7);
        ctx.fillStyle = isCur ? "#fff2e8" : fill;
        ctx.fill();
        ctx.strokeStyle = isCur ? C.orange : stroke;
        ctx.lineWidth = isCur ? 2.8 : 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = isCur ? C.orange : txtCol;
        ctx.textAlign = "center";
        ctx.fillText(txt, x, y + 4);
        if (nd.kids) nd.kids.forEach((k) => nodes(k.node));
      })(root);
    }

    reset();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — interactive information_gain code exercise
     ════════════════════════════════════════════════════════════ */
  LR.figs.codeex = function (mount) {
    LR.header(
      mount,
      "Run information_gain yourself",
      "The function below really executes against the seven measured fruits. Slide the threshold, press Run, and read the actual numbers."
    );

    // the tiny dataset from the worked example: 5 oranges, 2 lemons, by height
    const FRUITS = [
      { h: 6.2, cls: "orange" }, { h: 6.5, cls: "orange" }, { h: 6.8, cls: "orange" },
      { h: 6.9, cls: "orange" }, { h: 7.4, cls: "orange" },
      { h: 8.6, cls: "lemon" }, { h: 9.4, cls: "lemon" },
    ];
    // the best achievable IG for this data (checked live against the reader's split)
    const BEST = T.bestSplit(FRUITS, ["h"]);

    const CODE =
      "heights = np.array([6.2, 6.5, 6.8, 6.9, 7.4, 8.6, 9.4])\n" +
      "labels  = np.array(['O', 'O', 'O', 'O', 'O', 'L', 'L'])   # 5 oranges, 2 lemons\n" +
      "\n" +
      "def entropy(labels):\n" +
      "    _, counts = np.unique(labels, return_counts=True)\n" +
      "    p = counts / counts.sum()\n" +
      "    return -(p * np.log2(p)).sum()\n" +
      "\n" +
      "def information_gain(labels, left_mask):\n" +
      "    n = len(labels)\n" +
      "    left, right = labels[left_mask], labels[~left_mask]\n" +
      "    h_split = len(left)/n * entropy(left) + len(right)/n * entropy(right)\n" +
      "    return entropy(labels) - h_split\n" +
      "\n" +
      "information_gain(labels, heights <= threshold)";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);

    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);
    let thr = 7.0;
    LR.slider(controls, "threshold (cm)", 6.0, 10.0, 0.1, thr, (v) => { thr = v; }, (v) => LR.fmtF(v, 1));
    LR.button(controls, "Run ▸", run, "primary small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; information_gain(labels, heights &lt;= 7.0)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    // mini strip: the seven fruits on a height axis with the threshold
    const { cv, ctx, W, H } = LR.canvas(mount, 560, 130, {
      aria: "The seven fruits on a height number line with the chosen threshold",
    });
    cv.style.marginTop = "14px";
    const P = { x0: 40, y0: 12, w: W - 66, h: H - 58, xmin: 5.8, xmax: 9.8, ymin: 0, ymax: 1, xlabel: "height (cm)", yticks: [] };

    function labset(pts) {
      return "[" + pts.map((p) => "'" + (p.cls === "orange" ? "O" : "L") + "'").join(", ") + "]";
    }

    function run() {
      const s = T.splitEval(FRUITS, "h", thr);
      const atMax = s.ig >= BEST.ig - 1e-9;
      out.innerHTML =
        "&gt;&gt;&gt; information_gain(labels, heights &lt;= " + LR.fmtF(thr, 1) + ")\n" +
        "left  (h ≤ " + LR.fmtF(thr, 1) + "): " + labset(s.L) + "  H = " + LR.fmtF(s.HL, 4) + "\n" +
        "right (h &gt; " + LR.fmtF(thr, 1) + "): " + labset(s.R) + "  H = " + LR.fmtF(s.HR, 4) + "\n" +
        "H(parent) = " + LR.fmtF(s.H, 4) + ",  weighted children = " + LR.fmtF(s.HW, 4) + "\n" +
        "<b>" + LR.fmtF(s.ig, 4) + "</b>" +
        (atMax
          ? "   # maximum possible on this data: greedy splits exactly here"
          : "   # best achievable is " + LR.fmtF(BEST.ig, 4) + ", keep hunting");
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const SC = LR.plot(ctx, P);
      FRUITS.forEach(function (p) {
        LR.dot(ctx, SC.sx(p.h), SC.sy(0.5), 7, T.CC[p.cls], "#fff");
      });
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(SC.sx(thr), P.y0);
      ctx.lineTo(SC.sx(thr), P.y0 + P.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "center";
      ctx.fillText("threshold", SC.sx(thr), P.y0 + 10);
    }

    draw();
  };

  /* ── deterministic two-blob dataset with optional label noise ──
     Training sets get a small flip rate so an unconstrained tree
     really does memorize noise; validation sets are clean. */
  function blobs(seed, n, flip) {
    const rand = LR.rng(seed);
    const centres = [
      { w: 2.0, h: 2.4, cls: "A" },
      { w: 4.3, h: 3.9, cls: "B" },
    ];
    const pts = [];
    for (let i = 0; i < n; i++) {
      const c = centres[i % 2];
      const cls = rand() < (flip || 0) ? (c.cls === "A" ? "B" : "A") : c.cls;
      pts.push({
        w: Math.max(0.1, Math.min(6.3, c.w + LR.gauss(rand) * 0.85)),
        h: Math.max(0.1, Math.min(6.3, c.h + LR.gauss(rand) * 0.85)),
        cls,
      });
    }
    return pts;
  }
  const BOUNDS = { x0: 0, x1: 6.4, y0: 0, y1: 6.4 };

  function drawRegions(ctx, SC, regs, P) {
    regs.forEach(function (r) {
      const x = SC.sx(r.x0), y = SC.sy(r.y1);
      ctx.fillStyle = CLS_SOFT[r.node.label];
      ctx.fillRect(x, y, SC.sx(r.x1) - x, SC.sy(r.y0) - y);
      ctx.strokeStyle = "#d8d8d8";
      ctx.lineWidth = 0.8;
      ctx.strokeRect(x, y, SC.sx(r.x1) - x, SC.sy(r.y0) - y);
    });
  }

  function countLeaves(nd) {
    return nd.leaf ? 1 : countLeaves(nd.left) + countLeaves(nd.right);
  }

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — the depth dial: train vs validation
     ════════════════════════════════════════════════════════════ */
  LR.figs.depth = function (mount) {
    LR.header(
      mount,
      "The depth dial",
      "Slide the maximum depth and watch the decision regions fracture. Then, judging only by the curves, click the depth you would ship."
    );

    const TRAIN = blobs(31, 120, 0.07); // 7% of training labels flipped: class noise
    const VAL = blobs(77, 80, 0);       // clean validation set
    const DMAX = 10;

    // real trees at every depth, built once
    const trees = [null];
    const accTr = [null], accVl = [null], leaves = [null];
    for (let d = 1; d <= DMAX; d++) {
      const t = T.build(TRAIN, { maxDepth: d, minLeaf: 1 });
      trees.push(t);
      accTr.push(T.accuracy(t, TRAIN));
      accVl.push(T.accuracy(t, VAL));
      leaves.push(countLeaves(t));
    }
    let bestD = 1;
    for (let d = 2; d <= DMAX; d++) if (accVl[d] > accVl[bestD] + 1e-12) bestD = d;

    let depth = 2, picked = null;

    const bar = LR.controls(mount);
    LR.slider(bar, "max depth", 1, DMAX, 1, depth, function (v) { depth = Math.round(v); update(); }, (v) => String(Math.round(v)));

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Decision regions (train points shown)"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Accuracy vs depth · click your pick"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 330, { aria: "Decision region map of the tree at the chosen maximum depth" });
    const RC = LR.canvas(rb, 400, 330, { aria: "Training and validation accuracy versus tree depth; click a depth to commit to it" });

    const ro = LR.readout(mount, [
      { k: "tr", label: "train accuracy" },
      { k: "vl", label: "validation accuracy" },
      { k: "lv", label: "leaves" },
    ]);
    const msg = LR.msg(mount);

    const PM = { x0: 44, y0: 14, w: 400 - 62, h: 330 - 56, xmin: 0, xmax: 6.4, ymin: 0, ymax: 6.4, xlabel: "x₁", ylabel: "x₂" };
    const PC = { x0: 46, y0: 16, w: 400 - 66, h: 330 - 62, xmin: 0.5, xmax: DMAX + 0.5, ymin: 0.5, ymax: 1.02, xlabel: "max depth", ylabel: "accuracy", xticks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] };
    let SCc = null;

    LR.drag(RC.cv, RC.W, RC.H, {
      hit: (p) => SCc && p.x > PC.x0 && p.x < PC.x0 + PC.w && p.y > PC.y0 && p.y < PC.y0 + PC.h,
      down: function (p) {
        picked = Math.max(1, Math.min(DMAX, Math.round(SCc.inv(p.x, p.y).x)));
        if (picked === bestD || Math.abs(accVl[picked] - accVl[bestD]) < 0.005) {
          msg.show(
            "✓ Depth " + picked + " is (within a hair) the validation optimum: " +
            LR.fmtF(accVl[picked] * 100, 1) + "% validation accuracy vs " +
            LR.fmtF(accTr[picked] * 100, 1) + "% train. Deeper trees only memorize noise.",
            "good"
          );
        } else {
          msg.show(
            "You picked depth " + picked + " (validation " + LR.fmtF(accVl[picked] * 100, 1) +
            "%). The computed optimum is depth " + bestD + " at " + LR.fmtF(accVl[bestD] * 100, 1) +
            "%. Note how train accuracy keeps climbing anyway: it is a flattering lie.",
            "info"
          );
        }
        update();
      },
    });

    function update() {
      // region map
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      const SC = LR.plot(ctx, PM);
      drawRegions(ctx, SC, T.regions(trees[depth], BOUNDS), PM);
      TRAIN.forEach((p) => LR.dot(ctx, SC.sx(p.w), SC.sy(p.h), 3.4, CLS_COLOR[p.cls], "#fff"));
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.text;
      ctx.textAlign = "left";
      ctx.fillText("depth " + depth + " · " + leaves[depth] + " leaves", PM.x0 + 8, PM.y0 + 16);

      // accuracy chart
      const c2 = RC.ctx;
      c2.clearRect(0, 0, RC.W, RC.H);
      SCc = LR.plot(c2, PC);
      [["tr", accTr, C.green], ["vl", accVl, C.orange]].forEach(function (row) {
        const arr = row[1], col = row[2];
        c2.strokeStyle = col; c2.lineWidth = 2.2;
        c2.beginPath();
        for (let d = 1; d <= DMAX; d++) d === 1 ? c2.moveTo(SCc.sx(d), SCc.sy(arr[d])) : c2.lineTo(SCc.sx(d), SCc.sy(arr[d]));
        c2.stroke();
        for (let d = 1; d <= DMAX; d++) LR.dot(c2, SCc.sx(d), SCc.sy(arr[d]), 3, col);
      });
      c2.font = "600 11.5px Inter, sans-serif";
      c2.textAlign = "left";
      c2.fillStyle = C.green; c2.fillText("— train", PC.x0 + PC.w - 92, PC.y0 + 16);
      c2.fillStyle = C.orange; c2.fillText("— validation", PC.x0 + PC.w - 92, PC.y0 + 32);

      // current depth marker
      c2.strokeStyle = C.faint; c2.lineWidth = 1.3; c2.setLineDash([3, 3]);
      c2.beginPath(); c2.moveTo(SCc.sx(depth), SCc.sy(PC.ymin)); c2.lineTo(SCc.sx(depth), SCc.sy(PC.ymax)); c2.stroke();
      c2.setLineDash([]);
      if (picked !== null) {
        c2.strokeStyle = C.text; c2.lineWidth = 1.6; c2.setLineDash([5, 3]);
        c2.beginPath(); c2.moveTo(SCc.sx(picked), SCc.sy(PC.ymin)); c2.lineTo(SCc.sx(picked), SCc.sy(PC.ymax)); c2.stroke();
        c2.setLineDash([]);
        c2.fillStyle = C.text;
        c2.fillText("your pick", SCc.sx(picked) + 5, PC.y0 + 14);
      } else {
        c2.font = "700 12.5px Inter, sans-serif";
        c2.fillStyle = C.muted; c2.textAlign = "center";
        c2.fillText("click the depth you would ship", PC.x0 + PC.w / 2, PC.y0 + PC.h / 2 + 30);
      }

      ro.set("tr", LR.fmtF(accTr[depth] * 100, 1) + "%", C.green);
      ro.set("vl", LR.fmtF(accVl[depth] * 100, 1) + "%", C.orange);
      ro.set("lv", String(leaves[depth]));
    }

    update();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.2 — prune the overgrown tree
     ════════════════════════════════════════════════════════════ */
  LR.figs.pruning = function (mount) {
    LR.header(
      mount,
      "Prune it back to health",
      "This tree was grown too deep on noisy data. Click any internal node to collapse it into a leaf (click again to restore). Watch validation accuracy respond."
    );

    const TRAIN = blobs(8, 110, 0.08); // noisy training labels: the tree overgrows
    const VAL = blobs(64, 80, 0);      // clean validation set
    const tree = T.build(TRAIN, { maxDepth: 6, minLeaf: 2 });

    const pruned = new Set();
    const isLeaf = (nd) => nd.leaf || pruned.has(nd.id);
    const baseVl = T.accuracy(tree, VAL);
    const baseTr = T.accuracy(tree, TRAIN);

    const bar = LR.controls(mount);
    LR.button(bar, "Hint: best single prune 💡", hint);
    LR.button(bar, "Reset ⟲", function () { pruned.clear(); msg.hide(); update(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Overgrown decision tree whose internal nodes can be clicked to prune them, with live train and validation accuracy",
    });
    const ro = LR.readout(mount, [
      { k: "lv", label: "leaves" },
      { k: "tr", label: "train acc" },
      { k: "vl", label: "validation acc" },
      { k: "base", label: "unpruned validation" },
    ]);
    const msg = LR.msg(mount);

    function reachable() {
      const out = [];
      (function rec(nd) {
        out.push(nd);
        if (!isLeaf(nd)) { rec(nd.left); rec(nd.right); }
      })(tree);
      return out;
    }

    LR.drag(cv, W, H, {
      hit: () => true,
      down: function (p) {
        let hit = null;
        reachable().forEach(function (nd) {
          if (nd.__px !== undefined &&
              Math.abs(p.x - nd.__px) < nd.__bw / 2 + 3 &&
              Math.abs(p.y - nd.__py) < nd.__bh / 2 + 3) hit = nd;
        });
        if (!hit) return;
        if (hit.leaf) { msg.show("That node is already a true leaf; there is nothing to prune.", "info"); return; }
        if (pruned.has(hit.id)) pruned.delete(hit.id);
        else pruned.add(hit.id);
        const vl = T.accuracy(tree, VAL, isLeaf);
        if (vl > baseVl + 1e-9) {
          msg.show(
            "Validation accuracy is now " + LR.fmtF(vl * 100, 1) + "%, up from " + LR.fmtF(baseVl * 100, 1) +
            "% unpruned. You deleted memorized noise, not signal.",
            "good"
          );
        } else if (vl < baseVl - 1e-9) {
          msg.show(
            "Validation accuracy fell to " + LR.fmtF(vl * 100, 1) + "%. That branch was carrying real structure; restore it.",
            "bad"
          );
        } else {
          msg.show("No change in validation accuracy. That branch was neutral.", "info");
        }
        update();
      },
    });

    function hint() {
      // try pruning each currently-unpruned internal node; report the best single addition
      let best = null;
      reachable().forEach(function (nd) {
        if (isLeaf(nd)) return;
        pruned.add(nd.id);
        const vl = T.accuracy(tree, VAL, isLeaf);
        pruned.delete(nd.id);
        if (!best || vl > best.vl) best = { nd, vl };
      });
      if (best && best.vl >= T.accuracy(tree, VAL, isLeaf)) {
        pruned.add(best.nd.id);
        msg.show("Pruned the single node that helps validation most: now " + LR.fmtF(best.vl * 100, 1) + "%.", "good");
      } else {
        msg.show("No single prune improves validation right now.", "info");
      }
      update();
    }

    function update() {
      ctx.clearRect(0, 0, W, H);
      T.drawTree(ctx, tree, {
        x0: 8, y0: 26, w: W - 16, h: H - 60,
        boxH: 22,
        font: "600 10.5px Inter, sans-serif",
        edgeLabels: false,
        isLeaf,
        testText: (nd) => (nd.feat === "w" ? "x₁" : "x₂") + ">" + LR.fmtF(nd.thr, 1),
        leafText: (nd) => (pruned.has(nd.id) ? "✂ " : "") + nd.label + " (" + nd.n + ")",
        leafColor: (nd) => CLS_COLOR[nd.label],
        leafFill: (nd) => (pruned.has(nd.id) ? "#fff" : CLS_SOFT[nd.label]),
        leafMinW: 26,
      });
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("click an internal node to prune ✂ · click a pruned node to restore", 8, 14);

      const tr = T.accuracy(tree, TRAIN, isLeaf);
      const vl = T.accuracy(tree, VAL, isLeaf);
      let lv = 0;
      reachable().forEach((nd) => { if (isLeaf(nd)) lv++; });
      ro.set("lv", String(lv));
      ro.set("tr", LR.fmtF(tr * 100, 1) + "%  (was " + LR.fmtF(baseTr * 100, 1) + "%)", C.green);
      ro.set("vl", LR.fmtF(vl * 100, 1) + "%", vl > baseVl + 1e-9 ? C.green : C.orange);
      ro.set("base", LR.fmtF(baseVl * 100, 1) + "%");
    }

    update();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.3 — Gini vs entropy (Assignment 2 uses Gini)
     ════════════════════════════════════════════════════════════ */
  LR.figs.gini = function (mount) {
    LR.header(
      mount,
      "Two rulers for the same impurity",
      "Left: both curves. Right: both criteria scoring the same candidate split on the seven fruits. They peak together, vanish together, and usually agree."
    );

    const FRUITS = [
      { h: 6.2, cls: "orange" }, { h: 6.5, cls: "orange" }, { h: 6.8, cls: "orange" },
      { h: 6.9, cls: "orange" }, { h: 7.4, cls: "orange" },
      { h: 8.6, cls: "lemon" }, { h: 9.4, cls: "lemon" },
    ];
    const BEST_H = T.bestSplit(FRUITS, ["h"], 1, T.entropyOf);
    const BEST_G = T.bestSplit(FRUITS, ["h"], 1, T.giniOf);

    let thr = 7.0, halve = false;

    const bar = LR.controls(mount);
    LR.slider(bar, "threshold (cm)", 6.0, 10.0, 0.05, thr, function (v) { thr = v; draw(); }, (v) => LR.fmtF(v, 2));
    const hBtn = LR.button(bar, "Scale entropy ÷ 2: off", function () {
      halve = !halve;
      hBtn.textContent = "Scale entropy ÷ 2: " + (halve ? "on" : "off");
      hBtn.classList.toggle("on", halve);
      draw();
    });

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Impurity curves"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Same split, both scores"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 300, { aria: "Binary entropy and Gini impurity curves over class probability p" });
    const RC = LR.canvas(rb, 400, 300, { aria: "Seven fruits on a height axis with the current threshold and both impurity based gains" });

    const ro = LR.readout(mount, [
      { k: "ige", label: "gain (entropy)" },
      { k: "igg", label: "gain (Gini)" },
      { k: "agree", label: "best split under each" },
    ]);

    const PL = { x0: 48, y0: 16, w: 400 - 66, h: 300 - 60, xmin: 0, xmax: 1, ymin: 0, ymax: 1.04, xlabel: "p (fraction of one class)", ylabel: "impurity" };
    const PR = { x0: 40, y0: 16, w: 400 - 58, h: 300 - 60, xmin: 5.8, xmax: 9.8, ymin: 0, ymax: 1, xlabel: "height (cm)", yticks: [] };

    function draw() {
      // curves
      const c1 = LC.ctx;
      c1.clearRect(0, 0, LC.W, LC.H);
      const S1 = LR.plot(c1, PL);
      [[(p) => (halve ? T.entropyP(p) / 2 : T.entropyP(p)), C.orange], [T.giniP, C.purple]].forEach(function (row) {
        const f = row[0];
        c1.strokeStyle = row[1];
        c1.lineWidth = 2.2;
        c1.beginPath();
        for (let i = 0; i <= 200; i++) {
          const p = i / 200;
          i === 0 ? c1.moveTo(S1.sx(p), S1.sy(f(p))) : c1.lineTo(S1.sx(p), S1.sy(f(p)));
        }
        c1.stroke();
      });
      c1.font = "600 11.5px Inter, sans-serif";
      c1.textAlign = "left";
      c1.fillStyle = C.orange;
      c1.fillText("entropy H(p)" + (halve ? " ÷ 2" : ""), PL.x0 + 8, PL.y0 + 16);
      c1.fillStyle = C.purple;
      c1.fillText("Gini 2p(1−p)", PL.x0 + 8, PL.y0 + 32);

      // split scoring strip
      const c2 = RC.ctx;
      c2.clearRect(0, 0, RC.W, RC.H);
      const S2 = LR.plot(c2, PR);
      FRUITS.forEach((p) => LR.dot(c2, S2.sx(p.h), S2.sy(0.5), 7, T.CC[p.cls], "#fff"));
      c2.strokeStyle = C.text;
      c2.lineWidth = 2;
      c2.setLineDash([5, 4]);
      c2.beginPath(); c2.moveTo(S2.sx(thr), PR.y0); c2.lineTo(S2.sx(thr), PR.y0 + PR.h); c2.stroke();
      c2.setLineDash([]);

      const sE = T.splitEval(FRUITS, "h", thr, T.entropyOf);
      const sG = T.splitEval(FRUITS, "h", thr, T.giniOf);
      c2.font = "600 11.5px Inter, sans-serif";
      c2.textAlign = "left";
      c2.fillStyle = C.orange;
      c2.fillText("entropy gain = " + LR.fmtF(sE.ig, 3), PR.x0 + 8, PR.y0 + 20);
      c2.fillStyle = C.purple;
      c2.fillText("Gini gain     = " + LR.fmtF(sG.ig, 3), PR.x0 + 8, PR.y0 + 38);
      // best-split markers
      c2.strokeStyle = C.green;
      c2.lineWidth = 1.6;
      c2.setLineDash([2, 3]);
      c2.beginPath(); c2.moveTo(S2.sx(BEST_H.thr), PR.y0 + PR.h * 0.55); c2.lineTo(S2.sx(BEST_H.thr), PR.y0 + PR.h); c2.stroke();
      c2.setLineDash([]);
      c2.fillStyle = C.green;
      c2.textAlign = "center";
      c2.fillText("best split (both)", S2.sx(BEST_H.thr), PR.y0 + PR.h * 0.5);

      ro.set("ige", LR.fmtF(sE.ig, 4) + "  (max " + LR.fmtF(BEST_H.ig, 4) + ")", C.orange);
      ro.set("igg", LR.fmtF(sG.ig, 4) + "  (max " + LR.fmtF(BEST_G.ig, 4) + ")", C.purple);
      ro.set(
        "agree",
        "entropy → " + LR.fmtF(BEST_H.thr, 2) + " cm, Gini → " + LR.fmtF(BEST_G.thr, 2) + " cm" +
          (Math.abs(BEST_H.thr - BEST_G.thr) < 1e-9 ? " · they agree" : " · they differ"),
        Math.abs(BEST_H.thr - BEST_G.thr) < 1e-9 ? C.green : C.red
      );
    }

    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.4 — k-NN vs decision tree on a diagonal boundary
     ════════════════════════════════════════════════════════════ */
  LR.figs.knnvstree = function (mount) {
    LR.header(
      mount,
      "Two learners, one diagonal truth",
      "The true boundary is the diagonal x₂ = x₁. k-NN follows it; a tree can only approximate it with an axis-aligned staircase. Slide the depth."
    );

    // diagonal dataset with label noise
    const rand = LR.rng(1234);
    const pts = [];
    for (let i = 0; i < 70; i++) {
      const w = 0.3 + rand() * 5.8;
      const h = 0.3 + rand() * 5.8;
      const m = h - w + LR.gauss(rand) * 0.55;
      pts.push({ w, h, cls: m > 0 ? "A" : "B" });
    }

    let depth = 3;
    const KNN_K = 5;

    const bar = LR.controls(mount);
    LR.slider(bar, "tree max depth", 1, 8, 1, depth, function (v) { depth = Math.round(v); drawRight(); }, (v) => String(Math.round(v)));

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "k-NN (k = 5): similarity-based"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Decision tree: axis-aligned rules"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 330, { aria: "k nearest neighbour decision map hugging the diagonal boundary" });
    const RC = LR.canvas(rb, 400, 330, { aria: "Decision tree region map approximating the diagonal with axis aligned rectangles" });

    const ro = LR.readout(mount, [
      { k: "lv", label: "tree leaves" },
      { k: "note", label: "same data, different bet" },
    ]);

    const PP = { x0: 44, y0: 14, w: 400 - 62, h: 330 - 56, xmin: 0, xmax: 6.4, ymin: 0, ymax: 6.4, xlabel: "x₁", ylabel: "x₂" };

    function knnLabel(q) {
      const nb = pts
        .map((p) => ({ p, d: (p.w - q.w) ** 2 + (p.h - q.h) ** 2 }))
        .sort((a, b) => a.d - b.d)
        .slice(0, KNN_K);
      let a = 0;
      nb.forEach((n) => { if (n.p.cls === "A") a++; });
      return a * 2 > KNN_K ? "A" : "B";
    }

    function drawLeft() {
      const ctx = LC.ctx;
      ctx.clearRect(0, 0, LC.W, LC.H);
      // block map first, then axes over it
      const BLK = 6;
      for (let px = PP.x0; px < PP.x0 + PP.w; px += BLK) {
        for (let py = PP.y0; py < PP.y0 + PP.h; py += BLK) {
          const wx = PP.xmin + ((px + BLK / 2 - PP.x0) / PP.w) * (PP.xmax - PP.xmin);
          const wy = PP.ymin + ((PP.y0 + PP.h - (py + BLK / 2)) / PP.h) * (PP.ymax - PP.ymin);
          ctx.fillStyle = CLS_SOFT[knnLabel({ w: wx, h: wy })];
          ctx.fillRect(px, py, BLK, BLK);
        }
      }
      const SC = LR.plot(ctx, PP);
      pts.forEach((p) => LR.dot(ctx, SC.sx(p.w), SC.sy(p.h), 3.4, CLS_COLOR[p.cls], "#fff"));
      diag(ctx, SC);
    }

    function diag(ctx, SC) {
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(SC.sx(0), SC.sy(0));
      ctx.lineTo(SC.sx(6.4), SC.sy(6.4));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("true boundary", SC.sx(4.6), SC.sy(4.9));
    }

    function drawRight() {
      const ctx = RC.ctx;
      ctx.clearRect(0, 0, RC.W, RC.H);
      const tree = T.build(pts, { maxDepth: depth, minLeaf: 1 });
      const SC = LR.plot(ctx, PP);
      drawRegions(ctx, SC, T.regions(tree, BOUNDS), PP);
      pts.forEach((p) => LR.dot(ctx, SC.sx(p.w), SC.sy(p.h), 3.4, CLS_COLOR[p.cls], "#fff"));
      diag(ctx, SC);
      const lv = countLeaves(tree);
      ro.set("lv", lv + " at depth " + depth);
      ro.set(
        "note",
        depth <= 2
          ? "shallow: big rectangles, crude staircase"
          : depth <= 5
            ? "the staircase hugs the diagonal with more, smaller steps"
            : "many tiny slivers: the tree spends leaves imitating a line"
      );
    }

    drawLeft();
    drawRight();
  };
})();
