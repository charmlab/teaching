/* ══════════════════════════════════════════════════════════════
   figures-b.js — Bias-Variance & Ensembles lesson, sections 4–5
   Fig 4.1 bagging · Fig 4.2 forest · Fig 5.1 adaboost
   Fig 5.2 adacode · Fig 5.3 gradboost · Fig 5.4 contrast
   Uses LR.ens from figures-a.js. Real bootstrap resamples, real
   decision trees, real AdaBoost rounds; nothing staged.
   The 2-D demo datasets are constructed for these figures
   (the lecture describes the algorithms, not specific data).
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const E = LR.ens;

  const SOFT = { pos: "rgba(102,102,102,0.20)", neg: "rgba(150,150,150,0.17)" };
  const HARD = { pos: C.green, neg: C.purple };

  /* ── shared tree / stump core ───────────────────────────── */
  const T = (LR.trees = {});

  // two overlapping Gaussian blobs, labels +1 / −1
  T.makeBlobs = function (seed, nPer, cPos, cNeg, sd) {
    const rand = LR.rng(seed);
    const pts = [];
    for (let i = 0; i < nPer; i++)
      pts.push({ x: cPos[0] + LR.gauss(rand) * sd, y: cPos[1] + LR.gauss(rand) * sd, c: 1 });
    for (let i = 0; i < nPer; i++)
      pts.push({ x: cNeg[0] + LR.gauss(rand) * sd, y: cNeg[1] + LR.gauss(rand) * sd, c: -1 });
    return pts;
  };

  function giniImpurity(nPos, nNeg) {
    const n = nPos + nNeg;
    if (n === 0) return 0;
    const p = nPos / n;
    return 2 * p * (1 - p);
  }

  // best gini split of data[idx] on one feature; returns {thr, gain} or null
  function bestSplitOnFeature(data, idx, feat) {
    const vals = idx.map((i) => ({ v: feat === 0 ? data[i].x : data[i].y, c: data[i].c }));
    vals.sort((a, b) => a.v - b.v);
    const n = vals.length;
    let totPos = 0;
    vals.forEach((v) => { if (v.c === 1) totPos++; });
    const parent = giniImpurity(totPos, n - totPos);
    let leftPos = 0, best = null;
    for (let i = 0; i < n - 1; i++) {
      if (vals[i].c === 1) leftPos++;
      if (vals[i].v === vals[i + 1].v) continue;
      const nl = i + 1, nr = n - nl;
      const g = parent - (nl / n) * giniImpurity(leftPos, nl - leftPos) - (nr / n) * giniImpurity(totPos - leftPos, nr - (totPos - leftPos));
      if (g > 1e-12 && (!best || g > best.gain)) best = { thr: (vals[i].v + vals[i + 1].v) / 2, gain: g };
    }
    return best;
  }

  // recursive CART-style classifier. opts: {maxDepth, minLeaf, featSub, rand}
  T.buildTree = function (data, idx, opts, depth) {
    depth = depth || 0;
    let nPos = 0;
    idx.forEach((i) => { if (data[i].c === 1) nPos++; });
    const maj = nPos * 2 >= idx.length ? 1 : -1;
    if (depth >= opts.maxDepth || idx.length <= opts.minLeaf || nPos === 0 || nPos === idx.length)
      return { leaf: maj };
    // random forests: consider only a random subset of features at each split
    const feats = opts.featSub ? [opts.rand() < 0.5 ? 0 : 1] : [0, 1];
    let best = null;
    feats.forEach(function (f) {
      const s = bestSplitOnFeature(data, idx, f);
      if (s && (!best || s.gain > best.gain)) best = { feat: f, thr: s.thr, gain: s.gain };
    });
    if (!best) return { leaf: maj };
    const li = [], ri = [];
    idx.forEach(function (i) {
      const v = best.feat === 0 ? data[i].x : data[i].y;
      (v <= best.thr ? li : ri).push(i);
    });
    if (!li.length || !ri.length) return { leaf: maj };
    return {
      feat: best.feat,
      thr: best.thr,
      left: T.buildTree(data, li, opts, depth + 1),
      right: T.buildTree(data, ri, opts, depth + 1),
    };
  };

  T.predictTree = function (node, x, y) {
    while (node.leaf === undefined) {
      const v = node.feat === 0 ? x : y;
      node = v <= node.thr ? node.left : node.right;
    }
    return node.leaf;
  };

  // weighted decision stump: pred = s if (feat value ≤ thr) else −s
  // (implementation below as trainStumpW, exported as T.trainStump)
  T.predictStump = function (st, x, y) {
    const v = st.feat === 0 ? x : y;
    return v <= st.thr ? st.s : -st.s;
  };

  // one full AdaBoost trainer with explicit rounds
  T.adaTrainer = function (data) {
    const N = data.length;
    let w = new Array(N).fill(1 / N); // 1. initialize weights w₁⁽ⁿ⁾ > 0
    const rounds = [];
    return {
      weights: () => w.slice(),
      rounds,
      step: function () {
        // 2a. train the weak learner on the weighted data
        const stump = trainStumpW(data, w);
        const wrong = data.map((p, i) => T.predictStump(stump, p.x, p.y) !== p.c);
        // 2b. weighted error rate ε_m = Σ w·I[wrong] / Σ w
        let wSum = 0, eSum = 0;
        for (let i = 0; i < N; i++) { wSum += w[i]; if (wrong[i]) eSum += w[i]; }
        const eps = eSum / wSum;
        if (eps >= 0.5 || eps <= 0) return null; // stopping condition from the lecture
        // 2c. learner quality α_m = ln((1−ε)/ε)
        const alpha = Math.log((1 - eps) / eps);
        // 2d. weight update: w ← w·exp(α·I[wrong]).
        // (We also renormalize to sum 1; ε_m divides by Σw, so this changes
        // nothing mathematically — it just keeps the numbers readable.)
        let tot = 0;
        for (let i = 0; i < N; i++) { if (wrong[i]) w[i] *= Math.exp(alpha); tot += w[i]; }
        for (let i = 0; i < N; i++) w[i] /= tot;
        const round = { stump, eps, alpha, wrong, weights: w.slice() };
        rounds.push(round);
        return round;
      },
      // weighted-vote ensemble Y_M(x) = sign(Σ α_m y_m(x))
      predict: function (x, y, upTo) {
        const M = upTo === undefined ? rounds.length : upTo;
        let s = 0;
        for (let m = 0; m < M; m++) s += rounds[m].alpha * T.predictStump(rounds[m].stump, x, y);
        return s >= 0 ? 1 : -1;
      },
    };
  };

  // weighted stump without the buggy normalization above (clean version used everywhere)
  function trainStumpW(data, w) {
    let best = null;
    let wTot = 0;
    for (let i = 0; i < w.length; i++) wTot += w[i];
    for (let feat = 0; feat < 2; feat++) {
      const order = data.map((p, i) => ({ v: feat === 0 ? p.x : p.y, c: p.c, w: w[i] }));
      order.sort((a, b) => a.v - b.v);
      let errA = 0; // error of "everything predicted −1" for polarity s = +1
      order.forEach((o) => { if (o.c === 1) errA += o.w; });
      let curErr = errA;
      const consider = function (thr, err, s) {
        if (!best || err < best.err - 1e-12) best = { feat, thr, s, err };
      };
      consider(order[0].v - 1, curErr / wTot, 1);
      consider(order[0].v - 1, (wTot - curErr) / wTot, -1);
      for (let i = 0; i < order.length; i++) {
        curErr += order[i].c === 1 ? -order[i].w : order[i].w;
        if (i < order.length - 1 && order[i].v === order[i + 1].v) continue;
        const thr = i < order.length - 1 ? (order[i].v + order[i + 1].v) / 2 : order[i].v + 1;
        consider(thr, curErr / wTot, 1);
        consider(thr, (wTot - curErr) / wTot, -1);
      }
    }
    return best;
  }
  T.trainStump = trainStumpW;

  /* ── plot area shared by the 2-D figures ────────────────── */
  const DOM = { xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5 };

  function gridOf(P, blk) {
    // logical-space grid cell centres matching the pixel blocks
    const nx = Math.ceil(P.w / blk), ny = Math.ceil(P.h / blk);
    return { nx, ny, blk };
  }
  function cellCentre(P, g, ix, iy) {
    const px = P.x0 + ix * g.blk + g.blk / 2;
    const py = P.y0 + iy * g.blk + g.blk / 2;
    return {
      x: P.xmin + ((px - P.x0) / P.w) * (P.xmax - P.xmin),
      y: P.ymin + ((P.y0 + P.h - py) / P.h) * (P.ymax - P.ymin),
      px: P.x0 + ix * g.blk,
      py: P.y0 + iy * g.blk,
    };
  }

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — bagging demo (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.bagging = function (mount) {
    LR.header(
      mount,
      "Average away the wobble",
      "Every tree below was really trained on a real bootstrap resample of the same 80 points. Slide m: individual boundaries stay wobbly, the averaged boundary calms down, and the variance readout (measured at the × query) drops."
    );

    const data = T.makeBlobs(11, 40, [2.3, 3.9], [3.9, 2.4], 1.0);
    const n = data.length;
    const Q = { x: 3.1, y: 3.15 }; // fixed probe point for the variance readout

    // Variance machinery: KOUT datasets drawn from the population generator,
    // JIN bootstrap trees on each. The variance across the KOUT bags includes
    // both randomness sources, so the shortfall against the Var/m benchmark
    // is the REAL bootstrap-correlation effect the lecture describes, not a
    // simulation of it. The drawn trees are the k = 0 bag (the displayed data).
    const NDRAW = 40;
    const KOUT = 30, JIN = NDRAW;
    const trees = []; // the k = 0 trees, used for drawing
    const predsQ = []; // predsQ[k][j] = tree j of dataset k, evaluated at the probe
    for (let k = 0; k < KOUT; k++) {
      const dk = k === 0 ? data : T.makeBlobs(1100 + k * 91, 40, [2.3, 3.9], [3.9, 2.4], 1.0);
      const row = [];
      for (let j = 0; j < JIN; j++) {
        const rand = LR.rng(500 + k * 977 + j * 7);
        const idx = E.bootstrap(dk.length, rand);
        const tr = T.buildTree(dk, idx, { maxDepth: 7, minLeaf: 1, featSub: false, rand }, 0);
        row.push(T.predictTree(tr, Q.x, Q.y));
        if (k === 0) trees.push(tr);
      }
      predsQ.push(row);
    }
    // precomputed curves: Var and mean of a bag of m, over the KOUT reruns
    const bagVar = [], bagMean = [];
    for (let mm = 1; mm <= JIN; mm++) {
      const means = predsQ.map(function (row) {
        let s = 0;
        for (let j = 0; j < mm; j++) s += row[j];
        return s / mm;
      });
      bagVar.push(E.variance(means));
      bagMean.push(E.mean(means));
    }

    let m = 1;

    const bar = LR.controls(mount);
    LR.slider(bar, "m (bagged trees)", 1, NDRAW, 1, m, function (v) {
      m = Math.round(v);
      draw();
      updateRO();
    }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "Bagging demo: faint decision boundaries of individual bootstrap-trained trees and the bold averaged boundary, with a live variance readout",
    });
    const ro = LR.readout(mount, [
      { k: "v1", label: "Var of ONE tree at ×" },
      { k: "vm", label: "Var of the bag of m at ×" },
      { k: "th", label: "1/m benchmark (independent)" },
      { k: "bias", label: "mean prediction at × (bias proxy)" },
    ]);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: DOM.xmin, xmax: DOM.xmax, ymin: DOM.ymin, ymax: DOM.ymax, xlabel: "x₁", ylabel: "x₂" };
    const G = gridOf(P, 6);

    // cache per-tree grids for the first NDRAW trees (Int8: ±1)
    const grids = [];
    for (let ti = 0; ti < NDRAW; ti++) {
      const g = new Int8Array(G.nx * G.ny);
      for (let ix = 0; ix < G.nx; ix++) {
        for (let iy = 0; iy < G.ny; iy++) {
          const c = cellCentre(P, G, ix, iy);
          g[iy * G.nx + ix] = T.predictTree(trees[ti], c.x, c.y);
        }
      }
      grids.push(g);
    }
    // cumulative sums so the ensemble mean grid is O(cells) per slider move
    const cum = [new Int16Array(G.nx * G.ny)];
    for (let ti = 0; ti < NDRAW; ti++) {
      const prev = cum[ti], next = new Int16Array(G.nx * G.ny);
      for (let k = 0; k < next.length; k++) next[k] = prev[k] + grids[ti][k];
      cum.push(next);
    }

    function updateRO() {
      const v1 = bagVar[0]; // variance of a single tree's ±1 prediction at ×
      const vm = bagVar[m - 1]; // variance of the whole bag of m, over 30 reruns
      ro.set("v1", LR.fmtF(v1, 3));
      ro.set("vm", LR.fmtF(vm, 3) + "  (measured over " + KOUT + " reruns)", vm < v1 * 0.5 ? C.green : undefined);
      ro.set("th", LR.fmtF(v1 / m, 3) + " = Var/m" + (m > 4 && vm > (v1 / m) * 1.3 ? "  ← real gain is smaller: trees correlate" : ""));
      ro.set("bias", LR.fmtF(bagMean[m - 1], 3) + "  (does not move with m)", C.purple);
    }

    function drawBoundaryCells(grid, color, size) {
      // a cell is "boundary" if its label differs from its right or bottom neighbour
      ctx.fillStyle = color;
      for (let iy = 0; iy < G.ny; iy++) {
        for (let ix = 0; ix < G.nx; ix++) {
          const v = grid[iy * G.nx + ix];
          const r = ix + 1 < G.nx ? grid[iy * G.nx + ix + 1] : v;
          const b = iy + 1 < G.ny ? grid[(iy + 1) * G.nx + ix] : v;
          if ((v > 0) !== (r > 0) || (v > 0) !== (b > 0)) {
            const c = cellCentre(P, G, ix, iy);
            ctx.fillRect(c.px + G.blk / 2 - size / 2, c.py + G.blk / 2 - size / 2, size, size);
          }
        }
      }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // ensemble mean grid (average of ±1 votes of the first m trees)
      const sums = cum[m];
      for (let iy = 0; iy < G.ny; iy++) {
        for (let ix = 0; ix < G.nx; ix++) {
          const c = cellCentre(P, G, ix, iy);
          ctx.fillStyle = sums[iy * G.nx + ix] >= 0 ? SOFT.pos : SOFT.neg;
          ctx.fillRect(c.px, c.py, G.blk, G.blk);
        }
      }

      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      ctx.save();
      ctx.beginPath();
      ctx.rect(P.x0, P.y0, P.w, P.h);
      ctx.clip();

      // faint individual boundaries (first min(m, 12) trees)
      const show = Math.min(m, 12);
      for (let ti = 0; ti < show; ti++) drawBoundaryCells(grids[ti], "rgba(17,17,17,0.10)", 2.6);

      // bold averaged boundary: cells where the mean vote changes sign
      const meanGrid = new Int8Array(G.nx * G.ny);
      for (let k = 0; k < meanGrid.length; k++) meanGrid[k] = sums[k] >= 0 ? 1 : -1;
      drawBoundaryCells(meanGrid, C.orange, 5);

      ctx.restore();

      // data
      data.forEach(function (p) {
        LR.dot(ctx, sx(p.x), sy(p.y), 4.4, p.c === 1 ? HARD.pos : HARD.neg, "#ffffff");
      });

      // probe point
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 2.2;
      const qx = sx(Q.x), qy = sy(Q.y);
      ctx.beginPath();
      ctx.moveTo(qx - 7, qy - 7); ctx.lineTo(qx + 7, qy + 7);
      ctx.moveTo(qx - 7, qy + 7); ctx.lineTo(qx + 7, qy - 7);
      ctx.stroke();

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(17,17,17,0.55)";
      ctx.fillText("grey wobble: single trees (" + show + " shown)", P.x0 + 8, P.y0 + 16);
      ctx.fillStyle = C.orange;
      ctx.fillText("orange: the m-tree average, m = " + m, P.x0 + 8, P.y0 + 32);
    }

    draw();
    updateRO();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.2 — random forest vs a single deep tree
     ════════════════════════════════════════════════════════════ */
  LR.figs.forest = function (mount) {
    LR.header(
      mount,
      "One deep tree vs forty decorrelated ones",
      "Same data on both sides. The single tree memorizes; the forest averages. Toggle feature subsampling to watch the trees decorrelate and the test accuracy move."
    );

    // x₁ carries most of the signal here, deliberately: dominant features are
    // exactly the situation where plain bagging's trees stay correlated and
    // feature subsampling visibly helps
    const data = T.makeBlobs(23, 45, [2.0, 3.4], [4.3, 2.8], 1.0);
    const test = T.makeBlobs(24001, 80, [2.0, 3.4], [4.3, 2.8], 1.0); // held-out, same generator
    const n = data.length;

    // the single deep tree: full data, effectively unlimited depth
    const single = T.buildTree(data, data.map((_, i) => i), { maxDepth: 14, minLeaf: 1, featSub: false, rand: LR.rng(1) }, 0);

    // two precomputed forests: plain bagging vs bagging + feature subsampling
    function makeForest(featSub) {
      const f = [];
      for (let i = 0; i < 40; i++) {
        const rand = LR.rng(7000 + i * 13);
        const idx = E.bootstrap(n, rand);
        f.push(T.buildTree(data, idx, { maxDepth: 5, minLeaf: 1, featSub, rand }, 0));
      }
      return f;
    }
    const forests = { off: makeForest(false), on: makeForest(true) };
    let mode = "on";

    const bar = LR.controls(mount);
    const tog = LR.button(bar, "Feature subsampling: on", function () {
      mode = mode === "on" ? "off" : "on";
      tog.textContent = "Feature subsampling: " + mode;
      tog.classList.toggle("on", mode === "on");
      drawAll();
    }, "on");

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "One deep tree — jagged, memorized"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Random forest (40 trees) — averaged"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 340, { aria: "Decision regions of a single deep decision tree: jagged axis-aligned rectangles wrapped around individual points" });
    const RC = LR.canvas(rb, 400, 340, { aria: "Decision regions of a forty-tree random forest: smoother averaged boundary" });

    const ro = LR.readout(mount, [
      { k: "corr", label: "avg pairwise tree correlation" },
      { k: "accS", label: "single tree test accuracy" },
      { k: "accF", label: "forest test accuracy" },
    ]);
    const msg = LR.msg(mount);

    function forestVote(f, x, y) {
      let s = 0;
      for (const tr of f) s += T.predictTree(tr, x, y);
      return s / f.length; // mean vote in [−1, 1]
    }

    // avg Pearson correlation between tree predictions over sample locations
    function avgCorr(f) {
      const rand = LR.rng(99);
      const locs = [];
      for (let i = 0; i < 300; i++)
        locs.push({ x: DOM.xmin + rand() * (DOM.xmax - DOM.xmin), y: DOM.ymin + rand() * (DOM.ymax - DOM.ymin) });
      const preds = f.map((tr) => locs.map((l) => T.predictTree(tr, l.x, l.y)));
      let s = 0, cnt = 0;
      for (let a = 0; a < f.length; a++) {
        for (let b = a + 1; b < f.length; b++) {
          const pa = preds[a], pb = preds[b];
          const ma = E.mean(pa), mb = E.mean(pb);
          let cov = 0, va = 0, vb = 0;
          for (let i = 0; i < pa.length; i++) {
            cov += (pa[i] - ma) * (pb[i] - mb);
            va += (pa[i] - ma) ** 2;
            vb += (pb[i] - mb) ** 2;
          }
          if (va > 0 && vb > 0) { s += cov / Math.sqrt(va * vb); cnt++; }
        }
      }
      return s / cnt;
    }
    const corrs = { off: avgCorr(forests.off), on: avgCorr(forests.on) };

    function accuracy(predict) {
      let ok = 0;
      test.forEach((p) => { if (predict(p.x, p.y) === p.c) ok++; });
      return ok / test.length;
    }
    const accSingle = accuracy((x, y) => T.predictTree(single, x, y));
    const accs = {
      off: accuracy((x, y) => (forestVote(forests.off, x, y) >= 0 ? 1 : -1)),
      on: accuracy((x, y) => (forestVote(forests.on, x, y) >= 0 ? 1 : -1)),
    };

    function drawPane(CV, predictSoft) {
      const { ctx, W, H } = CV;
      ctx.clearRect(0, 0, W, H);
      const P = { x0: 44, y0: 12, w: W - 58, h: H - 52, xmin: DOM.xmin, xmax: DOM.xmax, ymin: DOM.ymin, ymax: DOM.ymax, xlabel: "x₁", ylabel: "x₂" };
      const G = gridOf(P, 5);
      for (let iy = 0; iy < G.ny; iy++) {
        for (let ix = 0; ix < G.nx; ix++) {
          const c = cellCentre(P, G, ix, iy);
          const v = predictSoft(c.x, c.y); // in [−1, 1]
          const a = 0.06 + 0.16 * Math.min(1, Math.abs(v));
          ctx.fillStyle = v >= 0 ? "rgba(102,102,102," + a + ")" : "rgba(150,150,150," + a + ")";
          ctx.fillRect(c.px, c.py, G.blk, G.blk);
        }
      }
      const SC = LR.plot(ctx, P);
      data.forEach(function (p) {
        LR.dot(ctx, SC.sx(p.x), SC.sy(p.y), 3.6, p.c === 1 ? HARD.pos : HARD.neg, "#ffffff");
      });
    }

    function drawAll() {
      drawPane(LC, (x, y) => T.predictTree(single, x, y));
      drawPane(RC, (x, y) => forestVote(forests[mode], x, y));
      ro.set("corr", LR.fmtF(corrs[mode], 3) + "  (off: " + LR.fmtF(corrs.off, 3) + " → on: " + LR.fmtF(corrs.on, 3) + ")", mode === "on" ? C.green : C.amber);
      ro.set("accS", LR.fmtF(accSingle * 100, 1) + "%");
      ro.set("accF", LR.fmtF(accs[mode] * 100, 1) + "%", accs[mode] > accSingle ? C.green : undefined);
      msg.show(
        mode === "on"
          ? "Feature subsampling on: each split sees a random feature subset, so the trees stop making the same choices. Correlation " + LR.fmtF(corrs.on, 3) + " vs " + LR.fmtF(corrs.off, 3) + " without it, and their errors cancel better."
          : "Feature subsampling off: this is plain bagging. The trees share the strongest splits, so they stay more correlated (" + LR.fmtF(corrs.off, 3) + ") and the averaging cancels less.",
        mode === "on" ? "good" : "info"
      );
    }
    drawAll();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — AdaBoost, round by round (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.adaboost = function (mount) {
    LR.header(
      mount,
      "AdaBoost, one round at a time",
      "Dot area = current weight. Each round really trains the best decision stump on the weighted data, computes ε and α, grows the weights of its mistakes, and adds a weighted vote to the ensemble."
    );

    const data = T.makeBlobs(29, 15, [2.2, 2.2], [3.9, 3.9], 0.9);
    const N = data.length;
    let trainer = T.adaTrainer(data);
    let maxRounds = 8;

    const bar = LR.controls(mount);
    const stepBtn = LR.button(bar, "Step ▸", step, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, LR.reducedMotion ? 1100 : 650);
    });
    LR.button(bar, "Reset ⟲", reset);
    LR.slider(bar, "rounds M", 1, 15, 1, maxRounds, function (v) { maxRounds = Math.round(v); }, (v) => String(Math.round(v)));

    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "AdaBoost stepping figure: training points sized by weight, the current decision stump, and the evolving weighted-vote ensemble boundary",
    });
    const narrate = LR.el("div", "stepper-narrate", "Press <b>Step</b> to run round 1. All " + N + " weights start equal at 1/" + N + " = " + LR.fmtF(1 / N, 3) + ".");
    mount.appendChild(narrate);
    const logBox = LR.el("div", "ada-log");
    logBox.innerHTML = "<table class='wtable'><thead><tr><th>round m</th><th>stump</th><th>ε_m</th><th>α_m = ln((1−ε)/ε)</th><th>ensemble train error</th></tr></thead><tbody></tbody></table>";
    mount.appendChild(logBox);
    const logBody = logBox.querySelector("tbody");

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: DOM.xmin, xmax: DOM.xmax, ymin: DOM.ymin, ymax: DOM.ymax, xlabel: "x₁", ylabel: "x₂" };
    const G = gridOf(P, 6);

    function ensembleErr() {
      let wrong = 0;
      data.forEach((p) => { if (trainer.predict(p.x, p.y) !== p.c) wrong++; });
      return wrong / N;
    }

    function step() {
      if (trainer.rounds.length >= maxRounds) {
        stopPlay();
        narrate.innerHTML = "<b>Stopped at M = " + maxRounds + " rounds.</b> Prediction rule: Y(x) = sign(Σ α_m y_m(x)). Raise the rounds slider and keep stepping, or reset.";
        return;
      }
      const r = trainer.step();
      if (!r) {
        stopPlay();
        narrate.innerHTML = "<b>No weak learner beats chance on these weights (ε ≥ 0.5): AdaBoost stops.</b> That is one of the lecture's three stopping conditions.";
        return;
      }
      const m = trainer.rounds.length;
      const err = ensembleErr();
      const grew = r.wrong.filter(Boolean).length;
      narrate.innerHTML =
        "<b>Round " + m + ".</b> Best stump: " + stumpDesc(r.stump) + ". Weighted error ε = <b>" + LR.fmtF(r.eps, 3) +
        "</b>, so its vote weight is α = ln(" + LR.fmtF(1 - r.eps, 3) + "/" + LR.fmtF(r.eps, 3) + ") = <b>" + LR.fmtF(r.alpha, 2) +
        "</b>. The " + grew + " misclassified points were multiplied by e^α = " + LR.fmtF(Math.exp(r.alpha), 2) + " (watch them grow). Ensemble training error: <b>" + LR.fmtF(err * 100, 1) + "%</b>.";
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + m + "</td><td>" + stumpDesc(r.stump) + "</td><td>" + LR.fmtF(r.eps, 3) + "</td><td>" + LR.fmtF(r.alpha, 3) + "</td><td>" + LR.fmtF(err * 100, 1) + "%</td>";
      logBody.appendChild(tr);
      if (err === 0) stopPlay();
      draw();
    }

    function stumpDesc(st) {
      const f = st.feat === 0 ? "x₁" : "x₂";
      return f + " ≤ " + LR.fmtF(st.thr, 2) + " → " + (st.s === 1 ? "+" : "−") + ", else " + (st.s === 1 ? "−" : "+");
    }

    function reset() {
      stopPlay();
      trainer = T.adaTrainer(data);
      logBody.innerHTML = "";
      narrate.innerHTML = "Press <b>Step</b> to run round 1. All " + N + " weights start equal at 1/" + N + " = " + LR.fmtF(1 / N, 3) + ".";
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const M = trainer.rounds.length;

      // ensemble regions
      if (M > 0) {
        for (let iy = 0; iy < G.ny; iy++) {
          for (let ix = 0; ix < G.nx; ix++) {
            const c = cellCentre(P, G, ix, iy);
            ctx.fillStyle = trainer.predict(c.x, c.y) === 1 ? SOFT.pos : SOFT.neg;
            ctx.fillRect(c.px, c.py, G.blk, G.blk);
          }
        }
      }

      const SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // current stump line
      const last = M > 0 ? trainer.rounds[M - 1] : null;
      if (last) {
        ctx.strokeStyle = C.orange;
        ctx.lineWidth = 2.4;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        if (last.stump.feat === 0) {
          ctx.moveTo(sx(last.stump.thr), sy(P.ymin));
          ctx.lineTo(sx(last.stump.thr), sy(P.ymax));
        } else {
          ctx.moveTo(sx(P.xmin), sy(last.stump.thr));
          ctx.lineTo(sx(P.xmax), sy(last.stump.thr));
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // points, sized by weight
      const w = trainer.weights();
      data.forEach(function (p, i) {
        const r = Math.max(3, Math.min(15, 4 + Math.sqrt(w[i] * N) * 3.2));
        const wrongNow = last && last.wrong[i];
        LR.dot(ctx, sx(p.x), sy(p.y), r, p.c === 1 ? HARD.pos : HARD.neg, wrongNow ? C.red : "#ffffff");
      });

      // legend
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = HARD.pos; ctx.fillText("● class +1", P.x0 + 8, P.y0 + 16);
      ctx.fillStyle = HARD.neg; ctx.fillText("● class −1", P.x0 + 8, P.y0 + 32);
      ctx.fillStyle = C.red; ctx.fillText("red ring: missed by this round's stump", P.x0 + 8, P.y0 + 48);
      if (last) {
        ctx.fillStyle = C.orange;
        ctx.fillText("dashed: round-" + M + " stump · shading: weighted-vote ensemble", P.x0 + 8, P.y0 + 64);
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — the weight update as a runnable exercise
     ════════════════════════════════════════════════════════════ */
  LR.figs.adacode = function (mount) {
    LR.header(
      mount,
      "Run one AdaBoost round yourself",
      "Fixed tiny dataset, your choice of stump. The code really executes: it computes ε and α for your stump and shows the weight update. Apply it and run another round on the new weights."
    );

    // fixed tiny dataset (1-D): x, label t ∈ {+1, −1}
    const X = [1, 2, 3, 4, 5];
    const TL = [1, 1, -1, 1, -1];
    let w = [0.2, 0.2, 0.2, 0.2, 0.2];
    let round = 1;
    let pending = null; // last computed update, not yet applied

    const CODE =
      "x = np.array([1, 2, 3, 4, 5])\n" +
      "t = np.array([+1, +1, -1, +1, -1])\n" +
      "w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])\n" +
      "\n" +
      "def boost_round(w, pred):\n" +
      "    wrong = pred != t\n" +
      "    eps = (w * wrong).sum() / w.sum()   # weighted error\n" +
      "    alpha = np.log((1 - eps) / eps)     # learner quality\n" +
      "    w_new = w * np.exp(alpha * wrong)   # grow the mistakes\n" +
      "    return eps, alpha, w_new / w_new.sum()";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);
    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);

    let thr = 2.5, pol = 1; // stump: predict +1 if x < thr (pol=1) or x > thr (pol=−1)
    LR.slider(controls, "stump threshold θ", 0.5, 5.5, 1, thr, (v) => { thr = v; drawMini(); }, (v) => LR.fmtF(v, 1));
    const polBtn = LR.button(controls, "rule: +1 if x < θ", function () {
      pol = -pol;
      polBtn.textContent = pol === 1 ? "rule: +1 if x < θ" : "rule: +1 if x > θ";
      drawMini();
    });
    LR.button(controls, "Run round ▸", run, "primary small");
    const applyBtn = LR.button(controls, "Apply update ↺", apply, "small");
    LR.button(controls, "Reset weights", function () {
      w = [0.2, 0.2, 0.2, 0.2, 0.2];
      round = 1;
      pending = null;
      out.innerHTML = "&gt;&gt;&gt; boost_round(w, pred)\n(press Run round)";
      drawMini();
    }, "small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; boost_round(w, pred)\n(press Run round)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    const { cv, ctx, W, H } = LR.canvas(mount, 560, 170, {
      aria: "Five one-dimensional points with dot sizes showing the current boosting weights and a movable stump threshold",
    });
    cv.style.marginTop = "14px";

    function stumpPred(x) { return (pol === 1 ? x < thr : x > thr) ? 1 : -1; }

    function run() {
      const pred = X.map(stumpPred);
      const wrong = pred.map((p, i) => p !== TL[i]);
      const wSum = w.reduce((a, b) => a + b, 0);
      const eSum = w.reduce((a, wi, i) => a + (wrong[i] ? wi : 0), 0);
      const eps = eSum / wSum;
      let lines = "&gt;&gt;&gt; boost_round(w, pred)   # round " + round + ", θ = " + LR.fmtF(thr, 1) + (pol === 1 ? ", +1 if x &lt; θ" : ", +1 if x &gt; θ") + "\n";
      lines += "pred  = [" + pred.map((p) => (p > 0 ? "+1" : "-1")).join(", ") + "]\n";
      lines += "wrong = [" + wrong.map((b) => (b ? "True " : "False")).join(", ") + "]\n";
      if (eps <= 0) {
        lines += "eps = 0.0 → this stump is already perfect on these weights; α → ∞.\nAdaBoost would stop here (nothing left to reweight).";
        pending = null;
      } else if (eps >= 0.5) {
        lines += "eps = " + LR.fmtF(eps, 3) + " ≥ 0.5 → not a weak learner on these weights.\nα = ln((1−ε)/ε) would be ≤ 0: pick a better stump.";
        pending = null;
      } else {
        const alpha = Math.log((1 - eps) / eps);
        const wNewRaw = w.map((wi, i) => (wrong[i] ? wi * Math.exp(alpha) : wi));
        const tot = wNewRaw.reduce((a, b) => a + b, 0);
        const wNew = wNewRaw.map((v) => v / tot);
        lines += "eps   = " + LR.fmtF(eSum, 3) + " / " + LR.fmtF(wSum, 3) + " = <b>" + LR.fmtF(eps, 3) + "</b>\n";
        lines += "alpha = ln(" + LR.fmtF(1 - eps, 3) + "/" + LR.fmtF(eps, 3) + ") = <b>" + LR.fmtF(alpha, 3) + "</b>   (e^alpha = " + LR.fmtF(Math.exp(alpha), 2) + ")\n";
        lines += "w_new = [" + wNewRaw.map((v) => LR.fmtF(v, 3)).join(", ") + "]  (before normalizing)\n";
        lines += "      = [" + wNew.map((v) => LR.fmtF(v, 3)).join(", ") + "]  (normalized)\n";
        lines += "press Apply update to make these the round-" + (round + 1) + " weights";
        pending = { wNew, alpha, eps };
      }
      out.innerHTML = lines;
      drawMini();
    }

    function apply() {
      if (!pending) { out.innerHTML += "\n(nothing to apply: run a valid round first)"; return; }
      w = pending.wNew;
      round += 1;
      pending = null;
      out.innerHTML += "\n<b>applied.</b> w = [" + w.map((v) => LR.fmtF(v, 3)).join(", ") + "] — now find the stump that attacks the heavy points.";
      drawMini();
    }

    function drawMini() {
      ctx.clearRect(0, 0, W, H);
      const P = { x0: 44, y0: 12, w: W - 62, h: H - 54, xmin: 0, xmax: 6, ymin: -1, ymax: 1, xlabel: "x", yticks: [0] };
      const SC = LR.plot(ctx, P);
      // stump line + rule
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(SC.sx(thr), P.y0);
      ctx.lineTo(SC.sx(thr), P.y0 + P.h);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.orange;
      ctx.textAlign = "center";
      ctx.fillText(pol === 1 ? "+1 side ←" : "→ +1 side", SC.sx(thr) + (pol === 1 ? -42 : 42), P.y0 + 14);
      X.forEach(function (x, i) {
        const r = Math.max(4, Math.min(18, Math.sqrt(w[i]) * 26));
        LR.dot(ctx, SC.sx(x), SC.sy(0), r, TL[i] === 1 ? HARD.pos : HARD.neg, stumpPred(x) !== TL[i] ? C.red : "#ffffff");
        ctx.font = "600 11px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText(LR.fmtF(w[i], 2), SC.sx(x), SC.sy(0) + 32);
      });
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText("dot area = weight · green = +1, purple = −1 · red ring = missed by your stump", P.x0 + 4, H - 6);
    }
    drawMini();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.3 — gradient boosting fits the residuals
     ════════════════════════════════════════════════════════════ */
  LR.figs.gradboost = function (mount) {
    LR.header(
      mount,
      "Each new learner fits what is left",
      "Top: the running model F. Bottom: the current residuals t − F and the regression stump that will be added next, scaled by η. Step and watch the residuals shrink toward noise."
    );

    // 1-D regression data
    const rand = LR.rng(77);
    const xs = [], ts = [];
    for (let i = 0; i < 26; i++) {
      const x = (i / 25) * 9;
      xs.push(x);
      ts.push(1.6 * Math.sin(0.9 * x) + 0.35 * x + LR.gauss(rand) * 0.35);
    }
    const n = xs.length;
    const tMean = E.mean(ts);

    let eta = 0.5;
    let stumps = []; // each: {thr, left, right}
    let F = ts.map(() => tMean);

    const bar = LR.controls(mount);
    const stepBtn = LR.button(bar, "Step ▸", step, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, LR.reducedMotion ? 1000 : 550);
    });
    LR.button(bar, "Reset ⟲", reset);
    LR.slider(bar, "learning rate η", 0.1, 1.0, 0.05, eta, function (v) {
      eta = v;
      reset(); // η changes the whole trajectory, so restart honestly
      msg.show("η changed to " + LR.fmtF(eta, 2) + ": run restarted, F₀ = mean(t) = " + LR.fmtF(tMean, 3) + ".", "info");
    }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 520, {
      aria: "Gradient boosting on one-dimensional regression data: the running fit above, the residuals and the next stump below",
    });
    const ro = LR.readout(mount, [
      { k: "m", label: "rounds m" },
      { k: "mse", label: "train MSE of F" },
      { k: "eta", label: "η" },
    ]);
    const msg = LR.msg(mount);

    // regression stump fit to (xs, r): split minimizing SSE, leaf means
    function fitStump(r) {
      const order = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]);
      let best = null;
      for (let cut = 1; cut < n; cut++) {
        if (xs[order[cut - 1]] === xs[order[cut]]) continue;
        let sl = 0, sr = 0;
        for (let i = 0; i < cut; i++) sl += r[order[i]];
        for (let i = cut; i < n; i++) sr += r[order[i]];
        const ml = sl / cut, mr = sr / (n - cut);
        let sse = 0;
        for (let i = 0; i < cut; i++) sse += (r[order[i]] - ml) ** 2;
        for (let i = cut; i < n; i++) sse += (r[order[i]] - mr) ** 2;
        if (!best || sse < best.sse) best = { thr: (xs[order[cut - 1]] + xs[order[cut]]) / 2, left: ml, right: mr, sse };
      }
      return best;
    }
    const stumpAt = (st, x) => (x <= st.thr ? st.left : st.right);
    function Fat(x) {
      let v = tMean;
      for (const st of stumps) v += eta * stumpAt(st, x);
      return v;
    }
    function mse() {
      let s = 0;
      for (let i = 0; i < n; i++) s += (ts[i] - F[i]) ** 2;
      return s / n;
    }

    function step() {
      if (stumps.length >= 30) { stopPlay(); return; }
      const r = ts.map((t, i) => t - F[i]); // r_m = t − F_{m−1}
      const st = fitStump(r);
      stumps.push(st);
      F = F.map((v, i) => v + eta * stumpAt(st, xs[i])); // F_m = F_{m−1} + η h_m
      draw();
    }
    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }
    function reset() {
      stopPlay();
      stumps = [];
      F = ts.map(() => tMean);
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const PT = { x0: 60, y0: 20, w: W - 84, h: 250, xmin: 0, xmax: 9, ymin: -2.2, ymax: 4.6, ylabel: "t and F(x)" };
      const PB = { x0: 60, y0: 315, w: W - 84, h: 150, xmin: 0, xmax: 9, ymin: -2.2, ymax: 2.2, xlabel: "x", ylabel: "residual t − F" };
      const ST = LR.plot(ctx, PT);
      const r = ts.map((t, i) => t - F[i]);
      const nextStump = fitStump(r);

      // top: data + F
      ctx.save();
      ctx.beginPath(); ctx.rect(PT.x0, PT.y0, PT.w, PT.h); ctx.clip();
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const x = (i / 300) * 9;
        const y = Fat(x);
        i === 0 ? ctx.moveTo(ST.sx(x), ST.sy(y)) : ctx.lineTo(ST.sx(x), ST.sy(y));
      }
      ctx.stroke();
      ctx.restore();
      for (let i = 0; i < n; i++) LR.dot(ctx, ST.sx(xs[i]), ST.sy(ts[i]), 4, C.text);
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.orange;
      ctx.fillText("F after m = " + stumps.length + " stumps" + (stumps.length === 0 ? " (F₀ = mean of t = " + LR.fmtF(tMean, 2) + ")" : ""), PT.x0 + 8, PT.y0 + 16);

      // bottom: residuals + next stump
      const SB = LR.plot(ctx, PB);
      ctx.strokeStyle = C.grid;
      for (let i = 0; i < n; i++) {
        LR.dot(ctx, SB.sx(xs[i]), SB.sy(Math.max(PB.ymin, Math.min(PB.ymax, r[i]))), 3.6, C.red);
      }
      if (nextStump) {
        ctx.save();
        ctx.beginPath(); ctx.rect(PB.x0, PB.y0, PB.w, PB.h); ctx.clip();
        ctx.strokeStyle = C.purple;
        ctx.lineWidth = 2.4;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(SB.sx(0), SB.sy(nextStump.left));
        ctx.lineTo(SB.sx(nextStump.thr), SB.sy(nextStump.left));
        ctx.lineTo(SB.sx(nextStump.thr), SB.sy(nextStump.right));
        ctx.lineTo(SB.sx(9), SB.sy(nextStump.right));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
        ctx.fillStyle = C.purple;
        ctx.fillText("next stump h (will be added × η = " + LR.fmtF(eta, 2) + ")", PB.x0 + 8, PB.y0 + 16);
      }
      ctx.fillStyle = C.red;
      ctx.fillText("residuals r = t − F", PB.x0 + PB.w - 150, PB.y0 + 16);

      ro.set("m", String(stumps.length));
      ro.set("mse", LR.fmtF(mse(), 4), C.orange);
      ro.set("eta", LR.fmtF(eta, 2));
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.4 — bagging vs boosting, side by side
     ════════════════════════════════════════════════════════════ */
  LR.figs.contrast = function (mount) {
    LR.header(
      mount,
      "Equal votes in parallel vs weighted votes in sequence",
      "Left: bagging trains 5 models at once on bootstrap resamples and gives each one vote. Right: boosting trains them one after another and pays each by quality. The α bars are real, from a real 5-round AdaBoost run."
    );

    // real αs: run AdaBoost 5 rounds on the same blobs as Fig 5.1
    const data = T.makeBlobs(29, 15, [2.2, 2.2], [3.9, 3.9], 0.9);
    const trainer = T.adaTrainer(data);
    const alphas = [];
    for (let i = 0; i < 5; i++) {
      const r = trainer.step();
      if (r) alphas.push(r.alpha);
    }

    const bar = LR.controls(mount);
    LR.button(bar, "Train ▸", play, "primary");
    LR.button(bar, "Reset ⟲", function () { prog = 0; if (anim) cancelAnimationFrame(anim); draw(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 380, {
      aria: "Schematic contrast of bagging (parallel training, equal votes) and boosting (sequential training, weighted votes with real alpha values)",
    });

    let prog = 0, anim = null; // 0..1

    function play() {
      if (anim) cancelAnimationFrame(anim);
      if (LR.reducedMotion) { prog = 1; draw(); return; }
      prog = 0;
      const tick = function () {
        prog = Math.min(1, prog + 0.012);
        draw();
        if (prog < 1) anim = requestAnimationFrame(tick);
      };
      tick();
    }

    function box(x, y, w, h, fill, label, sub) {
      ctx.fillStyle = fill;
      ctx.strokeStyle = "#cfcfcf";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = C.text;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x + w / 2, y + h / 2 + (sub ? -2 : 4));
      if (sub) {
        ctx.font = "500 10.5px Inter, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.fillText(sub, x + w / 2, y + h / 2 + 12);
      }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const mid = W / 2;
      // divider
      ctx.strokeStyle = "#e6e6e6";
      ctx.beginPath();
      ctx.moveTo(mid, 12);
      ctx.lineTo(mid, H - 12);
      ctx.stroke();

      ctx.font = "800 13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = C.text;
      ctx.fillText("BAGGING — parallel, equal votes", mid / 2, 26);
      ctx.fillText("BOOSTING — sequential, weighted votes", mid + mid / 2, 26);

      const bw = 58, bh = 40, gap = 76;
      // ── bagging pane ──
      box(mid / 2 - 40, 44, 80, 32, "#f2f2f2", "dataset D", "");
      for (let i = 0; i < 5; i++) {
        const cx = mid / 2 + (i - 2) * gap;
        const done = prog >= 1 || prog > 0.25; // all together
        const fill = prog === 0 ? "#ffffff" : done ? "rgba(102,102,102,0.15)" : "#ffffff";
        // sample arrow
        ctx.strokeStyle = "#bbbbbb";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(mid / 2, 78);
        ctx.lineTo(cx, 116);
        ctx.stroke();
        box(cx - bw / 2, 116, bw, bh, fill, "model " + (i + 1), "bootstrap Dᵢ");
        // equal vote bar
        const vh = 34 * (prog > 0.6 ? 1 : prog / 0.6);
        ctx.fillStyle = C.green;
        ctx.fillRect(cx - 10, 236 - vh, 20, vh);
        ctx.font = "600 10.5px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.muted;
        ctx.fillText("1", cx, 250);
      }
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.text;
      ctx.fillText("Y(x) = sign( Σ yₘ(x) )", mid / 2, 286);
      ctx.font = "500 11.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.fillText("all 5 train at the same time · mainly cuts variance", mid / 2, 306);

      // ── boosting pane ──
      for (let i = 0; i < 5; i++) {
        const cx = mid + mid / 2 + (i - 2) * gap;
        const t = Math.max(0, Math.min(1, (prog - i * 0.18) / 0.18)); // sequential fill
        const fill = t >= 1 ? "rgba(26,26,26,0.14)" : "#ffffff";
        box(cx - bw / 2, 116, bw, bh, fill, "model " + (i + 1), "ε" + (i + 1) + " → α" + (i + 1));
        if (i < 4) {
          const t2 = Math.max(0, Math.min(1, (prog - (i + 1) * 0.18) / 0.1));
          if (t2 > 0) LR.arrow(ctx, cx + bw / 2 + 2, 136, cx + gap - bw / 2 - 4, 136, C.orange, 1.8);
          if (t2 > 0) {
            ctx.font = "500 10px Inter, sans-serif";
            ctx.fillStyle = C.orange;
            ctx.textAlign = "center";
            ctx.fillText("reweight", cx + gap / 2, 128);
          }
        }
        // weighted vote bar: height ∝ real α
        const a = alphas[i] !== undefined ? alphas[i] : 0;
        const maxA = Math.max.apply(null, alphas);
        const vh = (34 * a) / maxA;
        const tv = Math.max(0, Math.min(1, (prog - i * 0.18 - 0.1) / 0.1));
        ctx.fillStyle = C.orange;
        ctx.fillRect(cx - 10, 236 - vh * tv, 20, vh * tv);
        ctx.font = "600 10.5px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText(LR.fmtF(a, 2), cx, 250);
      }
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.text;
      ctx.fillText("Y(x) = sign( Σ αₘ yₘ(x) )", mid + mid / 2, 286);
      ctx.font = "500 11.5px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.fillText("each model trains on the last one's mistakes · mainly cuts bias", mid + mid / 2, 306);

      ctx.font = "500 11px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.fillText("vote-weight bars: bagging gives every model 1; boosting pays by quality (real α from a 5-round run)", W / 2, H - 14);
    }
    draw();
  };
})();
