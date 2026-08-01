/* ══════════════════════════════════════════════════════════════
   figures-knn-b.js — kNN lesson figures, sections 3–5
   Fig 3.1 playground · Fig 3.2 regression · Fig 3.3 codeex
   Fig 4.1 validation · Fig 5.1 curse · Fig 5.2 standardize
   Fig 5.3 cost
   Uses the shared kNN core (LR.knn) from figures-knn-a.js.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const K = LR.knn;

  const CLS_COLOR = { A: C.green, B: C.purple, C: C.amber };
  const CLS_SOFT = { A: "rgba(47,158,68,0.20)", B: "rgba(112,72,232,0.17)", C: "rgba(240,162,2,0.22)" };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the k-NN classifier playground (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.playground = function (mount) {
    LR.header(
      mount,
      "The k dial",
      "Drag the query, slide k. Watch the neighbours, the vote, and the whole decision boundary respond. Jagged at k = 1, buttery at k = 25."
    );

    // three classes with deliberate overlap + a couple of noisy points
    const rand = LR.rng(17);
    const pts = [];
    for (let i = 0; i < 13; i++) pts.push({ x: 1.0 + rand() * 2.3, y: 3.4 + rand() * 2.3, cls: "A" });
    for (let i = 0; i < 13; i++) pts.push({ x: 3.4 + rand() * 2.4, y: 3.2 + rand() * 2.5, cls: "B" });
    for (let i = 0; i < 12; i++) pts.push({ x: 2.2 + rand() * 2.4, y: 0.6 + rand() * 2.2, cls: "C" });
    // class noise (real datasets have it; this is what small k overfits to)
    pts.push({ x: 4.6, y: 5.2, cls: "C" });
    pts.push({ x: 1.6, y: 1.2, cls: "B" });

    let k = 1;
    let q = { x: 3.0, y: 3.1 };

    const bar = LR.controls(mount);
    const kS = LR.slider(bar, "k (neighbours)", 1, 25, 2, k, function (v) {
      k = Math.round(v);
      boundaryDirty = true;
      draw();
    }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 470, {
      aria: "k-nearest-neighbour classifier playground with draggable query point, k slider, vote tally, and live decision boundary",
    });

    const tallyBox = LR.el("div", "vote-tally");
    mount.appendChild(tallyBox);

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 6.5, ymin: 0, ymax: 6.5, xlabel: "x₁", ylabel: "x₂" };
    let SC = null;

    // decision boundary cached on an offscreen canvas (depends on k, not on q)
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const offCtx = off.getContext("2d");
    let boundaryDirty = true;

    function renderBoundary() {
      offCtx.clearRect(0, 0, W, H);
      const BLK = 6;
      for (let px = P.x0; px < P.x0 + P.w; px += BLK) {
        for (let py = P.y0; py < P.y0 + P.h; py += BLK) {
          const wx = P.xmin + ((px + BLK / 2 - P.x0) / P.w) * (P.xmax - P.xmin);
          const wy = P.ymin + ((P.y0 + P.h - (py + BLK / 2)) / P.h) * (P.ymax - P.ymin);
          const v = K.vote(K.nearest(pts, { x: wx, y: wy }, k));
          offCtx.fillStyle = CLS_SOFT[v.label];
          offCtx.fillRect(px, py, BLK, BLK);
        }
      }
      boundaryDirty = false;
    }

    LR.drag(cv, W, H, {
      hit: (p) => SC && (p.x - SC.sx(q.x)) ** 2 + (p.y - SC.sy(q.y)) ** 2 < 2200,
      move: function (p) {
        const d = SC.inv(p.x, p.y);
        q.x = Math.max(0.15, Math.min(6.35, d.x));
        q.y = Math.max(0.15, Math.min(6.35, d.y));
        draw();
      },
    });

    function draw() {
      if (boundaryDirty) renderBoundary();
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(off, 0, 0, W, H);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      const nbrs = K.nearest(pts, q, k);
      const v = K.vote(nbrs);
      const nbrSet = new Set(nbrs.map((n) => n.p));

      // neighbour links
      ctx.strokeStyle = "rgba(232,89,12,0.45)";
      ctx.lineWidth = 1.4;
      nbrs.forEach(function (n) {
        ctx.beginPath();
        ctx.moveTo(sx(q.x), sy(q.y));
        ctx.lineTo(sx(n.p.x), sy(n.p.y));
        ctx.stroke();
      });

      // training points
      pts.forEach(function (p) {
        const isNbr = nbrSet.has(p);
        LR.dot(ctx, sx(p.x), sy(p.y), isNbr ? 7 : 5, CLS_COLOR[p.cls], isNbr ? C.orange : "#fff");
      });

      // query
      LR.dot(ctx, sx(q.x), sy(q.y), 10, CLS_COLOR[v.label], C.orange);
      ctx.font = "800 12px Inter, sans-serif";
      ctx.fillStyle = "#fff"; ctx.textAlign = "center";
      ctx.fillText("?", sx(q.x), sy(q.y) + 4);

      // tally pills
      let html = "<span style='font-weight:700'>vote (k = " + k + "):</span>";
      ["A", "B", "C"].forEach(function (c) {
        const n = v.tally[c] || 0;
        html += "<span class='vote-pill" + (c === v.label ? " winner" : "") + "'>" +
          "<span class='swatch' style='background:" + CLS_COLOR[c] + "'></span>" +
          c + " · " + n + (c === v.label ? " ✓" : "") + "</span>";
      });
      tallyBox.innerHTML = html;
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.2 — k-NN regression
     ════════════════════════════════════════════════════════════ */
  LR.figs.regression = function (mount) {
    LR.header(
      mount,
      "Vote → average",
      "Continuous targets, same neighbour logic. The predicted curve is drawn across every x, computed from the real data."
    );

    // 1-D data from a wiggly truth + noise
    const rand = LR.rng(41);
    const data = [];
    for (let i = 0; i < 12; i++) {
      const x = 0.3 + (i / 11) * 5.4 + (rand() - 0.5) * 0.2;
      const y = 2.6 + 1.6 * Math.sin(x * 1.25) + LR.gauss(rand) * 0.28;
      data.push({ x, y });
    }

    let k = 1, weighted = false;

    const bar = LR.controls(mount);
    LR.slider(bar, "k (neighbours)", 1, 8, 1, k, function (v) { k = Math.round(v); draw(); }, (v) => String(Math.round(v)));
    const wBtn = LR.button(bar, "Distance-weighted: off", function () {
      weighted = !weighted;
      wBtn.textContent = "Distance-weighted: " + (weighted ? "on" : "off");
      wBtn.classList.toggle("on", weighted);
      draw();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 420, {
      aria: "One-dimensional k-nearest-neighbour regression with k slider and distance-weighting toggle",
    });

    const P = { x0: 60, y0: 20, w: W - 84, h: H - 72, xmin: 0, xmax: 6, ymin: 0, ymax: 5.5, xlabel: "x", ylabel: "t" };

    function predict(x) {
      const nbrs = data
        .map((p) => ({ p, d: Math.abs(p.x - x) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, k);
      if (!weighted) {
        return nbrs.reduce((s, n) => s + n.p.y, 0) / nbrs.length;
      }
      // inverse-distance weights; snap to an exact hit
      let num = 0, den = 0;
      for (const n of nbrs) {
        if (n.d < 1e-9) return n.p.y;
        num += n.p.y / n.d;
        den += 1 / n.d;
      }
      return num / den;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const { sx, sy } = LR.plot(ctx, P);

      ctx.save();
      ctx.beginPath(); ctx.rect(P.x0, P.y0, P.w, P.h); ctx.clip();

      // predicted function
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let i = 0; i <= 300; i++) {
        const x = P.xmin + ((P.xmax - P.xmin) * i) / 300;
        const y = predict(x);
        i === 0 ? ctx.moveTo(sx(x), sy(y)) : ctx.lineTo(sx(x), sy(y));
      }
      ctx.stroke();
      ctx.restore();

      // data
      data.forEach((p) => LR.dot(ctx, sx(p.x), sy(p.y), 5, C.text));

      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "left";
      ctx.fillText(
        "prediction: " + (k === 1 ? "step function (nearest target)" : (weighted ? "inverse-distance weighted average of " + k : "plain average of " + k)),
        P.x0 + 10, P.y0 + 16
      );
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.3 — interactive knn_classify code exercise
     ════════════════════════════════════════════════════════════ */
  LR.figs.codeex = function (mount) {
    LR.header(
      mount,
      "Run the classifier yourself",
      "The function below actually executes against the fixed training set. Change k and the query, press Run."
    );

    // the tiny dataset from the worked vote (constructed for this exercise)
    const X = [
      { x: 1.0, y: 1.0, cls: "A" }, { x: 1.5, y: 2.0, cls: "A" },
      { x: 2.0, y: 1.0, cls: "A" }, { x: 1.0, y: 3.0, cls: "A" },
      { x: 4.0, y: 1.5, cls: "B" }, { x: 3.5, y: 0.5, cls: "B" },
      { x: 4.5, y: 2.5, cls: "B" },
    ];

    const CODE =
      "X_train = np.array([[1,1],[1.5,2],[2,1],[1,3],[4,1.5],[3.5,0.5],[4.5,2.5]])\n" +
      "y_train = np.array(['A','A','A','A','B','B','B'])\n" +
      "\n" +
      "def knn_classify(X_train, y_train, x_query, k):\n" +
      "    dists = np.sqrt(((X_train - x_query) ** 2).sum(axis=1))\n" +
      "    nearest = np.argsort(dists)[:k]      # indices of the k closest\n" +
      "    votes = Counter(y_train[nearest])    # tally the labels\n" +
      "    return votes.most_common(1)[0][0]    # majority wins";

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);

    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);
    let k = 3, qx = 3.0, qy = 1.0;
    const kSel = LR.el("label", "", "k = ");
    const sel = document.createElement("select");
    [1, 3, 5, 7].forEach((v) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v;
      if (v === 3) o.selected = true;
      sel.appendChild(o);
    });
    sel.style.cssText = "font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid #ccc";
    sel.addEventListener("change", () => { k = parseInt(sel.value, 10); });
    kSel.appendChild(sel);
    controls.appendChild(kSel);
    const sx = LR.slider(controls, "query x₁", 0, 5.5, 0.25, qx, (v) => { qx = v; }, (v) => LR.fmtF(v, 2));
    const sy = LR.slider(controls, "query x₂", 0, 4, 0.25, qy, (v) => { qy = v; }, (v) => LR.fmtF(v, 2));
    const runBtn = LR.button(controls, "Run ▸", run, "primary small");

    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; knn_classify(X_train, y_train, np.array([3.0, 1.0]), k=3)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    // mini canvas: dataset + query
    const { cv, ctx, W, H } = LR.canvas(mount, 500, 260, {
      aria: "Tiny training set with the current query point",
    });
    cv.style.marginTop = "14px";
    const P = { x0: 46, y0: 14, w: W - 62, h: H - 56, xmin: 0, xmax: 5.5, ymin: 0, ymax: 4, xlabel: "x₁", ylabel: "x₂" };

    function run() {
      const q = { x: qx, y: qy };
      const nbrs = K.nearest(X, q, k);
      const v = K.vote(nbrs);
      const tallyStr = ["A", "B"].map((c) => c + ": " + (v.tally[c] || 0)).join(", ");
      const distStr = nbrs.map((n) => n.p.cls + "@" + LR.fmtF(n.d, 2)).join("  ");
      out.innerHTML =
        "&gt;&gt;&gt; knn_classify(X_train, y_train, np.array([" + LR.fmtF(qx, 2) + ", " + LR.fmtF(qy, 2) + "]), k=" + k + ")\n" +
        "k nearest: " + distStr + "\n" +
        "votes: {" + tallyStr + "}\n" +
        "'<b>" + v.label + "</b>'";
      draw(nbrs, v);
    }

    function draw(nbrs, v) {
      ctx.clearRect(0, 0, W, H);
      const SC = LR.plot(ctx, P);
      const nbrSet = nbrs ? new Set(nbrs.map((n) => n.p)) : new Set();
      X.forEach(function (p) {
        const isN = nbrSet.has(p);
        LR.dot(ctx, SC.sx(p.x), SC.sy(p.y), isN ? 7 : 5, CLS_COLOR[p.cls], isN ? C.orange : "#fff");
      });
      LR.dot(ctx, SC.sx(qx), SC.sy(qy), 8, v ? CLS_COLOR[v.label] : "#fff", C.orange);
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("green = A, purple = B, orange ring = query", P.x0 + 8, P.y0 + 14);
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — the validation curve, with the reader in the loop
     ════════════════════════════════════════════════════════════ */
  LR.figs.validation = function (mount) {
    LR.header(
      mount,
      "Pick k like you mean it",
      "Real three-way split, real error curves. Click the k you would ship, judged only by the validation curve; the test curve stays hidden until you commit."
    );

    // dataset: two overlapping 2-D Gaussians, split 60/40/40
    const rand = LR.rng(97);
    function makePoint(cls) {
      const cx = cls === "A" ? 2.0 : 3.2, cy = cls === "A" ? 2.0 : 3.0;
      return { x: cx + LR.gauss(rand) * 0.85, y: cy + LR.gauss(rand) * 0.85, cls };
    }
    const all = [];
    for (let i = 0; i < 70; i++) all.push(makePoint("A"));
    for (let i = 0; i < 70; i++) all.push(makePoint("B"));
    // deterministic shuffle
    for (let i = all.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = all[i]; all[i] = all[j]; all[j] = t;
    }
    const TR = all.slice(0, 60), VL = all.slice(60, 100), TS = all.slice(100, 140);

    const KS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 39, 49, 59];
    function errOn(set, k) {
      let wrong = 0;
      for (const p of set) {
        // classify against TR; exclude the point itself when it is in TR
        const pool = set === TR ? TR.filter((t) => t !== p) : TR;
        const v = K.vote(K.nearest(pool, p, Math.min(k, pool.length)));
        if (v.label !== p.cls) wrong++;
      }
      return wrong / set.length;
    }
    const curves = { tr: KS.map((k) => errOn(TR, k)), vl: KS.map((k) => errOn(VL, k)), ts: KS.map((k) => errOn(TS, k)) };
    // NOTE: training error uses leave-self-out except at the k=1 story point,
    // where the lecture's "each point is its own neighbour" trap is shown directly:
    curves.tr[0] = 0; // k=1 training error with self included: exactly zero, the trap

    const bestIdx = curves.vl.indexOf(Math.min.apply(null, curves.vl));

    let picked = null, revealed = false;

    const bar = LR.controls(mount);
    LR.button(bar, "Reset ⟲", function () { picked = null; revealed = false; msg.hide(); draw(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 430, {
      aria: "Training and validation error versus k; click to choose k, then the hidden test curve is revealed",
    });
    const msg = LR.msg(mount);

    const ymax = Math.max(0.45, Math.max.apply(null, curves.vl) * 1.15);
    const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: 0, xmax: KS.length - 1, ymin: 0, ymax: ymax, xlabel: "k (position on the sweep " + KS.join(", ") + ")", ylabel: "error rate" };
    let SC = null;

    LR.drag(cv, W, H, {
      hit: (p) => SC && p.x > P.x0 && p.x < P.x0 + P.w && p.y > P.y0 && p.y < P.y0 + P.h,
      down: function (p) {
        const xi = Math.round(SC.inv(p.x, p.y).x);
        picked = Math.max(0, Math.min(KS.length - 1, xi));
        revealed = true;
        const pickedErr = curves.vl[picked];
        const bestErr = curves.vl[bestIdx];
        if (picked === bestIdx || pickedErr <= bestErr + 0.005) {
          msg.show("✓ k = " + KS[picked] + " is the validation optimum (error " + LR.fmtF(pickedErr * 100, 1) + "%). Test error at your k: " + LR.fmtF(curves.ts[picked] * 100, 1) + "%. That is the number you report, once.", "good");
        } else {
          msg.show("You picked k = " + KS[picked] + " (validation error " + LR.fmtF(pickedErr * 100, 1) + "%). The sweep's best is k = " + KS[bestIdx] + " at " + LR.fmtF(bestErr * 100, 1) + "%. Note the trap: k = 1 has zero training error and mediocre validation error.", "info");
        }
        draw();
      },
      move: function () {},
    });

    function plotCurve(arr, color, dash) {
      const { sx, sy } = SC;
      ctx.strokeStyle = color; ctx.lineWidth = 2.2;
      if (dash) ctx.setLineDash([6, 4]);
      ctx.beginPath();
      arr.forEach(function (v, i) {
        i === 0 ? ctx.moveTo(sx(i), sy(v)) : ctx.lineTo(sx(i), sy(v));
      });
      ctx.stroke();
      ctx.setLineDash([]);
      arr.forEach((v, i) => LR.dot(ctx, SC.sx(i), SC.sy(v), 3, color));
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      P.xticks = KS.map((_, i) => i).filter((i) => i % 2 === 0);
      SC = LR.plot(ctx, P);
      const { sx, sy } = SC;

      // custom x tick labels (actual k values)
      ctx.font = "10.5px Inter, sans-serif";
      ctx.fillStyle = "#fff";
      ctx.fillRect(P.x0 - 10, P.y0 + P.h + 6, P.w + 20, 14);
      ctx.fillStyle = C.faint; ctx.textAlign = "center";
      KS.forEach(function (k, i) {
        if (i % 2 === 0) ctx.fillText(String(k), sx(i), P.y0 + P.h + 16);
      });

      plotCurve(curves.tr, C.green);
      plotCurve(curves.vl, C.orange);
      if (revealed) plotCurve(curves.ts, C.red, true);

      // trap annotation at k=1
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "left";
      ctx.fillText("training error = 0 at k=1: the trap", sx(0) + 8, sy(curves.tr[0]) - 8);

      // legend
      ctx.fillStyle = C.green; ctx.fillText("— train", P.x0 + P.w - 130, P.y0 + 16);
      ctx.fillStyle = C.orange; ctx.fillText("— validation", P.x0 + P.w - 130, P.y0 + 32);
      ctx.fillStyle = revealed ? C.red : "#bbb";
      ctx.fillText(revealed ? "- - test (revealed)" : "- - test (hidden)", P.x0 + P.w - 130, P.y0 + 48);

      // best-k marker and the user's pick
      if (revealed) {
        LR.dot(ctx, sx(bestIdx), sy(curves.vl[bestIdx]), 7, "#fff", C.orange);
        ctx.fillStyle = C.orange;
        ctx.fillText("best on validation: k=" + KS[bestIdx], sx(bestIdx) + 10, sy(curves.vl[bestIdx]) + 4);
      }
      if (picked !== null) {
        ctx.strokeStyle = C.text; ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(sx(picked), sy(0)); ctx.lineTo(sx(picked), sy(P.ymax)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.text;
        ctx.fillText("your pick", sx(picked) + 6, P.y0 + 14);
      }
      if (!revealed) {
        ctx.font = "700 13px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "center";
        ctx.fillText("click the k you would ship", P.x0 + P.w / 2, P.y0 + P.h / 2 - 40);
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — the curse of dimensionality, sampled live
     ════════════════════════════════════════════════════════════ */
  LR.figs.curse = function (mount) {
    LR.header(
      mount,
      "Distance dies in high dimensions",
      "Real uniform samples in the d-cube, real pairwise distances. Watch the histogram collapse to a spike."
    );

    const DS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500];
    let di = 0;

    const bar = LR.controls(mount);
    LR.slider(bar, "d (dimensions)", 0, DS.length - 1, 1, 0, function (v) {
      di = Math.round(v);
      recompute();
    }, (v) => String(DS[Math.round(v)]));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "Histogram of pairwise distances between random points collapsing as dimension grows",
    });
    const ro = LR.readout(mount, [
      { k: "contrast", label: "(max − min) / min" },
      { k: "edge", label: "edge for 10 of 1000 nbrs: l = (k/n)^(1/d)" },
      { k: "verdict", label: "is “nearest” meaningful?" },
    ]);

    const NPTS = 80;

    let hist = null, contrast = 0;

    function recompute() {
      const d = DS[di];
      const rand = LR.rng(1000 + d);
      // sample points
      const pts = [];
      for (let i = 0; i < NPTS; i++) {
        const v = new Array(d);
        for (let j = 0; j < d; j++) v[j] = rand();
        pts.push(v);
      }
      // pairwise distances, normalized by the mean so the x-axis is comparable
      const dists = [];
      let min = Infinity, max = 0, sum = 0;
      for (let i = 0; i < NPTS; i++) {
        for (let j = i + 1; j < NPTS; j++) {
          let s = 0;
          for (let m = 0; m < d; m++) { const g = pts[i][m] - pts[j][m]; s += g * g; }
          const dist = Math.sqrt(s);
          dists.push(dist);
          if (dist < min) min = dist;
          if (dist > max) max = dist;
          sum += dist;
        }
      }
      const mean = sum / dists.length;
      contrast = (max - min) / min;

      // histogram of dist/mean over [0, 2]
      const BINS = 60;
      hist = new Array(BINS).fill(0);
      dists.forEach(function (dist) {
        const b = Math.max(0, Math.min(BINS - 1, Math.floor((dist / mean / 2) * BINS)));
        hist[b] += 1;
      });
      const peak = Math.max.apply(null, hist);
      hist = hist.map((h) => h / peak);

      draw();

      const l = Math.pow(10 / 1000, 1 / d);
      ro.set("contrast", LR.fmtF(contrast, 2) + (contrast < 0.6 ? "  ← everyone is “nearest”" : ""), contrast < 0.6 ? C.red : C.green);
      ro.set("edge", LR.fmtF(l, 3) + (l > 0.9 ? "  ← “local” spans the whole cube" : ""), l > 0.9 ? C.red : C.green);
      ro.set("verdict", d <= 5 ? "yes: distances discriminate" : d <= 50 ? "fading fast" : "no: near and far have merged", d <= 5 ? C.green : d <= 50 ? C.amber : C.red);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const P = { x0: 62, y0: 22, w: W - 90, h: H - 76, xmin: 0, xmax: 2, ymin: 0, ymax: 1.05, xlabel: "pairwise distance ÷ mean distance", ylabel: "relative frequency" };
      const { sx, sy } = LR.plot(ctx, P);

      const BINS = hist.length;
      const bw = P.w / BINS;
      for (let b = 0; b < BINS; b++) {
        const h = hist[b] * (P.h - 20);
        ctx.fillStyle = "rgba(112,72,232,0.55)";
        ctx.fillRect(P.x0 + b * bw + 1, P.y0 + P.h - h, bw - 2, h);
      }

      // marker at 1 (the mean)
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(sx(1), sy(0)); ctx.lineTo(sx(1), sy(1.02)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillStyle = C.orange; ctx.textAlign = "left";
      ctx.fillText("mean", sx(1) + 6, P.y0 + 16);

      ctx.fillStyle = C.text;
      ctx.fillText("d = " + DS[di] + "   (" + NPTS + " points, " + (NPTS * (NPTS - 1)) / 2 + " real pairwise distances)", P.x0 + 8, P.y0 + 16);
    }

    recompute();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — standardization changes who your neighbours are
     ════════════════════════════════════════════════════════════ */
  LR.figs.standardize = function (mount) {
    LR.header(
      mount,
      "Units are weights in disguise",
      "Salary spans tens of thousands; age spans tens. Raw Euclidean distance only hears the salary. Standardize and the neighbours change."
    );

    // people: salary ($k, 20–120) and age (20–65)
    const rand = LR.rng(52);
    const ppl = [];
    for (let i = 0; i < 16; i++) {
      ppl.push({ sal: 25 + rand() * 90, age: 21 + rand() * 42 });
    }
    const q = { sal: 62, age: 33 };

    const musd = (function () {
      const n = ppl.length;
      let ms = 0, ma = 0;
      ppl.forEach((p) => { ms += p.sal; ma += p.age; });
      ms /= n; ma /= n;
      let vs = 0, va = 0;
      ppl.forEach((p) => { vs += (p.sal - ms) ** 2; va += (p.age - ma) ** 2; });
      return { ms, ma, ss: Math.sqrt(vs / n), sa: Math.sqrt(va / n) };
    })();

    function nbrsRaw() {
      return ppl
        .map((p) => ({ p, d: Math.hypot(p.sal - q.sal, p.age - q.age) }))
        .sort((a, b) => a.d - b.d).slice(0, 3);
    }
    function nbrsStd() {
      const zq = { x: (q.sal - musd.ms) / musd.ss, y: (q.age - musd.ma) / musd.sa };
      return ppl
        .map((p) => ({ p, d: Math.hypot((p.sal - musd.ms) / musd.ss - zq.x, (p.age - musd.ma) / musd.sa - zq.y) }))
        .sort((a, b) => a.d - b.d).slice(0, 3);
    }

    const pane = LR.el("div", "dual-pane");
    mount.appendChild(pane);
    const lb = LR.el("div"); lb.appendChild(LR.el("div", "pane-label", "Raw units: salary owns the vote"));
    const rb = LR.el("div"); rb.appendChild(LR.el("div", "pane-label", "Standardized: both features speak"));
    pane.appendChild(lb); pane.appendChild(rb);
    const LC = LR.canvas(lb, 400, 330, { aria: "Raw-unit scatter where nearest neighbours are chosen almost entirely by salary" });
    const RC = LR.canvas(rb, 400, 330, { aria: "Standardized scatter where nearest neighbours reflect both features" });

    const ro = LR.readout(mount, [
      { k: "mu", label: "salary μ, σ" },
      { k: "mu2", label: "age μ, σ" },
      { k: "diff", label: "neighbour sets" },
    ]);

    function panel(cv, ctx, W, H, std) {
      const P = std
        ? { x0: 50, y0: 14, w: W - 66, h: H - 58, xmin: -2.5, xmax: 2.5, ymin: -2.5, ymax: 2.5, xlabel: "salary (standardized)", ylabel: "age (standardized)" }
        : { x0: 50, y0: 14, w: W - 66, h: H - 58, xmin: 0, xmax: 130, ymin: 0, ymax: 70, xlabel: "salary ($k)", ylabel: "age (years)" };
      const SC = LR.plot(ctx, P);
      const nbrs = std ? nbrsStd() : nbrsRaw();
      const nset = new Set(nbrs.map((n) => n.p));
      const toXY = std
        ? (p) => ({ x: (p.sal - musd.ms) / musd.ss, y: (p.age - musd.ma) / musd.sa })
        : (p) => ({ x: p.sal, y: p.age });
      const qxy = std ? { x: (q.sal - musd.ms) / musd.ss, y: (q.age - musd.ma) / musd.sa } : { x: q.sal, y: q.age };

      // links
      ctx.strokeStyle = "rgba(232,89,12,0.5)"; ctx.lineWidth = 1.4;
      nbrs.forEach(function (n) {
        const c = toXY(n.p);
        ctx.beginPath(); ctx.moveTo(SC.sx(qxy.x), SC.sy(qxy.y)); ctx.lineTo(SC.sx(c.x), SC.sy(c.y)); ctx.stroke();
      });
      ppl.forEach(function (p) {
        const c = toXY(p);
        LR.dot(ctx, SC.sx(c.x), SC.sy(c.y), nset.has(p) ? 6.5 : 4.5, nset.has(p) ? C.green : C.text, nset.has(p) ? C.orange : "#fff");
      });
      LR.dot(ctx, SC.sx(qxy.x), SC.sy(qxy.y), 8, "#fff", C.orange);
      return nbrs;
    }

    const n1 = panel(LC.cv, LC.ctx, LC.W, LC.H, false);
    const n2 = panel(RC.cv, RC.ctx, RC.W, RC.H, true);
    const shared = n1.filter((a) => n2.some((b) => b.p === a.p)).length;

    ro.set("mu", "μ = " + LR.fmtF(musd.ms, 1) + "k, σ = " + LR.fmtF(musd.ss, 1) + "k");
    ro.set("mu2", "μ = " + LR.fmtF(musd.ma, 1) + " yrs, σ = " + LR.fmtF(musd.sa, 1) + " yrs");
    ro.set("diff", shared + " of 3 neighbours survive standardization", shared < 3 ? C.red : C.green);
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.3 — the cost of laziness
     ════════════════════════════════════════════════════════════ */
  LR.figs.cost = function (mount) {
    LR.header(
      mount,
      "One prediction, itemized bill",
      "Training cost: zero. Query cost: a distance to every stored point, then a sort. Slide N and run the query."
    );

    let N = 500;
    const D = 16;

    const bar = LR.controls(mount);
    LR.slider(bar, "N (stored examples)", 100, 5000, 100, N, function (v) { N = Math.round(v); reset(); }, (v) => String(Math.round(v)));
    LR.button(bar, "Run one query ▸", run, "primary");
    LR.button(bar, "Reset ⟲", reset);

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 300, {
      aria: "Animated operation counter for a single k-nearest-neighbour query on a growing dataset",
    });
    const ro = LR.readout(mount, [
      { k: "train", label: "training ops" },
      { k: "dist", label: "distance ops O(N·D)" },
      { k: "sort", label: "sort ops O(N log N)" },
      { k: "total", label: "total for ONE prediction" },
    ]);

    let phase = "idle", prog = 0, raf = null;

    function reset() {
      if (raf) cancelAnimationFrame(raf);
      phase = "idle"; prog = 0;
      update(); draw();
    }
    function run() {
      if (raf) cancelAnimationFrame(raf);
      phase = "dist"; prog = 0;
      const speed = LR.reducedMotion ? 0.05 : 0.014;
      const tick = function () {
        prog += speed;
        if (phase === "dist" && prog >= 1) { phase = "sort"; prog = 0; }
        else if (phase === "sort" && prog >= 1) { phase = "done"; prog = 1; }
        update(); draw();
        if (phase !== "done") raf = requestAnimationFrame(tick);
      };
      tick();
    }

    function counts() {
      const distOps = N * D;
      const sortOps = Math.round(N * Math.log2(Math.max(2, N)));
      return { distOps, sortOps };
    }

    function update() {
      const { distOps, sortOps } = counts();
      const dDone = phase === "idle" ? 0 : phase === "dist" ? Math.round(distOps * prog) : distOps;
      const sDone = phase === "sort" ? Math.round(sortOps * prog) : phase === "done" ? sortOps : 0;
      ro.set("train", "0  (lazy: training = storing)", C.green);
      ro.set("dist", dDone.toLocaleString("en-CA") + " / " + distOps.toLocaleString("en-CA"), phase === "dist" ? C.orange : undefined);
      ro.set("sort", sDone.toLocaleString("en-CA") + " / " + sortOps.toLocaleString("en-CA"), phase === "sort" ? C.orange : undefined);
      ro.set("total", (dDone + sDone).toLocaleString("en-CA") + (phase === "done" ? ", for a single prediction" : ""), phase === "done" ? C.red : undefined);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const { distOps, sortOps } = counts();

      // stored dataset strip
      const rows = Math.min(N, 240);
      const perRow = Math.ceil(N / rows);
      const stripX = 30, stripW = W - 60, stripY = 60, stripH = 60;
      ctx.font = "600 12.5px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("stored training set: N = " + N.toLocaleString("en-CA") + " points × D = " + D + " features (all kept in memory, forever)", stripX, stripY - 12);

      const cw = stripW / rows;
      const scanned = phase === "idle" ? 0 : phase === "dist" ? Math.floor(rows * prog) : rows;
      for (let i = 0; i < rows; i++) {
        ctx.fillStyle = i < scanned ? C.orange : "#ececec";
        ctx.fillRect(stripX + i * cw, stripY, Math.max(1, cw - 0.6), stripH);
      }

      // phase captions
      ctx.font = "700 13.5px Inter, sans-serif";
      ctx.fillStyle = C.text;
      if (phase === "idle") ctx.fillText("press Run: the query arrives, and only now does KNN start working", stripX, stripY + stripH + 34);
      else if (phase === "dist") ctx.fillText("computing a " + D + "-dimensional distance to every stored point…", stripX, stripY + stripH + 34);
      else if (phase === "sort") ctx.fillText("sorting " + N.toLocaleString("en-CA") + " distances to find the k closest…", stripX, stripY + stripH + 34);
      else ctx.fillText("done. " + (distOps + sortOps).toLocaleString("en-CA") + " operations, one answer. Approximate KNN structures exist to dodge exactly this.", stripX, stripY + stripH + 34);

      // sort progress bar
      if (phase === "sort" || phase === "done") {
        const bw = stripW * (phase === "done" ? 1 : prog);
        ctx.fillStyle = "#ececec";
        ctx.fillRect(stripX, stripY + stripH + 50, stripW, 16);
        ctx.fillStyle = C.purple;
        ctx.fillRect(stripX, stripY + stripH + 50, bw, 16);
        ctx.font = "600 11px Inter, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.fillText("sort", stripX, stripY + stripH + 80);
      }
    }

    reset();
  };
})();
