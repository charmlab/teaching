/* ══════════════════════════════════════════════════════════════
   figures-b.js — Language Models lesson figures, sections 4–7
   Fig 4.1 densify · Fig 4.2 embedgeo · Fig 5.1 neurallm
   Fig 5.2 generalize · Fig 6.1 decoding · Fig 7.1 perplexity
   Uses the shared language core LR.lm from figures-a.js.
   All numbers computed live: the co-occurrence factorization and
   the neural trigram are trained in the browser; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const LM = LR.lm;

  /* ════════════════════════════════════════════════════════════
     shared: co-occurrence corpus + low-rank factorization
     (Fig 4.1). Synthetic mini-corpus written for this lesson so
     foods share contexts with foods and instruments with
     instruments, without being identical.
     ════════════════════════════════════════════════════════════ */
  const COOC_CORPUS = [
    "we ate pizza yesterday",
    "he ate a burger today",
    "she ordered sushi again",
    "we cooked pasta yesterday",
    "the pizza tastes great",
    "that burger tastes salty",
    "the sushi tastes fresh",
    "her pasta tastes great",
    "he ordered pizza again",
    "she ate the burger",
    "we ate sushi today",
    "he cooked the pasta",
    "she played guitar today",
    "he played the piano",
    "she practised violin again",
    "he played drums yesterday",
    "the guitar sounds loud",
    "that piano sounds soft",
    "her violin sounds sweet",
    "the drums sounds loud",
    "we practised the guitar",
    "she practised piano today",
  ];

  let coocCache = null;
  function coocModel() {
    if (coocCache) return coocCache;
    const sents = LM.sentences(COOC_CORPUS.join("\n"));
    const vlist = LM.vocab(sents);
    const idx = new Map(vlist.map(function (w, i) { return [w, i]; }));
    const n = vlist.length;
    // window ±2 co-occurrence counts (symmetric)
    const X = [];
    for (let i = 0; i < n; i++) X.push(new Array(n).fill(0));
    sents.forEach(function (s) {
      for (let i = 0; i < s.length; i++)
        for (let j = Math.max(0, i - 2); j <= Math.min(s.length - 1, i + 2); j++)
          if (i !== j) X[idx.get(s[i])][idx.get(s[j])] += 1;
    });
    // top eigenpairs of X² (PSD, so power iteration orders by |λ| of X)
    // gives the best rank-d factorization X ≈ R Rᵀ in the SVD sense.
    const X2 = [];
    for (let i = 0; i < n; i++) {
      X2.push(new Array(n).fill(0));
      for (let k = 0; k < n; k++) {
        const a = X[i][k];
        if (!a) continue;
        for (let j = 0; j < n; j++) X2[i][j] += a * X[k][j];
      }
    }
    const DMAX = 6;
    const rand = LR.rng(11);
    const B = X2.map(function (r) { return r.slice(); });
    const eigs = [];
    for (let e = 0; e < DMAX; e++) {
      let v = [];
      for (let i = 0; i < n; i++) v.push(rand() - 0.5);
      for (let it = 0; it < 600; it++) {
        const w = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
          const Bi = B[i];
          for (let j = 0; j < n; j++) w[i] += Bi[j] * v[j];
        }
        let nrm = 0;
        for (let i = 0; i < n; i++) nrm += w[i] * w[i];
        nrm = Math.sqrt(nrm) || 1;
        v = w.map(function (x) { return x / nrm; });
      }
      let ray = 0;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += B[i][j] * v[j];
        ray += v[i] * s;
      }
      eigs.push({ val: Math.max(ray, 0), vec: v });
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) B[i][j] -= ray * v[i] * v[j];
    }
    // word i, dimension k:  e_i[k] = σ_k^{1/2} · v_k[i],  σ_k = sqrt(eig_k of X²)
    function embed(w, d) {
      const i = idx.get(w);
      return eigs.slice(0, d).map(function (e) {
        return Math.pow(e.val, 0.25) * e.vec[i];
      });
    }
    coocCache = { vlist, idx, X, embed, DMAX };
    return coocCache;
  }

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — context counts shrink to a dense embedding
     ════════════════════════════════════════════════════════════ */
  LR.figs.densify = function (mount) {
    LR.header(
      mount,
      "Compress a context-count vector into a dense embedding",
      "The co-occurrence matrix of the 22-sentence mini-corpus is factorized (low-rank, computed here in the browser); each word keeps only d learned coordinates."
    );

    const M = coocModel();
    const bar = LR.controls(mount);
    let word = "pizza", d = 3;
    LM.select(bar, "word", ["pizza", "burger", "sushi", "pasta", "guitar", "piano", "violin", "drums"], "pizza", function (v) { word = v; render(); });
    LR.slider(bar, "d (dense dims)", 1, M.DMAX, 1, 3, function (v) { d = Math.round(v); render(); },
      function (v) { return String(Math.round(v)); });

    const { ctx, W, H } = LR.canvas(mount, 820, 240, {
      aria: "Bar chart of the selected word's context-count vector over the full vocabulary",
    });

    const denseTitle = LR.el("div", "fig-sub", "");
    denseTitle.style.marginTop = "10px";
    mount.appendChild(denseTitle);
    const denseRow = LR.el("div", "vcells");
    mount.appendChild(denseRow);

    const tblWrap = LR.el("div", "wtable-wrap");
    mount.appendChild(tblWrap);
    const msg = LR.msg(mount);

    const PAIRS = [
      ["pizza", "burger", "similar"],
      ["pizza", "sushi", "similar"],
      ["guitar", "piano", "similar"],
      ["pizza", "guitar", "different"],
      ["burger", "violin", "different"],
    ];

    function cellBg(v, vmax) {
      const a = Math.min(1, Math.abs(v) / (vmax || 1)) * 0.75 + 0.1;
      return v >= 0 ? "rgba(26,26,26," + a.toFixed(2) + ")" : "rgba(150,150,150," + a.toFixed(2) + ")";
    }

    function render() {
      // count-vector bar chart
      ctx.clearRect(0, 0, W, H);
      const row = M.X[M.idx.get(word)];
      const vmax = Math.max.apply(null, row.concat([1]));
      const P = {
        x0: 46, y0: 16, w: W - 66, h: H - 76,
        xmin: 0, xmax: M.vlist.length, ymin: 0, ymax: vmax + 1,
        ylabel: "co-occurrences", xticks: [], yticks: LR.ticks(0, vmax + 1, 4),
      };
      const { sx, sy } = LR.plot(ctx, P);
      const bw = (P.w / M.vlist.length) * 0.62;
      let nz = 0;
      M.vlist.forEach(function (w2, i) {
        const v = row[i];
        if (v > 0) nz++;
        const x = sx(i + 0.5);
        ctx.fillStyle = v > 0 ? C.orange : "#e9e9e9";
        const y = sy(v);
        ctx.fillRect(x - bw / 2, Math.min(y, sy(0) - 1), bw, Math.max(sy(0) - y, v > 0 ? 2 : 1.2));
        ctx.save();
        ctx.translate(x, P.y0 + P.h + 10);
        ctx.rotate(-Math.PI / 3);
        ctx.font = "10px Inter, sans-serif";
        ctx.fillStyle = v > 0 ? C.text : "#b5b5b5";
        ctx.textAlign = "right";
        ctx.fillText(w2, 0, 3);
        ctx.restore();
      });
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillStyle = C.text;
      ctx.textAlign = "left";
      ctx.fillText("c_" + word + " ∈ ℝ^" + M.vlist.length + "  (" + nz + " nonzero of " + M.vlist.length + ")", P.x0, 12);

      // dense vector cells
      const e = M.embed(word, d);
      const emax = Math.max.apply(null, e.map(Math.abs).concat([1e-9]));
      denseTitle.innerHTML = "dense embedding e_" + word + " ∈ ℝ<sup>" + d + "</sup> (learned coordinates, not named context words):";
      denseRow.innerHTML = "";
      e.forEach(function (v) {
        const cell = LR.el("span", "vcell", v.toFixed(2));
        cell.style.background = cellBg(v, emax);
        cell.style.color = Math.abs(v) / emax > 0.55 ? "#ffffff" : "#222222";
        denseRow.appendChild(cell);
      });

      // cosine table, count space vs d-space, all live
      let html = '<table class="wtable"><tr><th style="text-align:left">pair</th><th>cos in count space (ℝ^' + M.vlist.length + ')</th><th>cos in dense space (ℝ^' + d + ")</th></tr>";
      let worstOk = true;
      PAIRS.forEach(function (pr) {
        const full = LM.cosine(M.X[M.idx.get(pr[0])], M.X[M.idx.get(pr[1])]);
        const dd = LM.cosine(M.embed(pr[0], d), M.embed(pr[1], d));
        if (pr[2] === "different" && dd > 0.55) worstOk = false;
        const hl = pr[0] === word || pr[1] === word;
        html += "<tr" + (hl ? ' style="background:var(--accent-soft)"' : "") + '><td style="text-align:left">' +
          pr[0] + " · " + pr[1] + " <span style='color:var(--faint)'>(" + pr[2] + ")</span></td>" +
          "<td>" + full.toFixed(3) + "</td><td><b>" + dd.toFixed(3) + "</b></td></tr>";
      });
      html += "</table>";
      tblWrap.innerHTML = html;

      if (!worstOk) {
        msg.show("With d = " + d + ", even unrelated words look similar: too few dimensions and the geometry has no room to keep clusters apart. Raise d.", "bad");
      } else {
        msg.show("With d = " + d + ", similar pairs keep high cosine and different pairs stay low: " + M.vlist.length + " sparse counts became " + d + " dense coordinates without losing the neighbourhood structure.", "good");
      }
    }
    render();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.2 — embedding geometry and analogies (signature)
     Curated 2-D coordinates (flagged addition: hand-laid for
     legibility; the source only asserts the analogy property).
     Capitals are computed as country + shared offset + small
     jitter, so the parallel-arrow geometry is honest by
     construction and the jitter is what keeps ≈ from being =.
     ════════════════════════════════════════════════════════════ */
  LR.figs.embedgeo = function (mount) {
    LR.header(
      mount,
      "Walk around a word-embedding space",
      "Similarity mode: click two words for their cosine. Analogy mode: pick two country–capital pairs and watch the difference vectors line up."
    );

    const OFFSET = [1.5, 2.9];
    const COUNTRIES = {
      france: { xy: [-4.8, -6.2], cap: "paris", jit: [0.12, -0.18] },
      italy: { xy: [-2.6, -6.8], cap: "rome", jit: [-0.15, 0.14] },
      germany: { xy: [-6.8, -5.4], cap: "berlin", jit: [0.08, 0.12] },
      canada: { xy: [2.4, -6.6], cap: "ottawa", jit: [-0.12, 0.06] },
      japan: { xy: [5.4, -5.6], cap: "tokyo", jit: [0.1, -0.1] },
    };
    const WORDS = {
      pizza: { xy: [5.2, 6.6], g: "food" },
      burger: { xy: [6.2, 6.0], g: "food" },
      sushi: { xy: [4.6, 5.6], g: "food" },
      pasta: { xy: [5.9, 7.2], g: "food" },
      guitar: { xy: [-6.1, 5.9], g: "instrument" },
      piano: { xy: [-6.8, 5.2], g: "instrument" },
      violin: { xy: [-5.4, 6.6], g: "instrument" },
      drums: { xy: [-6.6, 6.8], g: "instrument" },
    };
    Object.keys(COUNTRIES).forEach(function (c) {
      const d = COUNTRIES[c];
      WORDS[c] = { xy: d.xy, g: "country" };
      WORDS[d.cap] = {
        xy: [d.xy[0] + OFFSET[0] + d.jit[0], d.xy[1] + OFFSET[1] + d.jit[1]],
        g: "capital",
      };
    });
    const GCOL = { food: C.orange, instrument: C.purple, country: C.green, capital: "#575757" };
    const NAMES = Object.keys(WORDS);
    const PAIRNAMES = Object.keys(COUNTRIES).map(function (c) {
      return COUNTRIES[c].cap + " − " + c;
    });

    const bar = LR.controls(mount);
    let mode = "sim";
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = {};
    [["sim", "similarity"], ["ana", "analogy"]].forEach(function (m) {
      btns[m[0]] = LR.button(group, m[1], function () {
        mode = m[0];
        Object.keys(btns).forEach(function (k) { btns[k].classList.toggle("on", k === mode); });
        selA.parentElement.style.display = selB.parentElement.style.display = mode === "ana" ? "" : "none";
        render();
      });
    });
    btns.sim.classList.add("on");
    let pairA = PAIRNAMES[0], pairB = PAIRNAMES[1];
    const selA = LM.select(bar, "pair 1", PAIRNAMES, pairA, function (v) { pairA = v; render(); });
    const selB = LM.select(bar, "pair 2", PAIRNAMES, pairB, function (v) { pairB = v; render(); });
    selA.parentElement.style.display = selB.parentElement.style.display = "none";

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 520, {
      aria: "Two dimensional word embedding scatter plot with food, instrument, country, and capital clusters; click words to compare them",
    });

    const ro = LR.readout(mount, [
      { k: "sel", label: "selection" },
      { k: "cos", label: "cosine similarity" },
      { k: "near", label: "nearest to computed point" },
    ]);
    const msg = LR.msg(mount);

    let picked = ["pizza", "burger"];
    const P = { x0: 52, y0: 16, w: W - 72, h: H - 72, xmin: -9, xmax: 9.5, ymin: -9, ymax: 9.5, xlabel: "embedding dimension 1", ylabel: "embedding dimension 2" };
    let SX = null, SY = null;

    function vsub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
    function vadd(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
    function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const pl = LR.plot(ctx, P);
      SX = pl.sx; SY = pl.sy;
      // origin cross
      ctx.strokeStyle = "#d8d8d8";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(SX(0), P.y0); ctx.lineTo(SX(0), P.y0 + P.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(P.x0, SY(0)); ctx.lineTo(P.x0 + P.w, SY(0)); ctx.stroke();
      ctx.setLineDash([]);

      if (mode === "sim") {
        // vectors from the origin to the two picked words + angle readout
        picked.forEach(function (w, i) {
          const p = WORDS[w].xy;
          LR.arrow(ctx, SX(0), SY(0), SX(p[0]), SY(p[1]), i === 0 ? C.orange : C.purple, 2.2);
        });
        const a = WORDS[picked[0]].xy, b = WORDS[picked[1]].xy;
        const cos = LM.cosine(a, b);
        ro.set("sel", picked[0] + " , " + picked[1]);
        ro.set("cos", cos.toFixed(3), cos > 0.85 ? C.green : cos < 0.3 ? C.red : C.text);
        ro.set("near", "–");
        msg.show(
          cos > 0.85
            ? "cos ≈ " + cos.toFixed(2) + ": these two point the same way, the geometry calls them similar."
            : cos < 0.3
              ? "cos ≈ " + cos.toFixed(2) + ": nearly unrelated directions. Click any two dots to compare others."
              : "cos ≈ " + cos.toFixed(2) + ": somewhat related. Click any two dots to compare others.",
          "info"
        );
      } else {
        // analogy: cap1 - c1 + c2, drawn as parallel arrows
        const c1 = pairA.split(" − ")[1], cap1 = pairA.split(" − ")[0];
        const c2 = pairB.split(" − ")[1], cap2 = pairB.split(" − ")[0];
        const d1 = vsub(WORDS[cap1].xy, WORDS[c1].xy);
        const q = vadd(WORDS[c2].xy, d1); // e(cap1) - e(c1) + e(c2)
        LR.arrow(ctx, SX(WORDS[c1].xy[0]), SY(WORDS[c1].xy[1]), SX(WORDS[cap1].xy[0]), SY(WORDS[cap1].xy[1]), C.orange, 2.4);
        LR.arrow(ctx, SX(WORDS[c2].xy[0]), SY(WORDS[c2].xy[1]), SX(WORDS[cap2].xy[0]), SY(WORDS[cap2].xy[1]), C.green, 2.4);
        // the translated arrow (dashed purple) and the computed point
        ctx.setLineDash([6, 5]);
        LR.arrow(ctx, SX(WORDS[c2].xy[0]), SY(WORDS[c2].xy[1]), SX(q[0]), SY(q[1]), C.purple, 2);
        ctx.setLineDash([]);
        ctx.strokeStyle = C.purple;
        ctx.lineWidth = 2.4;
        const qx = SX(q[0]), qy = SY(q[1]);
        ctx.beginPath(); ctx.moveTo(qx - 7, qy - 7); ctx.lineTo(qx + 7, qy + 7); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(qx - 7, qy + 7); ctx.lineTo(qx + 7, qy - 7); ctx.stroke();
        // nearest word to q (excluding the inputs, like word2vec demos do)
        let best = null;
        NAMES.forEach(function (w) {
          if (w === cap1 || w === c1 || w === c2) return;
          const dd = dist(WORDS[w].xy, q);
          if (!best || dd < best.d) best = { w, d: dd };
        });
        const d2 = vsub(WORDS[cap2].xy, WORDS[c2].xy);
        ro.set("sel", "e(" + cap1 + ") − e(" + c1 + ") + e(" + c2 + ")");
        ro.set("cos", LM.cosine(d1, d2).toFixed(3) + " (between the two arrows)", C.text);
        ro.set("near", best.w + " (dist " + best.d.toFixed(2) + ")", best.w === cap2 ? C.green : C.red);
        msg.show(
          "The purple × is the computed point. Its nearest word is “" + best.w + "”" +
          (best.w === cap2 ? ", the right capital: the country→capital arrow is (almost) the same arrow everywhere." : "."),
          best.w === cap2 ? "good" : "info"
        );
      }

      // points + labels on top
      NAMES.forEach(function (w) {
        const p = WORDS[w].xy;
        const hot = mode === "sim" && picked.indexOf(w) >= 0;
        LR.dot(ctx, SX(p[0]), SY(p[1]), hot ? 6.5 : 4.5, GCOL[WORDS[w].g], hot ? C.text : null);
        ctx.font = (hot ? "700 " : "600 ") + "11.5px Inter, sans-serif";
        ctx.fillStyle = hot ? C.text : "#444444";
        ctx.textAlign = "center";
        ctx.fillText(w, SX(p[0]), SY(p[1]) - 9);
      });
      // legend
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      let lx = P.x0 + 10;
      [["foods", C.orange], ["instruments", C.purple], ["countries/capitals", C.green]].forEach(function (l) {
        LR.dot(ctx, lx, P.y0 + 12, 4, l[1]);
        ctx.fillStyle = C.muted;
        ctx.fillText(l[0], lx + 8, P.y0 + 15.5);
        lx += ctx.measureText(l[0]).width + 34;
      });
    }

    LR.drag(cv, W, H, {
      down: function (p) {
        if (mode !== "sim" || !SX) return;
        let best = null;
        NAMES.forEach(function (w) {
          const d = Math.hypot(SX(WORDS[w].xy[0]) - p.x, SY(WORDS[w].xy[1]) - p.y);
          if (!best || d < best.d) best = { w, d };
        });
        if (best && best.d < 26) {
          picked = [picked[1], best.w];
          render();
        }
      },
    });

    render();
  };

  /* ════════════════════════════════════════════════════════════
     shared: the neural trigram, trained live in the browser
     Corpus designed so dog and cat share contexts, but “the cat
     is …” never occurs; the count model must answer 0 there.
     ════════════════════════════════════════════════════════════ */
  const PET_CORPUS = [
    "the dog is eating",
    "a dog is eating",
    "the dog is barking",
    "the dog is running",
    "a dog is sleeping",
    "the dog was eating",
    "the dog sat down",
    "a dog sat down",
    "the dog was running",
    "a dog was sleeping",
    "the cat sat down",
    "a cat sat down",
    "the cat was sleeping",
    "a cat was eating",
  ];

  let petCache = null;
  function petModel() {
    if (petCache) return petCache;
    const D = 4, EPOCHS = 200, ETA = 0.5; // settings picked so the model fits
    // the data without fully memorizing it (that is what leaves room for
    // generalization to the unseen “the cat is …” context)
    const sents = LM.sentences(PET_CORPUS.join("\n"));
    const vlist = [LM.BOS].concat(LM.vocab(sents), [LM.EOS]);
    const idx = new Map(vlist.map(function (w, i) { return [w, i]; }));
    const V = vlist.length;
    const triples = [];
    sents.forEach(function (s) {
      const seq = [LM.BOS, LM.BOS].concat(s, [LM.EOS]);
      for (let i = 2; i < seq.length; i++)
        triples.push([idx.get(seq[i - 2]), idx.get(seq[i - 1]), idx.get(seq[i])]);
    });
    const rand = LR.rng(42);
    const E = [], Wm = [], b = new Array(V).fill(0);
    for (let i = 0; i < V; i++) {
      E.push([]); Wm.push([]);
      for (let j = 0; j < D; j++) E[i].push((rand() - 0.5) * 0.8);
      for (let j = 0; j < 2 * D; j++) Wm[i].push((rand() - 0.5) * 0.8);
    }
    function forward(i2, i1) {
      const c = E[i2].concat(E[i1]);
      const z = Wm.map(function (row, k) {
        let s = b[k];
        for (let j = 0; j < 2 * D; j++) s += row[j] * c[j];
        return s;
      });
      return { c, z, p: LM.softmax(z) };
    }
    let loss = 0;
    const N = triples.length;
    for (let ep = 0; ep < EPOCHS; ep++) {
      const gE = E.map(function (r) { return r.map(function () { return 0; }); });
      const gW = Wm.map(function (r) { return r.map(function () { return 0; }); });
      const gb = new Array(V).fill(0);
      loss = 0;
      triples.forEach(function (tr) {
        const f = forward(tr[0], tr[1]);
        loss += -Math.log(f.p[tr[2]]);
        for (let k = 0; k < V; k++) {
          const dz = f.p[k] - (k === tr[2] ? 1 : 0);
          gb[k] += dz;
          for (let j = 0; j < 2 * D; j++) gW[k][j] += dz * f.c[j];
        }
        for (let j = 0; j < 2 * D; j++) {
          let s = 0;
          for (let k = 0; k < V; k++) s += (f.p[k] - (k === tr[2] ? 1 : 0)) * Wm[k][j];
          gE[j < D ? tr[0] : tr[1]][j % D] += s;
        }
      });
      for (let k = 0; k < V; k++) {
        b[k] -= (ETA * gb[k]) / N;
        for (let j = 0; j < 2 * D; j++) Wm[k][j] -= (ETA * gW[k][j]) / N;
        for (let j = 0; j < D; j++) E[k][j] -= (ETA * gE[k][j]) / N;
      }
    }
    petCache = { D, V, vlist, idx, E, forward, sents, epochs: EPOCHS, triples: N, loss: loss / N };
    return petCache;
  }

  function petCorpusDetails(mount) {
    const det = LR.el("details");
    det.style.cssText = "font-family:var(--sans);font-size:13px;margin-top:10px";
    det.innerHTML = "<summary style='cursor:pointer;font-weight:700'>show the 14-sentence training corpus</summary>";
    const pre = LR.el("div");
    pre.style.cssText = "font-family:var(--mono);font-size:12.5px;line-height:1.8;columns:2;column-gap:26px;background:#fafafa;border:1px solid var(--line);border-radius:8px;padding:8px 14px;margin-top:6px";
    pre.innerHTML = PET_CORPUS.map(function (s) {
      return /cat/.test(s) ? "<b>" + s + "</b>" : s;
    }).join("<br>");
    det.appendChild(pre);
    det.insertAdjacentHTML("beforeend", "<div style='color:var(--faint);font-size:12px;margin-top:4px'>cat sentences in bold: “the cat is …” appears nowhere.</div>");
    mount.appendChild(det);
  }

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — the neural trigram next-word predictor (signature)
     ════════════════════════════════════════════════════════════ */
  LR.figs.neurallm = function (mount) {
    const M = petModel();
    LR.header(
      mount,
      "Two embeddings in, one distribution out",
      "This exact model trained in your browser just now: " + M.epochs + " epochs on " + M.triples +
      " (context → next word) examples, final cross-entropy " + M.loss.toFixed(2) + " nats. Every number below is its real state."
    );

    const bar = LR.controls(mount);
    let w2 = "the", w1 = "dog";
    const opts = M.vlist.filter(function (w) { return w !== LM.EOS; });
    LM.select(bar, "w(t−2)", opts, w2, function (v) { w2 = v; render(); });
    LM.select(bar, "w(t−1)", opts, w1, function (v) { w1 = v; render(); });

    const pipe = LR.el("div");
    pipe.style.cssText = "font-family:var(--mono);font-size:13px;display:flex;flex-direction:column;gap:7px;margin:6px 0 2px";
    mount.appendChild(pipe);

    const distTitle = LR.el("div", "fig-sub", "");
    distTitle.style.marginTop = "8px";
    mount.appendChild(distTitle);
    const distBox = LR.el("div");
    mount.appendChild(distBox);

    const ro = LR.readout(mount, [
      { k: "E", label: "E params d|V|" },
      { k: "W", label: "W params 2d|V|" },
      { k: "tot", label: "total 3d|V|" },
      { k: "table", label: "count-table cells |V|³" },
    ]);
    ro.set("E", M.D + "×" + M.V + " = " + M.D * M.V);
    ro.set("W", 2 * M.D + "×" + M.V + " = " + 2 * M.D * M.V);
    ro.set("tot", String(3 * M.D * M.V), C.green);
    ro.set("table", M.V + "³ = " + Math.pow(M.V, 3), C.red);

    function cellRow(labelHtml, vec, vmax) {
      const row = LR.el("div");
      row.style.cssText = "display:flex;align-items:center;gap:8px;flex-wrap:wrap";
      const lab = LR.el("span", "", labelHtml);
      lab.style.cssText = "min-width:150px;text-align:right";
      row.appendChild(lab);
      const cells = LR.el("span", "vcells");
      vec.forEach(function (v) {
        const cell = LR.el("span", "vcell", v.toFixed(2));
        const a = Math.min(1, Math.abs(v) / (vmax || 1)) * 0.75 + 0.08;
        cell.style.background = v >= 0 ? "rgba(26,26,26," + a.toFixed(2) + ")" : "rgba(150,150,150," + a.toFixed(2) + ")";
        cell.style.color = Math.abs(v) / vmax > 0.55 ? "#ffffff" : "#222222";
        cells.appendChild(cell);
      });
      row.appendChild(cells);
      return row;
    }

    function render() {
      const i2 = M.idx.get(w2), i1 = M.idx.get(w1);
      const f = M.forward(i2, i1);
      const vmax = Math.max.apply(null, f.c.map(Math.abs).concat([1e-9]));
      pipe.innerHTML = "";
      pipe.appendChild(cellRow("e(" + LM.showTok(w2) + ") = E[" + i2 + "] →", M.E[i2], vmax));
      pipe.appendChild(cellRow("e(" + LM.showTok(w1) + ") = E[" + i1 + "] →", M.E[i1], vmax));
      pipe.appendChild(cellRow("c = concat ∈ ℝ<sup>" + 2 * M.D + "</sup> →", f.c, vmax));
      const zrow = LR.el("div", "", "z = Wc + b &nbsp;(" + M.V + " scores) → softmax ↓");
      zrow.style.cssText = "font-family:var(--sans);font-size:12.5px;font-weight:700;color:var(--muted);margin-left:158px";
      pipe.appendChild(zrow);

      distTitle.innerHTML = "p<sub>θ</sub>( · | " + LM.showTok(w2) + ", " + LM.showTok(w1) + "), top 8 of " + M.V + " words:";
      distBox.innerHTML = "";
      const ranked = M.vlist
        .map(function (w, i) { return { w, p: f.p[i] }; })
        .filter(function (r) { return r.w !== LM.BOS; })
        .sort(function (a, b) { return b.p - a.p; })
        .slice(0, 8);
      const bl = LM.barList(distBox, ranked.map(function (r) { return { key: r.w, label: LM.showTok(r.w) }; }));
      ranked.forEach(function (r, i) {
        bl.set(r.w, r.p, { win: i === 0 });
      });
    }
    render();
    petCorpusDetails(mount);
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.2 — generalization: count model vs neural model
     ════════════════════════════════════════════════════════════ */
  LR.figs.generalize = function (mount) {
    const M = petModel();
    LR.header(
      mount,
      "The unseen context: exact zero vs shared geometry",
      "Both models trained on the same corpus. “the dog …” has plenty of data; “the cat is …” was never seen, and only one of the two models can cope."
    );

    const triCounts = LM.counts(M.sents, 3);

    const bar = LR.controls(mount);
    let ctxPick = "the cat";
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = {};
    ["the cat", "the dog"].forEach(function (c) {
      btns[c] = LR.button(group, "context: " + c, function () {
        ctxPick = c;
        Object.keys(btns).forEach(function (k) { btns[k].classList.toggle("on", k === ctxPick); });
        render();
      });
    });
    btns[ctxPick].classList.add("on");

    const panes = LR.el("div", "dual-pane");
    mount.appendChild(panes);
    const paneL = LR.el("div");
    const paneR = LR.el("div");
    panes.appendChild(paneL);
    panes.appendChild(paneR);

    const ro = LR.readout(mount, [
      { k: "cnt", label: "count p(is | ctx)" },
      { k: "nn", label: "neural p(is | ctx)" },
      { k: "cos", label: "cos(e_cat, e_dog)" },
    ]);
    const msg = LR.msg(mount);
    ro.set("cos", LM.cosine(M.E[M.idx.get("cat")], M.E[M.idx.get("dog")]).toFixed(3), C.purple);

    const SHOW = ["is", "was", "sat", "eating", "sleeping", "running", "barking"];

    function render() {
      const ctxArr = ctxPick.split(" ");
      const row = LM.rowOf(triCounts, ctxArr);
      const f = M.forward(M.idx.get(ctxArr[0]), M.idx.get(ctxArr[1]));

      paneL.innerHTML = '<div class="pane-label">count trigram: count(ctx, w) / count(ctx, ·)</div>';
      paneR.innerHTML = '<div class="pane-label">neural trigram: softmax(Wc + b)</div>';
      const blL = LM.barList(paneL, SHOW.map(function (w) { return { key: w, label: w }; }));
      const blR = LM.barList(paneR, SHOW.map(function (w) { return { key: w, label: w }; }));
      SHOW.forEach(function (w) {
        const c = row ? row.next.get(w) || 0 : 0;
        const tot = row ? row.total : 0;
        const pc = tot ? c / tot : 0;
        blL.set(w, pc, {
          text: tot ? c + "/" + tot + " = " + pc.toFixed(3) : "0/0",
          cut: c === 0,
        });
        const pn = f.p[M.idx.get(w)];
        blR.set(w, pn, { text: pn.toFixed(3), win: w === "is" && pn > 0 });
        const rowEl = blR.row(w);
        if (w === "is") rowEl.style.outline = "2px solid " + C.green + "33";
        const rowElL = blL.row(w);
        if (w === "is") rowElL.style.outline = "2px solid " + C.red + "33";
      });

      const cIs = row ? row.next.get("is") || 0 : 0;
      const tot = row ? row.total : 0;
      ro.set("cnt", (tot ? cIs + "/" + tot + " = " : "") + (tot ? (cIs / tot).toFixed(3) : "0"), cIs === 0 ? C.red : C.green);
      const pIs = f.p[M.idx.get("is")];
      ro.set("nn", pIs.toFixed(3), C.green);

      if (ctxPick === "the cat") {
        msg.show(
          "“the cat is” never occurs in training, so the count model says exactly 0. The neural model still gives “is” p ≈ " +
          pIs.toFixed(2) + ", because e_cat ≈ e_dog puts the cat context near dog contexts where “is” was common. Dogs taught it about cats.",
          "good"
        );
      } else {
        msg.show(
          "“the dog” has real data, so both models agree roughly: counting works fine when the counts exist. The difference only shows on unseen contexts.",
          "info"
        );
      }
    }
    render();
    petCorpusDetails(mount);
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — the decoding explorer
     Fixed next-word logits, handcrafted for the context “the
     weather is …” (flagged addition: the source defines the
     decoding rules; the specific token/logit values are mine).
     ════════════════════════════════════════════════════════════ */
  LR.figs.decoding = function (mount) {
    LR.header(
      mount,
      "One distribution, four ways to pick a word",
      "The logits are fixed; everything else, temperature reshaping, cutoffs, samples, is computed live from the controls."
    );

    const TOKENS = [
      { w: "nice", z: 2.0 },
      { w: "sunny", z: 1.6 },
      { w: "cold", z: 1.3 },
      { w: "rainy", z: 1.0 },
      { w: "mild", z: 0.5 },
      { w: "strange", z: 0.0 },
      { w: "okay", z: -0.4 },
      { w: "loud", z: -1.2 },
      { w: "purple", z: -2.2 },
      { w: "algebra", z: -3.0 },
    ];
    const zs = TOKENS.map(function (t) { return t.z; });

    const bar = LR.controls(mount);
    let mode = "temp", tau = 1.0, k = 4, topp = 0.9;
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = {};
    [["greedy", "greedy"], ["temp", "sample (τ)"], ["topk", "top-k"], ["topp", "top-p"]].forEach(function (m) {
      btns[m[0]] = LR.button(group, m[1], function () {
        mode = m[0];
        Object.keys(btns).forEach(function (kk) { btns[kk].classList.toggle("on", kk === mode); });
        clearSamples();
        render();
      });
    });
    btns.temp.classList.add("on");
    LR.slider(bar, "τ", 0.25, 3, 0.05, 1.0, function (v) { tau = v; clearSamples(); render(); }, function (v) { return v.toFixed(2); });
    LR.slider(bar, "k", 1, 10, 1, 4, function (v) { k = Math.round(v); clearSamples(); render(); }, function (v) { return String(Math.round(v)); });
    LR.slider(bar, "p", 0.1, 1, 0.05, 0.9, function (v) { topp = v; clearSamples(); render(); }, function (v) { return v.toFixed(2); });

    mount.appendChild(LR.el("div", "fig-sub", "p<sub>τ</sub>(next word | “the weather is …”):"));
    const distBox = LR.el("div");
    mount.appendChild(distBox);
    const bl = LM.barList(distBox, TOKENS.map(function (t) { return { key: t.w, label: t.w }; }));

    const ro = LR.readout(mount, [
      { k: "kept", label: "tokens kept" },
      { k: "mass", label: "kept probability mass" },
    ]);

    const sbar = LR.controls(mount);
    LR.button(sbar, "Sample 20 tokens ▸", sample20, "primary");
    LR.button(sbar, "Clear", clearSamples);
    const strip = LR.el("div", "sample-strip");
    strip.setAttribute("aria-live", "polite");
    mount.appendChild(strip);
    const msg = LR.msg(mount);

    let seedCounter = 1;
    const tally = new Map();

    function currentDist() {
      const p = LM.softmax(zs, mode === "greedy" ? 1 : tau);
      const order = p.map(function (v, i) { return i; }).sort(function (a, b) { return p[b] - p[a]; });
      let keep = new Set();
      if (mode === "greedy") {
        keep.add(order[0]);
      } else if (mode === "topk") {
        order.slice(0, k).forEach(function (i) { keep.add(i); });
      } else if (mode === "topp") {
        let acc = 0;
        for (let oi = 0; oi < order.length; oi++) {
          keep.add(order[oi]);
          acc += p[order[oi]];
          if (acc >= topp - 1e-12) break;
        }
      } else {
        order.forEach(function (i) { keep.add(i); });
      }
      let mass = 0;
      keep.forEach(function (i) { mass += p[i]; });
      return { p, keep, mass, top: order[0] };
    }

    function render() {
      const d = currentDist();
      TOKENS.forEach(function (t, i) {
        const kept = d.keep.has(i);
        bl.set(t.w, d.p[i], {
          text: d.p[i].toFixed(3) + (kept && mode !== "temp" && mode !== "greedy" ? " → " + (d.p[i] / d.mass).toFixed(3) : ""),
          cut: !kept,
          win: mode === "greedy" && i === d.top,
        });
      });
      ro.set("kept", d.keep.size + " / " + TOKENS.length);
      ro.set("mass", (100 * d.mass).toFixed(1) + "%", d.mass < 1 - 1e-9 ? C.orange : C.green);
      if (mode === "greedy") {
        msg.show("Greedy is the argmax: it returns “" + TOKENS[d.top].w + "” every single time. Deterministic, and the reason greedy chatbots loop.", "info");
      } else if (mode === "temp") {
        msg.show(tau < 0.6
          ? "Low τ sharpens: the top words hog the mass and samples get repetitive, approaching greedy as τ → 0."
          : tau > 1.6
            ? "High τ flattens: the tail (“purple”, “algebra”) gets real probability, and samples get chaotic."
            : "τ ≈ 1 is the model's own distribution. Move the slider both ways and watch the bars reshape.", "info");
      } else if (mode === "topk") {
        msg.show("Top-k keeps the " + k + " most probable tokens and renormalizes (second number on each kept bar). The tail is simply deleted before sampling.", "info");
      } else {
        msg.show("Top-p keeps the smallest set reaching " + (100 * topp).toFixed(0) + "% of the mass: " + d.keep.size + " tokens here. Sharpen or flatten with τ and the kept set adapts by itself.", "info");
      }
    }

    function sample20() {
      const d = currentDist();
      const kept = Array.from(d.keep);
      const probs = kept.map(function (i) { return d.p[i] / d.mass; });
      const rand = LR.rng(2000 + 131 * seedCounter++);
      for (let s = 0; s < 20; s++) {
        const w = TOKENS[kept[LM.sample(probs, rand)]].w;
        tally.set(w, (tally.get(w) || 0) + 1);
      }
      strip.innerHTML = "";
      Array.from(tally.entries())
        .sort(function (a, b) { return b[1] - a[1]; })
        .forEach(function (e) {
          strip.appendChild(LR.el("span", "sample-chip", e[0] + " ×" + e[1]));
        });
    }

    function clearSamples() {
      tally.clear();
      strip.innerHTML = "";
    }

    render();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 7.1 — the perplexity meter
     Bigram model (add-α smoothed; smoothing is a standard fix,
     flagged as beyond the slides) vs the uniform model, whose
     perplexity is exactly |V|.
     ════════════════════════════════════════════════════════════ */
  LR.figs.perplexity = function (mount) {
    LR.header(
      mount,
      "Bits of surprise, token by token",
      "A bigram model trained on the picnic corpus reads held-out text. Each token contributes −log₂ p; the average exponentiates into perplexity."
    );

    const sents = LM.sentences(LM.PICNIC);
    const table = LM.counts(sents, 2);
    const V = LM.vocab(sents).length + 1; // + EOS as an outcome
    const ALPHA = 0.1;

    const HELD = {
      "held-out: “the dog sat under the tree”": "the dog sat under the tree",
      "scrambled: “tree the under dog sat the”": "tree the under dog sat the",
      "held-out: “we ate cheese in the rain”": "we ate cheese in the rain",
    };

    const bar = LR.controls(mount);
    let pick = Object.keys(HELD)[0];
    LM.select(bar, "text", Object.keys(HELD), pick, function (v) { pick = v; render(); });

    const tblWrap = LR.el("div");
    mount.appendChild(tblWrap);

    const ro = LR.readout(mount, [
      { k: "avg", label: "avg bits ℓ̄" },
      { k: "ppl", label: "PPL = 2^ℓ̄" },
      { k: "uni", label: "uniform model PPL" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      const words = LM.tokenize(HELD[pick]);
      const seq = [LM.BOS].concat(words, [LM.EOS]);
      let bits = 0;
      let html = '<table class="ppl-table"><tr><th>t</th><th>context</th><th>token w<sub>t</sub></th><th>p(w<sub>t</sub> | w<sub>t−1</sub>)</th><th>−log₂ p (bits)</th></tr>';
      const uniBits = Math.log2(V);
      for (let i = 1; i < seq.length; i++) {
        const p = LM.pNext(table, [seq[i - 1]], seq[i], ALPHA, V);
        const bi = -Math.log2(p);
        bits += bi;
        html += "<tr><td>" + i + "</td><td>" + LM.showTok(seq[i - 1]) + "</td><td><b>" + LM.showTok(seq[i]) + "</b></td>" +
          "<td>" + p.toFixed(3) + "</td><td" + (bi > uniBits ? ' class="bad"' : "") + ">" + bi.toFixed(2) + "</td></tr>";
      }
      const M2 = seq.length - 1;
      const avg = bits / M2;
      const ppl = Math.pow(2, avg);
      html += '<tr><td colspan="4" style="text-align:right;font-family:var(--sans);font-weight:700">average over ' + M2 + " tokens:</td><td><b>" + avg.toFixed(2) + "</b></td></tr>";
      html += "</table>";
      tblWrap.innerHTML = html;

      ro.set("avg", avg.toFixed(2) + " bits");
      ro.set("ppl", ppl.toFixed(1), ppl < V / 2 ? C.green : C.red);
      ro.set("uni", "exactly |V| = " + V, C.faint);

      if (/scrambled/.test(pick)) {
        msg.show("Same six words, order destroyed: PPL ≈ " + ppl.toFixed(0) + ", worse than the know-nothing uniform model (" + V + "). The bigram model learned word order, and this text violates it constantly. Red cells cost more bits than a uniform guess.", "bad");
      } else {
        msg.show("PPL ≈ " + ppl.toFixed(1) + ": on this text the model behaves as if choosing among ~" + Math.round(ppl) + " plausible words per step, far better than the uniform model's " + V + ". The expensive rows are the genuinely unpredictable words.", "good");
      }
    }
    render();
  };
})();
