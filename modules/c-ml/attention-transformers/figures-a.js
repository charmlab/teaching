/* ══════════════════════════════════════════════════════════════
   figures-a.js — Attention & Transformers lesson, technical half
   Fig 0.1 highlighter · Fig 1.1 qkv (signature) · Fig 1.2 heatmap
   Fig 1.3 attncode · Fig 2.1 multihead
   Also defines the shared attention core (LR.attn) used by
   figures-b.js: matmul, softmax rows, the attention pipeline,
   masking, the toy "Amir went to class" example, the demo
   sentence embeddings, matrix rendering, and colour ramps.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared attention core (also used by figures-b.js) ──── */
  const AT = (LR.attn = {
    matmul: function (A, B) {
      return A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
    },
    transpose: function (A) {
      return A[0].map((_, j) => A.map((r) => r[j]));
    },
    add: function (A, B) {
      return A.map((r, i) => r.map((v, j) => v + B[i][j]));
    },
    scale: function (A, s) {
      return A.map((r) => r.map((v) => v * s));
    },
    softmaxRow: function (r) {
      const m = Math.max(...r);
      if (!isFinite(m)) return r.map(() => NaN); // all -inf (cannot happen with diag allowed)
      const e = r.map((v) => (isFinite(v) ? Math.exp(v - m) : 0));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    },
    softmax: function (M) {
      return M.map(AT.softmaxRow);
    },

    /* full pipeline used by the QKV builder, code demo, mask demo and
       block walkthrough. mask entries are 0 (allowed) or -Infinity. */
    attention: function (X, Wq, Wk, Wv, opts) {
      opts = opts || {};
      const Q = AT.matmul(X, Wq);
      const K = AT.matmul(X, Wk);
      const V = AT.matmul(X, Wv);
      const dk = Wk[0].length;
      const raw = AT.matmul(Q, AT.transpose(K));
      const masked = opts.mask ? AT.add(raw, opts.mask) : raw;
      const useScale = opts.noScale ? 1 : 1 / Math.sqrt(dk);
      const scaled = AT.scale(masked, useScale);
      const A = AT.softmax(scaled);
      const O = AT.matmul(A, V);
      return { Q, K, V, dk, raw, masked, scaled, A, O };
    },

    causalMask: function (T) {
      return Array.from({ length: T }, (_, i) =>
        Array.from({ length: T }, (_, j) => (j > i ? -Infinity : 0))
      );
    },

    layerNorm: function (M) {
      // per-row normalization to zero mean, unit variance (no learned scale/shift)
      return M.map(function (r) {
        const mu = r.reduce((a, b) => a + b, 0) / r.length;
        const va = r.reduce((a, b) => a + (b - mu) * (b - mu), 0) / r.length;
        const sd = Math.sqrt(va + 1e-8);
        return r.map((v) => (v - mu) / sd);
      });
    },

    // sinusoidal positional encoding, T positions × d dims (source formula)
    posenc: function (T, d) {
      const P = [];
      for (let i = 0; i < T; i++) {
        const row = [];
        for (let k = 0; k < d; k++) {
          const j = Math.floor(k / 2);
          const angle = i / Math.pow(10000, (2 * j) / d);
          row.push(k % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
        }
        P.push(row);
      }
      return P;
    },

    /* ── the lecture's toy example: "Amir went to class" ─────
       N=4, d_model=5, d_k=2 as in the slides. The specific matrix
       entries below are chosen by us for clean arithmetic (the
       source states the setup but not the numbers); everything
       downstream is computed from them live. */
    TOKS: ["Amir", "went", "to", "class"],
    X0: [
      [1, 0, 1, 0, 0], // Amir
      [0, 1, 0, 0, 1], // went
      [0, 0, 0, 1, 0], // to
      [1, 0, 0, 1, 1], // class
    ],
    WQ: [[1, 0], [0, 1], [0, 1], [1, 0], [0, 0]],
    WK: [[1, 0], [0, 1], [1, 0], [0, 1], [1, 1]],
    WV: [[1, 0], [0, 2], [0, 1], [0, 1], [2, 0]],

    /* ── demo sentence for the highlighter / heatmap / heads ──
       8 tokens, 8 dims: four feature pairs = [person, action, place,
       position]. Hand-crafted for pedagogy (flagged in captions);
       attention weights are genuinely computed from these vectors. */
    SENT: ["Amir", "went", "to", "class", "and", "he", "listened", "carefully"],
    sentE: function () {
      const feats = [
        // [ person ,  action ,  place  ]
        [2.0, 1.4, 0.0, 0.0, 0.0, 0.0], // Amir
        [0.0, 0.0, 1.8, 1.2, 0.4, 0.0], // went
        [0.0, 0.0, 0.0, 0.0, 0.9, 0.7], // to
        [0.0, 0.0, 0.0, 0.0, 1.8, 1.3], // class
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], // and
        [1.8, 1.6, 0.0, 0.0, 0.0, 0.0], // he
        [0.0, 0.0, 1.5, 1.7, 0.0, 0.0], // listened
        [0.0, 0.0, 1.0, 1.2, 0.0, 0.0], // carefully
      ];
      return feats.map(function (f, i) {
        return f.concat([Math.cos(0.6 * i), Math.sin(0.6 * i)]); // position pair
      });
    },
    HEADS: [
      { name: "Head 1 · reference", dims: [0, 1], desc: "reads the person features: pronouns find their names" },
      { name: "Head 2 · action", dims: [2, 3], desc: "reads the action features: verbs and their modifiers find each other" },
      { name: "Head 3 · sequence", dims: [6, 7], desc: "reads the position features: each token leans on its neighbours" },
      { name: "Head 4 · location", dims: [4, 5], desc: "reads the place features: where things happen clusters together" },
    ],
    headAttn: function (E, dims) {
      // one head = attention restricted to one feature pair (its "projection")
      const S = E.map((r) => dims.map((d) => r[d]));
      const raw = AT.matmul(S, AT.transpose(S));
      const scaled = AT.scale(raw, 1 / Math.sqrt(dims.length));
      return { A: AT.softmax(scaled), S };
    },

    /* ── colour ramps (single hue, no rainbow) ──────────────── */
    ramp: function (v) {
      // v in [0,1] → white → orange
      const t = Math.max(0, Math.min(1, v));
      const r = Math.round(255 + (232 - 255) * t);
      const g = Math.round(255 + (89 - 255) * t);
      const b = Math.round(255 + (12 - 255) * t);
      return "rgb(" + r + "," + g + "," + b + ")";
    },
    rampSigned: function (v, vmax) {
      // v in [-vmax, vmax] → purple ← white → orange
      const t = Math.max(-1, Math.min(1, v / (vmax || 1)));
      if (t >= 0) return AT.ramp(t);
      const u = -t;
      const r = Math.round(255 + (112 - 255) * u);
      const g = Math.round(255 + (72 - 255) * u);
      const b = Math.round(255 + (232 - 255) * u);
      return "rgb(" + r + "," + g + "," + b + ")";
    },
    txtOn: function (bg) {
      // readable text colour for a ramp background
      const m = bg.match(/(\d+),(\d+),(\d+)/);
      if (!m) return "#111";
      const lum = 0.299 * +m[1] + 0.587 * +m[2] + 0.114 * +m[3];
      return lum > 150 ? "#111" : "#fff";
    },

    /* ── matrix table renderer ──────────────────────────────── */
    // opts: {label, sub, rows, cols, heat:'pos'|'signed'|null, vmax, d (decimals),
    //        onCell(i,j,td), mask (true where cell is struck), big}
    matrix: function (M, opts) {
      opts = opts || {};
      const box = LR.el("div", "mtx-box");
      if (opts.label) box.appendChild(LR.el("div", "mtx-cap", opts.label));
      const tbl = LR.el("table", "mtx" + (opts.big ? " big" : ""));
      let vmax = opts.vmax;
      if (!vmax && opts.heat) {
        vmax = 0;
        M.forEach((r) => r.forEach((v) => { if (isFinite(v)) vmax = Math.max(vmax, Math.abs(v)); }));
        vmax = vmax || 1;
      }
      if (opts.cols) {
        const tr = LR.el("tr");
        tr.appendChild(LR.el("th", null, ""));
        opts.cols.forEach((c) => tr.appendChild(LR.el("th", null, c)));
        tbl.appendChild(tr);
      }
      M.forEach(function (row, i) {
        const tr = LR.el("tr");
        if (opts.rows) tr.appendChild(LR.el("th", null, opts.rows[i]));
        else if (opts.cols) tr.appendChild(LR.el("th", null, ""));
        row.forEach(function (v, j) {
          const td = LR.el("td");
          if (!isFinite(v)) {
            td.textContent = v < 0 ? "−∞" : "∞";
            td.className = "mtx-inf";
          } else {
            td.textContent = LR.fmtF(v, opts.d === undefined ? 2 : opts.d);
            if (opts.heat === "pos") {
              const bg = AT.ramp(v / vmax);
              td.style.background = bg;
              td.style.color = AT.txtOn(bg);
            } else if (opts.heat === "signed") {
              const bg = AT.rampSigned(v, vmax);
              td.style.background = bg;
              td.style.color = AT.txtOn(bg);
            }
          }
          if (opts.onCell) opts.onCell(i, j, td);
          tr.appendChild(td);
        });
        tbl.appendChild(tr);
      });
      box.appendChild(tbl);
      if (opts.sub) box.appendChild(LR.el("div", "mtx-sub", opts.sub));
      return box;
    },
    op: function (sym) {
      return LR.el("div", "mtx-op", sym);
    },
    fmtRow: function (r, d) {
      return "[" + r.map((v) => LR.fmtF(v, d === undefined ? 3 : d)).join(", ") + "]";
    },
  });

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — attention as a learned highlighter
     ══════════════════════════════════════════════════════════ */
  LR.figs.highlighter = function (mount) {
    LR.header(
      mount,
      "Attention as a learned highlighter",
      "Click a word to make it the query. Every other word is shaded by its softmax attention weight, computed live from dot-product similarity."
    );

    const E = AT.sentE();
    const T = E.length;
    let qi = 5; // start on "he": the payoff case

    const sentBox = LR.el("div", "attn-sent");
    sentBox.setAttribute("role", "group");
    sentBox.setAttribute("aria-label", "Sentence; click a word to select it as the attention query");
    mount.appendChild(sentBox);

    const bar = LR.el("div", "attn-weightbar");
    mount.appendChild(bar);

    const ro = LR.readout(mount, [
      { k: "q", label: "query" },
      { k: "top", label: "strongest match" },
      { k: "sum", label: "Σ weights" },
    ]);

    const btns = AT.SENT.map(function (w, i) {
      const b = LR.el("button", "attn-tok", w);
      b.type = "button";
      b.addEventListener("click", function () { qi = i; render(); });
      sentBox.appendChild(b);
      return b;
    });

    function weights(i) {
      // softmax over the *other* tokens (the hook excludes self-attention;
      // Fig 1.1 onward includes it)
      const d = E[0].length;
      const scores = E.map(function (e, j) {
        if (j === i) return -Infinity;
        let s = 0;
        for (let k = 0; k < d; k++) s += E[i][k] * e[k];
        return s / Math.sqrt(d);
      });
      return AT.softmaxRow(scores);
    }

    function render() {
      const w = weights(qi);
      let best = -1, bi = -1, sum = 0;
      w.forEach(function (v, j) {
        if (j === qi) return;
        sum += v;
        if (v > best) { best = v; bi = j; }
      });
      btns.forEach(function (b, j) {
        b.classList.toggle("sel", j === qi);
        b.setAttribute("aria-pressed", j === qi ? "true" : "false");
        if (j === qi) {
          b.style.background = "#111";
          b.style.color = "#fff";
          b.removeAttribute("data-w");
          b.setAttribute("aria-label", AT.SENT[j] + ": the current query word");
        } else {
          const bg = AT.ramp(w[j] / (best || 1));
          b.style.background = bg;
          b.style.color = AT.txtOn(bg);
          b.setAttribute("data-w", LR.fmtF(w[j], 2));
          b.setAttribute("aria-label", AT.SENT[j] + ": attention weight " + LR.fmtF(w[j], 3));
        }
      });
      bar.innerHTML = "";
      w.forEach(function (v, j) {
        if (j === qi) return;
        const item = LR.el("div", "wb-item");
        item.appendChild(LR.el("div", "wb-lab", AT.SENT[j]));
        const track = LR.el("div", "wb-track");
        const fill = LR.el("div", "wb-fill");
        fill.style.width = Math.round(v * 100) + "%";
        if (j === bi) fill.style.background = C.orange;
        track.appendChild(fill);
        item.appendChild(track);
        item.appendChild(LR.el("div", "wb-val", LR.fmtF(v, 3)));
        bar.appendChild(item);
      });
      ro.set("q", '"' + AT.SENT[qi] + '"');
      ro.set("top", '"' + AT.SENT[bi] + '" (' + LR.fmtF(best, 3) + ")", C.orange);
      ro.set("sum", LR.fmtF(sum, 3) + " (softmax)");
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — the query/key/value builder (signature figure)
     ══════════════════════════════════════════════════════════ */
  LR.figs.qkv = function (mount) {
    LR.header(
      mount,
      "Self-attention, matrix by matrix",
      "Step through X → Q, K, V → scores → softmax → output on the toy sentence. Click any cell of X to edit it (cycles −1, 0, 1, 2); everything downstream recomputes."
    );

    // deep copy so edits do not disturb other figures
    const X = AT.X0.map((r) => r.slice());
    let stage = 0;

    const STAGES = [
      { name: "1 · X", text: "The input: one embedding row per token, X ∈ ℝ<sup>4×5</sup> (T = 4, d<sub>model</sub> = 5). Each row is an independent token; no row knows the others exist yet. Click cells to edit." },
      { name: "2 · Q, K, V", text: "Three learned projections read the same X: Q = XW<sub>Q</sub> asks, K = XW<sub>K</sub> labels, V = XW<sub>V</sub> carries content. Each is 4×2 here (d<sub>k</sub> = 2)." },
      { name: "3 · QKᵀ", text: "Every query meets every key: the raw score matrix QKᵀ is T×T = 4×4. Entry (i, j) is how strongly token i's question matches token j's label." },
      { name: "4 · ÷ √dₖ", text: "Temperature scaling: divide by √d<sub>k</sub> = √2 ≈ 1.41 so the scores stay moderate and the upcoming softmax stays soft and trainable." },
      { name: "5 · softmax → A", text: "Softmax each row: A is the attention matrix. Each row is a probability distribution over the sentence, and the row sums printed below are computed, not asserted." },
      { name: "6 · O = AV", text: "Blend the values with the attention weights: O = AV. Same shape story as the theory: 4 rows in, 4 rows out, but each output row is now a context-aware mixture of the whole sentence." },
    ];

    const pills = LR.el("div", "stage-pills");
    mount.appendChild(pills);
    const pillBtns = STAGES.map(function (s, i) {
      const b = LR.el("button", "stage-pill", s.name);
      b.type = "button";
      b.addEventListener("click", function () { stage = i; render(); });
      pills.appendChild(b);
      return b;
    });

    const narrate = LR.el("div", "stepper-narrate");
    mount.appendChild(narrate);
    const stageBox = LR.el("div", "mtx-stage");
    mount.appendChild(stageBox);

    const bar = LR.controls(mount);
    LR.button(bar, "◂ Back", function () { stage = Math.max(0, stage - 1); render(); });
    LR.button(bar, "Step ▸", function () { stage = Math.min(STAGES.length - 1, stage + 1); render(); }, "primary");
    LR.button(bar, "Reset X ⟲", function () {
      AT.X0.forEach((r, i) => r.forEach((v, j) => (X[i][j] = v)));
      render();
    });

    const CYCLE = [-1, 0, 1, 2];
    function editableX(hi) {
      return AT.matrix(X, {
        label: "X  (4×5)",
        rows: AT.TOKS,
        d: 0,
        heat: hi ? "signed" : null,
        onCell: function (i, j, td) {
          td.classList.add("mtx-edit");
          td.tabIndex = 0;
          td.setAttribute("role", "button");
          td.setAttribute("aria-label", "X row " + AT.TOKS[i] + " column " + (j + 1) + " value " + X[i][j] + "; activate to change");
          const bump = function () {
            X[i][j] = CYCLE[(CYCLE.indexOf(X[i][j]) + 1) % CYCLE.length];
            render();
          };
          td.addEventListener("click", bump);
          td.addEventListener("keydown", function (e) {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); bump(); }
          });
        },
      });
    }

    function render() {
      const R = AT.attention(X, AT.WQ, AT.WK, AT.WV);
      pillBtns.forEach((b, i) => b.classList.toggle("on", i === stage));
      narrate.innerHTML = STAGES[stage].text;
      stageBox.innerHTML = "";
      const row = LR.el("div", "mtx-row");
      stageBox.appendChild(row);

      if (stage === 0) {
        row.appendChild(editableX(false));
      } else if (stage === 1) {
        row.appendChild(editableX(false));
        row.appendChild(AT.op("×ᴡ→"));
        row.appendChild(AT.matrix(R.Q, { label: "Q = XW_Q  (4×2)", rows: AT.TOKS, d: 0 }));
        row.appendChild(AT.matrix(R.K, { label: "K = XW_K  (4×2)", rows: AT.TOKS, d: 0 }));
        row.appendChild(AT.matrix(R.V, { label: "V = XW_V  (4×2)", rows: AT.TOKS, d: 0 }));
      } else if (stage === 2) {
        row.appendChild(AT.matrix(R.Q, { label: "Q", rows: AT.TOKS, d: 0 }));
        row.appendChild(AT.op("·"));
        row.appendChild(AT.matrix(AT.transpose(R.K), { label: "Kᵀ", d: 0 }));
        row.appendChild(AT.op("="));
        row.appendChild(AT.matrix(R.raw, { label: "QKᵀ  (4×4 raw scores)", rows: AT.TOKS, cols: AT.TOKS, d: 0, heat: "pos" }));
      } else if (stage === 3) {
        row.appendChild(AT.matrix(R.raw, { label: "QKᵀ", rows: AT.TOKS, cols: AT.TOKS, d: 0 }));
        row.appendChild(AT.op("÷ √2 →"));
        row.appendChild(AT.matrix(R.scaled, { label: "scaled scores", rows: AT.TOKS, cols: AT.TOKS, d: 2, heat: "pos" }));
      } else if (stage === 4) {
        row.appendChild(AT.matrix(R.scaled, { label: "scaled scores", rows: AT.TOKS, cols: AT.TOKS, d: 2 }));
        row.appendChild(AT.op("softmax→"));
        const sums = R.A.map((r) => r.reduce((a, b) => a + b, 0));
        row.appendChild(AT.matrix(R.A, {
          label: "A  (attention matrix)",
          rows: AT.TOKS, cols: AT.TOKS, d: 3, heat: "pos", vmax: 1,
          sub: "row sums: " + sums.map((s) => LR.fmtF(s, 2)).join(" · "),
        }));
      } else {
        row.appendChild(AT.matrix(R.A, { label: "A", rows: AT.TOKS, cols: AT.TOKS, d: 3, heat: "pos", vmax: 1 }));
        row.appendChild(AT.op("·"));
        row.appendChild(AT.matrix(R.V, { label: "V", rows: AT.TOKS, d: 0 }));
        row.appendChild(AT.op("="));
        row.appendChild(AT.matrix(R.O, {
          label: "O = AV  (4×2 output)",
          rows: AT.TOKS, d: 3, heat: "signed",
          sub: "each row: a context-aware blend of all four value vectors",
        }));
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.2 — the T×T attention heatmap, with scaling toggle
     ══════════════════════════════════════════════════════════ */
  LR.figs.heatmap = function (mount) {
    LR.header(
      mount,
      "The attention matrix as a heatmap",
      "Rows are queries, columns are keys. Hover or focus a cell to read the weight; toggle the 1/√dₖ scaling to watch the softmax saturate without it."
    );

    const E = AT.sentE();
    const T = E.length;
    const d = E[0].length;
    let scaled = true;

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bOn = LR.button(grp, "with 1/√dₖ", function () { scaled = true; render(); }, "on");
    const bOff = LR.button(grp, "without (raw scores)", function () { scaled = false; render(); });

    const wrap = LR.el("div", "mtx-stage");
    mount.appendChild(wrap);
    const ro = LR.readout(mount, [
      { k: "cell", label: "hovered cell" },
      { k: "max", label: "max |score| pre-softmax" },
      { k: "peak", label: "largest weight in A" },
    ]);
    const msg = LR.msg(mount);

    function compute() {
      const raw = AT.matmul(E, AT.transpose(E));
      const s = scaled ? AT.scale(raw, 1 / Math.sqrt(d)) : raw;
      return { scores: s, A: AT.softmax(s) };
    }

    function render() {
      bOn.classList.toggle("on", scaled);
      bOff.classList.toggle("on", !scaled);
      const { scores, A } = compute();
      let mx = 0, peak = 0;
      scores.forEach((r) => r.forEach((v) => (mx = Math.max(mx, Math.abs(v)))));
      A.forEach((r) => r.forEach((v) => (peak = Math.max(peak, v))));

      wrap.innerHTML = "";
      const sums = A.map((r) => r.reduce((a, b) => a + b, 0));
      wrap.appendChild(AT.matrix(A, {
        label: "A = softmax(" + (scaled ? "EEᵀ/√" + d : "EEᵀ") + ")   (" + T + "×" + T + ")",
        rows: AT.SENT, cols: AT.SENT, d: 2, heat: "pos", vmax: 1, big: true,
        sub: "row sums: " + sums.map((s) => LR.fmtF(s, 2)).join(" · "),
        onCell: function (i, j, td) {
          td.tabIndex = 0;
          const label = '"' + AT.SENT[i] + '" attends to "' + AT.SENT[j] + '" with weight ' + LR.fmtF(A[i][j], 3);
          td.setAttribute("aria-label", label);
          const show = function () { ro.set("cell", label, C.orange); };
          td.addEventListener("mouseenter", show);
          td.addEventListener("focus", show);
        },
      }));

      ro.set("max", LR.fmtF(mx, 2) + (scaled ? "  (kept moderate)" : "  (unscaled: large)"), scaled ? C.green : C.red);
      ro.set("peak", LR.fmtF(peak, 3));
      if (scaled) {
        msg.show("Scores divided by √dₖ = √" + d + " ≈ " + LR.fmtF(Math.sqrt(d), 2) + ": the weights stay soft, so second-best matches still contribute and gradients flow.", "good");
      } else {
        msg.show("No scaling: the biggest scores reach " + LR.fmtF(mx, 1) + " and the softmax rows spike toward one-hot. This is the saturation the 1/√dₖ prevents.", "bad");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.3 — live self_attention code run
     ══════════════════════════════════════════════════════════ */
  LR.figs.attncode = function (mount) {
    LR.header(
      mount,
      "self_attention, executed for real",
      "Runs the exact pipeline from the Python above on the toy matrices. The mask checkbox passes causal_mask(4) as M (section 4)."
    );

    const bar = LR.controls(mount);
    let useMask = false;
    const lab = LR.el("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.setAttribute("aria-label", "apply causal mask");
    cb.addEventListener("change", function () { useMask = cb.checked; });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(" M = causal_mask(4)"));
    bar.appendChild(lab);
    LR.button(bar, "Run ▸", run, "primary");

    const out = LR.el("pre", "run-output attn-runout", "press Run to execute self_attention(X, Wq, Wk, Wv)");
    mount.appendChild(out);

    function pad(s, n) { s = String(s); while (s.length < n) s = " " + s; return s; }

    function run() {
      const R = AT.attention(AT.X0, AT.WQ, AT.WK, AT.WV, { mask: useMask ? AT.causalMask(4) : null });
      const L = [];
      L.push(">>> A, O = self_attention(X, Wq, Wk, Wv" + (useMask ? ", M=causal_mask(4)" : "") + ")");
      L.push(">>> A                    # rows sum to 1");
      R.A.forEach(function (r, i) {
        L.push("  " + pad('"' + AT.TOKS[i] + '"', 8) + "  [" + r.map((v) => LR.fmtF(v, 3)).join("  ") + "]" +
          "   Σ = " + LR.fmtF(r.reduce((a, b) => a + b, 0), 3));
      });
      L.push(">>> O                    # each row = A[i] @ V");
      R.O.forEach(function (r, i) {
        L.push("  " + pad('"' + AT.TOKS[i] + '"', 8) + "  [" + r.map((v) => LR.fmtF(v, 3)).join("  ") + "]");
      });
      if (useMask) {
        L.push("# upper-triangle weights are exactly 0: e^{-inf} = 0, rows renormalized");
      } else {
        L.push('# check the "class" row against the worked example: α = [0.434, 0.106, 0.026, 0.434]');
      }
      out.textContent = L.join("\n");
    }
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — multi-head attention: four lenses
     ══════════════════════════════════════════════════════════ */
  LR.figs.multihead = function (mount) {
    LR.header(
      mount,
      "Four heads, four relational lenses",
      "Each head runs the identical attention formula through its own projection, so each one highlights a different relationship in the same sentence."
    );

    const E = AT.sentE();
    const T = E.length;
    let view = 0; // 0..3 heads, 4 = combine

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const btns = AT.HEADS.map(function (h, i) {
      return LR.button(grp, "H" + (i + 1), function () { view = i; render(); });
    });
    btns.push(LR.button(grp, "Combine", function () { view = 4; render(); }));

    const narrate = LR.el("div", "stepper-narrate");
    mount.appendChild(narrate);
    const wrap = LR.el("div", "mtx-stage");
    mount.appendChild(wrap);

    // fixed seeded W_O (8×8) for the combine view: real numbers, chosen once
    const rand = LR.rng(657);
    const WO = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => Math.round((rand() - 0.5) * 10) / 10));

    function render() {
      btns.forEach((b, i) => b.classList.toggle("on", i === view));
      wrap.innerHTML = "";
      if (view < 4) {
        const h = AT.HEADS[view];
        const { A } = AT.headAttn(E, h.dims);
        narrate.innerHTML = "<b>" + h.name + "</b>: " + h.desc + '. Note "he" in the reference head, "carefully" in the action head, the near-diagonal band in the sequence head. Rows with nothing to find through this lens go flat (uniform).';
        wrap.appendChild(AT.matrix(A, {
          label: h.name + "  ·  A⁽" + (view + 1) + "⁾ = softmax(Q⁽ʰ⁾K⁽ʰ⁾ᵀ/√2)",
          rows: AT.SENT, cols: AT.SENT, d: 2, heat: "pos", vmax: 1, big: true,
        }));
      } else {
        narrate.innerHTML = "<b>Combine</b>: each head's output O⁽ʰ⁾ = A⁽ʰ⁾V⁽ʰ⁾ is 8×2; concatenating the four gives 8×8, and W<sub>O</sub> projects the stack back to d<sub>model</sub>. One token row now carries all four relational readings at once.";
        const outs = AT.HEADS.map(function (h) {
          const { A, S } = AT.headAttn(E, h.dims);
          return AT.matmul(A, S);
        });
        const concat = outs[0].map((_, i) => outs.reduce((acc, O) => acc.concat(O[i]), []));
        const mha = AT.matmul(concat, WO);
        const row = LR.el("div", "mtx-row");
        row.appendChild(AT.matrix(concat, { label: "Concat(O⁽¹⁾…O⁽⁴⁾)  (8×8)", rows: AT.SENT, d: 1, heat: "signed" }));
        row.appendChild(AT.op("× W_O →"));
        row.appendChild(AT.matrix(mha, { label: "MHA(X)  (8×8)", rows: AT.SENT, d: 1, heat: "signed" }));
        wrap.appendChild(row);
      }
    }
    render();
  };
})();
