/* ══════════════════════════════════════════════════════════════
   figures-b.js — Attention & Transformers lesson, sections 3–7
   Fig 3.1 posenc · Fig 3.2 block (signature) · Fig 4.1 mask
   (signature) · Fig 5.1 pipeline · Fig 6.1 patches · Fig 7.1 cost
   Fig 7.2 quadratic · Fig 7.3 window · Fig 7.4 chinchilla
   Uses the shared attention core LR.attn from figures-a.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const AT = LR.attn;

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — positional encoding: the failure and the fix
     ══════════════════════════════════════════════════════════ */
  LR.figs.posenc = function (mount) {
    LR.header(
      mount,
      "Order-blindness, and the sinusoidal fix",
      'Left: the attention output for "not" in two opposite sentences, with and without positional encoding. Right: the P matrix itself.'
    );

    // small embeddings for good / not / bad (d_model = 4); identity projections
    // keep the demo about position, not about learned W's
    const EMB = {
      good: [1.5, 0.3, 0.8, 0.2],
      not: [0.2, 1.6, 0.1, 0.9],
      bad: [1.4, 0.4, -0.9, 0.3],
    };
    const S1 = ["good", "not", "bad"];
    const S2 = ["bad", "not", "good"];
    let usePE = false;

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bOff = LR.button(grp, "encoding OFF", function () { usePE = false; render(); }, "on");
    const bOn = LR.button(grp, "encoding ON  (z = x + p)", function () { usePE = true; render(); });

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left);
    grid.appendChild(right);

    left.appendChild(LR.el("div", "pane-label", "attention output for “not”, both sentences"));
    const outBox = LR.el("div");
    left.appendChild(outBox);
    const msg = LR.msg(left);

    right.appendChild(LR.el("div", "pane-label", "sinusoidal P: 32 positions × 16 dims"));
    const { cv, ctx, W, H } = LR.canvas(right, 380, 240, {
      aria: "Heatmap of the sinusoidal positional encoding matrix: fast stripes in early dimensions, slow stripes in later dimensions",
    });

    function outputs(sent) {
      const P = AT.posenc(3, 4);
      const X = sent.map(function (w, i) {
        return EMB[w].map((v, k) => v + (usePE ? P[i][k] : 0));
      });
      // identity projections: Q = K = V = X (the point is the positions)
      const raw = AT.matmul(X, AT.transpose(X));
      const A = AT.softmax(AT.scale(raw, 1 / Math.sqrt(4)));
      return AT.matmul(A, X);
    }

    function render() {
      bOff.classList.toggle("on", !usePE);
      bOn.classList.toggle("on", usePE);
      const o1 = outputs(S1)[1]; // "not" sits at position 1 in both
      const o2 = outputs(S2)[1];
      const diff = Math.max(...o1.map((v, k) => Math.abs(v - o2[k])));
      outBox.innerHTML = "";
      outBox.appendChild(AT.matrix([o1, o2], {
        rows: ['"good, not bad"', '"bad, not good"'],
        d: 3, heat: "signed",
        sub: "max |difference| = " + LR.fmtF(diff, 4),
      }));
      if (diff < 1e-9) {
        msg.show("Identical to the last digit. Without position, attention sees a bag of tokens: two opposite sentences produce the same representation.", "bad");
      } else {
        msg.show("The outputs separate (max difference " + LR.fmtF(diff, 3) + "): adding p_i makes the same word at a different position present a different vector.", "good");
      }
    }

    function drawP() {
      const T = 32, d = 16;
      const P = AT.posenc(T, d);
      ctx.clearRect(0, 0, W, H);
      const x0 = 46, y0 = 14, w = W - 60, h = H - 48;
      const cw = w / d, ch = h / T;
      for (let i = 0; i < T; i++)
        for (let k = 0; k < d; k++) {
          ctx.fillStyle = AT.rampSigned(P[i][k], 1);
          ctx.fillRect(x0 + k * cw, y0 + i * ch, cw + 0.5, ch + 0.5);
        }
      ctx.strokeStyle = C.axis;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(x0, y0, w, h);
      ctx.fillStyle = C.text;
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("dimension k  (fast stripes → slow stripes)", x0 + w / 2, H - 12);
      ctx.save();
      ctx.translate(12, y0 + h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("position i  (0 → 31)", 0, 0);
      ctx.restore();
      // ticks
      ctx.fillStyle = LR.C.faint;
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "right";
      [0, 8, 16, 24, 31].forEach(function (i) {
        ctx.fillText(String(i), x0 - 5, y0 + (i + 0.5) * ch + 3);
      });
      ctx.textAlign = "center";
      [0, 4, 8, 12, 15].forEach(function (k) {
        ctx.fillText(String(k), x0 + (k + 0.5) * cw, y0 + h + 12);
      });
    }

    render();
    drawP();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.2 — the transformer block, executed step by step
     ══════════════════════════════════════════════════════════ */
  LR.figs.block = function (mount) {
    LR.header(
      mount,
      "One transformer block, on real numbers",
      "Step through the six sub-steps on the toy sentence; every matrix shown is computed. Then stack blocks and reread."
    );

    /* fixed demo weights so the block computes end to end.
       Flagged addition: W_O, W1/W2 (FFN) and the plain LayerNorm are
       our choices; the source gives the block structure, not toy weights. */
    const WO = [
      [0.5, 0.0, 0.5, 0.0, 0.5],
      [0.0, 0.5, 0.0, 0.5, 0.0],
    ];
    const rand = LR.rng(15);
    const W1 = Array.from({ length: 5 }, () => Array.from({ length: 8 }, () => Math.round((rand() - 0.5) * 12) / 10));
    const W2 = Array.from({ length: 8 }, () => Array.from({ length: 5 }, () => Math.round((rand() - 0.5) * 12) / 10));

    function ffn(M) {
      const h = AT.matmul(M, W1).map((r) => r.map((v) => Math.max(0, v))); // ReLU
      return AT.matmul(h, W2);
    }
    function oneBlock(Xin) {
      const R = AT.attention(Xin, AT.WQ, AT.WK, AT.WV);
      const H = AT.matmul(R.O, WO); // back to d_model
      const U = AT.layerNorm(AT.add(Xin, H));
      const Xp = AT.layerNorm(AT.add(U, ffn(U)));
      return { A: R.A, H, U, Xp };
    }

    const P = AT.posenc(4, 5);
    const X = AT.X0;
    const Xt = AT.add(X, P);

    const STEPS = [
      { key: "X", label: "X", eq: "X: raw token embeddings, T×d_model = 4×5. Four independent rows; no order, no context.", shape: "4×5" },
      { key: "XP", label: "X + P", eq: "X̃ = X + P, inject order. Each row gains its position's sinusoidal fingerprint, so identical words at different positions now differ.", shape: "4×5" },
      { key: "MHA", label: "MHA", eq: "MHA(X̃), mix across tokens. The only step where rows talk to each other: attention blends value vectors, and W_O returns to d_model. (A shown alongside.)", shape: "4×5" },
      { key: "N1", label: "Add & Norm", eq: "U = Norm(X̃ + MHA(X̃)): the residual keeps the original signal in the mix; normalization keeps every row well-scaled so deep stacks stay trainable.", shape: "4×5" },
      { key: "FFN", label: "FFN", eq: "FFN(U) = W₂ σ(W₁u + b₁) + b₂, process each token. The same two-layer MLP applied to every row independently: no cross-token talk here.", shape: "4×5" },
      { key: "N2", label: "Add & Norm", eq: "X′ = Norm(U + FFN(U)): the block's output. Same shape as the input, ready to feed the next block.", shape: "4×5" },
    ];
    let step = 0;
    let stack = 1;

    // pipeline diagram
    const { cv, ctx, W, H } = LR.canvas(mount, 820, 96, {
      aria: "Transformer block pipeline diagram: X, plus positional encoding, multi-head attention, add and norm, feed-forward, add and norm, X prime",
    });

    const narrate = LR.el("div", "stepper-narrate");
    mount.appendChild(narrate);
    const stageBox = LR.el("div", "mtx-stage");
    mount.appendChild(stageBox);

    const bar = LR.controls(mount);
    LR.button(bar, "◂ Back", function () { step = Math.max(0, step - 1); render(); });
    LR.button(bar, "Step ▸", function () { step = Math.min(STEPS.length - 1, step + 1); render(); }, "primary");
    const stackCtl = LR.slider(bar, "stack L blocks", 1, 6, 1, 1, function (v) { stack = v; renderStack(); }, (v) => "L = " + v);
    const stackBox = LR.el("div");
    mount.appendChild(stackBox);

    function drawPipe() {
      ctx.clearRect(0, 0, W, H);
      const boxes = ["X", "+P", "MHA", "Add&Norm", "FFN", "Add&Norm", "X′"];
      const bw = 92, gap = (W - 40 - boxes.length * bw) / (boxes.length - 1);
      boxes.forEach(function (t, i) {
        const x = 20 + i * (bw + gap);
        const active = (step === 0 && i === 0) || (step > 0 && i === step) || (step === STEPS.length - 1 && i === boxes.length - 1);
        const isOut = i === boxes.length - 1;
        const hot = i === (step === 0 ? 0 : step) || (step === 5 && isOut);
        ctx.fillStyle = hot ? C.orange : "#ffffff";
        ctx.strokeStyle = hot ? C.orange : "#bbbbbb";
        ctx.lineWidth = 1.6;
        roundRect(ctx, x, 26, bw, 44, 9);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = hot ? "#ffffff" : C.text;
        ctx.font = "700 12px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(t, x + bw / 2, 52);
        ctx.fillStyle = LR.C.faint;
        ctx.font = "10px Inter, sans-serif";
        ctx.fillText("4×5", x + bw / 2, 84);
        if (i < boxes.length - 1) {
          LR.arrow(ctx, x + bw + 3, 48, x + bw + gap - 3, 48, "#999999", 1.6);
        }
      });
      // residual arcs
      ctx.strokeStyle = C.purple;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 3]);
      arc(1, 3); // X̃ → first Add&Norm
      arc(3, 5); // U → second Add&Norm
      ctx.setLineDash([]);
      function arc(i, j) {
        const x1 = 20 + i * (bw + gap) + bw / 2;
        const x2 = 20 + j * (bw + gap) + bw / 2;
        ctx.beginPath();
        ctx.moveTo(x1, 26);
        ctx.quadraticCurveTo((x1 + x2) / 2, 2, x2, 26);
        ctx.stroke();
      }
      ctx.fillStyle = C.purple;
      ctx.font = "600 10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("residual connections", W / 2, 12);
    }
    function roundRect(c, x, y, w, h, r) {
      c.beginPath();
      c.moveTo(x + r, y);
      c.arcTo(x + w, y, x + w, y + h, r);
      c.arcTo(x + w, y + h, x, y + h, r);
      c.arcTo(x, y + h, x, y, r);
      c.arcTo(x, y, x + w, y, r);
      c.closePath();
    }

    function render() {
      drawPipe();
      const s = STEPS[step];
      narrate.innerHTML = "<b>Step " + (step + 1) + " of 6 · " + s.label + " (" + s.shape + ").</b> " + s.eq;
      stageBox.innerHTML = "";
      const row = LR.el("div", "mtx-row");
      stageBox.appendChild(row);
      const B = oneBlock(Xt);
      if (step === 0) {
        row.appendChild(AT.matrix(X, { label: "X", rows: AT.TOKS, d: 0, heat: "signed" }));
      } else if (step === 1) {
        row.appendChild(AT.matrix(P, { label: "P (sinusoidal)", rows: ["p₀", "p₁", "p₂", "p₃"], d: 2 }));
        row.appendChild(AT.op("+X→"));
        row.appendChild(AT.matrix(Xt, { label: "X̃ = X + P", rows: AT.TOKS, d: 2, heat: "signed" }));
      } else if (step === 2) {
        row.appendChild(AT.matrix(B.A, { label: "A (4×4)", rows: AT.TOKS, cols: AT.TOKS, d: 2, heat: "pos", vmax: 1 }));
        row.appendChild(AT.op("→"));
        row.appendChild(AT.matrix(B.H, { label: "MHA(X̃) = (AV)W_O", rows: AT.TOKS, d: 2, heat: "signed" }));
      } else if (step === 3) {
        row.appendChild(AT.matrix(B.U, { label: "U = Norm(X̃ + MHA(X̃))", rows: AT.TOKS, d: 2, heat: "signed", sub: "each row: mean 0, variance 1 (check any row)" }));
      } else if (step === 4) {
        row.appendChild(AT.matrix(ffn(B.U), { label: "FFN(U), applied row-wise", rows: AT.TOKS, d: 2, heat: "signed" }));
      } else {
        row.appendChild(AT.matrix(B.Xp, { label: "X′ = Norm(U + FFN(U))", rows: AT.TOKS, d: 2, heat: "signed", sub: "same 4×5 shape as X: ready for the next block" }));
      }
    }

    function renderStack() {
      stackBox.innerHTML = "";
      let cur = Xt;
      for (let l = 0; l < stack; l++) cur = oneBlock(cur).Xp;
      const READ = [
        "one pass: token-level features",
        "two passes: local structure starts to assemble",
        "three passes: syntax-level relations",
        "four passes: relations of relations",
        "five passes: meaning-level abstraction",
        "six passes: deeply reread, at more compute per token",
      ];
      const box = LR.el("div", "mtx-row");
      box.appendChild(AT.matrix(cur, {
        label: "after " + stack + " block" + (stack > 1 ? "s" : "") + " (same block reapplied)",
        rows: AT.TOKS, d: 2, heat: "signed",
        sub: "stacking is rereading: " + READ[stack - 1],
      }));
      stackBox.appendChild(box);
    }

    render();
    renderStack();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — masking: BERT vs GPT (signature figure)
     ══════════════════════════════════════════════════════════ */
  LR.figs.mask = function (mount) {
    LR.header(
      mount,
      "One formula, two families",
      "Toggle the mask M added to the scores. Full attention is the encoder-only family (BERT); the causal mask is the decoder-only family (GPT)."
    );

    let causal = false;
    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bFull = LR.button(grp, "full (bidirectional)", function () { causal = false; render(); }, "on");
    const bCaus = LR.button(grp, "causal (lower-triangular)", function () { causal = true; render(); });
    const fam = LR.el("span", "family-tag");
    bar.appendChild(fam);

    const wrap = LR.el("div", "mtx-stage");
    mount.appendChild(wrap);
    const msg = LR.msg(mount);

    function render() {
      bFull.classList.toggle("on", !causal);
      bCaus.classList.toggle("on", causal);
      fam.textContent = causal ? "→ decoder-only · GPT, Llama, Claude" : "→ encoder-only · BERT, RoBERTa";
      fam.style.color = causal ? C.purple : C.green;

      const M = causal ? AT.causalMask(4) : Array.from({ length: 4 }, () => [0, 0, 0, 0]);
      const R = AT.attention(AT.X0, AT.WQ, AT.WK, AT.WV, { mask: M });
      const sums = R.A.map((r) => r.reduce((a, b) => a + b, 0));

      wrap.innerHTML = "";
      const row = LR.el("div", "mtx-row");
      wrap.appendChild(row);
      row.appendChild(AT.matrix(M, {
        label: "mask M",
        rows: AT.TOKS, cols: AT.TOKS, d: 0,
        onCell: function (i, j, td) { if (!isFinite(M[i][j])) td.classList.add("mtx-dead"); },
      }));
      row.appendChild(AT.op("+scores→"));
      row.appendChild(AT.matrix(R.masked, {
        label: "QKᵀ + M",
        rows: AT.TOKS, cols: AT.TOKS, d: 0,
        onCell: function (i, j, td) { if (!isFinite(R.masked[i][j])) td.classList.add("mtx-dead"); },
      }));
      row.appendChild(AT.op("softmax→"));
      row.appendChild(AT.matrix(R.A, {
        label: "A",
        rows: AT.TOKS, cols: AT.TOKS, d: 3, heat: "pos", vmax: 1,
        sub: "row sums: " + sums.map((s) => LR.fmtF(s, 2)).join(" · "),
        onCell: function (i, j, td) { if (causal && j > i) td.classList.add("mtx-dead"); },
      }));

      if (causal) {
        msg.show('Forbidden cells hold −∞, exponentiate to exactly 0, and each row renormalizes over the past. "went" now reads only "Amir" and itself: α = [' +
          R.A[1].map((v) => LR.fmtF(v, 3)).join(", ") + "], matching the worked example.", "info");
      } else {
        msg.show("All-zeros mask: every token attends across the whole sentence in both directions. Great for understanding tasks; useless for generation, because it can peek at the future it is supposed to predict.", "good");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — the training-pipeline map
     ══════════════════════════════════════════════════════════ */
  LR.figs.pipeline = function (mount) {
    LR.header(
      mount,
      "Pretraining → SFT → preference tuning",
      "Click a stage. The bar under each is its relative cost; the bracket shows what “fine-tuning” actually touches."
    );

    const STAGES = [
      {
        name: "1 · Pretraining", cost: 1.0, costLabel: "months · thousands of GPUs · tens of $M",
        buys: "Broad knowledge and fluency: predict the next token on internet-scale text. This is what you are really paying for in a frontier model.",
        breaks: "Bakes in stale facts and biases. Whatever the corpus believed at training time is frozen into the weights.",
      },
      {
        name: "2 · Supervised fine-tuning", cost: 0.06, costLabel: "comparatively cheap",
        buys: "Instruction following: imitate curated instruction-answer pairs. Turns a text continuer into something that answers you.",
        breaks: "Learns the style of good answers, not guaranteed correctness. Polished format is not verified truth.",
      },
      {
        name: "3 · Preference tuning (RLHF/DPO)", cost: 0.06, costLabel: "comparatively cheap",
        buys: "Helpful, harmless tone: learn from human “which answer is better?” comparisons. Decides usability and safety.",
        breaks: "Can induce sycophancy and over-refusal: models learn what raters liked, and raters like agreeable confidence.",
      },
    ];
    let open = 0;

    const rowBox = LR.el("div", "stage-cards");
    mount.appendChild(rowBox);
    const detail = LR.el("div", "stage-detail");
    mount.appendChild(detail);

    const cards = STAGES.map(function (s, i) {
      const card = LR.el("button", "stage-card");
      card.type = "button";
      card.setAttribute("aria-expanded", "false");
      card.appendChild(LR.el("div", "stage-name", s.name));
      const track = LR.el("div", "stage-costtrack");
      const fill = LR.el("div", "stage-costfill");
      fill.style.width = Math.max(4, Math.round(s.cost * 100)) + "%";
      track.appendChild(fill);
      card.appendChild(track);
      card.appendChild(LR.el("div", "stage-costlab", s.costLabel));
      card.addEventListener("click", function () { open = i; render(); });
      rowBox.appendChild(card);
      if (i < 2) rowBox.appendChild(LR.el("div", "stage-arrow", "→"));
      return card;
    });

    const bracket = LR.el("div", "ft-bracket", "└─────────── “fine-tuning” = touching stages 2–3, on top of someone else's pretraining ───────────┘");
    mount.appendChild(bracket);

    function render() {
      cards.forEach(function (c, i) {
        c.classList.toggle("open", i === open);
        c.setAttribute("aria-expanded", i === open ? "true" : "false");
      });
      const s = STAGES[open];
      detail.innerHTML =
        "<div class='sd-row'><span class='sd-k buys'>What it buys</span> " + s.buys + "</div>" +
        "<div class='sd-row'><span class='sd-k breaks'>Failure mode</span> " + s.breaks + "</div>";
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.1 — an image becomes a sentence (ViT patches)
     ══════════════════════════════════════════════════════════ */
  LR.figs.patches = function (mount) {
    LR.header(
      mount,
      "Patches are tokens",
      "Slide the patch size: the image is cut into patches, each patch flattens to a token, and the sequence feeds the same transformer block."
    );

    const N = 16; // image is N×N "pixels", drawn from a deterministic function
    function pix(i, j) {
      // a bright square plus a diagonal gradient: something for patches to differ on
      const inSq = i >= 4 && i < 11 && j >= 5 && j < 12 ? 0.75 : 0;
      return Math.min(1, 0.12 + inSq + 0.25 * (i + j) / (2 * N));
    }

    let p = 4;
    const bar = LR.controls(mount);
    LR.slider(bar, "patch size", 0, 2, 1, 1, function (v) { p = [2, 4, 8][v]; render(); }, function (v) { return [2, 4, 8][v] + "×" + [2, 4, 8][v]; });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 300, {
      aria: "An image cut into patches on the left; the flattened sequence of patch tokens on the right",
    });
    const ro = LR.readout(mount, [
      { k: "T", label: "sequence length T" },
      { k: "pairs", label: "attention pairs T²" },
    ]);

    function render() {
      ctx.clearRect(0, 0, W, H);
      const S = 224, x0 = 26, y0 = 30, cell = S / N;
      // image
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) {
          const v = pix(i, j);
          ctx.fillStyle = AT.ramp(v);
          ctx.fillRect(x0 + j * cell, y0 + i * cell, cell + 0.5, cell + 0.5);
        }
      // patch grid
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 1.4;
      const np = N / p;
      for (let g = 0; g <= np; g++) {
        ctx.beginPath(); ctx.moveTo(x0 + g * p * cell, y0); ctx.lineTo(x0 + g * p * cell, y0 + S); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x0, y0 + g * p * cell); ctx.lineTo(x0 + S, y0 + g * p * cell); ctx.stroke();
      }
      ctx.fillStyle = C.text;
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("image: " + N + "×" + N + ", cut into " + np + "×" + np + " patches", x0 + S / 2, 20);

      // arrow
      LR.arrow(ctx, x0 + S + 16, y0 + S / 2, x0 + S + 66, y0 + S / 2, C.orange, 2.2);
      ctx.fillStyle = LR.C.muted;
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText("flatten", x0 + S + 41, y0 + S / 2 - 10);

      // token strip: the patch sequence, row-major
      const T = np * np;
      const tx = x0 + S + 92, ty = 46;
      const ts = 26, perRow = Math.floor((W - tx - 16) / (ts + 6));
      ctx.fillStyle = C.text;
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("token sequence (T = " + T + "):", tx, 26);
      const shown = Math.min(T, perRow * 6);
      for (let t = 0; t < shown; t++) {
        const pi = Math.floor(t / np), pj = t % np;
        const gx = tx + (t % perRow) * (ts + 6);
        const gy = ty + Math.floor(t / perRow) * (ts + 14);
        // draw the patch miniature (real pixels)
        const sub = ts / p;
        for (let a = 0; a < p; a++)
          for (let b = 0; b < p; b++) {
            ctx.fillStyle = AT.ramp(pix(pi * p + a, pj * p + b));
            ctx.fillRect(gx + b * sub, gy + a * sub, sub + 0.4, sub + 0.4);
          }
        ctx.strokeStyle = "#999999";
        ctx.lineWidth = 1;
        ctx.strokeRect(gx, gy, ts, ts);
        ctx.fillStyle = LR.C.faint;
        ctx.font = "9px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(t), gx + ts / 2, gy + ts + 9);
      }
      if (shown < T) {
        ctx.fillStyle = LR.C.muted;
        ctx.font = "600 12px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("… +" + (T - shown) + " more", tx, ty + 6 * (ts + 14) + 4);
      }
      ctx.fillStyle = LR.C.muted;
      ctx.font = "11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("→ embed each patch, add positions, feed the identical block from section 3", tx, H - 12);

      ro.set("T", T + " tokens");
      ro.set("pairs", (T * T).toLocaleString("en-CA") + (p === 2 ? "  (smaller patches, longer sequence: the n² tax)" : ""));
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.1 — tokenization and the cost calculator
     ══════════════════════════════════════════════════════════ */
  LR.figs.cost = function (mount) {
    LR.header(
      mount,
      "Tokens in, tokens out, dollars per month",
      "cost = tokens_in × p_in + tokens_out × p_out, computed live. The preset is the lecture's manual-stuffing example."
    );

    const ta = document.createElement("textarea");
    ta.className = "cost-text";
    ta.rows = 2;
    ta.value = "Paste any text here to estimate its token count. English averages about one token per four characters, so this sentence alone is already a few dozen tokens.";
    ta.setAttribute("aria-label", "Text to estimate token count for");
    mount.appendChild(ta);
    const tokLab = LR.el("div", "cost-toklab");
    mount.appendChild(tokLab);

    let tin = 12000, tout = 500, pin = 3, pout = 15, qpm = 100000, rag = false;

    const bar = LR.controls(mount);
    const sIn = LR.slider(bar, "tokens in", 100, 20000, 100, tin, function (v) { tin = v; render(); }, (v) => (v / 1000).toFixed(1) + "k");
    const sOut = LR.slider(bar, "tokens out", 50, 2000, 50, tout, function (v) { tout = v; render(); }, (v) => String(v));
    const sPin = LR.slider(bar, "$/1M in", 0.5, 30, 0.5, pin, function (v) { pin = v; render(); }, (v) => "$" + v);
    const sPout = LR.slider(bar, "$/1M out", 1, 60, 1, pout, function (v) { pout = v; render(); }, (v) => "$" + v);
    const sQ = LR.slider(bar, "queries/mo", 10000, 1000000, 10000, qpm, function (v) { qpm = v; render(); }, (v) => (v / 1000) + "k");

    const bar2 = LR.controls(mount);
    LR.button(bar2, "Preset: 12k-token manual, $3/$15", function () {
      tin = 12000; tout = 500; pin = 3; pout = 15; qpm = 100000;
      sIn.set(tin); sOut.set(tout); sPin.set(pin); sPout.set(pout); sQ.set(qpm);
      render();
    });
    const ragBtn = LR.button(bar2, "Use RAG (input ÷ 20)", function () { rag = !rag; render(); });

    const ro = LR.readout(mount, [
      { k: "in", label: "effective tokens in" },
      { k: "call", label: "cost per call" },
      { k: "month", label: "monthly bill" },
    ]);
    const msg = LR.msg(mount);

    function money(v, d) {
      return "$" + v.toLocaleString("en-CA", { minimumFractionDigits: d, maximumFractionDigits: d });
    }

    ta.addEventListener("input", render);

    function render() {
      const est = Math.ceil(ta.value.length / 4);
      tokLab.innerHTML = ta.value.length + " characters ≈ <b>" + est + " tokens</b> (1 token ≈ 4 characters of English)";

      ragBtn.classList.toggle("on", rag);
      const effIn = rag ? Math.ceil(tin / 20) : tin;
      const perCall = (effIn * pin) / 1e6 + (tout * pout) / 1e6;
      const monthly = perCall * qpm;
      ro.set("in", effIn.toLocaleString("en-CA") + (rag ? " (RAG: two relevant paragraphs, not the manual)" : ""));
      ro.set("call", money(perCall, 4), C.orange);
      ro.set("month", money(monthly, 0) + " at " + qpm.toLocaleString("en-CA") + " queries", monthly > 2000 ? C.red : C.green);

      const inShare = (effIn * pin) / 1e6 / perCall;
      if (rag) {
        msg.show("Retrieval cut input tokens ~20×. Same model, same question, " + money(perCall, 4) + " per call instead of stuffing the manual: the bill fell to " + money(monthly, 0) + "/month.", "good");
      } else if (inShare > 0.6) {
        msg.show("The input term is " + Math.round(inShare * 100) + "% of every call: this bill is dominated by stuffed context. That is what RAG attacks.", "info");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.2 — the n² attention-cost curve
     ══════════════════════════════════════════════════════════ */
  LR.figs.quadratic = function (mount) {
    LR.header(
      mount,
      "Attention cost grows like n²",
      "Relative attention compute versus context length, normalized so an 8k-token context costs 1. The green line is the linear world we do not live in."
    );

    let n = 8192;
    const BASE = 8192, NMAX = 32768;

    const bar = LR.controls(mount);
    const sN = LR.slider(bar, "context length", 1024, NMAX, 1024, n, function (v) { n = v; render(); }, (v) => (v / 1024) + "k");
    LR.button(bar, "Double it", function () { n = Math.min(NMAX, n * 2); sN.set(n); render(); }, "primary");

    const { cv, ctx, W, H } = LR.canvas(mount, 720, 340, {
      aria: "Plot of relative attention compute versus context length: a quadratic curve versus a linear reference line",
    });
    const ro = LR.readout(mount, [
      { k: "quad", label: "n² (attention)" },
      { k: "lin", label: "if it were linear" },
      { k: "rule", label: "rule" },
    ]);

    function render() {
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 64, y0: 16, w: W - 90, h: H - 72,
        xmin: 0, xmax: NMAX / 1024, ymin: 0, ymax: 16,
        xlabel: "context length n (thousands of tokens)",
        ylabel: "relative attention compute (×8k baseline)",
        xticks: [0, 4, 8, 16, 24, 32], yticks: [0, 4, 8, 12, 16],
      };
      const { sx, sy } = LR.plot(ctx, P);

      // linear reference
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(sx(0), sy(0));
      ctx.lineTo(sx(NMAX / 1024), sy(NMAX / BASE));
      ctx.stroke();
      ctx.setLineDash([]);

      // quadratic curve
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      for (let t = 0; t <= 120; t++) {
        const nn = (NMAX * t) / 120;
        const y = Math.pow(nn / BASE, 2);
        const px = sx(nn / 1024), py = sy(Math.min(y, 16.4));
        if (t === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // labels on curves
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.orange;
      ctx.textAlign = "left";
      ctx.fillText("(n/8k)²  attention", sx(20) + 6, sy(Math.pow(20480 / BASE, 2)) - 8);
      ctx.fillStyle = C.green;
      ctx.fillText("n/8k  linear", sx(26) + 6, sy(26624 / BASE) - 8);

      // marker at current n
      const q = Math.pow(n / BASE, 2), l = n / BASE;
      LR.dot(ctx, sx(n / 1024), sy(Math.min(q, 16.4)), 6, C.orange, "#ffffff");
      LR.dot(ctx, sx(n / 1024), sy(l), 5, "#ffffff", C.green);
      // gap line
      ctx.strokeStyle = C.red;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(n / 1024), sy(l));
      ctx.lineTo(sx(n / 1024), sy(Math.min(q, 16.4)));
      ctx.stroke();
      ctx.setLineDash([]);

      ro.set("quad", LR.fmtF(q, 2) + "× baseline at n = " + (n / 1024) + "k", C.orange);
      ro.set("lin", LR.fmtF(l, 2) + "×", C.green);
      ro.set("rule", "double n → 4× attention compute");
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.3 — the context window as working memory
     ══════════════════════════════════════════════════════════ */
  LR.figs.window = function (mount) {
    LR.header(
      mount,
      "Everything competes for the window",
      "A fixed 8k-token window holds the system prompt, retrieved docs, the conversation, and the answer. Overflow does not degrade gracefully: it becomes invisible."
    );

    const CAP = 8000, SYS = 600, ANS = 400;
    let docs = 1500, conv = 3000, factPos = 0.5;

    const bar = LR.controls(mount);
    LR.slider(bar, "retrieved docs", 0, 6000, 100, docs, function (v) { docs = v; render(); }, (v) => v + " tok");
    LR.slider(bar, "conversation so far", 0, 12000, 200, conv, function (v) { conv = v; render(); }, (v) => v + " tok");
    LR.slider(bar, "key-fact position", 0, 1, 0.01, factPos, function (v) { factPos = v; render(); }, (v) => Math.round(v * 100) + "%");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 260, {
      aria: "A fixed-capacity context window bar with segments for system prompt, retrieved documents, conversation, and answer; overflow falls off the left, and a retrieval-strength curve dips in the middle",
    });
    const ro = LR.readout(mount, [
      { k: "used", label: "window" },
      { k: "lost", label: "fallen out" },
      { k: "fact", label: "key-fact retrieval" },
    ]);
    const msg = LR.msg(mount);

    // stylized "lost in the middle" retrieval strength (flagged in the caption):
    // strong near both edges of the window, weak in the centre
    function strength(pos) {
      return 0.3 + 0.7 * Math.pow(Math.abs(pos - 0.5) * 2, 1.6);
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const total = SYS + docs + conv + ANS;
      const overflow = Math.max(0, total - CAP);
      const keptConv = conv - overflow;

      const bx = 40, bw = W - 80, by = 150, bh = 46;
      const scale = bw / CAP;

      // the U-curve above the bar
      ctx.strokeStyle = C.purple;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let t = 0; t <= 100; t++) {
        const px = bx + (bw * t) / 100;
        const py = 118 - strength(t / 100) * 62;
        if (t === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.fillStyle = C.purple;
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("retrieval strength across the window (stylized “lost in the middle”)", bx, 40);
      ctx.strokeStyle = "#dddddd";
      ctx.beginPath(); ctx.moveTo(bx, 118); ctx.lineTo(bx + bw, 118); ctx.stroke();

      // overflow, spilling off the left edge
      if (overflow > 0) {
        const ow = Math.min(overflow * scale, bx - 6);
        ctx.fillStyle = "#ececec";
        ctx.fillRect(bx - ow, by + 8, ow, bh - 16);
        ctx.strokeStyle = "#b4b4b4";
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(bx - ow, by + 8, ow, bh - 16);
        ctx.setLineDash([]);
        ctx.fillStyle = LR.C.muted;
        ctx.font = "10.5px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("oldest turns: invisible", bx - ow, by + bh + 16);
      }

      // window frame
      ctx.strokeStyle = "#111111";
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, by, bw, bh);

      // segments
      let x = bx;
      const segs = [
        { n: "system", v: SYS, c: C.purple },
        { n: "retrieved docs", v: docs, c: C.green },
        { n: "conversation (kept)", v: keptConv, c: C.orange },
        { n: "answer", v: ANS, c: "#b4b4b4" },
      ];
      ctx.font = "600 10.5px Inter, sans-serif";
      segs.forEach(function (s) {
        const w = s.v * scale;
        if (w <= 0) return;
        ctx.fillStyle = s.c;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x, by, w, bh);
        ctx.globalAlpha = 1;
        if (w > 56) {
          ctx.fillStyle = "#ffffff";
          ctx.textAlign = "center";
          ctx.fillText(s.n, x + w / 2, by + bh / 2 + 4);
        }
        x += w;
      });
      const free = CAP - (total - overflow);
      if (free > 0) {
        ctx.fillStyle = LR.C.faint;
        ctx.textAlign = "center";
        ctx.font = "10.5px Inter, sans-serif";
        ctx.fillText("free", x + (free * scale) / 2, by + bh / 2 + 4);
      }

      // capacity labels
      ctx.fillStyle = C.text;
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("0", bx, by + bh + 16);
      ctx.textAlign = "right";
      ctx.fillText(CAP.toLocaleString("en-CA") + "-token window", bx + bw, by + bh + 16);
      ctx.textAlign = "center";
      ctx.fillText("no memory between calls unless you re-send it", bx + bw / 2, by + bh + 34);

      // the key fact marker
      const fx = bx + factPos * bw;
      const st = strength(factPos);
      ctx.fillStyle = C.red;
      ctx.beginPath();
      ctx.moveTo(fx, by - 12);
      ctx.lineTo(fx - 6, by - 22);
      ctx.lineTo(fx + 6, by - 22);
      ctx.closePath();
      ctx.fill();
      ctx.font = "700 10.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("key fact", fx, by - 26);
      LR.dot(ctx, fx, 118 - st * 62, 5, C.red, "#ffffff");

      ro.set("used", Math.min(total, CAP).toLocaleString("en-CA") + " / " + CAP.toLocaleString("en-CA") + " tokens");
      ro.set("lost", overflow > 0 ? overflow.toLocaleString("en-CA") + " tokens of oldest conversation" : "nothing (yet)", overflow > 0 ? C.red : C.green);
      const grade = st > 0.75 ? "strong" : st > 0.5 ? "moderate" : "weak";
      ro.set("fact", grade + " (" + LR.fmtF(st, 2) + ")", st > 0.75 ? C.green : st > 0.5 ? C.amber : C.red);

      if (overflow > 0) {
        msg.show(overflow.toLocaleString("en-CA") + " tokens have fallen out of the window. The model does not half-remember them; they are gone from this call entirely.", "bad");
      } else if (st <= 0.5) {
        msg.show("The key fact sits mid-window, where attention is weakest: present but poorly used. Retrieval that places the right chunk near an edge beats a bigger window.", "info");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 7.4 — scaling laws and the Chinchilla trade-off
     ══════════════════════════════════════════════════════════ */
  LR.figs.chinchilla = function (mount) {
    LR.header(
      mount,
      "Fixed compute, one dial: parameters vs tokens",
      "Compute ≈ 6·N·D, so a budget fixes the product. Slide the split and read the stylized loss; the minimum is not at the big-model end."
    );

    /* Stylized loss surface. Flagged addition: this is the parametric fit
       form from the Chinchilla paper, L(N, D) = E + A/N^a + B/D^b with the
       paper's fitted constants, used here as a qualitative illustration of
       the source's statement that smaller models trained longer can win. */
    const E0 = 1.69, A0 = 406.4, ALPHA = 0.34, B0 = 410.7, BETA = 0.28;
    function loss(N, D) {
      return E0 + A0 / Math.pow(N, ALPHA) + B0 / Math.pow(D, BETA);
    }

    let logC = 22;      // log10 of compute budget in FLOPs
    let logN = 9.8;     // log10 of parameters

    const bar = LR.controls(mount);
    LR.slider(bar, "compute budget", 21, 24, 0.5, logC, function (v) { logC = v; render(); },
      (v) => "10^" + v + " FLOPs");
    const sN = LR.slider(bar, "parameters N", 8, 12, 0.05, logN, function (v) { logN = v; render(); },
      (v) => fmtBig(Math.pow(10, v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 720, 360, {
      aria: "Stylized loss versus model size for a fixed compute budget: a U-shaped curve whose minimum sits between the small-model and big-model extremes",
    });
    const ro = LR.readout(mount, [
      { k: "N", label: "your model" },
      { k: "D", label: "training tokens D = C/6N" },
      { k: "L", label: "stylized loss" },
      { k: "opt", label: "budget optimum" },
    ]);
    const msg = LR.msg(mount);

    function fmtBig(v) {
      if (v >= 1e12) return LR.fmtF(v / 1e12, 1) + "T";
      if (v >= 1e9) return LR.fmtF(v / 1e9, 1) + "B";
      if (v >= 1e6) return LR.fmtF(v / 1e6, 0) + "M";
      return LR.fmtF(v, 0);
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const Cb = Math.pow(10, logC);
      const N = Math.pow(10, logN);
      const D = Cb / (6 * N);
      const L = loss(N, D);

      // scan the allocation curve, find the optimum
      const xs = [], ys = [];
      let bestL = Infinity, bestLogN = 8;
      for (let t = 0; t <= 160; t++) {
        const ln = 8 + (4 * t) / 160;
        const n = Math.pow(10, ln);
        const d = Cb / (6 * n);
        const l = loss(n, d);
        xs.push(ln); ys.push(l);
        if (l < bestL) { bestL = l; bestLogN = ln; }
      }
      const ymin = Math.floor((bestL - 0.15) * 10) / 10;
      const ymax = Math.ceil(Math.min(Math.max(...ys), bestL + 1.6) * 10) / 10;

      const P = { x0: 64, y0: 18, w: W - 92, h: H - 82, xmin: 8, xmax: 12, ymin: ymin, ymax: ymax };
      // frame + custom log-scale x labels
      const sx = (x) => P.x0 + ((x - P.xmin) / (P.xmax - P.xmin)) * P.w;
      const sy = (y) => P.y0 + P.h - ((y - P.ymin) / (P.ymax - P.ymin)) * P.h;
      ctx.strokeStyle = LR.C.grid;
      ctx.lineWidth = 1;
      ctx.font = "11px Inter, sans-serif";
      ctx.fillStyle = LR.C.faint;
      [8, 9, 10, 11, 12].forEach(function (lx) {
        ctx.beginPath(); ctx.moveTo(sx(lx), P.y0); ctx.lineTo(sx(lx), P.y0 + P.h); ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText(fmtBig(Math.pow(10, lx)), sx(lx), P.y0 + P.h + 16);
      });
      LR.ticks(ymin, ymax, 5).forEach(function (ty) {
        const py = sy(ty);
        if (py < P.y0 || py > P.y0 + P.h) return;
        ctx.beginPath(); ctx.moveTo(P.x0, py); ctx.lineTo(P.x0 + P.w, py); ctx.stroke();
        ctx.textAlign = "right";
        ctx.fillText(LR.fmt(ty, 2), P.x0 - 7, py + 3.5);
      });
      ctx.strokeStyle = LR.C.axis;
      ctx.lineWidth = 1.4;
      ctx.strokeRect(P.x0, P.y0, P.w, P.h);
      ctx.fillStyle = C.text;
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("parameters N (log scale)", P.x0 + P.w / 2, P.y0 + P.h + 36);
      ctx.save();
      ctx.translate(P.x0 - 44, P.y0 + P.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("stylized pretraining loss", 0, 0);
      ctx.restore();

      // zone labels
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = LR.C.muted;
      ctx.textAlign = "left";
      ctx.fillText("too small: capacity-starved", P.x0 + 8, P.y0 + 16);
      ctx.textAlign = "right";
      ctx.fillText("too big: undertrained (data-starved)", P.x0 + P.w - 8, P.y0 + 16);

      // the loss curve for this budget
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      xs.forEach(function (lx, i) {
        const py = sy(Math.min(ys[i], ymax));
        if (i === 0) ctx.moveTo(sx(lx), py);
        else ctx.lineTo(sx(lx), py);
      });
      ctx.stroke();

      // optimum + current markers
      LR.dot(ctx, sx(bestLogN), sy(bestL), 5.5, C.green, "#ffffff");
      ctx.fillStyle = C.green;
      ctx.font = "700 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("compute-optimal", sx(bestLogN), sy(bestL) + 22);
      LR.dot(ctx, sx(logN), sy(Math.min(L, ymax)), 6.5, C.orange, "#ffffff");

      const Dopt = Cb / (6 * Math.pow(10, bestLogN));
      ro.set("N", fmtBig(N) + " params");
      ro.set("D", fmtBig(D) + " tokens (" + LR.fmtF(D / N, 1) + " tokens/param)");
      ro.set("L", LR.fmtF(L, 3), C.orange);
      ro.set("opt", fmtBig(Math.pow(10, bestLogN)) + " params · " + fmtBig(Dopt) + " tokens", C.green);

      if (logN > bestLogN + 0.5) {
        msg.show("Undertrained: at this budget, all these parameters see only " + LR.fmtF(D / N, 1) + " tokens each. The smaller compute-optimal model (green) reaches lower loss and is cheaper to serve.", "bad");
      } else if (logN < bestLogN - 0.5) {
        msg.show("Capacity-starved: the model is too small to use this data budget. Loss falls if you move parameters up toward the optimum.", "info");
      } else {
        msg.show("Near compute-optimal: the Chinchilla regime, roughly balanced parameters and data. This is why strong modern models are often mid-size but trained long.", "good");
      }
    }
    render();
  };
})();
