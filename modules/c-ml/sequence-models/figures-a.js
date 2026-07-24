/* ══════════════════════════════════════════════════════════════
   figures-a.js — Sequence Models lesson figures, sections 0–2
   Fig 0.1 window · Fig 1.1 unroll (signature) · Fig 1.2 runsum
   Fig 1.3 rnncode · Fig 2.1 lmstep
   Also defines the shared sequence-model core (LR.seq) used by
   figures-b.js: activations, softmax, heat strips, and the tiny
   word-level RNN language model that is genuinely trained in the
   browser for Fig 2.1. All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared sequence core (also used by figures-b.js) ────── */
  const SEQ = (LR.seq = {
    sigmoid: function (z) { return 1 / (1 + Math.exp(-z)); },
    // derivative of tanh at pre-activation z, given h = tanh(z)
    dtanhFromH: function (h) { return 1 - h * h; },

    softmax: function (zs) {
      let mx = -Infinity;
      zs.forEach(function (z) { if (z > mx) mx = z; });
      const es = zs.map(function (z) { return Math.exp(z - mx); });
      const s = es.reduce(function (a, b) { return a + b; }, 0);
      return es.map(function (e) { return e / s; });
    },

    zeros: function (n) { return new Array(n).fill(0); },

    /* heat colour for a value in roughly [-1, 1]:
       positive → orange, negative → purple, magnitude → opacity */
    heatColor: function (v, vmax) {
      const m = Math.min(1, Math.abs(v) / (vmax || 1));
      return v >= 0
        ? "rgba(232, 89, 12, " + (0.08 + 0.85 * m).toFixed(3) + ")"
        : "rgba(112, 72, 232, " + (0.08 + 0.85 * m).toFixed(3) + ")";
    },

    /* build a DOM heat strip; returns {set(values)} */
    heatStrip: function (parent, n, label, vmax) {
      const wrap = LR.el("div");
      wrap.style.margin = "10px 0 4px";
      if (label) wrap.appendChild(LR.el("span", "hstrip-label", label));
      const strip = LR.el("span", "hstrip");
      strip.setAttribute("role", "img");
      strip.setAttribute("aria-label", label || "hidden state values");
      const cells = [];
      for (let i = 0; i < n; i++) {
        const c = LR.el("span", "hcell");
        strip.appendChild(c);
        cells.push(c);
      }
      wrap.appendChild(strip);
      parent.appendChild(wrap);
      return {
        set: function (vals) {
          for (let i = 0; i < n; i++) {
            cells[i].style.background = SEQ.heatColor(vals[i], vmax || 1);
            cells[i].title = "h[" + i + "] = " + LR.fmtF(vals[i], 2);
          }
        },
      };
    },
  });

  /* ══════════════════════════════════════════════════════════
     The tiny word-level RNN language model behind Fig 2.1.
     Really trained here (BPTT + SGD) on a toy corpus, seeded so
     every visitor gets the same trained weights. Vocabulary and
     corpus are mine (the source deck names no toy corpus); the
     architecture is the lecture's vanilla RNN + softmax head.
     ══════════════════════════════════════════════════════════ */
  SEQ.VOCAB = ["<s>", "the", "cat", "dog", "sat", "ran", "on", "mat", "."];

  SEQ.lm = function () {
    if (SEQ._lm) return SEQ._lm;
    const V = SEQ.VOCAB.length; // 9
    const H = 12;
    const id = {};
    SEQ.VOCAB.forEach(function (w, i) { id[w] = i; });
    const SENTS = [
      ["the", "cat", "sat", "on", "the", "mat", "."],
      ["the", "dog", "ran", "on", "the", "mat", "."],
      ["the", "dog", "sat", "on", "the", "mat", "."],
      ["the", "cat", "ran", "on", "the", "mat", "."],
      ["the", "cat", "sat", "."],
      ["the", "dog", "ran", "."],
    ].map(function (s) { return s.map(function (w) { return id[w]; }); });

    const rand = LR.rng(14);
    const mat = function (r, c, s) {
      const M = [];
      for (let i = 0; i < r; i++) {
        const row = [];
        for (let j = 0; j < c; j++) row.push((rand() * 2 - 1) * s);
        M.push(row);
      }
      return M;
    };
    const Wxh = mat(H, V, 0.35); // input (one-hot column) → hidden
    const Whh = mat(H, H, 0.35); // hidden → hidden
    const Why = mat(V, H, 0.35); // hidden → logits
    const bh = SEQ.zeros(H);
    const by = SEQ.zeros(V);

    function forwardStep(hPrev, x) {
      const a = SEQ.zeros(H);
      for (let i = 0; i < H; i++) {
        let s = Wxh[i][x] + bh[i];
        for (let j = 0; j < H; j++) s += Whh[i][j] * hPrev[j];
        a[i] = Math.tanh(s);
      }
      const z = SEQ.zeros(V);
      for (let k = 0; k < V; k++) {
        let s = by[k];
        for (let j = 0; j < H; j++) s += Why[k][j] * a[j];
        z[k] = s;
      }
      return { h: a, p: SEQ.softmax(z) };
    }

    // full BPTT over one sentence; returns loss, applies SGD grads
    function trainSeq(seq, lr) {
      const xs = [0].concat(seq.slice(0, -1)); // <s>, w1 … w_{n-1}
      const ys = seq;                          // w1 … wn
      const T = xs.length;
      const hs = [SEQ.zeros(H)];
      const ps = [];
      let loss = 0;
      for (let t = 0; t < T; t++) {
        const f = forwardStep(hs[t], xs[t]);
        hs.push(f.h);
        ps.push(f.p);
        loss -= Math.log(Math.max(f.p[ys[t]], 1e-12));
      }
      // grads
      const gWxh = mat(H, V, 0), gWhh = mat(H, H, 0), gWhy = mat(V, H, 0);
      const gbh = SEQ.zeros(H), gby = SEQ.zeros(V);
      let dhNext = SEQ.zeros(H);
      for (let t = T - 1; t >= 0; t--) {
        const dz = ps[t].slice();
        dz[ys[t]] -= 1; // softmax + cross-entropy
        const h = hs[t + 1];
        const dh = dhNext.slice();
        for (let k = 0; k < V; k++) {
          gby[k] += dz[k];
          for (let j = 0; j < H; j++) {
            gWhy[k][j] += dz[k] * h[j];
            dh[j] += Why[k][j] * dz[k];
          }
        }
        const da = SEQ.zeros(H);
        for (let i = 0; i < H; i++) da[i] = dh[i] * (1 - h[i] * h[i]);
        dhNext = SEQ.zeros(H);
        for (let i = 0; i < H; i++) {
          gbh[i] += da[i];
          gWxh[i][xs[t]] += da[i];
          for (let j = 0; j < H; j++) {
            gWhh[i][j] += da[i] * hs[t][j];
            dhNext[j] += Whh[i][j] * da[i];
          }
        }
      }
      // clip (the lecture's own patch, used honestly) then SGD
      let n2 = 0;
      const all = [gWxh, gWhh, gWhy];
      all.forEach(function (M) { M.forEach(function (r) { r.forEach(function (v) { n2 += v * v; }); }); });
      gbh.forEach(function (v) { n2 += v * v; });
      gby.forEach(function (v) { n2 += v * v; });
      const norm = Math.sqrt(n2);
      const sc = norm > 5 ? 5 / norm : 1;
      for (let i = 0; i < H; i++) {
        bh[i] -= lr * sc * gbh[i];
        for (let j = 0; j < V; j++) Wxh[i][j] -= lr * sc * gWxh[i][j];
        for (let j = 0; j < H; j++) Whh[i][j] -= lr * sc * gWhh[i][j];
      }
      for (let k = 0; k < V; k++) {
        by[k] -= lr * sc * gby[k];
        for (let j = 0; j < H; j++) Why[k][j] -= lr * sc * gWhy[k][j];
      }
      return loss / T;
    }

    let finalLoss = 0;
    for (let e = 0; e < 260; e++) {
      finalLoss = 0;
      for (let s = 0; s < SENTS.length; s++) finalLoss += trainSeq(SENTS[s], 0.1);
      finalLoss /= SENTS.length;
    }

    SEQ._lm = {
      V: V,
      H: H,
      id: id,
      vocab: SEQ.VOCAB,
      loss: finalLoss,
      h0: SEQ.zeros(H),
      step: forwardStep,
    };
    return SEQ._lm;
  };

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — the fixed-window failure
     ══════════════════════════════════════════════════════════ */
  LR.figs.window = function (mount) {
    LR.header(
      mount,
      "The fixed window forgets the subject",
      "The model must predict the word after “he” while seeing only the last k words. Slide k."
    );

    const WORDS = ["Amir", "went", "to", "class", "and", "then", "he", "returned", "home"];
    const TARGET = 7; // predicting the token at this position ("returned")
    let k = 2;

    const bar = LR.controls(mount);
    const slider = LR.slider(bar, "window size k", 1, 8, 1, k, function (v) {
      k = Math.round(v);
      render();
    }, function (v) { return String(Math.round(v)); });
    LR.button(bar, "Trigram (k = 2)", function () { k = 2; slider.set(2); render(); }, "small");
    LR.button(bar, "Wide enough (k = 7)", function () { k = 7; slider.set(7); render(); }, "small");

    const row = LR.el("div", "tok-row");
    row.setAttribute("aria-label", "sentence tokens with the visible window highlighted");
    mount.appendChild(row);

    const ro = LR.readout(mount, [
      { k: "sees", label: "model sees" },
      { k: "subj", label: "subject in view" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      row.innerHTML = "";
      const lo = TARGET - k; // window covers tokens lo … TARGET-1
      WORDS.forEach(function (w, i) {
        const t = LR.el("span", "tok", w);
        if (i === TARGET) {
          t.className = "tok target";
          t.textContent = "? " + w + " ?";
          t.title = "the word being predicted";
        } else if (i > TARGET) {
          t.classList.add("faded");
        } else if (i >= lo && i < TARGET) {
          t.classList.add("win");
          if (i === 0) t.classList.add("subj");
        } else {
          t.classList.add("faded");
          if (i === 0) t.classList.add("subj");
        }
        row.appendChild(t);
      });
      const seen = WORDS.slice(Math.max(0, lo), TARGET);
      ro.set("sees", "[" + seen.join(", ") + "]");
      const inView = lo <= 0;
      ro.set("subj", inView ? "yes" : "no", inView ? C.green : C.red);
      if (inView) {
        msg.show("k = " + k + ": “Amir” is inside the window, so the model can connect “he” back to its subject. The cost: this window must be this wide for every prediction, and some subject is always one word further back.", "good");
      } else {
        msg.show("k = " + k + ": the model conditions on " + JSON.stringify(seen.join(" ")) + " and nothing else. “Amir” does not exist for it, and no training data can teach a model about words it cannot see.", "bad");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — the RNN unrolling animation (signature figure)
     A real tanh RNN with a 2-d hidden state; fold/unfold + step.
     ══════════════════════════════════════════════════════════ */
  LR.figs.unroll = function (mount) {
    LR.header(
      mount,
      "One cell with a loop = a chain across time",
      "Unroll the self-loop into a chain, then step through the sequence and watch hₜ update for real."
    );

    // the actual network (fixed, visible weights; tanh activation)
    const Wh = [[0.6, -0.4], [0.3, 0.5]];
    const Wx = [0.8, -0.5];
    const xs = [1.0, 0.5, -1.0, 0.2, 0.8];
    const T = xs.length;
    const hs = [[0, 0]];
    for (let t = 0; t < T; t++) {
      const hp = hs[t];
      hs.push([
        Math.tanh(Wh[0][0] * hp[0] + Wh[0][1] * hp[1] + Wx[0] * xs[t]),
        Math.tanh(Wh[1][0] * hp[0] + Wh[1][1] * hp[1] + Wx[1] * xs[t]),
      ]);
    }

    let unrolled = false;
    let step = 0;    // how many inputs have been consumed (0…T)
    let u = 0;       // fold/unfold tween parameter 0…1
    let anim = null;

    const bar = LR.controls(mount);
    const toggleBtn = LR.button(bar, "Unroll ▸", function () {
      unrolled = !unrolled;
      toggleBtn.textContent = unrolled ? "Fold ⟲" : "Unroll ▸";
      if (LR.reducedMotion) { u = unrolled ? 1 : 0; render(); return; }
      if (anim) cancelAnimationFrame(anim);
      const t0 = performance.now();
      const from = u, to = unrolled ? 1 : 0;
      const tick = function (now) {
        const q = Math.min(1, (now - t0) / 380);
        u = from + (to - from) * (q * q * (3 - 2 * q));
        render();
        if (q < 1) anim = requestAnimationFrame(tick);
      };
      anim = requestAnimationFrame(tick);
    }, "primary");
    const stepBtn = LR.button(bar, "Step ▸", function () {
      if (!unrolled) { toggleBtn.click(); return; }
      if (step < T) { step += 1; render(); updateNarrate(); }
    });
    LR.button(bar, "Reset ⟲", function () { step = 0; render(); updateNarrate(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 320, {
      aria: "An RNN cell with a self-loop unrolling into a chain of five time steps",
    });

    const narrate = LR.el("div", "seq-narrate",
      "The folded view is the definition: one cell, its output looped back in. Press <b>Unroll ▸</b>.");
    mount.appendChild(narrate);

    function updateNarrate() {
      if (step === 0) {
        narrate.innerHTML = "Unrolled: the same cell copied once per time step, h₀ = [0, 0] on the left. Press <b>Step ▸</b> to feed x₁.";
      } else {
        const h = hs[step], hp = hs[step - 1];
        narrate.innerHTML =
          "h<sub>" + step + "</sub> = tanh(Wₕ h<sub>" + (step - 1) + "</sub> + Wₓ x<sub>" + step + "</sub>) = tanh(Wₕ [" +
          LR.fmtF(hp[0], 2) + ", " + LR.fmtF(hp[1], 2) + "] + Wₓ · " + LR.fmtF(xs[step - 1], 1) +
          ") = <b>[" + LR.fmtF(h[0], 2) + ", " + LR.fmtF(h[1], 2) + "]</b>, computed with the same Wₕ = [[0.6, −0.4], [0.3, 0.5]] as every other step.";
      }
    }

    const CELLW = 92, CELLH = 64;
    function cellRect(i) {
      // folded position (all cells stacked at centre) → chain position
      const cx0 = W / 2 - CELLW / 2;
      const cy = 128;
      const gap = (W - 90 - T * CELLW) / (T - 1);
      const cxi = 45 + i * (CELLW + gap);
      return { x: cx0 + (cxi - cx0) * u, y: cy, w: CELLW, h: CELLH };
    }

    function drawCell(r, label, active, ghost) {
      ctx.save();
      if (ghost !== undefined) ctx.globalAlpha = ghost;
      ctx.fillStyle = active ? "#fff5ec" : "#fff";
      ctx.strokeStyle = active ? C.orange : C.axis;
      ctx.lineWidth = active ? 2.4 : 1.6;
      const rr = 10;
      ctx.beginPath();
      ctx.moveTo(r.x + rr, r.y);
      ctx.arcTo(r.x + r.w, r.y, r.x + r.w, r.y + r.h, rr);
      ctx.arcTo(r.x + r.w, r.y + r.h, r.x, r.y + r.h, rr);
      ctx.arcTo(r.x, r.y + r.h, r.x, r.y, rr);
      ctx.arcTo(r.x, r.y, r.x + r.w, r.y, rr);
      ctx.closePath();
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.text;
      ctx.font = "700 13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, r.x + r.w / 2, r.y + r.h / 2 - 6);
      ctx.font = "11px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.fillText("tanh(Wₕh + Wₓx)", r.x + r.w / 2, r.y + r.h / 2 + 12);
      ctx.restore();
    }

    function drawHBars(r, h, t) {
      // two mini bars + numbers above a cell
      const bx = r.x + r.w / 2 - 26, by = r.y - 46, bw = 22, bh = 26;
      for (let i = 0; i < 2; i++) {
        const x = bx + i * 30;
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(x, by, bw, bh);
        const v = h[i]; // in [-1,1]
        const half = bh / 2;
        ctx.fillStyle = v >= 0 ? C.orange : C.purple;
        const len = Math.abs(v) * half;
        ctx.fillRect(x, by + half - (v >= 0 ? len : 0), bw, len);
        ctx.strokeStyle = C.axis;
        ctx.beginPath(); ctx.moveTo(x, by + half); ctx.lineTo(x + bw, by + half); ctx.stroke();
      }
      ctx.fillStyle = C.text;
      ctx.font = "700 10.5px " + "JetBrains Mono, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("h" + sub(t) + " = [" + LR.fmtF(h[0], 2) + ", " + LR.fmtF(h[1], 2) + "]", r.x + r.w / 2, by - 7);
    }

    function sub(n) {
      const S = ["₀", "₁", "₂", "₃", "₄", "₅"];
      return S[n] || String(n);
    }

    function render() {
      ctx.clearRect(0, 0, W, H);

      // banner: the shared weights, printed once
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.orange;
      ctx.textAlign = "center";
      ctx.fillText("one copy of the weights, reused at every step:  Wₕ = [[0.6, −0.4], [0.3, 0.5]],  Wₓ = [0.8, −0.5]", W / 2, 22);

      if (u < 0.02) {
        /* folded: single cell + self-loop */
        const r = cellRect(0);
        // self-loop arc
        ctx.strokeStyle = C.orange;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.arc(r.x + r.w / 2, r.y - 26, 34, Math.PI * 0.15, Math.PI * 0.85, true);
        ctx.stroke();
        LR.arrow(ctx, r.x + r.w / 2 - 27, r.y - 47, r.x + r.w / 2 - 33, r.y - 20, C.orange, 2.2);
        ctx.fillStyle = C.orange;
        ctx.font = "600 12px Inter, sans-serif";
        ctx.fillText("h fed back (one step later)", r.x + r.w / 2, r.y - 72);
        drawCell(r, "RNN cell", false);
        // input / output arrows
        LR.arrow(ctx, r.x + r.w / 2, r.y + r.h + 44, r.x + r.w / 2, r.y + r.h + 6, C.axis, 2);
        ctx.fillStyle = C.muted;
        ctx.fillText("xₜ (one token per step)", r.x + r.w / 2, r.y + r.h + 62);
        LR.arrow(ctx, r.x + r.w + 6, r.y + r.h / 2, r.x + r.w + 52, r.y + r.h / 2, C.green, 2);
        ctx.fillStyle = C.green;
        ctx.textAlign = "left";
        ctx.fillText("hₜ out", r.x + r.w + 58, r.y + r.h / 2 + 4);
        return;
      }

      /* unrolled chain */
      const rects = [];
      for (let i = 0; i < T; i++) rects.push(cellRect(i));

      // h0 marker
      const r0 = rects[0];
      ctx.fillStyle = C.muted;
      ctx.font = "700 12px JetBrains Mono, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("h₀ = [0, 0]", r0.x - 2, r0.y - 12);

      for (let i = 0; i < T; i++) {
        const r = rects[i];
        // inter-cell arrow with shared-weight label
        if (i > 0) {
          const pr = rects[i - 1];
          LR.arrow(ctx, pr.x + pr.w + 2, pr.y + CELLH / 2, r.x - 3, r.y + CELLH / 2, C.orange, 2.2);
          if (u > 0.85) {
            ctx.fillStyle = C.orange;
            ctx.font = "700 11px Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("Wₕ", (pr.x + pr.w + r.x) / 2, pr.y + CELLH / 2 - 8);
          }
        }
        drawCell(r, "t = " + (i + 1), i === step - 1, u);
        if (u > 0.85) {
          // input arrow + value
          LR.arrow(ctx, r.x + r.w / 2, r.y + r.h + 40, r.x + r.w / 2, r.y + r.h + 6, i < step ? C.green : "#bbb", 2);
          ctx.fillStyle = i < step ? C.text : C.faint;
          ctx.font = "700 12px JetBrains Mono, Menlo, monospace";
          ctx.textAlign = "center";
          ctx.fillText("x" + sub(i + 1) + " = " + LR.fmtF(xs[i], 1), r.x + r.w / 2, r.y + r.h + 58);
          ctx.fillStyle = C.faint;
          ctx.font = "600 10.5px Inter, sans-serif";
          ctx.fillText("Wₓ", r.x + r.w / 2 + 20, r.y + r.h + 24);
          // hidden state above, once computed
          if (i < step) drawHBars(r, hs[i + 1], i + 1);
        }
      }
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.2 — the running-summary demo
     h_{t+1} = w·x_{t+1} + w·h_t, the lecture's simplest example.
     ══════════════════════════════════════════════════════════ */
  LR.figs.runsum = function (mount) {
    LR.header(
      mount,
      "The hidden state is memory: a running sum",
      "Step a number sequence through hₜ₊₁ = w·xₜ₊₁ + w·hₜ and watch the summary accumulate."
    );

    const PRESETS = {
      A: [2, 1, 3, -2, 4],
      B: [1, 1, 1, 1, 1, 1],
    };
    let xs = PRESETS.A.slice();
    let w = 1.0;
    let t = 0; // steps consumed
    let playing = null;

    const bar = LR.controls(mount);
    LR.button(bar, "Step ▸", stepOnce, "primary");
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(stepOnce, LR.reducedMotion ? 1100 : 650);
    });
    LR.button(bar, "Reset ⟲", function () { stopPlay(); t = 0; render(); });
    const wSlider = LR.slider(bar, "shared weight w", 0.5, 1.5, 0.05, w, function (v) {
      w = v; t = Math.min(t, xs.length); render();
    });
    LR.button(bar, "x = [2, 1, 3, −2, 4]", function () { xs = PRESETS.A.slice(); stopPlay(); t = 0; render(); }, "small");
    LR.button(bar, "x = [1, 1, 1, 1, 1, 1]", function () { xs = PRESETS.B.slice(); stopPlay(); t = 0; render(); }, "small");

    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }
    function stepOnce() {
      if (t >= xs.length) { stopPlay(); return; }
      t += 1;
      render();
    }

    const tableWrap = LR.el("div", "seq-table-wrap");
    mount.appendChild(tableWrap);
    const { cv, ctx, W, H } = LR.canvas(mount, 780, 240, {
      aria: "Bar chart of the hidden state h over time compared with the true running sum",
    });
    const msg = LR.msg(mount);

    function series() {
      const hs = [0], sums = [0];
      for (let i = 0; i < xs.length; i++) {
        hs.push(w * xs[i] + w * hs[i]);
        sums.push(sums[i] + xs[i]);
      }
      return { hs, sums };
    }

    function render() {
      const { hs, sums } = series();

      // trace table
      let html = "<table class='seq-table'><tr><th>t</th>";
      for (let i = 1; i <= xs.length; i++) html += "<th>" + i + "</th>";
      html += "</tr><tr><th>xₜ</th>";
      for (let i = 0; i < xs.length; i++)
        html += "<td class='" + (i === t - 1 ? "cur" : i < t ? "" : "dim") + "'>" + xs[i] + "</td>";
      html += "</tr><tr><th>hₜ</th>";
      for (let i = 1; i <= xs.length; i++)
        html += "<td class='" + (i === t ? "cur" : i < t ? "" : "dim") + "'>" + (i <= t ? LR.fmtF(hs[i], 2) : "·") + "</td>";
      html += "</tr><tr><th>Σ x</th>";
      for (let i = 1; i <= xs.length; i++)
        html += "<td class='" + (i <= t ? "" : "dim") + "'>" + (i <= t ? LR.fmt(sums[i], 2) : "·") + "</td>";
      html += "</tr></table>";
      tableWrap.innerHTML = html;

      // chart
      ctx.clearRect(0, 0, W, H);
      let lo = 0, hi = 1;
      hs.concat(sums).forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      const pad = Math.max(1, (hi - lo) * 0.15);
      const P = {
        x0: 56, y0: 14, w: W - 80, h: H - 62,
        xmin: 0.4, xmax: xs.length + 0.6, ymin: lo - pad, ymax: hi + pad,
        xlabel: "time step t", ylabel: "value",
        xticks: xs.map(function (_, i) { return i + 1; }),
      };
      const { sx, sy } = LR.plot(ctx, P);
      // zero line
      ctx.strokeStyle = "#cfcfcf";
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(sx(P.xmin), sy(0)); ctx.lineTo(sx(P.xmax), sy(0)); ctx.stroke();
      ctx.setLineDash([]);
      // bars for h_t
      const bw = Math.min(34, (P.w / xs.length) * 0.5);
      for (let i = 1; i <= t; i++) {
        const x = sx(i) - bw / 2;
        ctx.fillStyle = i === t ? C.orange : "rgba(232,89,12,0.45)";
        const y0 = sy(0), y1 = sy(hs[i]);
        ctx.fillRect(x, Math.min(y0, y1), bw, Math.abs(y0 - y1));
      }
      // running-sum markers
      for (let i = 1; i <= t; i++) LR.dot(ctx, sx(i), sy(sums[i]), 4.5, "#fff", C.green);
      // legend
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.orange;
      ctx.fillText("■ hₜ (the hidden state)", P.x0 + 8, P.y0 + 14);
      ctx.fillStyle = C.green;
      ctx.fillText("○ running sum Σ x", P.x0 + 8, P.y0 + 30);

      if (t === 0) { msg.hide(); return; }
      if (Math.abs(w - 1) < 1e-9) {
        msg.show("w = 1: after " + t + " step" + (t > 1 ? "s" : "") + ", h" + t + " = " + LR.fmt(hs[t], 2) +
          " and the running sum is " + LR.fmt(sums[t], 2) + ". They agree exactly: the hidden state IS the memory of every input so far.", "good");
      } else {
        // flagged addition: the w≠1 fade/swell observation is my bridge to
        // section 3; the source's example itself is only the w = 1 running sum.
        const fac = Math.pow(w, t);
        msg.show("w = " + LR.fmt(w, 2) + ": x₁'s contribution to h" + t + " has been multiplied by w^" + t + " = " +
          LR.fmtF(fac, 3) + ". " + (w < 1 ? "Old inputs fade a little more each step." : "Old inputs swell a little more each step.") +
          " Section 3 turns exactly this into the vanishing/exploding gradient story.", "info");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.3 — rnn_step as a live code exercise
     The displayed Python semantics executed for real (sigmoid).
     ══════════════════════════════════════════════════════════ */
  LR.figs.rnncode = function (mount) {
    LR.header(
      mount,
      "Run rnn_step yourself",
      "The loop from the text executes with your weights; every number in the trace is computed."
    );

    let Wh = 0.5, Wx = 1.0;
    let xs = [1.0, 0.5, -0.3, 0.9];

    const bar = LR.controls(mount);
    const whS = LR.slider(bar, "Wh (memory weight)", -1, 1.5, 0.05, Wh, function (v) { Wh = v; run(); });
    const wxS = LR.slider(bar, "Wx (input weight)", 0, 2, 0.05, Wx, function (v) { Wx = v; run(); });
    LR.button(bar, "x = [1.0, 0.5, −0.3, 0.9]", function () { xs = [1.0, 0.5, -0.3, 0.9]; run(); }, "small");
    LR.button(bar, "x = [2, 2, 2, 2, 2, 2] (saturating)", function () { xs = [2, 2, 2, 2, 2, 2]; run(); }, "small");
    LR.button(bar, "Run ▸", run, "primary");

    const ex = LR.el("div", "code-exercise");
    const pre = LR.el("pre", "code");
    pre.setAttribute("data-lang", "python");
    const codeEl = LR.el("code");
    pre.appendChild(codeEl);
    ex.appendChild(pre);
    const out = LR.el("div", "run-output", "");
    ex.appendChild(out);
    mount.appendChild(ex);
    const msg = LR.msg(mount);

    function run() {
      const src =
        "h = 0.0\n" +
        "for x in [" + xs.map(function (v) { return LR.fmt(v, 1); }).join(", ") + "]:\n" +
        "    h = rnn_step(h, x, Wh=" + LR.fmt(Wh, 2) + ", Wx=" + LR.fmt(Wx, 2) + ", b=0.0)";
      codeEl.innerHTML = LR.highlight(src);

      let h = 0.0;
      const lines = ["  t      x        z = Wh*h + Wx*x        h = sigmoid(z)"];
      let sat = false;
      for (let i = 0; i < xs.length; i++) {
        const z = Wh * h + Wx * xs[i];
        h = SEQ.sigmoid(z);
        const sp = SEQ.sigmoid(z) * (1 - SEQ.sigmoid(z));
        if (sp < 0.05) sat = true;
        lines.push(
          "  " + (i + 1) + "   " + pad(LR.fmtF(xs[i], 1), 5) + "   " + pad(LR.fmtF(z, 4), 9) +
          "                " + LR.fmtF(h, 4)
        );
      }
      out.textContent = lines.join("\n");

      if (Math.abs(Wh) < 1e-9) {
        msg.show("Wh = 0: the memory wire is cut. Each hₜ now depends on xₜ alone, and the network is back to a windowless, history-free model.", "bad");
      } else if (sat) {
        msg.show("Note the h values pinned near 1: the sigmoid is saturating, so its slope σ′(z) is nearly 0 there. Remember this for section 3, where that slope multiplies into every backward step.", "info");
      } else {
        msg.show("With Wh = 0.5, Wx = 1.0, the first two rows are the worked example from the text: h₁ = 0.7311, h₂ = 0.7038.", "good");
      }
    }
    function pad(s, n) { while (s.length < n) s = " " + s; return s; }
    run();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — the RNN language-model stepper
     A real RNN LM trained in-browser; teacher forcing vs free
     generation, hidden state carried forward, live softmax.
     ══════════════════════════════════════════════════════════ */
  LR.figs.lmstep = function (mount) {
    LR.header(
      mount,
      "Current token + memory → next-token probabilities",
      "This small RNN was trained in your browser just now (its final training loss is printed below). Step it."
    );

    const lm = SEQ.lm();
    const SENT = ["the", "cat", "sat", "on", "the", "mat", "."];
    let mode = "teacher"; // or "gen"
    let h = lm.h0.slice();
    let fed = [];         // token ids consumed so far (inputs)
    let probs = null;     // current output distribution
    let done = false;

    const bar = LR.controls(mount);
    const tg = LR.el("div", "toggle-group");
    const btnT = LR.button(tg, "Teacher forcing", function () { setMode("teacher"); });
    const btnG = LR.button(tg, "Free generation", function () { setMode("gen"); });
    bar.appendChild(tg);
    LR.button(bar, "Step ▸", step, "primary");
    LR.button(bar, "Reset ⟲", reset);

    function setMode(m) {
      mode = m;
      btnT.classList.toggle("on", m === "teacher");
      btnG.classList.toggle("on", m === "gen");
      reset();
    }

    const row = LR.el("div", "tok-row");
    row.setAttribute("aria-label", "tokens fed to the model so far");
    mount.appendChild(row);
    const strip = SEQ.heatStrip(mount, lm.H, "hidden state hₜ", 1);
    const { cv, ctx, W, H } = LR.canvas(mount, 780, 230, {
      aria: "Bar chart of the model's softmax distribution over the next token",
    });
    const narrate = LR.el("div", "seq-narrate", "");
    mount.appendChild(narrate);
    const ro = LR.readout(mount, [
      { k: "loss", label: "final training loss / token" },
      { k: "step", label: "step" },
    ]);
    ro.set("loss", LR.fmtF(lm.loss, 3));

    function reset() {
      h = lm.h0.slice();
      fed = [0]; // <s>
      const f = lm.step(h, 0);
      h = f.h; probs = f.p;
      done = false;
      narrate.innerHTML = mode === "teacher"
        ? "Fed <b>&lt;s&gt;</b> (start). The bars show p(next token | &lt;s&gt;). In teacher forcing, the true corpus sentence “" + SENT.join(" ") + "” will be fed in no matter what the model prefers."
        : "Fed <b>&lt;s&gt;</b> (start). In free generation, whatever token the model picks (greedy: its own argmax; decoding rules were last lecture) is fed back in as the next input.";
      render();
    }

    function trueNextId() {
      // in teacher mode, position of next true token in SENT
      const n = fed.length - 1; // tokens fed after <s>
      return n < SENT.length ? lm.id[SENT[n]] : null;
    }

    function step() {
      if (done) return;
      let nextId, tag;
      if (mode === "teacher") {
        nextId = trueNextId();
        if (nextId === null) { done = true; narrate.innerHTML = "<b>End of sentence.</b> Every step was a clean supervised problem: real context in, true next token as the target."; render(); return; }
        const top = argmax(probs);
        tag = "Fed “" + lm.vocab[nextId] + "” (the true token, p = " + LR.fmtF(probs[nextId], 2) + ")" +
          (top === nextId ? ", which was also the model's top guess." : ", even though the model's top guess was “" + lm.vocab[top] + "” (p = " + LR.fmtF(probs[top], 2) + ").");
      } else {
        nextId = argmax(probs);
        tag = "Model picked “" + lm.vocab[nextId] + "” (p = " + LR.fmtF(probs[nextId], 2) + ") and feeds its own pick back in. Its mistakes, if any, are now part of its context.";
      }
      fed.push(nextId);
      const f = lm.step(h, nextId);
      h = f.h; probs = f.p;
      narrate.innerHTML = tag;
      if (lm.vocab[nextId] === "." || fed.length > 14) {
        done = true;
        if (mode === "gen") narrate.innerHTML = tag + " <b>Sentence finished:</b> “" + fed.slice(1).map(function (i) { return lm.vocab[i]; }).join(" ") + "”";
      }
      render();
    }

    function argmax(p) {
      let b = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
      return b;
    }

    function render() {
      // token row
      row.innerHTML = "";
      fed.forEach(function (id, i) {
        const t = LR.el("span", "tok " + (mode === "teacher" ? "fed" : i === 0 ? "fed" : "gen"), lm.vocab[id] === "<s>" ? "⟨s⟩" : lm.vocab[id]);
        row.appendChild(t);
      });
      if (mode === "teacher") {
        for (let n = fed.length - 1; n < SENT.length; n++) {
          const t = LR.el("span", "tok faded", SENT[n]);
          row.appendChild(t);
        }
      } else if (!done) {
        row.appendChild(LR.el("span", "tok target", "?"));
      }

      strip.set(h);
      ro.set("step", String(fed.length - 1));

      // softmax bars
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 56, y0: 16, w: W - 82, h: H - 66,
        xmin: 0, xmax: lm.V, ymin: 0, ymax: 1,
        ylabel: "p(next token)", xticks: [], yticks: [0, 0.25, 0.5, 0.75, 1],
      };
      const { sx, sy } = LR.plot(ctx, P);
      const trueId = mode === "teacher" && !done ? trueNextId() : null;
      const top = argmax(probs);
      const bw = (P.w / lm.V) * 0.62;
      for (let i = 0; i < lm.V; i++) {
        const x = sx(i + 0.5) - bw / 2;
        ctx.fillStyle = i === top ? C.orange : "rgba(232,89,12,0.35)";
        ctx.fillRect(x, sy(probs[i]), bw, sy(0) - sy(probs[i]));
        if (i === trueId) {
          ctx.strokeStyle = C.green;
          ctx.lineWidth = 2.4;
          ctx.strokeRect(x - 1.5, sy(probs[i]) - 1.5, bw + 3, sy(0) - sy(probs[i]) + 3);
        }
        ctx.fillStyle = C.text;
        ctx.font = "600 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(lm.vocab[i] === "<s>" ? "⟨s⟩" : lm.vocab[i], sx(i + 0.5), P.y0 + P.h + 16);
        if (probs[i] > 0.04) {
          ctx.fillStyle = C.muted;
          ctx.font = "10.5px JetBrains Mono, Menlo, monospace";
          ctx.fillText(LR.fmtF(probs[i], 2), sx(i + 0.5), sy(probs[i]) - 5);
        }
      }
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.orange;
      ctx.fillText("■ model's distribution (argmax solid)", P.x0 + 6, P.y0 + 12);
      if (trueId !== null) {
        ctx.fillStyle = C.green;
        ctx.fillText("□ true next token (what teacher forcing feeds)", P.x0 + 6, P.y0 + 27);
      }
    }

    setMode("teacher");
  };
})();
