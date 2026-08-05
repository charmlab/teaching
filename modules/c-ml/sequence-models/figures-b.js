/* ══════════════════════════════════════════════════════════════
   figures-b.js — Sequence Models lesson figures, sections 3–5
   Fig 3.1 apow (signature) · Fig 3.2 bpttflow · Fig 3.3 clip
   Fig 4.1 lstm (signature) · Fig 5.1 chain · Fig 5.2 seq2seq
   Uses the shared LR.seq core from figures-a.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const SEQ = LR.seq;

  /* format a possibly tiny/huge positive number honestly */
  function sci(v) {
    if (v === 0) return "0";
    if (v >= 0.001 && v < 10000) return LR.fmtF(v, v < 1 ? 5 : 2);
    const e = Math.floor(Math.log10(v));
    const m = v / Math.pow(10, e);
    return LR.fmtF(m, 2) + "e" + e;
  }

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — vanishing/exploding gradient simulator (signature)
     a^t on a log axis, with the lecture's 0.9^50 / 1.1^50 marks.
     ══════════════════════════════════════════════════════════ */
  LR.figs.apow = function (mount) {
    LR.header(
      mount,
      "aᵀ: the arithmetic of deep time",
      "Each backward step multiplies the gradient by roughly a. Slide a and the number of steps T."
    );

    let a = 0.9, T = 50;

    const bar = LR.controls(mount);
    const aS = LR.slider(bar, "per-step multiplier a", 0.5, 1.5, 0.01, a, function (v) { a = v; render(); });
    const tS = LR.slider(bar, "steps T", 1, 100, 1, T, function (v) { T = Math.round(v); render(); },
      function (v) { return String(Math.round(v)); });
    LR.button(bar, "0.9⁵⁰ (vanish)", function () { a = 0.9; T = 50; aS.set(a); tS.set(T); render(); }, "small");
    LR.button(bar, "1.1⁵⁰ (explode)", function () { a = 1.1; T = 50; aS.set(a); tS.set(T); render(); }, "small");
    LR.button(bar, "a = 1 (knife edge)", function () { a = 1.0; aS.set(a); render(); }, "small");

    const { cv, ctx, W, H } = LR.canvas(mount, 800, 330, {
      aria: "Log-scale plot of a to the power t, collapsing toward zero for a below 1 and blowing up for a above 1",
    });
    const ro = LR.readout(mount, [
      { k: "val", label: "aᵀ" },
      { k: "verdict", label: "verdict" },
    ]);
    const msg = LR.msg(mount);

    const YMIN = -8, YMAX = 8; // log10 range
    function render() {
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 64, y0: 16, w: W - 92, h: H - 68,
        xmin: 0, xmax: Math.max(T, 50), ymin: YMIN, ymax: YMAX,
        xlabel: "steps back through time, t",
        ylabel: "gradient scale aᵗ (log axis, 10ʸ)",
        yticks: [-8, -6, -4, -2, 0, 2, 4, 6, 8],
      };
      const { sx, sy } = LR.plot(ctx, P);

      // the stable line a^t = 1
      ctx.strokeStyle = "#bbb";
      ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(sx(0), sy(0)); ctx.lineTo(sx(P.xmax), sy(0)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.faint;
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("aᵗ = 1 (the only stable line)", sx(1), sy(0) - 6);

      // reference curves 0.9 and 1.1 (grey), with the lecture's landmarks
      [[0.9, "0.9⁵⁰ ≈ " + sci(Math.pow(0.9, 50))], [1.1, "1.1⁵⁰ ≈ " + sci(Math.pow(1.1, 50))]].forEach(function (ref) {
        const ra = ref[0];
        ctx.strokeStyle = "#c9c9c9";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        for (let t = 0; t <= P.xmax; t++) {
          const y = Math.max(YMIN, Math.min(YMAX, t * Math.log10(ra)));
          if (t === 0) ctx.moveTo(sx(t), sy(y)); else ctx.lineTo(sx(t), sy(y));
        }
        ctx.stroke();
        if (P.xmax >= 50) {
          const y50 = Math.max(YMIN, Math.min(YMAX, 50 * Math.log10(ra)));
          LR.dot(ctx, sx(50), sy(y50), 4, "#fff", C.purple);
          ctx.fillStyle = C.purple;
          ctx.font = "700 11px Inter, sans-serif";
          ctx.textAlign = ra < 1 ? "left" : "left";
          ctx.fillText(ref[1], sx(50) + 8, sy(y50) + (ra < 1 ? 12 : -6));
        }
      });

      // the live curve
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      let clipped = false;
      for (let t = 0; t <= T; t++) {
        let y = t * Math.log10(a);
        if (y < YMIN || y > YMAX) clipped = true;
        y = Math.max(YMIN, Math.min(YMAX, y));
        if (t === 0) ctx.moveTo(sx(t), sy(y)); else ctx.lineTo(sx(t), sy(y));
      }
      ctx.stroke();
      const yT = Math.max(YMIN, Math.min(YMAX, T * Math.log10(a)));
      LR.dot(ctx, sx(T), sy(yT), 5.5, C.orange);

      const val = Math.pow(a, T);
      ro.set("val", sci(val) + (clipped ? "  (off the chart)" : ""), C.orange);
      if (Math.abs(a - 1) < 0.005) {
        ro.set("verdict", "stable, but only by luck", C.green);
        msg.show("a = 1 exactly: the product neither grows nor shrinks. Nothing in training pins a here; it is a knife edge, not a place to live.", "info");
      } else if (a < 1) {
        ro.set("verdict", "vanishing", C.red);
        msg.show("a = " + LR.fmt(a, 2) + " < 1: after " + T + " steps the gradient is scaled by " + sci(val) +
          ". Early inputs get essentially no credit or blame for late predictions, so long-range dependencies cannot be learned.", "bad");
      } else {
        ro.set("verdict", "exploding", C.red);
        msg.show("a = " + LR.fmt(a, 2) + " > 1: after " + T + " steps the gradient is scaled by " + sci(val) +
          ". Updates that size make optimization unstable: the loss spikes instead of falling.", "bad");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.2 — BPTT flow on a real unrolled scalar RNN
     Forward pass with tanh, then the true product of local
     factors σ'(z_k)·w flowing backward from the last step.
     ══════════════════════════════════════════════════════════ */
  LR.figs.bpttflow = function (mount) {
    LR.header(
      mount,
      "A gradient's journey back through time",
      "Forward pass left to right, then step the error signal backward and watch the product build."
    );

    // fixed input sequence; tanh scalar RNN (σ in the text is a generic
    // squashing activation; tanh makes both regimes reachable with one w)
    const xs = [0.4, -0.2, 0.5, 0.1, -0.4, 0.3, 0.2, -0.1];
    const T = xs.length;
    const wx = 0.6;
    let w = 0.7;
    let ptr = T; // gradient has reached h_ptr (T = just left the loss)

    const bar = LR.controls(mount);
    LR.button(bar, "Step backward ◂", function () { if (ptr > 1) { ptr -= 1; render(); } }, "primary");
    LR.button(bar, "All the way ◂◂", function () { ptr = 1; render(); });
    LR.button(bar, "Reset ⟲", function () { ptr = T; render(); });
    const wS = LR.slider(bar, "recurrent weight w", 0.3, 2.5, 0.05, w, function (v) { w = v; render(); });
    LR.button(bar, "w = 0.7 (vanishes)", function () { w = 0.7; wS.set(w); render(); }, "small");
    LR.button(bar, "w = 2.2 (explodes)", function () { w = 2.2; wS.set(w); render(); }, "small");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Unrolled RNN chain with a gradient signal flowing backward, fading or amplifying at each hop",
    });
    const narrate = LR.el("div", "seq-narrate", "");
    mount.appendChild(narrate);

    function compute() {
      const hs = [0], zs = [0];
      for (let t = 0; t < T; t++) {
        const z = w * hs[t] + wx * xs[t];
        zs.push(z);
        hs.push(Math.tanh(z));
      }
      // local backward factor crossing from h_k to h_{k-1} is tanh'(z_k)·w
      const fac = [null];
      for (let k = 1; k <= T; k++) fac.push(SEQ.dtanhFromH(hs[k]) * w);
      // g[t] = ∂h_T/∂h_t
      const g = new Array(T + 1).fill(1);
      for (let t = T - 1; t >= 1; t--) g[t] = g[t + 1] * fac[t + 1];
      return { hs, fac, g };
    }

    function render() {
      const { hs, fac, g } = compute();
      ctx.clearRect(0, 0, W, H);

      const cw = 66, ch = 44, y = 96;
      const gap = (W - 60 - T * cw) / (T - 1);
      const cx = function (t) { return 30 + (t - 1) * (cw + gap); }; // cell t: 1…T

      // loss node
      const lx = cx(T) + cw / 2;
      ctx.fillStyle = C.red;
      ctx.font = "700 13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("loss", lx, 24);
      LR.arrow(ctx, lx, 58, lx, 30, C.red, 2);

      for (let t = 1; t <= T; t++) {
        const x = cx(t);
        // cell
        ctx.fillStyle = t >= ptr ? "#fff5ec" : "#fff";
        ctx.strokeStyle = t >= ptr ? C.orange : C.axis;
        ctx.lineWidth = t === ptr ? 2.4 : 1.4;
        ctx.fillRect(x, y - ch / 2, cw, ch);
        ctx.strokeRect(x, y - ch / 2, cw, ch);
        ctx.fillStyle = C.text;
        ctx.font = "700 12px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("h" + t, x + cw / 2, y - 2);
        ctx.font = "10.5px JetBrains Mono, Menlo, monospace";
        ctx.fillStyle = C.muted;
        ctx.fillText(LR.fmtF(hs[t], 2), x + cw / 2, y + 13);

        // forward arrow
        if (t > 1) {
          LR.arrow(ctx, cx(t - 1) + cw, y + 8, x, y + 8, "#c9c9c9", 1.6);
        }
        // backward arrow + factor label
        if (t > 1 && t >= ptr + 1) {
          const mag = Math.abs(g[t - 1]);
          const width = Math.max(1.2, Math.min(7, 2 + Math.log10(Math.max(mag, 1e-9)) + 3));
          LR.arrow(ctx, x, y - 14, cx(t - 1) + cw, y - 14, C.purple, width);
          ctx.fillStyle = C.purple;
          ctx.font = "700 10.5px JetBrains Mono, Menlo, monospace";
          ctx.fillText("×" + LR.fmtF(fac[t], 2), (x + cx(t - 1) + cw) / 2, y - 24);
        }
      }

      // gradient magnitude bars (log scale)
      const by0 = 300, bh = 120;
      const LMIN = -6, LMAX = 3;
      ctx.strokeStyle = C.grid;
      ctx.beginPath(); ctx.moveTo(24, by0 - bh * ((0 - LMIN) / (LMAX - LMIN))); ctx.lineTo(W - 24, by0 - bh * ((0 - LMIN) / (LMAX - LMIN))); ctx.stroke();
      ctx.fillStyle = C.faint;
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("|∂h₈/∂hₜ| = 1", 26, by0 - bh * ((0 - LMIN) / (LMAX - LMIN)) - 4);
      for (let t = 1; t <= T; t++) {
        const x = cx(t);
        const known = t >= ptr;
        const mag = Math.abs(g[t]);
        const lv = Math.max(LMIN, Math.min(LMAX, Math.log10(Math.max(mag, 1e-12))));
        const hpx = bh * ((lv - LMIN) / (LMAX - LMIN));
        ctx.fillStyle = known ? (mag < 0.01 ? "rgba(224,49,49,0.75)" : C.orange) : "#e6e6e6";
        ctx.fillRect(x + 8, by0 - hpx, cw - 16, hpx);
        if (known) {
          ctx.fillStyle = C.text;
          ctx.font = "700 10px JetBrains Mono, Menlo, monospace";
          ctx.textAlign = "center";
          ctx.fillText(sci(mag), x + cw / 2, by0 - hpx - 5);
        }
      }
      ctx.fillStyle = C.muted;
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("|∂h₈/∂hₜ|, log scale:", 26, by0 - bh - 14);

      // narration
      if (ptr === T) {
        narrate.innerHTML = "The gradient leaves the loss at step 8 with size 1. Press <b>Step backward ◂</b>. Each hop to the left multiplies by that hop's true local factor tanh′(zₖ)·w, exactly the σ′(zₖ)·w product from the text.";
      } else {
        const terms = [];
        for (let k = T; k > ptr; k--) terms.push(LR.fmtF(fac[k], 2));
        narrate.innerHTML = "∂h₈/∂h" + ptr + " = " + terms.join(" × ") + " = <b>" + sci(Math.abs(g[ptr])) + "</b>" +
          (Math.abs(g[ptr]) < 0.01 ? ". Barely anything left: whatever happened at step " + ptr + " gets almost no credit or blame for the loss at step 8."
            : Math.abs(g[ptr]) > 20 ? ". Already huge: an update scaled like this is a wrecking ball, not a step." : ".");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.3 — gradient clipping demo
     Drag g; when ‖g‖ > η it is rescaled to length η, same angle.
     ══════════════════════════════════════════════════════════ */
  LR.figs.clip = function (mount) {
    LR.header(
      mount,
      "Clipping: cap the length, keep the direction",
      "Drag the gradient's tip. Outside the circle, the clip rule g ← η·g/‖g‖ fires."
    );

    let g = { x: 3, y: 4 };
    let eta = 2;

    const bar = LR.controls(mount);
    LR.slider(bar, "threshold η", 0.5, 6, 0.1, eta, function (v) { eta = v; render(); });
    LR.button(bar, "g = (3, 4), the worked example", function () { g = { x: 3, y: 4 }; render(); }, "small");

    const { cv, ctx, W, H } = LR.canvas(mount, 560, 380, {
      aria: "A draggable gradient vector and its clipped version inside a threshold circle",
    });
    const ro = LR.readout(mount, [
      { k: "norm", label: "‖g‖" },
      { k: "clipped", label: "after clipping" },
      { k: "dir", label: "direction" },
    ]);
    const msg = LR.msg(mount);

    const R = 6.5; // world half-range
    const P = { x0: 60, y0: 14, w: W - 90, h: H - 68 };
    const sx = function (x) { return P.x0 + ((x + R) / (2 * R)) * P.w; };
    const sy = function (y) { return P.y0 + P.h - ((y + R) / (2 * R)) * P.h; };
    const inv = function (px, py) {
      return { x: ((px - P.x0) / P.w) * 2 * R - R, y: ((P.y0 + P.h - py) / P.h) * 2 * R - R };
    };

    LR.drag(cv, W, H, {
      hit: function (p) {
        const wpt = inv(p.x, p.y);
        return Math.hypot(wpt.x - g.x, wpt.y - g.y) < 1.2;
      },
      move: function (p) {
        const wpt = inv(p.x, p.y);
        g.x = Math.max(-R + 0.3, Math.min(R - 0.3, wpt.x));
        g.y = Math.max(-R + 0.3, Math.min(R - 0.3, wpt.y));
        render();
      },
    });

    function render() {
      ctx.clearRect(0, 0, W, H);
      LR.plot(ctx, {
        x0: P.x0, y0: P.y0, w: P.w, h: P.h,
        xmin: -R, xmax: R, ymin: -R, ymax: R,
        xlabel: "g₁", ylabel: "g₂",
        xticks: [-6, -4, -2, 0, 2, 4, 6], yticks: [-6, -4, -2, 0, 2, 4, 6],
      });

      // threshold circle
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.ellipse(sx(0), sy(0), (eta / (2 * R)) * P.w, (eta / (2 * R)) * P.h, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.green;
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("‖g‖ = η", sx(eta * 0.72), sy(eta * 0.76));

      const norm = Math.hypot(g.x, g.y);
      const over = norm > eta;
      const cg = over ? { x: (g.x * eta) / norm, y: (g.y * eta) / norm } : g;

      // original vector
      ctx.save();
      if (over) ctx.setLineDash([5, 4]);
      LR.arrow(ctx, sx(0), sy(0), sx(g.x), sy(g.y), over ? "#9a9a9a" : C.orange, 2.4);
      ctx.restore();
      // clipped vector
      if (over) LR.arrow(ctx, sx(0), sy(0), sx(cg.x), sy(cg.y), C.orange, 3);

      // drag handle
      LR.dot(ctx, sx(g.x), sy(g.y), 7, "#fff", over ? "#9a9a9a" : C.orange);
      ctx.fillStyle = C.text;
      ctx.font = "700 11.5px JetBrains Mono, Menlo, monospace";
      ctx.textAlign = "left";
      ctx.fillText("g = (" + LR.fmtF(g.x, 1) + ", " + LR.fmtF(g.y, 1) + ")", sx(g.x) + 10, sy(g.y) - 8);

      const ang = (Math.atan2(g.y, g.x) * 180) / Math.PI;
      ro.set("norm", LR.fmtF(norm, 2), over ? C.red : C.green);
      ro.set("clipped", over
        ? "(" + LR.fmtF(cg.x, 2) + ", " + LR.fmtF(cg.y, 2) + "), norm " + LR.fmtF(Math.hypot(cg.x, cg.y), 2)
        : "unchanged");
      ro.set("dir", LR.fmtF(ang, 1) + "° before and after");
      if (over) {
        msg.show("‖g‖ = " + LR.fmtF(norm, 2) + " > η = " + LR.fmtF(eta, 1) + ": the update is rescaled to length η. Same direction, sane size. Note what clipping cannot do: it never makes a small gradient bigger.", "good");
      } else {
        msg.show("‖g‖ = " + LR.fmtF(norm, 2) + " ≤ η = " + LR.fmtF(eta, 1) + ": clipping leaves g completely untouched. A vanished gradient lives here, inside the circle, beyond clipping's reach.", "info");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — the LSTM gate explorer (signature figure)
     Left: one gated cell computed live. Right: persistence f^t.
     ══════════════════════════════════════════════════════════ */
  LR.figs.lstm = function (mount) {
    LR.header(
      mount,
      "Three gates and a conveyor belt",
      "Slide the gates and watch memory being discarded, stored, and exposed, with every number computed."
    );

    let f = 0.9, i = 0.5, o = 0.8;
    let Cprev = 1.0, Ccand = 0.8;

    const bar = LR.controls(mount);
    const fS = LR.slider(bar, "forget gate f", 0, 1, 0.01, f, function (v) { f = v; render(); });
    const iS = LR.slider(bar, "input gate i", 0, 1, 0.01, i, function (v) { i = v; render(); });
    const oS = LR.slider(bar, "output gate o", 0, 1, 0.01, o, function (v) { o = v; render(); });
    const cpS = LR.slider(bar, "old memory Cₜ₋₁", -2, 2, 0.1, Cprev, function (v) { Cprev = v; render(); });
    const ccS = LR.slider(bar, "candidate C̃ₜ", -1, 1, 0.05, Ccand, function (v) { Ccand = v; render(); });

    const bar2 = LR.controls(mount);
    function preset(name, ff, ii, oo) {
      LR.button(bar2, name, function () {
        f = ff; i = ii; o = oo;
        fS.set(f); iS.set(i); oS.set(o);
        render();
      }, "small");
    }
    preset("Remember everything (f=1, i=0)", 1, 0, 1);
    preset("Forget fast (f=0.2)", 0.2, 0.5, 1);
    preset("Overwrite (f=0, i=1)", 0, 1, 1);
    preset("Keep but hide (o=0)", 1, 0, 0);

    const panes = LR.el("div", "dual-pane");
    mount.appendChild(panes);
    const left = LR.el("div");
    left.appendChild(LR.el("div", "pane-label", "one LSTM step, live"));
    const right = LR.el("div");
    right.appendChild(LR.el("div", "pane-label", "how long a stored value survives"));
    panes.appendChild(left);
    panes.appendChild(right);

    const L = LR.canvas(left, 400, 300, { aria: "LSTM cell diagram with live values at the forget, input, and output gates" });
    const Rt = LR.canvas(right, 400, 300, { aria: "Decay curve of stored memory under the forget gate, f to the power t" });
    const msg = LR.msg(mount);

    function render() {
      const Ct = f * Cprev + i * Ccand;
      const tC = Math.tanh(Ct);
      const ht = o * tC;

      /* ── left: the cell ── */
      const ctx = L.ctx;
      ctx.clearRect(0, 0, L.W, L.H);
      const beltY = 64;

      // conveyor belt
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(18, beltY); ctx.lineTo(L.W - 60, beltY); ctx.stroke();
      LR.arrow(ctx, L.W - 66, beltY, L.W - 30, beltY, C.green, 3);
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.green;
      ctx.textAlign = "left";
      ctx.fillText("cell state: the conveyor belt", 18, beltY - 38);

      // C_{t-1} in
      ctx.font = "700 12.5px JetBrains Mono, Menlo, monospace";
      ctx.fillStyle = C.text;
      ctx.fillText("Cₜ₋₁ = " + LR.fmtF(Cprev, 2), 18, beltY - 12);
      // forget gate node
      gateNode(ctx, 150, beltY, "×", C.red);
      ctx.fillStyle = C.red;
      ctx.textAlign = "center";
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillText("forget f = " + LR.fmtF(f, 2), 150, beltY + 32);
      ctx.font = "600 10.5px JetBrains Mono, Menlo, monospace";
      ctx.fillStyle = C.muted;
      ctx.fillText("f·Cₜ₋₁ = " + LR.fmtF(f * Cprev, 2), 150, beltY - 16);

      // input gate node
      gateNode(ctx, 252, beltY, "+", C.purple);
      ctx.fillStyle = C.purple;
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillText("input i = " + LR.fmtF(i, 2), 252, beltY + 32);
      ctx.font = "600 10.5px JetBrains Mono, Menlo, monospace";
      ctx.fillStyle = C.muted;
      ctx.fillText("+ i·C̃ = " + LR.fmtF(i * Ccand, 2), 252, beltY - 16);
      // candidate feed
      LR.arrow(ctx, 252, beltY + 74, 252, beltY + 12, C.purple, 2);
      ctx.fillStyle = C.purple;
      ctx.font = "600 11px JetBrains Mono, Menlo, monospace";
      ctx.fillText("C̃ₜ = " + LR.fmtF(Ccand, 2) + " (from hₜ₋₁, xₜ)", 252, beltY + 90);

      // C_t out
      ctx.fillStyle = C.green;
      ctx.font = "700 12.5px JetBrains Mono, Menlo, monospace";
      ctx.textAlign = "right";
      ctx.fillText("Cₜ = " + LR.fmtF(Ct, 2), L.W - 14, beltY - 12);

      // output branch
      const oy = 210;
      LR.arrow(ctx, L.W - 46, beltY + 6, L.W - 46, oy - 34, "#c9c9c9", 1.8);
      ctx.fillStyle = C.muted;
      ctx.textAlign = "center";
      ctx.font = "600 10.5px JetBrains Mono, Menlo, monospace";
      ctx.fillText("tanh(Cₜ) = " + LR.fmtF(tC, 2), L.W - 80, oy - 42);
      gateNode(ctx, L.W - 46, oy - 18, "×", C.orange);
      ctx.fillStyle = C.orange;
      ctx.font = "700 11.5px Inter, sans-serif";
      ctx.fillText("output o = " + LR.fmtF(o, 2), L.W - 90, oy + 4);
      LR.arrow(ctx, L.W - 46, oy - 4, L.W - 46, oy + 30, C.orange, 2.4);
      ctx.fillStyle = C.text;
      ctx.font = "700 13px JetBrains Mono, Menlo, monospace";
      ctx.fillText("hₜ = o·tanh(Cₜ) = " + LR.fmtF(ht, 2), L.W - 120, oy + 52);

      // the belt equation, printed from the live numbers
      ctx.fillStyle = C.text;
      ctx.textAlign = "left";
      ctx.font = "700 12px JetBrains Mono, Menlo, monospace";
      ctx.fillText("Cₜ = f·Cₜ₋₁ + i·C̃ₜ = " + LR.fmtF(f * Cprev, 2) + " + " + LR.fmtF(i * Ccand, 2) + " = " + LR.fmtF(Ct, 2), 18, 278);

      /* ── right: persistence f^t ── */
      const cx2 = Rt.ctx;
      cx2.clearRect(0, 0, Rt.W, Rt.H);
      const P = {
        x0: 46, y0: 16, w: Rt.W - 66, h: Rt.H - 66,
        xmin: 0, xmax: 30, ymin: 0, ymax: 1.05,
        xlabel: "steps since the value was stored", ylabel: "remaining fraction fᵗ",
        yticks: [0, 0.25, 0.5, 0.75, 1],
      };
      const { sx, sy } = LR.plot(cx2, P);
      // reference: f = 1 (perfect belt)
      cx2.strokeStyle = C.green;
      cx2.lineWidth = 1.6;
      cx2.setLineDash([5, 4]);
      cx2.beginPath(); cx2.moveTo(sx(0), sy(1)); cx2.lineTo(sx(30), sy(1)); cx2.stroke();
      cx2.setLineDash([]);
      cx2.fillStyle = C.green;
      cx2.font = "600 10.5px Inter, sans-serif";
      cx2.textAlign = "left";
      cx2.fillText("f = 1: remembered forever", sx(1), sy(1) + 14);
      // reference: vanilla-ish f = 0.8
      cx2.strokeStyle = "#c9c9c9";
      cx2.beginPath();
      for (let t = 0; t <= 30; t++) {
        const y = Math.pow(0.8, t);
        if (t === 0) cx2.moveTo(sx(t), sy(y)); else cx2.lineTo(sx(t), sy(y));
      }
      cx2.stroke();
      cx2.fillStyle = C.faint;
      cx2.fillText("f = 0.8", sx(6.4), sy(Math.pow(0.8, 6)) - 6);
      // live curve
      cx2.strokeStyle = C.orange;
      cx2.lineWidth = 2.6;
      cx2.beginPath();
      for (let t = 0; t <= 30; t++) {
        const y = Math.pow(f, t);
        if (t === 0) cx2.moveTo(sx(t), sy(y)); else cx2.lineTo(sx(t), sy(y));
      }
      cx2.stroke();
      const f30 = Math.pow(f, 30);
      LR.dot(cx2, sx(30), sy(f30), 5, C.orange);
      cx2.fillStyle = C.orange;
      cx2.font = "700 11px JetBrains Mono, Menlo, monospace";
      cx2.textAlign = "right";
      cx2.fillText("f³⁰ = " + sci(f30), sx(29.4), sy(f30) - 10);

      // message
      if (o === 0 && f > 0.95) {
        msg.show("Output gate closed: hₜ = 0, so downstream layers see nothing. But look at the belt: Cₜ = " + LR.fmtF(Ct, 2) + " is still riding along. The memory is kept, just not exposed. Remembering and reporting are separate decisions.", "info");
      } else if (f >= 0.995 && i <= 0.05) {
        msg.show("Forget gate open, input gate shut: Cₜ ≈ Cₜ₋₁. The value is copied forward, not squashed forward. This near-identity path is what a vanilla RNN never offers.", "good");
      } else if (f <= 0.3) {
        msg.show("Forget gate mostly closed: after 30 steps only f³⁰ = " + sci(f30) + " of a stored value remains. Sometimes that is exactly right; the point is the model gets to learn when.", "info");
      } else {
        msg.hide();
      }
    }

    function gateNode(ctx, x, y, sym, color) {
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "800 14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(sym, x, y + 5);
    }

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — bidirectional and stacked cells
     Click a token; toggle the backward pass and a second layer.
     ══════════════════════════════════════════════════════════ */
  LR.figs.chain = function (mount) {
    LR.header(
      mount,
      "What can this position see?",
      "Pick a token, then change the architecture and watch its context grow."
    );

    const WORDS = ["Though", "he", "later", "went", "home", ",", "Amir", "went", "to", "class"];
    const N = WORDS.length;
    let sel = 8; // "to"
    let bidir = false;
    let stacked = false;

    const bar = LR.controls(mount);
    const bBtn = LR.button(bar, "Backward pass: off", function () {
      bidir = !bidir;
      bBtn.textContent = "Backward pass: " + (bidir ? "on" : "off");
      bBtn.classList.toggle("on", bidir);
      render();
    });
    const sBtn = LR.button(bar, "Second layer: off", function () {
      stacked = !stacked;
      sBtn.textContent = "Second layer: " + (stacked ? "on" : "off");
      sBtn.classList.toggle("on", stacked);
      render();
    });

    const row = LR.el("div", "tok-row");
    row.setAttribute("aria-label", "sentence tokens; click one to select it");
    const btns = WORDS.map(function (w, idx) {
      const b = LR.el("button", "tok", w);
      b.type = "button";
      b.addEventListener("click", function () { sel = idx; render(); });
      row.appendChild(b);
      return b;
    });
    mount.appendChild(row);

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 250, {
      aria: "Chains of recurrent cells over the sentence, forward, optionally backward, optionally stacked",
    });
    const ro = LR.readout(mount, [
      { k: "sees", label: "this position's context" },
      { k: "cost", label: "passes over the sequence" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      btns.forEach(function (b, idx) {
        const reach = bidir || idx <= sel;
        b.className = "tok" + (idx === sel ? " win" : reach ? "" : " faded");
      });

      ctx.clearRect(0, 0, W, H);
      const cw = 56, ch = 34;
      const gap = (W - 60 - N * cw) / (N - 1);
      const cx = function (i) { return 30 + i * (cw + gap); };

      function layerRow(y, color, dir, label) {
        ctx.fillStyle = color;
        ctx.font = "700 11px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(label, 30, y - ch / 2 - 6);
        for (let i2 = 0; i2 < N; i2++) {
          const x = cx(i2);
          ctx.fillStyle = i2 === sel ? "#fff5ec" : "#fff";
          ctx.strokeStyle = i2 === sel ? C.orange : "#bbb";
          ctx.lineWidth = i2 === sel ? 2.2 : 1.2;
          ctx.fillRect(x, y - ch / 2, cw, ch);
          ctx.strokeRect(x, y - ch / 2, cw, ch);
          if (i2 > 0) {
            const x1 = cx(i2 - 1) + cw, x2 = x;
            if (dir === "fwd" || dir === "both") LR.arrow(ctx, x1, y - 6, x2, y - 6, color, 1.8);
            if (dir === "bwd" || dir === "both") LR.arrow(ctx, x2, y + 8, x1, y + 8, C.green, 1.8);
          }
        }
      }

      const y1 = stacked ? 190 : 150;
      layerRow(y1, C.orange, bidir ? "both" : "fwd", "layer 1" + (bidir ? "  (forward + backward)" : "  (forward only)"));
      if (stacked) {
        const y2 = 92;
        layerRow(y2, C.purple, bidir ? "both" : "fwd", "layer 2 (reads layer 1's states as its input sequence)");
        for (let i2 = 0; i2 < N; i2++) {
          LR.arrow(ctx, cx(i2) + cw / 2, y1 - ch / 2, cx(i2) + cw / 2, y2 + ch / 2 + 4, "#c9c9c9", 1.5);
        }
      }

      // token labels under layer 1
      ctx.font = "600 11px Inter, sans-serif";
      ctx.textAlign = "center";
      for (let i2 = 0; i2 < N; i2++) {
        const reach = bidir || i2 <= sel;
        ctx.fillStyle = i2 === sel ? C.orange : reach ? C.text : "#c4c4c4";
        ctx.fillText(WORDS[i2], cx(i2) + cw / 2, y1 + ch / 2 + 18);
      }

      const seen = bidir ? N : sel + 1;
      ro.set("sees", seen + " of " + N + " tokens", seen === N ? C.green : C.red);
      ro.set("cost", String((bidir ? 2 : 1) * (stacked ? 2 : 1)));

      const parts = [];
      if (!bidir && sel < N - 1) {
        parts.push("Forward only: “" + WORDS[sel] + "” is represented using its prefix alone. If the disambiguating words come later (they do in this sentence), tough luck.");
      }
      if (bidir) {
        parts.push("Bidirectional: a flipped copy reads right to left, so every position sees the whole sentence. The catch: the future must already exist, fine for analyzing text, impossible while generating it.");
      }
      if (stacked) {
        parts.push("Stacked: layer 2 treats layer 1's hidden states as its input sequence, better features for more compute. In practice, usually just two or three layers.");
      }
      msg.show(parts.join(" "), bidir ? "good" : "info");
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.2 — the seq2seq bottleneck
     A real encoder RNN; everything funnels through h_T.
     ══════════════════════════════════════════════════════════ */
  LR.figs.seq2seq = function (mount) {
    LR.header(
      mount,
      "The whole input through one vector",
      "The encoder reads T tokens into hidden states; the decoder receives only the last one."
    );

    const D = 4; // encoder hidden size
    let T = 6;
    let attn = false;

    // real encoder weights + per-position inputs, fixed by seed
    const rand = LR.rng(57);
    const Whh = [], Wxh = [];
    for (let a = 0; a < D; a++) {
      const r1 = [], r2 = [];
      for (let b = 0; b < D; b++) { r1.push((rand() * 2 - 1) * 0.55); r2.push((rand() * 2 - 1) * 0.9); }
      Whh.push(r1); Wxh.push(r2);
    }
    const XS = [];
    for (let t = 0; t < 10; t++) {
      const v = [];
      for (let a = 0; a < D; a++) v.push(rand() * 2 - 1);
      XS.push(v);
    }

    function encode(Tn) {
      const hs = [SEQ.zeros(D)];
      for (let t = 0; t < Tn; t++) {
        const hp = hs[t], x = XS[t], h = SEQ.zeros(D);
        for (let a = 0; a < D; a++) {
          let s = 0;
          for (let b = 0; b < D; b++) s += Whh[a][b] * hp[b] + Wxh[a][b] * x[b];
          h[a] = Math.tanh(s);
        }
        hs.push(h);
      }
      return hs.slice(1);
    }

    const bar = LR.controls(mount);
    LR.slider(bar, "input length T", 2, 10, 1, T, function (v) { T = Math.round(v); render(); },
      function (v) { return String(Math.round(v)); });
    const aBtn = LR.button(bar, "Attention preview: off", function () {
      attn = !attn;
      aBtn.textContent = "Attention preview: " + (attn ? "on" : "off");
      aBtn.classList.toggle("on", attn);
      render();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 330, {
      aria: "Encoder hidden states funnelling into a single vector handed to the decoder",
    });
    const ro = LR.readout(mount, [
      { k: "in", label: "encoder produces" },
      { k: "through", label: "decoder receives" },
    ]);
    const msg = LR.msg(mount);

    function heatCol(x, y, vals, big) {
      const s = big ? 17 : 13;
      for (let a = 0; a < D; a++) {
        ctx.fillStyle = SEQ.heatColor(vals[a], 1);
        ctx.fillRect(x, y + a * (s + 2), s, s);
        ctx.strokeStyle = "rgba(0,0,0,0.15)";
        ctx.strokeRect(x, y + a * (s + 2), s, s);
      }
      return { w: s, h: D * (s + 2) - 2 };
    }

    function render() {
      const hs = encode(T);
      ctx.clearRect(0, 0, W, H);

      const encY = 60;
      const encSpan = W * 0.55;
      const step = encSpan / T;
      const bx = 34;

      ctx.fillStyle = C.text;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("encoder (a real RNN, values computed live)", bx, 26);

      const encPos = [];
      for (let t = 0; t < T; t++) {
        const x = bx + t * step;
        const dims = heatCol(x, encY, hs[t]);
        encPos.push({ x: x + dims.w / 2, y: encY + dims.h });
        if (t > 0) LR.arrow(ctx, bx + (t - 1) * step + dims.w + 2, encY + dims.h / 2, x - 3, encY + dims.h / 2, "#c9c9c9", 1.5);
        ctx.fillStyle = C.muted;
        ctx.font = "600 10.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("w" + (t + 1), x + dims.w / 2, encY + dims.h + 16);
        ctx.fillStyle = C.faint;
        ctx.font = "600 9.5px Inter, sans-serif";
        ctx.fillText("h" + (t + 1), x + dims.w / 2, encY - 8);
      }

      // the funnel to h_T
      const hx = W * 0.66, hy = 92;
      const last = encPos[T - 1];
      ctx.fillStyle = "rgba(232, 89, 12, 0.12)";
      ctx.beginPath();
      ctx.moveTo(bx, encY - 4);
      ctx.lineTo(bx + encSpan - step + 14, encY - 4);
      ctx.lineTo(hx, hy);
      ctx.lineTo(hx, hy + 70);
      ctx.lineTo(bx, encY + 66);
      ctx.closePath();
      ctx.fill();

      const hDims = heatCol(hx, hy, hs[T - 1], true);
      ctx.strokeStyle = C.orange;
      ctx.lineWidth = 2.2;
      ctx.strokeRect(hx - 3, hy - 3, hDims.w + 6, hDims.h + 6);
      ctx.fillStyle = C.orange;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("h_T", hx + hDims.w / 2, hy - 12);
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.fillText("the bottleneck", hx + hDims.w / 2, hy + hDims.h + 18);

      // decoder
      const decY = 240, decN = 4;
      const dx0 = W * 0.42, dstep = 78;
      ctx.fillStyle = C.text;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("decoder (generates the output one token at a time)", dx0 - 8, decY - 32);
      for (let t = 0; t < decN; t++) {
        const x = dx0 + t * dstep;
        ctx.fillStyle = "#fff";
        ctx.strokeStyle = C.axis;
        ctx.lineWidth = 1.4;
        ctx.fillRect(x, decY - 17, 56, 34);
        ctx.strokeRect(x, decY - 17, 56, 34);
        ctx.fillStyle = C.text;
        ctx.font = "700 11.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("y" + (t + 1), x + 28, decY + 4);
        if (t > 0) LR.arrow(ctx, x - dstep + 56 + 2, decY, x - 3, decY, "#c9c9c9", 1.5);
        // handoff arrows: everything the decoder knows about the input
        if (t === 0) LR.arrow(ctx, hx + hDims.w / 2, hy + hDims.h + 26, x + 28, decY - 20, C.orange, 2.4);
        // attention preview: direct lines back to every encoder state
        if (attn) {
          ctx.save();
          ctx.globalAlpha = 0.45;
          for (let e = 0; e < T; e++) {
            ctx.strokeStyle = C.purple;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(x + 28, decY - 17);
            ctx.lineTo(encPos[e].x, encPos[e].y + 4);
            ctx.stroke();
          }
          ctx.restore();
        }
      }

      ro.set("in", T + " states × " + D + " numbers = " + T * D + " numbers");
      ro.set("through", D + " numbers, no matter how long the input", C.orange);
      if (attn) {
        msg.show("The preview of next lecture: give the decoder direct access to ALL the encoder states, h₁ … h_T, and let it choose which matter for the token it is producing right now. Retrieval instead of preservation.", "good");
      } else if (T >= 8) {
        msg.show("T = " + T + ": " + (T * D) + " numbers of computed context, and exactly " + D + " of them survive the handoff. The longer the input, the more the final state has to forget.", "bad");
      } else {
        msg.show("Slide T up. The input grows; the handoff does not. One fixed-size vector must summarize the entire sequence for the decoder.", "info");
      }
    }
    render();
  };
})();
