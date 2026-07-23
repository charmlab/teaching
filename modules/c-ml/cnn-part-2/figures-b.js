/* ══════════════════════════════════════════════════════════════
   figures-b.js — CNNs in Practice (Lecture 12), sections 4.2–7
   Fig 4.2 flops · Fig 4.3 layerreport · Fig 4.4 skipconn
   Fig 5.1 transfer (signature) · Fig 6.1 filters
   Fig 6.2 adversarial · Fig 7.1 segmentation
   Uses LR.cnn2 (this lesson's convolution/sizing core) from
   figures-a.js. All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const CNN2 = LR.cnn2;

  /* ════════════════════════════════════════════════════════════
     Fig 4.2 — parameters vs connections vs FLOPS for a conv layer
     params = (K²J + 1)·M,  connections = Ŵ·Ĥ·M·(K²J + 1),
     FLOPS ≈ 2 × connections. Dense layer shown for contrast.
     ════════════════════════════════════════════════════════════ */
  LR.figs.flops = function (mount) {
    LR.header(
      mount,
      "Memory is parameters, speed is connections",
      "One conv layer, 'same'-style padding P = (K−1)/2. Grow the input: the parameter bar does not move, the connection bar climbs, because the shared kernel is re-executed at every output position."
    );

    let Win = 32, K = 3, M = 16, J = 3, S = 1;

    const bar = LR.controls(mount);
    LR.slider(bar, "input size W", 8, 256, 8, Win, function (v) { Win = Math.round(v); draw(); }, (v) => Math.round(v) + "×" + Math.round(v));
    LR.slider(bar, "kernel K", 1, 7, 2, K, function (v) { K = Math.round(v); draw(); }, (v) => Math.round(v) + "×" + Math.round(v));
    LR.slider(bar, "filters M", 4, 64, 4, M, function (v) { M = Math.round(v); draw(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "input depth J", 1, 16, 1, J, function (v) { J = Math.round(v); draw(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "stride S", 1, 2, 1, S, function (v) { S = Math.round(v); draw(); }, (v) => String(Math.round(v)));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 300, {
      aria: "Log-scale bars comparing the conv layer's parameter count, connection count, and a dense layer with the same input and output sizes",
    });
    const ro = LR.readout(mount, [
      { k: "out", label: "output volume" },
      { k: "p", label: "params (memory)" },
      { k: "c", label: "connections" },
      { k: "f", label: "≈ FLOPS (speed)" },
    ]);
    const msg = LR.msg(mount);

    function calc() {
      const P = (K - 1) / 2;
      const Wo = Math.floor((Win - K + 2 * P) / S) + 1;
      const per = K * K * J + 1;
      const params = per * M;
      const conns = Wo * Wo * M * per;
      // dense layer with the same input (W²J) and output (Wo²M) sizes
      const dense = Win * Win * J * Wo * Wo * M + Wo * Wo * M;
      return { P, Wo, per, params, conns, dense };
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const r = calc();
      const rows = [
        { label: "conv params  (K²J+1)·M", v: r.params, color: C.green },
        { label: "conv connections  Ŵ²·M·(K²J+1)", v: r.conns, color: C.orange },
        { label: "dense layer, same in/out (params = connections)", v: r.dense, color: C.purple },
      ];
      const lmax = Math.log10(Math.max.apply(null, rows.map((x) => x.v))) * 1.08;
      const x0 = 40, bw = W - x0 - 150, rowH = 76, y0 = 42;

      ctx.save();
      rows.forEach(function (row, i) {
        const y = y0 + i * rowH;
        ctx.font = "600 12.5px Inter, sans-serif";
        ctx.fillStyle = C.text;
        ctx.textAlign = "left";
        ctx.fillText(row.label, x0, y - 8);
        const frac = Math.max(0.02, Math.log10(Math.max(1.5, row.v)) / lmax);
        ctx.fillStyle = row.color;
        ctx.fillRect(x0, y, frac * bw, 30);
        ctx.font = "700 13px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.text;
        ctx.fillText(row.v.toLocaleString("en-CA"), x0 + frac * bw + 10, y + 21);
      });
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.faint;
      ctx.textAlign = "left";
      ctx.fillText("bar length ∝ log₁₀ of the count", x0, y0 + 3 * rowH - 18);
      ctx.restore();

      const ratio = Math.round(r.conns / r.params);
      ro.set("out", r.Wo + " × " + r.Wo + " × " + M);
      ro.set("p", r.params.toLocaleString("en-CA"), C.green);
      ro.set("c", r.conns.toLocaleString("en-CA"), C.orange);
      ro.set("f", "~" + CNN2.fmtBig(2 * r.conns) + " per image", C.red);
      msg.show(
        "Each of the " + r.params.toLocaleString("en-CA") + " shared weights is re-executed at " +
        r.Wo + "×" + r.Wo + " positions, so the forward pass does " + ratio.toLocaleString("en-CA") +
        "× more work than it has parameters. FLOPS (≈ 2 per connection: one multiply, one add) set the frames per second; parameters only set the download size.",
        "info"
      );
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.3 — layer_report, executed live
     The printed report runs the sizing + counting formulas for
     real. Presets: LeNet-5 C1/C2/C3 (from the lecture's stack).
     ════════════════════════════════════════════════════════════ */
  LR.figs.layerreport = function (mount) {
    LR.header(
      mount,
      "Run layer_report yourself",
      "Pick a preset or set the arguments, then Run: the output-size, weight, and connection arithmetic executes for real. C1 should reproduce the worked LeNet-5 table."
    );

    /* flagged addition: the "AlexNet-scale" preset's 96 11×11 stride-4
       filters are AlexNet-like values chosen by me for scale; the lecture
       gives AlexNet's totals (8 layers, 60M params) but not its per-layer
       configuration. */
    const PRESETS = {
      "LeNet-5 C1": { W: 32, H: 32, J: 1, K: 5, M: 6, P: 0, S: 1 },
      "LeNet-5 C2": { W: 14, H: 14, J: 6, K: 5, M: 16, P: 0, S: 1 },
      "LeNet-5 C3": { W: 5, H: 5, J: 16, K: 5, M: 120, P: 0, S: 1 },
      "AlexNet-scale conv1": { W: 256, H: 256, J: 3, K: 11, M: 96, P: 0, S: 4 },
    };

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);
    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);

    let cfg = Object.assign({}, PRESETS["LeNet-5 C1"]);

    function makeSelect(labelText, options, value, oninput) {
      const lab = LR.el("label", "", labelText + " ");
      const sel = document.createElement("select");
      options.forEach(function (o) {
        const opt = document.createElement("option");
        opt.value = String(o); opt.textContent = String(o);
        sel.appendChild(opt);
      });
      sel.value = String(value);
      sel.style.cssText = "font-family:var(--mono);font-size:13px;padding:2px 6px;border-radius:6px;border:1px solid #cccccc";
      sel.setAttribute("aria-label", labelText);
      sel.addEventListener("change", function () { oninput(sel.value); });
      lab.appendChild(sel);
      controls.appendChild(lab);
      return sel;
    }

    const presetSel = makeSelect("preset", Object.keys(PRESETS), "LeNet-5 C1", function (v) {
      cfg = Object.assign({}, PRESETS[v]);
      syncSelects();
    });
    const sW = makeSelect("W=H", [5, 8, 14, 28, 32, 64, 128, 256], cfg.W, (v) => { cfg.W = cfg.H = +v; });
    const sJ = makeSelect("J", [1, 3, 6, 16, 64], cfg.J, (v) => { cfg.J = +v; });
    const sK = makeSelect("K", [1, 3, 5, 7, 11], cfg.K, (v) => { cfg.K = +v; });
    const sM = makeSelect("M", [6, 16, 32, 96, 120], cfg.M, (v) => { cfg.M = +v; });
    const sP = makeSelect("P", [0, 1, 2], cfg.P, (v) => { cfg.P = +v; });
    const sS = makeSelect("S", [1, 2, 4], cfg.S, (v) => { cfg.S = +v; });
    function syncSelects() {
      sW.value = String(cfg.W); sJ.value = String(cfg.J); sK.value = String(cfg.K);
      sM.value = String(cfg.M); sP.value = String(cfg.P); sS.value = String(cfg.S);
    }
    LR.button(controls, "Run ▸", run, "primary small");

    const CODE = "report = layer_report(W=W, H=H, J=J, K=K, M=M, P=P, S=S)   # the function above";
    const pre = LR.el("pre", "code");
    const codeEl = LR.el("code");
    codeEl.innerHTML = LR.highlight(CODE);
    pre.appendChild(codeEl);
    wrap.appendChild(pre);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; layer_report(W=32, H=32, J=1, K=5, M=6, P=0, S=1)\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    // small log-scale bar pair: params vs connections
    const { cv, ctx, W, H } = LR.canvas(mount, 560, 130, {
      aria: "Two log-scale bars comparing the layer's parameters against its connections",
    });
    cv.style.marginTop = "14px";

    function run() {
      const c = cfg;
      const Wo = Math.floor((c.W - c.K + 2 * c.P) / c.S) + 1;
      const Ho = Math.floor((c.H - c.K + 2 * c.P) / c.S) + 1;
      if (Wo < 1 || Ho < 1) {
        out.innerHTML =
          "&gt;&gt;&gt; layer_report(W=" + c.W + ", H=" + c.H + ", J=" + c.J + ", K=" + c.K +
          ", M=" + c.M + ", P=" + c.P + ", S=" + c.S + ")\n" +
          "ValueError: kernel (K=" + c.K + ") larger than the padded input (" +
          (c.W + 2 * c.P) + "): the filter does not fit even once";
        ctx.clearRect(0, 0, W, H);
        return;
      }
      const per = c.K * c.K * c.J + 1;
      const params = per * c.M;
      const conns = Wo * Ho * c.M * per;

      out.innerHTML =
        "&gt;&gt;&gt; layer_report(W=" + c.W + ", H=" + c.H + ", J=" + c.J + ", K=" + c.K +
        ", M=" + c.M + ", P=" + c.P + ", S=" + c.S + ")\n" +
        "output:      " + Wo + " x " + Ho + " x " + c.M +
        "   # (" + c.W + " − " + c.K + " + 2·" + c.P + ")/" + c.S + " + 1 = " + Wo + "\n" +
        "params:      " + params.toLocaleString("en-CA") +
        "   # (" + c.K + "²·" + c.J + " + 1)·" + c.M + "\n" +
        "connections: " + conns.toLocaleString("en-CA") +
        "  (~" + (2 * conns).toLocaleString("en-CA") + " FLOPS)";

      ctx.clearRect(0, 0, W, H);
      const rows = [
        { label: "params", v: params, color: C.green },
        { label: "connections", v: conns, color: C.orange },
      ];
      const lmax = Math.log10(Math.max(params, conns)) * 1.1;
      const x0 = 120, bw = W - x0 - 116;
      rows.forEach(function (row, i) {
        const y = 22 + i * 48;
        ctx.font = "600 12px Inter, sans-serif";
        ctx.fillStyle = C.text; ctx.textAlign = "right";
        ctx.fillText(row.label, x0 - 10, y + 18);
        const frac = Math.max(0.02, Math.log10(Math.max(1.5, row.v)) / lmax);
        ctx.fillStyle = row.color;
        ctx.fillRect(x0, y, frac * bw, 26);
        ctx.font = "700 12px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.text; ctx.textAlign = "left";
        ctx.fillText(row.v.toLocaleString("en-CA"), x0 + frac * bw + 8, y + 18);
      });
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.fillStyle = C.faint; ctx.textAlign = "left";
      ctx.fillText("log scale", x0, 118);
    }
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.4 — skip connections keep deep stacks alive
     flagged addition: the per-layer pass-through model below
     (each block transmits a fixed fraction g without skips; the
     identity path transmits 1 with skips) is a conceptual
     illustration built by me; the source states only that skip
     connections enabled 100+ layer networks.
     ════════════════════════════════════════════════════════════ */
  LR.figs.skipconn = function (mount) {
    LR.header(
      mount,
      "The identity highway",
      "Each block passes on a fraction g of the signal it receives. Without skips, depth multiplies the fractions; with skips, y = F(x) + x always contains an undiminished copy. Bars show what reaches each depth."
    );

    let L = 12, g = 0.75, skips = true;

    const bar = LR.controls(mount);
    LR.slider(bar, "depth (blocks)", 2, 30, 1, L, function (v) { L = Math.round(v); draw(); }, (v) => String(Math.round(v)));
    LR.slider(bar, "per-block pass-through g", 0.5, 0.95, 0.05, g, function (v) { g = v; draw(); }, (v) => LR.fmtF(v, 2));
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const bOn = LR.button(group, "skips on", function () { skips = true; bOn.classList.add("on"); bOff.classList.remove("on"); draw(); }, "small on");
    const bOff = LR.button(group, "skips off", function () { skips = false; bOff.classList.add("on"); bOn.classList.remove("on"); draw(); }, "small");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 380, {
      aria: "Bar chart of signal strength reaching each block of a deep stack, with and without skip connections, plus a diagram of one residual block",
    });
    const ro = LR.readout(mount, [
      { k: "deep", label: "signal reaching the deepest block" },
      { k: "form", label: "formula" },
    ]);
    const msg = LR.msg(mount);

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // ── residual block diagram, top ──
      const dy = 40;
      ctx.save();
      ctx.font = "700 13px 'JetBrains Mono', Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillStyle = C.text;
      ctx.fillText("x", 90, dy + 5);
      LR.arrow(ctx, 104, dy, 208, dy, C.axis, 1.8);
      ctx.fillStyle = "#f7f7f7";
      ctx.strokeStyle = C.text; ctx.lineWidth = 1.6;
      ctx.fillRect(210, dy - 20, 110, 40); ctx.strokeRect(210, dy - 20, 110, 40);
      ctx.fillStyle = C.text;
      ctx.fillText("F(x)", 265, dy + 5);
      LR.arrow(ctx, 322, dy, 424, dy, C.axis, 1.8);
      // plus node
      ctx.beginPath(); ctx.arc(440, dy, 13, 0, Math.PI * 2);
      ctx.fillStyle = skips ? "rgba(102,102,102,0.14)" : "#f2f2f2"; ctx.fill();
      ctx.strokeStyle = skips ? C.green : "#c0c0c0"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = skips ? C.green : "#b5b5b5";
      ctx.fillText("+", 440, dy + 5);
      LR.arrow(ctx, 455, dy, 540, dy, C.axis, 1.8);
      ctx.fillStyle = C.text;
      ctx.fillText(skips ? "y = F(x) + x" : "y = F(x)", 625, dy + 5);
      // skip arc
      ctx.strokeStyle = skips ? C.green : "#d9d9d9";
      ctx.lineWidth = skips ? 2.4 : 1.6;
      if (!skips) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(150, dy - 2);
      ctx.quadraticCurveTo(295, dy - 58, 436, dy - 14);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = skips ? C.green : "#b5b5b5";
      ctx.fillText(skips ? "skip: identity, ×1" : "skip removed", 295, dy - 44);
      ctx.restore();

      // ── bars: signal reaching block k ──
      const P = {
        x0: 62, y0: 118, w: W - 96, h: H - 118 - 62,
        xmin: 0.5, xmax: L + 0.5, ymin: 0, ymax: 1.05,
        xlabel: "block index k (depth into the network)",
        ylabel: "fraction of signal arriving",
        xticks: (function () {
          const st = Math.max(1, Math.ceil(L / 10));
          const t = [];
          for (let k = 1; k <= L; k += st) t.push(k);
          return t;
        })(),
        yticks: [0, 0.25, 0.5, 0.75, 1],
      };
      const { sx, sy } = LR.plot(ctx, P);
      const bwid = Math.min(22, (P.w / L) * 0.62);

      for (let k = 1; k <= L; k++) {
        const plain = Math.pow(g, k);
        const withSkip = 1; // the identity path delivers the signal undiminished
        const v = skips ? withSkip : plain;
        ctx.fillStyle = skips ? "rgba(102,102,102,0.75)" : "rgba(26,26,26,0.8)";
        ctx.fillRect(sx(k) - bwid / 2, sy(v), bwid, sy(0) - sy(v));
        if (!skips) {
          // ghost of the skip case for contrast
          ctx.strokeStyle = "rgba(102,102,102,0.5)";
          ctx.lineWidth = 1.4;
          ctx.setLineDash([3, 3]);
          ctx.strokeRect(sx(k) - bwid / 2, sy(1), bwid, sy(0) - sy(1));
          ctx.setLineDash([]);
        }
      }

      ctx.save();
      ctx.font = "600 12px Inter, sans-serif";
      ctx.textAlign = "left";
      if (!skips) {
        ctx.fillStyle = C.orange;
        ctx.fillText("no skips: g^k decays geometrically", P.x0 + 10, P.y0 + 18);
        ctx.fillStyle = C.green;
        ctx.fillText("dashed: what skips would preserve", P.x0 + 10, P.y0 + 36);
      } else {
        ctx.fillStyle = C.green;
        ctx.fillText("with skips: the identity path carries the signal at full strength to any depth", P.x0 + 10, P.y0 + 18);
      }
      ctx.restore();

      const deepPlain = Math.pow(g, L);
      ro.set("deep", skips ? "1.00 (identity path)" : LR.fmtF(deepPlain, deepPlain < 0.001 ? 5 : 3), skips ? C.green : C.red);
      ro.set("form", skips ? "y = F(x) + x per block" : "g^" + L + " = " + LR.fmtF(deepPlain, deepPlain < 0.001 ? 5 : 3));
      if (!skips && deepPlain < 0.01) {
        msg.show("At depth " + L + ", each block passing " + LR.fmtF(g, 2) + " leaves " + LR.fmtF(deepPlain, 5) + " of the signal: effectively nothing arrives, and the same multiplication kills the gradient on the way back. This is why plain 100-layer stacks refuse to train.", "bad");
      } else if (skips) {
        msg.show("Every block's output contains an untouched copy of its input, so both the forward signal and the backward gradient have a multiplication-free path through all " + L + " blocks. The blocks only learn residual corrections. That is ResNet's whole trick.", "good");
      } else {
        msg.show("Shallow stacks survive without skips (that is why LeNet and AlexNet never needed them). Push the depth slider toward 30 and watch the bars vanish.", "info");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 5.1 — the transfer-learning workbench (signature figure)
     Freeze/unfreeze a pre-trained backbone stage by stage; sliders
     for data amount and similarity drive the lecture's rules.
     flagged addition: the per-stage parameter counts are
     ResNet-scale illustrative values chosen by me; the source
     gives the rules, not a specific backbone's numbers.
     ════════════════════════════════════════════════════════════ */
  LR.figs.transfer = function (mount) {
    LR.header(
      mount,
      "Freeze, replace, fine-tune",
      "A backbone pre-trained on ImageNet (five stages) plus a new head for your task. Click stages to freeze 🔒 or train 🔥 them; the sliders drive the lecture's recommendation."
    );

    const STAGES = [
      { name: "stage 1", what: "edges, colours", params: 0.2e6 },
      { name: "stage 2", what: "textures", params: 0.7e6 },
      { name: "stage 3", what: "motifs, patterns", params: 2.1e6 },
      { name: "stage 4", what: "parts", params: 4.7e6 },
      { name: "stage 5", what: "objects", params: 3.9e6 },
    ];
    const HEAD = { name: "new head", params: 0.05e6 };
    const TOTAL = STAGES.reduce((s, x) => s + x.params, 0) + HEAD.params;

    let frozen = [true, true, true, true, false];  // stage i frozen?
    let logN = 3.3;      // log10 of new examples
    let sim = 60;        // similarity to ImageNet, %

    const bar = LR.controls(mount);
    LR.slider(bar, "new labelled examples", 2, 6, 0.1, logN, function (v) { logN = v; sync(); }, (v) => CNN2.fmtBig(Math.round(Math.pow(10, v))));
    LR.slider(bar, "similarity to ImageNet (%)", 0, 100, 5, sim, function (v) { sim = Math.round(v); sync(); }, (v) => Math.round(v) + "%");
    LR.button(bar, "Apply recommendation", function () {
      const rec = recommend();
      frozen = STAGES.map((_, i) => i < rec.freeze);
      sync();
    }, "primary");

    // ---- pipeline ----
    const pipe = LR.el("div", "tl-pipeline");
    mount.appendChild(pipe);
    const stageEls = STAGES.map(function (s, i) {
      const b = LR.el("button", "tl-stage");
      b.type = "button";
      b.setAttribute("aria-label", s.name + " (" + s.what + "), click to toggle frozen or trainable");
      b.addEventListener("click", function () { frozen[i] = !frozen[i]; sync(); });
      pipe.appendChild(b);
      if (i < STAGES.length - 1 || true) pipe.appendChild(LR.el("span", "tl-arrow", "→"));
      return b;
    });
    const headEl = LR.el("button", "tl-stage tl-head");
    headEl.type = "button";
    headEl.setAttribute("aria-label", "new classification head, always trainable");
    headEl.addEventListener("click", function () {
      note.show("The head is freshly initialized for your classes (the pre-trained one answered ImageNet's 1,000), so it always trains.", "info");
    });
    pipe.appendChild(headEl);

    const recBox = LR.el("div", "tl-rec");
    mount.appendChild(recBox);

    const ro = LR.readout(mount, [
      { k: "trainable", label: "trainable params" },
      { k: "lr", label: "learning rate" },
    ]);
    const note = LR.msg(mount);

    function recommend() {
      // the lecture's rules, made mechanical:
      // fewer examples → freeze more; more dissimilar → fine-tune (unfreeze) more;
      // always a smaller learning rate than from-scratch training.
      let freeze = logN < 3 ? 5 : logN < 4 ? 4 : logN < 5 ? 3 : 1;   // stages to freeze, by data
      if (sim < 35) freeze -= 1;         // dissimilar → fine-tune deeper
      if (sim > 75) freeze += 1;         // very similar → freeze more
      freeze = Math.max(0, Math.min(5, freeze));
      // smaller LR the less we unfreeze (and always < from-scratch ~0.1)
      const lr = freeze >= 4 ? 0.001 : freeze >= 2 ? 0.003 : 0.01;
      return { freeze, lr };
    }

    function sync() {
      const rec = recommend();
      stageEls.forEach(function (el, i) {
        el.className = "tl-stage " + (frozen[i] ? "frozen" : "hot");
        el.innerHTML =
          "<span class='tl-ico'>" + (frozen[i] ? "🔒" : "🔥") + "</span>" +
          "<b>" + STAGES[i].name + "</b>" +
          "<span class='tl-what'>" + STAGES[i].what + "</span>" +
          "<span class='tl-p'>" + CNN2.fmtBig(STAGES[i].params) + "</span>";
      });
      headEl.className = "tl-stage tl-head hot";
      headEl.innerHTML =
        "<span class='tl-ico'>🔥</span><b>new head</b>" +
        "<span class='tl-what'>your classes, re-initialized</span>" +
        "<span class='tl-p'>" + CNN2.fmtBig(HEAD.params) + "</span>";

      const trainable = STAGES.reduce((s, x, i) => s + (frozen[i] ? 0 : x.params), 0) + HEAD.params;
      const N = Math.round(Math.pow(10, logN));
      ro.set("trainable", CNN2.fmtBig(trainable) + " of " + CNN2.fmtBig(TOTAL) + " (" + Math.round((100 * trainable) / TOTAL) + "%)", C.orange);
      ro.set("lr", "~" + rec.lr + "  (from-scratch would use ~0.1)", C.green);

      const nFrozen = frozen.filter(Boolean).length;
      recBox.innerHTML =
        "<b>Recommendation for " + CNN2.fmtBig(N) + " examples at " + sim + "% similarity:</b> freeze " +
        rec.freeze + " of 5 stages, fine-tune the rest plus the head, learning rate ~" + rec.lr +
        " <span style='color:var(--faint)'>(rules: fewer examples → freeze more; more dissimilar → fine-tune more; always a smaller rate than from-scratch, because the weights already sit near an optimum)</span>." +
        (nFrozen === rec.freeze
          ? " <b style='color:var(--green)'>Your setting matches.</b>"
          : " You currently freeze " + nFrozen + ".");

      // evaluate the *manual* setting against the data budget
      if (trainable > 40 * N) {
        note.show("Warning: " + CNN2.fmtBig(trainable) + " trainable parameters against " + CNN2.fmtBig(N) + " examples. That ratio is a memorization recipe; freeze more stages or find more data.", "bad");
      } else if (nFrozen === 5 && sim < 35) {
        note.show("Everything frozen but the data is very dissimilar: the backbone's later, photo-specific features will not fit it. Unfreeze the later stages so they can adapt.", "info");
      } else if (nFrozen < rec.freeze) {
        note.show("You are fine-tuning more than the data supports (" + CNN2.fmtBig(N) + " examples). It can work with strong augmentation, but the safer default is freezing " + rec.freeze + " stages.", "info");
      } else {
        note.show("Reasonable setup: the frozen stages act as a fixed feature extractor learned from ImageNet, and your " + CNN2.fmtBig(N) + " examples only have to train " + CNN2.fmtBig(trainable) + " parameters.", "good");
      }
    }
    sync();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.1 — what filters learn, layer by layer
     Procedurally generated swatches (no real network screenshots,
     no dataset images): edges/colours → textures → parts/objects.
     ════════════════════════════════════════════════════════════ */
  LR.figs.filters = function (mount) {
    LR.header(
      mount,
      "Representation learning, made visible",
      "Synthetic sketches of what filter visualizations look like at each depth. Click a swatch: the panel shows the kind of image patch that would excite it most."
    );

    const SW = 64; // swatch pixel size

    function grating(g, angle, freq, c1, c2) {
      for (let y = 0; y < SW; y++)
        for (let x = 0; x < SW; x++) {
          const t = (x * Math.cos(angle) + y * Math.sin(angle)) * freq;
          const v = 0.5 + 0.5 * Math.sin(t);
          g.fillStyle = mix(c1, c2, v);
          g.fillRect(x, y, 1, 1);
        }
    }
    function mix(a, b, t) {
      const pa = hex(a), pb = hex(b);
      return "rgb(" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",") + ")";
    }
    function hex(h) {
      return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    function checker(g, size, c1, c2) {
      for (let y = 0; y < SW; y += size)
        for (let x = 0; x < SW; x += size) {
          g.fillStyle = ((x / size + y / size) % 2 === 0) ? c1 : c2;
          g.fillRect(x, y, size, size);
        }
    }
    function rings(g, c1, c2) {
      for (let y = 0; y < SW; y++)
        for (let x = 0; x < SW; x++) {
          const r = Math.hypot(x - SW / 2, y - SW / 2);
          g.fillStyle = mix(c1, c2, 0.5 + 0.5 * Math.sin(r * 0.55));
          g.fillRect(x, y, 1, 1);
        }
    }
    function dots(g, c1, c2) {
      g.fillStyle = c1; g.fillRect(0, 0, SW, SW);
      g.fillStyle = c2;
      for (let y = 8; y < SW; y += 16)
        for (let x = 8; x < SW; x += 16) {
          g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2); g.fill();
        }
    }
    function eye(g) {
      g.fillStyle = "#dddddd"; g.fillRect(0, 0, SW, SW);
      g.fillStyle = "#ffffff";
      g.beginPath(); g.ellipse(SW / 2, SW / 2, 24, 13, 0, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "#3c3c3c"; g.lineWidth = 2; g.stroke();
      g.fillStyle = "#494949";
      g.beginPath(); g.arc(SW / 2, SW / 2, 8, 0, Math.PI * 2); g.fill();
      g.fillStyle = "#111111";
      g.beginPath(); g.arc(SW / 2, SW / 2, 3.5, 0, Math.PI * 2); g.fill();
    }
    function wheel(g) {
      g.fillStyle = "#d8d8d8"; g.fillRect(0, 0, SW, SW);
      g.strokeStyle = "#222222"; g.lineWidth = 5;
      g.beginPath(); g.arc(SW / 2, SW / 2, 22, 0, Math.PI * 2); g.stroke();
      g.lineWidth = 2.5;
      for (let a = 0; a < 6; a++) {
        g.beginPath();
        g.moveTo(SW / 2, SW / 2);
        g.lineTo(SW / 2 + 20 * Math.cos((a * Math.PI) / 3), SW / 2 + 20 * Math.sin((a * Math.PI) / 3));
        g.stroke();
      }
      g.fillStyle = "#222222";
      g.beginPath(); g.arc(SW / 2, SW / 2, 4, 0, Math.PI * 2); g.fill();
    }
    function face(g) {
      g.fillStyle = "#d5d5d5"; g.fillRect(0, 0, SW, SW);
      g.fillStyle = "#acacac";
      g.beginPath(); g.ellipse(SW / 2, SW / 2 + 2, 20, 25, 0, 0, Math.PI * 2); g.fill();
      g.fillStyle = "#222222";
      g.beginPath(); g.arc(SW / 2 - 8, SW / 2 - 6, 3, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.arc(SW / 2 + 8, SW / 2 - 6, 3, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "#222222"; g.lineWidth = 2;
      g.beginPath(); g.arc(SW / 2, SW / 2 + 8, 8, 0.25 * Math.PI, 0.75 * Math.PI); g.stroke();
    }
    function feather(g) {
      g.fillStyle = "#e5e5e5"; g.fillRect(0, 0, SW, SW);
      g.strokeStyle = "#5d5d5d"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(10, 54); g.quadraticCurveTo(SW / 2, SW / 2, 54, 10); g.stroke();
      g.lineWidth = 1.4;
      for (let t = 0.12; t < 0.95; t += 0.09) {
        const x = 10 + t * 44, y = 54 - t * 44;
        g.beginPath(); g.moveTo(x, y); g.lineTo(x - 7, y - 8); g.stroke();
        g.beginPath(); g.moveTo(x, y); g.lineTo(x + 8, y + 7); g.stroke();
      }
    }

    const ROWS = [
      {
        label: "early layers · edges, colours, gradients (small receptive fields)",
        items: [
          { name: "vertical edge", draw: (g) => grating(g, 0, 0.6, "#111111", "#f2f2f2"), excite: "any strong vertical boundary: a doorframe, a table leg, the side of a building" },
          { name: "45° edge", draw: (g) => grating(g, Math.PI / 4, 0.6, "#111111", "#f2f2f2"), excite: "diagonal boundaries: rooflines, shadows, a cat's ear outline" },
          { name: "horizontal edge", draw: (g) => grating(g, Math.PI / 2, 0.6, "#111111", "#f2f2f2"), excite: "horizons, shelf edges, closed eyelids" },
          { name: "fine grating", draw: (g) => grating(g, 0.35, 1.5, "#222222", "#e8e8e8"), excite: "high-frequency stripes: fabric weave, blinds, fur" },
          { name: "colour contrast", draw: (g) => grating(g, Math.PI / 4, 0.35, "#1a1a1a", "#666666"), excite: "opponent-colour boundaries: red fruit against leaves" },
          { name: "colour gradient", draw: (g) => grating(g, Math.PI / 2, 0.18, "#969696", "#adadad"), excite: "smooth colour transitions: sky at sunset, curved shaded surfaces" },
        ],
      },
      {
        label: "middle layers · textures and motifs (larger receptive fields)",
        items: [
          { name: "checker texture", draw: (g) => checker(g, 8, "#3a3a3a", "#d9d9d9"), excite: "regular grids: windows on a building, tiles, waffle patterns" },
          { name: "dot texture", draw: (g) => dots(g, "#e1e1e1", "#606060"), excite: "repeated blobs: leopard spots, berries, gravel" },
          { name: "concentric rings", draw: (g) => rings(g, "#3c3c3c", "#dddddd"), excite: "circular structure: eyes, wheels, cups seen from above" },
          { name: "cross-hatch", draw: (g) => { grating(g, 0.3, 0.9, "#333333", "#eeeeee"); g.globalAlpha = 0.5; grating(g, 0.3 + Math.PI / 2, 0.9, "#333333", "#eeeeee"); g.globalAlpha = 1; }, excite: "woven textures: baskets, fences, mesh" },
        ],
      },
      {
        label: "late layers · parts and objects (receptive fields span the image)",
        items: [
          { name: "eye-like", draw: eye, excite: "an eye, almost regardless of the animal it belongs to" },
          { name: "wheel-like", draw: wheel, excite: "wheels, gears, round instruments: circular things with radial structure" },
          { name: "face-like", draw: face, excite: "frontal faces: two dark blobs over a curve, at the right spacing" },
          { name: "feather / leaf", draw: feather, excite: "branching structure: feathers, ferns, leaf veins" },
        ],
      },
    ];

    const gal = LR.el("div", "filter-gallery");
    mount.appendChild(gal);
    const panel = LR.el("div", "filter-panel");
    let selBtn = null;

    ROWS.forEach(function (row, ri) {
      gal.appendChild(LR.el("div", "fg-label", row.label));
      const strip = LR.el("div", "fg-row");
      gal.appendChild(strip);
      row.items.forEach(function (it) {
        const b = LR.el("button", "fg-swatch");
        b.type = "button";
        b.setAttribute("aria-label", "filter sketch: " + it.name + ", click for what excites it");
        const cvS = document.createElement("canvas");
        cvS.width = SW; cvS.height = SW;
        it.draw(cvS.getContext("2d"));
        b.appendChild(cvS);
        b.appendChild(LR.el("span", "fg-name", it.name));
        b.addEventListener("click", function () {
          if (selBtn) selBtn.classList.remove("sel");
          selBtn = b; b.classList.add("sel");
          panel.innerHTML =
            "<b>" + it.name + "</b> (" + ["early", "middle", "late"][ri] + " layer), most excited by: " + it.excite + "." +
            (ri === 0
              ? " Filters like this are near-universal across tasks, which is why section 5 freezes them first."
              : ri === 2
              ? " Features this specific are married to the training task: the first to re-train when your data is dissimilar."
              : " Textures bridge the two: still fairly generic, but starting to specialize.");
          panel.classList.add("show");
        });
        strip.appendChild(b);
      });
    });
    mount.appendChild(panel);
    panel.innerHTML = "Click any swatch. Notice the progression: generic geometry first, task-specific structure last, exactly the receptive-field story from Part I.";
    panel.classList.add("show");
  };

  /* ════════════════════════════════════════════════════════════
     Fig 6.2 — the adversarial slider
     A real (hand-built) linear classifier over a synthetic image;
     the perturbation is aimed along the classifier's weak
     direction (toward "vulture"), FGSM-style: x' = x + ε·sign(d).
     flagged addition: the toy image, the three class templates,
     and the softmax temperature are constructions of mine; the
     lecture supplies the phenomenon and the panda/vulture example.
     ════════════════════════════════════════════════════════════ */
  LR.figs.adversarial = function (mount) {
    LR.header(
      mount,
      "An invisible nudge, a confident mistake",
      "The classifier below is real: three weight templates, scores computed live. Slide ε and watch the label flip while the image barely changes. The middle panel magnifies the perturbation 10×."
    );

    const N = 14;
    // synthetic "panda" stand-in: pale face, dark ears, eye patches, nose
    function makePanda() {
      const m = Array.from({ length: N }, () => new Array(N).fill(0.15));
      const put = (i, j, v) => { if (i >= 0 && i < N && j >= 0 && j < N) m[i][j] = v; };
      // face disc
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++)
          if (Math.hypot(i - 7.5, j - 6.5) < 5.6) m[i][j] = 0.88;
      // ears
      [[2, 2], [2, 3], [3, 2], [2, 10], [2, 11], [3, 11]].forEach((p) => put(p[0], p[1], 0.05));
      // eye patches
      [[6, 4], [6, 5], [7, 4], [7, 5]].forEach((p) => put(p[0], p[1], 0.1));
      [[6, 8], [6, 9], [7, 8], [7, 9]].forEach((p) => put(p[0], p[1], 0.1));
      // nose + mouth
      [[9, 6], [9, 7], [10, 6], [10, 7]].forEach((p) => put(p[0], p[1], 0.08));
      return m;
    }
    const X0 = makePanda();

    // class templates (weights): panda = matched filter for the clean image;
    // vulture = diagonal wing band; teapot = round body + spout blob.
    // Templates are centred and unit-normalized, and each class carries a
    // calibrated bias so the clean-image margins are realistic-small (a
    // matched filter with no bias would win by a landslide no invisible
    // perturbation could close). Verified: clean → panda ~87%, the label
    // flips to vulture near ε ≈ 0.05, and ε = 0.12 gives vulture ~95%.
    function centre(m) {
      let mean = 0;
      m.forEach((r) => r.forEach((v) => (mean += v)));
      mean /= N * N;
      return m.map((r) => r.map((v) => v - mean));
    }
    function normalize(m) {
      let ss = 0;
      m.forEach((r) => r.forEach((v) => (ss += v * v)));
      const s = Math.sqrt(ss) || 1;
      return m.map((r) => r.map((v) => v / s));
    }
    const Wpanda = normalize(centre(X0));
    const Wvult = normalize(centre(Array.from({ length: N }, (_, i) =>
      Array.from({ length: N }, (_, j) => (Math.abs(i + j - 13) < 2 ? 0.9 : 0.1) + (i < 3 && j > 9 ? 0.5 : 0))
    )));
    const Wtea = normalize(centre(Array.from({ length: N }, (_, i) =>
      Array.from({ length: N }, (_, j) => (Math.hypot(i - 9, j - 7) < 4 ? 0.9 : 0.1) + (i > 5 && i < 8 && j > 10 ? 0.6 : 0))
    )));
    function dot2(w, x) {
      let s = 0;
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) s += w[i][j] * x[i][j];
      return s;
    }
    // biases: clean logit margins of 0.7 (vulture) and 1.2 (teapot)
    const BIAS = [
      0,
      dot2(Wpanda, X0) - dot2(Wvult, X0) - 0.7,
      dot2(Wpanda, X0) - dot2(Wtea, X0) - 1.2,
    ];
    const CLASSES = [
      { name: "panda", w: Wpanda, b: BIAS[0], color: C.green },
      { name: "vulture", w: Wvult, b: BIAS[1], color: C.red },
      { name: "teapot", w: Wtea, b: BIAS[2], color: C.purple },
    ];

    // adversarial direction: toward vulture, away from panda (FGSM-style)
    const DIR = [];
    for (let i = 0; i < N; i++) {
      DIR.push([]);
      for (let j = 0; j < N; j++) DIR[i].push(Math.sign(Wvult[i][j] - Wpanda[i][j]) || 0);
    }

    let eps = 0;

    const bar = LR.controls(mount);
    LR.slider(bar, "perturbation strength ε", 0, 0.12, 0.002, eps, function (v) { eps = v; draw(); }, (v) => LR.fmtF(v, 3));
    LR.button(bar, "clean image", function () { eps = 0; sEl.set(0); draw(); }, "small");
    const sEl = { set: function () {} }; // replaced below
    // grab the slider's setter (last slider added)
    const sliders = bar.querySelectorAll("input[type=range]");
    const epsInput = sliders[sliders.length - 1];
    sEl.set = function (v) { epsInput.value = v; epsInput.dispatchEvent(new Event("input")); };

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 340, {
      aria: "Adversarial example demo: clean synthetic image, magnified perturbation, perturbed image, and live class scores that flip from panda to vulture as epsilon grows",
    });
    const ro = LR.readout(mount, [
      { k: "pred", label: "prediction" },
      { k: "linf", label: "max pixel change" },
    ]);
    const msg = LR.msg(mount);

    function perturbed() {
      return X0.map((row, i) => row.map((v, j) => Math.max(0, Math.min(1, v + eps * DIR[i][j]))));
    }
    function scores(x) {
      return CLASSES.map((c) => dot2(c.w, x) + c.b);
    }
    function softmax(s) {
      const t = 3; // temperature: readable confidences
      const m = Math.max.apply(null, s);
      const e = s.map((v) => Math.exp((v - m) * t));
      const Z = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / Z);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const Xp = perturbed();
      const sc = scores(Xp);
      const pr = softmax(sc);
      const top = pr.indexOf(Math.max.apply(null, pr));

      const cell = 15;
      // clean
      CNN2.label(ctx, 26, 40, "clean image x");
      CNN2.grid(ctx, 26, 50, cell, X0, { stroke: "#e8e8e8" });
      ctx.save();
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "left";
      ctx.fillText("“panda”", 26, 50 + N * cell + 18);
      ctx.restore();

      // + perturbation (magnified ×10)
      ctx.save();
      ctx.font = "700 16px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "center";
      ctx.fillText("+", 262, 50 + (N * cell) / 2);
      ctx.restore();
      CNN2.label(ctx, 288, 40, "ε · direction (shown 10×)");
      const pmag = X0.map((row, i) => row.map((_, j) => eps * DIR[i][j] * 10));
      CNN2.grid(ctx, 288, 50, cell, pmag, {
        fill: (v) => CNN2.ramp(v, 1),
        stroke: "#e8e8e8",
      });
      ctx.save();
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.muted; ctx.textAlign = "left";
      ctx.fillText("true scale: ±" + LR.fmtF(eps, 3) + " per pixel", 288, 50 + N * cell + 18);
      ctx.restore();

      // = perturbed
      ctx.save();
      ctx.font = "700 16px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "center";
      ctx.fillText("=", 524, 50 + (N * cell) / 2);
      ctx.restore();
      CNN2.label(ctx, 548, 40, "perturbed image x′");
      CNN2.grid(ctx, 548, 50, cell, Xp, { stroke: "#e8e8e8" });
      ctx.save();
      ctx.font = "700 12px Inter, sans-serif";
      ctx.fillStyle = CLASSES[top].color; ctx.textAlign = "left";
      ctx.fillText("“" + CLASSES[top].name + "”", 548, 50 + N * cell + 18);
      ctx.restore();

      // score bars
      const bx = 26, by = 300, bwid = 740;
      CNN2.label(ctx, bx, by - 8, "class probabilities (live: softmax of wᵀx′ per class)");
      let xacc = bx;
      CLASSES.forEach(function (c, i) {
        const w2 = Math.max(2, pr[i] * bwid);
        ctx.fillStyle = c.color;
        ctx.globalAlpha = i === top ? 0.9 : 0.35;
        ctx.fillRect(xacc, by, w2, 22);
        ctx.globalAlpha = 1;
        if (pr[i] > 0.06) {
          ctx.save();
          ctx.font = "700 11.5px Inter, sans-serif";
          ctx.fillStyle = "#ffffff";
          ctx.textAlign = "left";
          ctx.fillText(c.name + " " + Math.round(pr[i] * 100) + "%", xacc + 8, by + 15.5);
          ctx.restore();
        }
        xacc += w2;
      });

      ro.set("pred", CLASSES[top].name + " (" + Math.round(pr[top] * 100) + "%)", CLASSES[top].color);
      ro.set("linf", LR.fmtF(eps, 3) + " on a 0–1 scale", eps > 0 ? C.amber : C.green);

      if (eps === 0) {
        msg.show("Clean image: the classifier says panda, correctly and confidently. Now add the smallest ε you can.", "good");
      } else if (top === 0) {
        msg.show("Still panda, but look at the bars: the vulture score is climbing with every step of ε, because the perturbation points exactly along the weight difference w_vulture − w_panda.", "info");
      } else {
        msg.show("Flipped: the model now calls this a " + CLASSES[top].name + " at " + Math.round(pr[top] * 100) + "% confidence, from a change of at most " + LR.fmtF(eps, 3) + " per pixel that you would never notice at true scale. Accuracy on clean data never promised robustness to chosen perturbations.", "bad");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 7.1 — classification vs semantic segmentation
     One synthetic scene, two outputs: a single label, and a
     per-pixel class mask (the mask is computed from the same
     analytic scene description that draws the image).
     ════════════════════════════════════════════════════════════ */
  LR.figs.segmentation = function (mount) {
    LR.header(
      mount,
      "One label, or a label for every pixel",
      "The same synthetic scene answered two ways. Move your pointer over either pane: classification gives the same answer everywhere, segmentation answers per pixel."
    );

    const KLASSES = [
      { name: "sky", photo: "#d9d9d9", mask: "#d3d3d3" },
      { name: "sun", photo: "#c3c3c3", mask: "#adadad" },
      { name: "tree", photo: "#6c6c6c", mask: "#666666" },
      { name: "house", photo: "#808080", mask: "#1a1a1a" },
      { name: "road", photo: "#8a8a8a", mask: "#969696" },
      { name: "grass", photo: "#adadad", mask: "#cfcfcf" },
    ];

    // analytic scene: class index at normalized coords (u, v) in [0,1]
    function classAt(u, v) {
      // sun
      if (Math.hypot((u - 0.82) * 1.3, v - 0.16) < 0.09) return 1;
      // house: body + roof
      if (u > 0.14 && u < 0.44 && v > 0.42 && v < 0.72) return 3;
      if (v > 0.28 && v < 0.42 && Math.abs(u - 0.29) < (v - 0.28) * 1.18) return 3;
      // tree: trunk + crown
      if (u > 0.62 && u < 0.66 && v > 0.5 && v < 0.72) return 2;
      if (Math.hypot((u - 0.64) * 1.1, (v - 0.42) * 1.5) < 0.14) return 2;
      // road: band across the bottom
      if (v > 0.8) return 4;
      // grass below the horizon, sky above
      if (v > 0.72) return 5;
      return 0;
    }

    let hover = null; // {pane: 0|1, u, v}

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 350, {
      aria: "A synthetic scene shown twice: once with a single classification label, once as a per-pixel segmentation mask with a legend of class pixel counts",
    });

    const legend = LR.el("div", "seg-legend");
    mount.appendChild(legend);
    const ro = LR.readout(mount, [
      { k: "cls", label: "classification says" },
      { k: "seg", label: "segmentation at cursor" },
    ]);

    const PW = 360, PH = 240, LX = 30, RX = 430, PY = 56;

    cv.addEventListener("pointermove", function (e) {
      const p = LR.evtXY(cv, W, H, e);
      hover = null;
      if (p.y >= PY && p.y < PY + PH) {
        if (p.x >= LX && p.x < LX + PW) hover = { pane: 0, u: (p.x - LX) / PW, v: (p.y - PY) / PH };
        else if (p.x >= RX && p.x < RX + PW) hover = { pane: 1, u: (p.x - RX) / PW, v: (p.y - PY) / PH };
      }
      draw();
    });
    cv.addEventListener("pointerleave", function () { hover = null; draw(); });

    // pixel counts for the legend, computed once on a sampling grid
    const counts = new Array(KLASSES.length).fill(0);
    const GN = 120;
    for (let a = 0; a < GN; a++)
      for (let b = 0; b < GN; b++) counts[classAt((b + 0.5) / GN, (a + 0.5) / GN)] += 1;
    const totalPx = GN * GN;

    function drawScene(x0, asMask) {
      const step = 4; // draw in 4-px blocks (fast, and honest: same classAt)
      for (let y = 0; y < PH; y += step)
        for (let x = 0; x < PW; x += step) {
          const k = classAt((x + step / 2) / PW, (y + step / 2) / PH);
          ctx.fillStyle = asMask ? KLASSES[k].mask : KLASSES[k].photo;
          ctx.fillRect(x0 + x, PY + y, step, step);
        }
      // photo-only decoration: window + door so it reads as a scene
      if (!asMask) {
        ctx.fillStyle = "#525252";
        ctx.fillRect(x0 + 0.3 * PW, PY + 0.56 * PH, 0.05 * PW, 0.16 * PH); // door
        ctx.fillStyle = "#e0e0e0";
        ctx.fillRect(x0 + 0.19 * PW, PY + 0.5 * PH, 0.06 * PW, 0.07 * PH); // window
      }
      ctx.strokeStyle = "#c9c9c9";
      ctx.lineWidth = 1.4;
      ctx.strokeRect(x0 - 0.5, PY - 0.5, PW + 1, PH + 1);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      CNN2.label(ctx, LX, PY - 24, "classification: what is in the image?");
      CNN2.label(ctx, RX, PY - 24, "semantic segmentation: a class for every pixel");
      drawScene(LX, false);
      drawScene(RX, true);

      // classification: one label chip over the left pane
      ctx.save();
      ctx.font = "700 14px Inter, sans-serif";
      const chip = "“house” (one label for all " + totalPx.toLocaleString("en-CA") + " sampled pixels)";
      const tw = ctx.measureText(chip).width;
      ctx.fillStyle = "rgba(17,17,17,0.85)";
      ctx.fillRect(LX + 10, PY + 10, tw + 20, 28);
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "left";
      ctx.fillText(chip, LX + 20, PY + 29);
      ctx.restore();

      // hover crosshair on both panes at the same scene position
      if (hover) {
        const k = classAt(hover.u, hover.v);
        [LX, RX].forEach(function (x0) {
          const hx = x0 + hover.u * PW, hy = PY + hover.v * PH;
          ctx.strokeStyle = C.text; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.arc(hx, hy, 7, 0, Math.PI * 2); ctx.stroke();
        });
        ro.set("cls", "“house”, everywhere", C.text);
        ro.set("seg", KLASSES[k].name, KLASSES[k].mask);
      } else {
        ro.set("cls", "“house”, one label total");
        ro.set("seg", "hover a pane to query a pixel");
      }

      // legend with live pixel counts
      legend.innerHTML = KLASSES.map(function (k2, i) {
        return "<span class='seg-key'><span class='seg-swatch' style='background:" + k2.mask + "'></span>" +
          k2.name + " <b>" + Math.round((100 * counts[i]) / totalPx) + "%</b></span>";
      }).join("");
    }
    draw();
  };
})();
