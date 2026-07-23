/* ══════════════════════════════════════════════════════════════
   figures-a.js — CNNs in Practice (Lecture 12), sections 1–4.1
   Fig 1.1 onebyone · Fig 2.1 datasets · Fig 3.1 scaling
   Fig 4.1 timeline (signature)
   Defines LR.cnn2: this lesson's copy of the Part I convolution /
   sizing core (each lesson carries its own copy) plus the shared
   lecture-12 data tables. All numbers computed live.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── LR.cnn2: convolution core copied from the Part I lesson ── */
  const CNN2 = (LR.cnn2 = {
    outSize: function (W, K, P, S) {
      return (W - K + 2 * P) / S + 1;
    },

    pad: function (img, P) {
      if (!P) return img;
      const W = img[0].length;
      const blank = new Array(W + 2 * P).fill(0);
      const out = [];
      for (let i = 0; i < P; i++) out.push(blank.slice());
      img.forEach(function (row) {
        out.push(new Array(P).fill(0).concat(row, new Array(P).fill(0)));
      });
      for (let i = 0; i < P; i++) out.push(blank.slice());
      return out;
    },

    // 2-D convolution as CNNs compute it (sliding dot product)
    conv2d: function (img, ker, S, P) {
      S = S || 1;
      const im = CNN2.pad(img, P || 0);
      const K = ker.length;
      const Ho = Math.floor((im.length - K) / S) + 1;
      const Wo = Math.floor((im[0].length - K) / S) + 1;
      const out = [];
      for (let i = 0; i < Ho; i++) {
        const row = [];
        for (let j = 0; j < Wo; j++) {
          let s = 0;
          for (let a = 0; a < K; a++)
            for (let b = 0; b < K; b++) s += im[i * S + a][j * S + b] * ker[a][b];
          row.push(s);
        }
        out.push(row);
      }
      return out;
    },

    relu: function (m) {
      return m.map((row) => row.map((v) => Math.max(0, v)));
    },

    maxAbs: function (m) {
      let mx = 0;
      m.forEach((row) => row.forEach((v) => { if (Math.abs(v) > mx) mx = Math.abs(v); }));
      return mx;
    },

    // Part I's shared test image: bright block on dark background
    IMG8: [
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ],

    // Part I's preset kernels (standard textbook examples)
    KER: {
      vedge: { name: "Vertical edge", k: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
      hedge: { name: "Horizontal edge", k: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
      blur: { name: "Blur", k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] },
      sharpen: { name: "Sharpen", k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    },

    gray: function (v) {
      const s = Math.round(245 - Math.max(0, Math.min(1, v)) * 215);
      return "rgb(" + s + "," + s + "," + s + ")";
    },
    ramp: function (v, vmax) {
      if (!vmax) return "#ffffff";
      const a = Math.min(1, Math.abs(v) / vmax) * 0.85;
      return v >= 0 ? "rgba(26,26,26," + a.toFixed(3) + ")" : "rgba(150,150,150," + a.toFixed(3) + ")";
    },

    fmtV: function (v) {
      if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
      return (Math.round(v * 10) / 10).toFixed(1);
    },
    fmtK: function (v) {
      if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
      return LR.fmtF(v, 2).replace("0.", ".");
    },

    // human-friendly big numbers: 9,760 · 60K · 1.2M · 50 GB etc.
    fmtBig: function (n) {
      if (n >= 1e9) return LR.fmt(n / 1e9, n >= 1e10 ? 0 : 1) + "B";
      if (n >= 1e6) return LR.fmt(n / 1e6, n >= 1e7 ? 0 : 1) + "M";
      if (n >= 1e4) return LR.fmt(n / 1e3, 0) + "K";
      return n.toLocaleString("en-CA");
    },

    grid: function (ctx, x0, y0, cell, g, o) {
      o = o || {};
      ctx.save();
      ctx.font = o.font || "600 " + Math.max(9, Math.min(13, Math.round(cell * 0.4))) + "px 'JetBrains Mono', Menlo, monospace";
      ctx.textAlign = "center";
      for (let r = 0; r < g.length; r++) {
        for (let c = 0; c < g[r].length; c++) {
          const v = g[r][c];
          const x = x0 + c * cell, y = y0 + r * cell;
          ctx.fillStyle = o.fill ? o.fill(v, r, c) : CNN2.gray(v);
          ctx.fillRect(x, y, cell - 1, cell - 1);
          ctx.strokeStyle = o.stroke || "#d8d8d8";
          ctx.lineWidth = 0.8;
          ctx.strokeRect(x, y, cell - 1, cell - 1);
          if (o.num) {
            const t = o.num(v, r, c);
            if (t !== null && t !== undefined) {
              ctx.fillStyle = o.numColor ? o.numColor(v, r, c) : C.text;
              ctx.fillText(t, x + cell / 2, y + cell / 2 + 4);
            }
          }
        }
      }
      ctx.restore();
    },

    label: function (ctx, x, y, text, color) {
      ctx.save();
      ctx.font = "600 12px Inter, -apple-system, sans-serif";
      ctx.fillStyle = color || C.muted;
      ctx.textAlign = "left";
      ctx.fillText(text, x, y);
      ctx.restore();
    },

    /* ── lecture-12 ground-truth tables (from the source slides) ── */
    DATASETS: [
      {
        key: "mnist", name: "MNIST", sub: "handwritten digits (scanned zip codes)",
        classes: 10, train: 60000, test: 10000,
        res: "28 × 28 × 1", pixels: 28 * 28 * 1, sizeMB: 60,
        note: "digits centred and size-normalized",
      },
      {
        key: "cifar10", name: "CIFAR-10", sub: "labelled subset of the 80M tiny images",
        classes: 10, train: 60000, test: 6000,
        /* flagged addition: the source gives CIFAR's colour and 160 MB size but
           not its 32×32 resolution; the 32×32×3 figure is standard knowledge
           added by me. */
        res: "32 × 32 × 3", pixels: 32 * 32 * 3, sizeMB: 160,
        note: "colour, everyday objects",
      },
      {
        key: "cifar100", name: "CIFAR-100", sub: "same images, 100 finer classes",
        classes: 100, train: 60000, test: 6000,
        res: "32 × 32 × 3", pixels: 32 * 32 * 3, sizeMB: 160,
        note: "colour, fine-grained labels",
      },
      {
        key: "imagenet", name: "ImageNet (2009)", sub: "WordNet synsets, Mechanical Turk labels",
        classes: 1000, train: 1200000, test: 150000,
        /* flagged addition: the source's architecture table lists 256×256×3 as
           the AlexNet/GoogLeNet image size; using it as ImageNet's working
           resolution here is my inference. */
        res: "~256 × 256 × 3", pixels: 256 * 256 * 3, sizeMB: 50000,
        note: "1,000 of ~22,000 available synsets",
      },
    ],

    /* architecture lineage: every number from the lecture's comparison
       table and architecture notes. err = ILSVRC top-5 / test error where
       the lecture reports it; null = not given in the source. */
    ARCHS: [
      {
        key: "lenet1", name: "LeNet-1", year: 1988,
        depth: "a few weight layers (first of the LeNet line)",
        params: 9760, paramsTxt: "~9,760", err: null,
        cats: "10", img: "~16 × 16", train: "7,291",
        idea: "Convolution + average pooling (subsampling) + fully connected, trained end to end for USPS zip-code digits: the existence proof.",
      },
      {
        key: "lenet5", name: "LeNet-5", year: 1998,
        /* flagged addition: "5 weight layers" is read off the source's C1,
           C2, C3, FC1, FC2 stack; the source does not state the depth as a
           number. */
        depth: "5 weight layers (C1, C2, C3, FC1, FC2)",
        params: 60000, paramsTxt: "~60,000", err: null,
        cats: "10", img: "32 × 32", train: "60,000",
        idea: "The mature recipe: conv, subsample, conv, subsample, dense, softmax; originally tanh where modern nets use ReLU. Deployed for envelope sorting.",
      },
      {
        key: "alexnet", name: "AlexNet", year: 2012,
        depth: "8 weight layers",
        params: 60000000, paramsTxt: "60M", err: "16.4% top-5",
        cats: "1,000", img: "256 × 256 × 3", train: "1.2M",
        idea: "Two parallel branches split across two GTX 580 3GB GPUs, 5 to 6 days of training. Its ILSVRC win set off the modern deep-learning boom.",
      },
      {
        key: "googlenet", name: "GoogLeNet", year: 2014,
        depth: "22 weight layers",
        params: 4000000, paramsTxt: "4M", err: "6.6% test error",
        cats: "1,000", img: "256 × 256 × 3", train: "1.2M",
        idea: "Fully convolutional (no giant dense layers), factorized convolutions, and 1×1 bottlenecks: deeper than AlexNet with 15× fewer parameters.",
      },
      {
        key: "resnet", name: "ResNet", year: 2015,
        depth: "100+ layers",
        params: null, paramsTxt: "not given in the lecture", err: "so strong that evaluation moved to top-1",
        cats: "1,000", img: "256 × 256 × 3", train: "1.2M",
        idea: "Skip connections (y = F(x) + x) keep very deep stacks trainable. Many remaining “errors” traced to wrong ground-truth labels; the benchmark was effectively retired.",
      },
    ],
  });

  /* ════════════════════════════════════════════════════════════
     Fig 1.1 — the 1×1 convolution as a channel mixer
     Four real feature maps (computed with the Part I core) mixed
     into I output channels by 1×1×4 filters. Everything live.
     ════════════════════════════════════════════════════════════ */
  LR.figs.onebyone = function (mount) {
    LR.header(
      mount,
      "Mix channels, keep the grid",
      "Four input channels (real convolutions of Part I's test image) feed 1×1 filters. Slide the output depth, pick an output channel, and click any cell of the big map to audit its arithmetic."
    );

    // input channels: real feature maps from the Part I core
    const SRC = ["vedge", "hedge", "blur", "sharpen"];
    const chans = SRC.map((k) => CNN2.conv2d(CNN2.IMG8, CNN2.KER[k].k, 1, 0)); // four 6×6 maps
    const M = 4, N = 6;

    /* flagged addition: the specific 1×1 mixing weights below are examples
       chosen by me (the source describes the mechanism, not numeric filters). */
    const WSETS = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0.5, 0.5, 0, 0],
      [1, -1, 0, 0],
      [0, 0, 1, -1],
      [0.25, 0.25, 0.25, 0.25],
      [1, 0, -1, 0],
      [0, 1, 0, -1],
    ];

    let I = 2;           // output depth
    let sel = 0;         // selected output channel
    let pix = { i: 1, j: 3 };  // inspected pixel (on the square's edge: non-zero responses)

    const bar = LR.controls(mount);
    LR.slider(bar, "output depth I (number of 1×1 filters)", 1, 8, 1, I, function (v) {
      I = Math.round(v);
      if (sel >= I) sel = I - 1;
      draw();
    }, (v) => String(Math.round(v)));
    LR.button(bar, "◂ channel", function () { sel = (sel + I - 1) % I; draw(); }, "small");
    LR.button(bar, "channel ▸", function () { sel = (sel + 1) % I; draw(); }, "small");

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 400, {
      aria: "One by one convolution demo: four input channels mixed into a chosen number of output channels, with the weighted sum at one pixel shown in full",
    });

    const calc = LR.el("div", "det-calc");
    mount.appendChild(calc);

    const ro = LR.readout(mount, [
      { k: "vin", label: "input volume" },
      { k: "vout", label: "output volume" },
      { k: "params", label: "params (1²·I·J + I)" },
    ]);
    const msg = LR.msg(mount);

    // layout constants
    const CIN = 13, INX = 26, INY = 66;               // 2×2 grid of input maps
    const BIGC = 26, BX = 360, BY = 80;               // selected output map, large
    const STX = 636, STY = 96, STC = 14;              // output stack slices

    function outMap(c) {
      const w = WSETS[c];
      const out = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          let s = 0;
          for (let m = 0; m < M; m++) s += w[m] * chans[m][i][j];
          row.push(s);
        }
        out.push(row);
      }
      return out;
    }

    LR.drag(cv, W, H, {
      hit: function (p) {
        // click the big output map → move the inspected pixel
        if (p.x >= BX && p.x < BX + N * BIGC && p.y >= BY && p.y < BY + N * BIGC) {
          pix = { i: Math.floor((p.y - BY) / BIGC), j: Math.floor((p.x - BX) / BIGC) };
          draw();
          return false;
        }
        // click an output stack slice → select that channel
        for (let c = I - 1; c >= 0; c--) {
          const off = c * 12;
          if (p.x >= STX + off && p.x < STX + off + N * STC && p.y >= STY + off && p.y < STY + off + N * STC) {
            sel = c; draw();
            return false;
          }
        }
        return false;
      },
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const w = WSETS[sel];
      const out = outMap(sel);
      const omax = Math.max(CNN2.maxAbs(out), 1e-9);

      // input maps, 2×2 grid
      CNN2.label(ctx, INX, INY - 26, "input: 6 × 6 × 4 volume");
      CNN2.label(ctx, INX, INY - 10, "(four real feature maps)", C.faint);
      const names = ["ch1 · v-edge", "ch2 · h-edge", "ch3 · blur", "ch4 · sharpen"];
      for (let m = 0; m < M; m++) {
        const gx = INX + (m % 2) * (N * CIN + 44);
        const gy = INY + Math.floor(m / 2) * (N * CIN + 42);
        const cmax = Math.max(CNN2.maxAbs(chans[m]), 1e-9);
        CNN2.grid(ctx, gx, gy, CIN, chans[m], { fill: (v) => CNN2.ramp(v, cmax) });
        // highlight the inspected pixel on each channel
        ctx.strokeStyle = C.text; ctx.lineWidth = 1.6;
        ctx.strokeRect(gx + pix.j * CIN - 1, gy + pix.i * CIN - 1, CIN, CIN);
        ctx.save();
        ctx.font = "600 10.5px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText(names[m], gx, gy + N * CIN + 13);
        // the mixing weight for this channel, beside the map
        ctx.font = "700 11.5px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = w[m] === 0 ? C.faint : w[m] > 0 ? C.orange : C.purple;
        ctx.fillText("w=" + CNN2.fmtK(w[m]), gx, gy - 5);
        ctx.restore();
      }

      // arrow input → filter → output
      LR.arrow(ctx, 250, 190, 292, 190, "rgba(0,0,0,0.3)", 2);
      ctx.save();
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "center";
      ctx.fillText("1×1×4", 322, 176);
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.fillText("filter " + (sel + 1) + " of " + I, 322, 192);
      ctx.restore();
      LR.arrow(ctx, 344, 190, BX - 8, 190, "rgba(26,26,26,0.55)", 2);

      // selected output channel, large
      CNN2.label(ctx, BX, BY - 26, "output channel " + (sel + 1) + " (6 × 6)");
      CNN2.label(ctx, BX, BY - 10, "click a cell to inspect it", C.faint);
      CNN2.grid(ctx, BX, BY, BIGC, out, {
        fill: (v) => CNN2.ramp(v, omax),
        num: (v) => CNN2.fmtV(v),
        numColor: () => C.text,
      });
      ctx.strokeStyle = C.text; ctx.lineWidth = 2.2;
      ctx.strokeRect(BX + pix.j * BIGC - 1, BY + pix.i * BIGC - 1, BIGC, BIGC);

      // output stack
      CNN2.label(ctx, STX - 4, STY - 26, "output: 6 × 6 × " + I);
      CNN2.label(ctx, STX - 4, STY - 10, "spatial size unchanged", C.faint);
      for (let c = I - 1; c >= 0; c--) {
        const off = c * 12;
        const m2 = outMap(c);
        const mmax = Math.max(CNN2.maxAbs(m2), 1e-9);
        CNN2.grid(ctx, STX + off, STY + off, STC, m2, {
          fill: (v) => CNN2.ramp(v, mmax),
          stroke: "#cfcfcf",
        });
        ctx.strokeStyle = c === sel ? C.orange : "#bbbbbb";
        ctx.lineWidth = c === sel ? 2.4 : 1.2;
        ctx.strokeRect(STX + off - 1, STY + off - 1, N * STC + 1, N * STC + 1);
      }

      // the audited pixel, term by term (real numbers)
      let z = 0;
      const terms = [];
      for (let m = 0; m < M; m++) {
        const a = chans[m][pix.i][pix.j];
        z += w[m] * a;
        terms.push("(" + CNN2.fmtK(w[m]) + ")(" + CNN2.fmtV(a) + ")");
      }
      calc.innerHTML =
        "z<sub>out</sub>[" + pix.i + "," + pix.j + "] = " + terms.join(" + ") +
        " = <b style='color:" + (z >= 0 ? C.orange : C.purple) + "'>" + CNN2.fmtV(z) + "</b>" +
        " &nbsp;<span style='color:var(--faint)'>(one weighted sum straight down the 4 channels)</span>";

      const params = (1 * 1 * M + 1) * I;
      ro.set("vin", "6 × 6 × 4");
      ro.set("vout", "6 × 6 × " + I, C.orange);
      ro.set("params", "(1·1·4 + 1)·" + I + " = " + params, C.green);

      if (I < M) {
        msg.show("Compression: " + M + " channels squeezed to " + I + " for only " + params + " parameters, spatial grid untouched. Every later layer now convolves a thinner volume. This is GoogLeNet's bottleneck move.", "info");
      } else if (I > M) {
        msg.show("Expansion: more 1×1 filters (" + I + ") than input channels (" + M + "), so the depth grew. Same mechanism, opposite direction.", "info");
      } else {
        msg.show("Depth preserved (4 → 4), but each output channel is still a new learned mixture of all four inputs: cross-channel features for " + params + " parameters.", "info");
      }
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — the dataset ladder: MNIST · CIFAR · ImageNet
     Cards + a log-scale bar chart, re-sortable by attribute.
     All numbers from the lecture; thumbnails drawn procedurally.
     ════════════════════════════════════════════════════════════ */
  LR.figs.datasets = function (mount) {
    LR.header(
      mount,
      "Three generations of vision data",
      "Pick an attribute: the bars (log scale) and cards re-sort. Thumbnails are synthetic sketches in each dataset's visual style, drawn in this page, not real dataset images."
    );

    const DS = CNN2.DATASETS;
    const ATTRS = {
      classes: { label: "classes", get: (d) => d.classes, fmt: (v) => String(v) },
      train: { label: "training images", get: (d) => d.train, fmt: (v) => CNN2.fmtBig(v) },
      pixels: { label: "pixels per image", get: (d) => d.pixels, fmt: (v) => v.toLocaleString("en-CA") },
      size: { label: "disk size", get: (d) => d.sizeMB, fmt: (v) => (v >= 1000 ? LR.fmt(v / 1000, 0) + " GB" : v + " MB") },
    };
    let attr = "train";

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const aBtns = {};
    Object.keys(ATTRS).forEach(function (key) {
      aBtns[key] = LR.button(group, ATTRS[key].label, function () {
        attr = key;
        for (const k2 in aBtns) aBtns[k2].classList.toggle("on", k2 === attr);
        sync();
      }, "small" + (key === attr ? " on" : ""));
    });

    // ---- cards ----
    const cardsBox = LR.el("div", "ds-cards");
    mount.appendChild(cardsBox);
    const cardEls = DS.map(function (d) {
      const card = LR.el("div", "ds-card");
      const th = document.createElement("canvas");
      th.width = 168; th.height = 112;
      th.className = "ds-thumb";
      th.setAttribute("role", "img");
      th.setAttribute("aria-label", "synthetic sample sketch in the style of " + d.name);
      card.appendChild(th);
      card.appendChild(LR.el("div", "ds-name", d.name));
      card.appendChild(LR.el("div", "ds-sub", d.sub));
      const stats = LR.el("div", "ds-stats");
      stats.innerHTML =
        row("classes", String(d.classes)) +
        row("train / test", CNN2.fmtBig(d.train) + " / " + CNN2.fmtBig(d.test)) +
        row("resolution", d.res) +
        row("disk size", ATTRS.size.fmt(d.sizeMB));
      card.appendChild(stats);
      cardsBox.appendChild(card);
      drawThumb(th, d.key);
      return card;
    });
    function row(k, v) {
      return "<div class='ds-row'><span>" + k + "</span><b>" + v + "</b></div>";
    }

    // ---- procedural thumbnails (synthetic, no real dataset images) ----
    function drawThumb(cvT, key) {
      const g = cvT.getContext("2d");
      const rand = LR.rng(key === "mnist" ? 11 : key === "cifar10" ? 22 : key === "cifar100" ? 33 : 44);
      g.fillStyle = "#f2f2f2";
      g.fillRect(0, 0, 168, 112);
      if (key === "mnist") {
        // 3×2 tiles of white squiggles on black: digit-like strokes
        for (let t = 0; t < 6; t++) {
          const x = 6 + (t % 3) * 54, y = 6 + Math.floor(t / 3) * 52;
          g.fillStyle = "#0d0d0d";
          g.fillRect(x, y, 48, 46);
          g.strokeStyle = "#f5f5f5";
          g.lineWidth = 4.5;
          g.lineCap = "round";
          g.beginPath();
          let px = x + 12 + rand() * 10, py = y + 8 + rand() * 6;
          g.moveTo(px, py);
          for (let s = 0; s < 3; s++) {
            const nx = x + 8 + rand() * 32, ny = y + 10 + (s + 1) * 11 + rand() * 5;
            g.quadraticCurveTo(px + (rand() - 0.5) * 26, (py + ny) / 2, nx, ny);
            px = nx; py = ny;
          }
          g.stroke();
        }
      } else if (key === "cifar10" || key === "cifar100") {
        // 3×2 tiles of chunky low-res colour blocks
        const pals = [
          ["#c1c1c1", "#737373", "#a7a7a7"], ["#949494", "#626262", "#d6d6d6"],
          ["#8a8a8a", "#b5b5b5", "#585858"], ["#6e6e6e", "#cacaca", "#888888"],
          ["#757575", "#b3b3b3", "#707070"], ["#929292", "#c9c9c9", "#4c4c4c"],
        ];
        for (let t = 0; t < 6; t++) {
          const x = 6 + (t % 3) * 54, y = 6 + Math.floor(t / 3) * 52;
          const pal = pals[(t + (key === "cifar100" ? 3 : 0)) % pals.length];
          for (let a = 0; a < 6; a++)
            for (let b = 0; b < 6; b++) {
              g.fillStyle = pal[Math.floor(rand() * pal.length)];
              g.fillRect(x + b * 8, y + a * 8, 8, 8);
            }
          // one blocky "object" blob in the middle
          g.fillStyle = pal[0];
          g.fillRect(x + 12, y + 14, 24, 18);
        }
      } else {
        // ImageNet: 2 larger "photos": gradient sky + silhouette shapes
        for (let t = 0; t < 2; t++) {
          const x = 6 + t * 82, y = 8, w2 = 74, h2 = 96;
          const grad = g.createLinearGradient(0, y, 0, y + h2);
          grad.addColorStop(0, t === 0 ? "#cfcfcf" : "#d5d5d5");
          grad.addColorStop(1, t === 0 ? "#ededed" : "#e9e9e9");
          g.fillStyle = grad;
          g.fillRect(x, y, w2, h2);
          g.fillStyle = t === 0 ? "#737373" : "#6f6f6f";
          g.beginPath();
          g.moveTo(x, y + h2);
          g.lineTo(x, y + h2 - 22 - rand() * 12);
          g.quadraticCurveTo(x + w2 / 2, y + h2 - 46 - rand() * 14, x + w2, y + h2 - 18 - rand() * 10);
          g.lineTo(x + w2, y + h2);
          g.closePath();
          g.fill();
          // "animal" blob with legs
          g.fillStyle = "#3a3a3a";
          const ax = x + 16 + rand() * 30, ay = y + h2 - 30;
          g.beginPath();
          g.ellipse(ax, ay, 13, 8, 0, 0, Math.PI * 2);
          g.fill();
          g.fillRect(ax - 9, ay + 4, 3, 10);
          g.fillRect(ax + 6, ay + 4, 3, 10);
          g.beginPath();
          g.arc(ax + 13, ay - 7, 5, 0, Math.PI * 2);
          g.fill();
        }
      }
    }

    // ---- bar chart ----
    const { cv, ctx, W, H } = LR.canvas(mount, 820, 230, {
      aria: "Log-scale bar chart comparing MNIST, CIFAR, and ImageNet on the selected attribute",
    });
    const msg = LR.msg(mount);

    function sync() {
      const A = ATTRS[attr];
      const order = DS.slice().sort((a, b) => A.get(a) - A.get(b));
      // reorder cards to match
      order.forEach(function (d, i) {
        cardEls[DS.indexOf(d)].style.order = String(i);
      });

      ctx.clearRect(0, 0, W, H);
      const vmax = Math.max.apply(null, DS.map(A.get));
      const lmax = Math.log10(vmax) * 1.06;
      const x0 = 150, bw = W - x0 - 130, rowH = 44, y0 = 30;

      ctx.save();
      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.muted;
      ctx.textAlign = "left";
      ctx.fillText(A.label + " (log scale)", x0, 18);
      order.forEach(function (d, i) {
        const v = A.get(d);
        const frac = Math.max(0.03, Math.log10(Math.max(1.5, v)) / lmax);
        const y = y0 + i * rowH;
        ctx.fillStyle = C.text;
        ctx.font = "700 13px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(d.name, x0 - 12, y + 21);
        ctx.fillStyle = d.key === "imagenet" ? C.orange : d.key === "mnist" ? C.green : C.purple;
        ctx.fillRect(x0, y, frac * bw, 30);
        ctx.font = "700 12.5px 'JetBrains Mono', Menlo, monospace";
        ctx.textAlign = "left";
        ctx.fillStyle = C.text;
        ctx.fillText(A.fmt(v), x0 + frac * bw + 10, y + 21);
      });
      ctx.restore();

      const lo = A.get(order[0]), hi = A.get(order[order.length - 1]);
      const ratio = hi / lo;
      msg.show(
        "From " + order[0].name + " to " + order[order.length - 1].name + ": " +
        A.fmt(lo) + " → " + A.fmt(hi) +
        (ratio >= 2 ? ", a " + (ratio >= 100 ? Math.round(ratio).toLocaleString("en-CA") : LR.fmt(ratio, 1)) + "× jump." : ".") +
        (attr === "size" ? " That jump in raw bytes is why ImageNet needed industrial labelling (Mechanical Turk) and serious compute to be usable at all." : ""),
        "info"
      );
    }
    sync();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the scale-together playground
     Stylized validation-error model. flagged addition: the curve
     shapes and constants below are qualitative illustrations built
     by me around the lecture's claim that data, compute, and model
     size must scale together (and its scaling-laws reference); they
     are not measured numbers from any paper.
     ════════════════════════════════════════════════════════════ */
  LR.figs.scaling = function (mount) {
    LR.header(
      mount,
      "Scale the ingredients together",
      "Three budgets: data, model, compute. The curve shows (stylized) validation error against training compute for your current data and model; the diagnosis names the binding constraint."
    );

    // sliders operate on log10 of each quantity
    let logD = 4.5, logN = 5.5, logC = 1.5;

    const bar = LR.controls(mount);
    LR.slider(bar, "dataset size (images)", 3, 7, 0.1, logD, function (v) { logD = v; draw(); }, (v) => CNN2.fmtBig(Math.round(Math.pow(10, v))));
    LR.slider(bar, "model size (parameters)", 4, 8, 0.1, logN, function (v) { logN = v; draw(); }, (v) => CNN2.fmtBig(Math.round(Math.pow(10, v))));
    LR.slider(bar, "training compute (GPU-hours)", 0, 4, 0.1, logC, function (v) { logC = v; draw(); }, (v) => CNN2.fmtBig(Math.round(Math.pow(10, v))));

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 380, {
      aria: "Stylized validation error versus training compute, with training error shown as a second curve; the gap between them opens when the model outgrows the data",
    });
    const ro = LR.readout(mount, [
      { k: "val", label: "validation error (stylized)" },
      { k: "gap", label: "overfit gap (val − train)" },
      { k: "diag", label: "binding constraint" },
    ]);
    const msg = LR.msg(mount);

    // stylized error components, in "percent error" units
    function comps(lD, lN, lC) {
      const D = Math.pow(10, lD), N = Math.pow(10, lN), Cc = Math.pow(10, lC);
      const E0 = 2;                                        // irreducible floor
      const fModel = 24 * Math.pow(1e4 / N, 0.25);         // capacity-limited
      const fData = 24 * Math.pow(1e3 / D, 0.3);           // data-limited
      const need = 0.5 * Math.sqrt((N / 1e4) * (D / 1e3)); // compute to finish training
      const under = Math.max(0, 14 * Math.log10(Math.max(1, need / Cc))); // undertrained
      const over = Math.max(0, 7 * Math.log10(Math.max(1, N / (20 * D)))); // model ≫ data
      const train = Math.min(60, E0 + fModel * 0.7 + under);              // train error: no data penalty
      const val = Math.min(60, E0 + fModel + fData + under + over);
      return { val, train, fModel, fData, under, over, need };
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 62, y0: 20, w: W - 92, h: H - 78,
        xmin: 0, xmax: 4, ymin: 0, ymax: 60,
        xlabel: "training compute (GPU-hours, log₁₀ scale)",
        ylabel: "error (%), stylized",
        xticks: [0, 1, 2, 3, 4],
      };
      const { sx, sy } = LR.plot(ctx, P);

      // validation + training curves across the compute range
      const curves = [
        { key: "val", color: C.orange, width: 2.6 },
        { key: "train", color: C.green, width: 2 },
      ];
      curves.forEach(function (cu) {
        ctx.strokeStyle = cu.color;
        ctx.lineWidth = cu.width;
        ctx.beginPath();
        let first = true;
        for (let lc = 0; lc <= 4.001; lc += 0.05) {
          const y = comps(logD, logN, lc)[cu.key];
          first ? ctx.moveTo(sx(lc), sy(y)) : ctx.lineTo(sx(lc), sy(y));
          first = false;
        }
        ctx.stroke();
      });

      // shade the val–train gap at the current compute onward
      const here = comps(logD, logN, logC);
      ctx.save();
      ctx.fillStyle = "rgba(61,61,61,0.08)";
      ctx.beginPath();
      let first = true;
      for (let lc = 0; lc <= 4.001; lc += 0.05) {
        const y = comps(logD, logN, lc).val;
        first ? ctx.moveTo(sx(lc), sy(y)) : ctx.lineTo(sx(lc), sy(y));
        first = false;
      }
      for (let lc = 4; lc >= -0.001; lc -= 0.05) {
        ctx.lineTo(sx(lc), sy(comps(logD, logN, lc).train));
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // marker at current compute
      LR.dot(ctx, sx(logC), sy(here.val), 6.5, C.orange, "#ffffff");
      LR.dot(ctx, sx(logC), sy(here.train), 5.5, C.green, "#ffffff");
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(0,0,0,0.3)"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(logC), sy(0)); ctx.lineTo(sx(logC), sy(here.val)); ctx.stroke();
      ctx.setLineDash([]);

      ctx.save();
      ctx.font = "600 12.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = C.orange;
      ctx.fillText("validation error", P.x0 + 10, sy(comps(logD, logN, 0.35).val) - 10);
      ctx.fillStyle = C.green;
      ctx.fillText("training error", P.x0 + 10, sy(comps(logD, logN, 0.1).train) + 18);
      ctx.restore();

      // diagnosis: the largest error component at the marker
      const gap = here.val - here.train;
      let diag, tip;
      if (here.under >= Math.max(here.fModel, here.fData + here.over, 4)) {
        diag = "compute-limited (undertrained)";
        tip = "Training stopped before convergence: it needs about " + CNN2.fmtBig(Math.round(here.need)) + " GPU-hours at this model and data scale. More compute moves you down the curve.";
      } else if (here.over > 2 || (here.fData > here.fModel + 2.5)) {
        diag = "data-limited" + (here.over > 2 ? " (overfitting)" : "");
        tip = here.over > 2
          ? "The model (" + CNN2.fmtBig(Math.round(Math.pow(10, logN))) + " params) dwarfs the data (" + CNN2.fmtBig(Math.round(Math.pow(10, logD))) + " images): the red gap between validation and training error is memorization. Grow the data, not the model."
          : "Error is pinned by the dataset: more images (or transfer learning) would help more than anything else.";
      } else if (here.fModel > here.fData + 2.5) {
        diag = "model-limited";
        tip = "The dataset has more to teach than this model can hold. A bigger model (with the compute to train it) is the move, exactly the LeNet-to-AlexNet jump.";
      } else {
        diag = "balanced";
        tip = "Data, model, and compute are roughly in balance: this is the scaling-laws regime where growing all three together keeps buying accuracy.";
      }

      ro.set("val", LR.fmtF(here.val, 1) + "%", C.orange);
      ro.set("gap", LR.fmtF(gap, 1) + "%", gap > 8 ? C.red : C.green);
      ro.set("diag", diag, diag === "balanced" ? C.green : C.amber);
      msg.show(tip, diag === "balanced" ? "good" : "info");
    }
    draw();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 4.1 — the architecture timeline (signature figure)
     LeNet-1 → ResNet with the lecture's exact numbers, plus a
     linked log-scale parameter chart across the generations.
     ════════════════════════════════════════════════════════════ */
  LR.figs.timeline = function (mount) {
    LR.header(
      mount,
      "Five architectures, one arms race",
      "Click a node on the timeline (or use the buttons). The lower chart plots parameter count on a log scale; the two top-5 / test errors the lecture reports are marked."
    );

    const A = CNN2.ARCHS;
    let sel = 2; // start on AlexNet, the hinge of the story

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const nBtns = A.map(function (d, i) {
      return LR.button(group, d.name, function () { sel = i; draw(); }, "small" + (i === sel ? " on" : ""));
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 350, {
      aria: "Architecture timeline from LeNet-1 in 1988 to ResNet in 2015, with a linked log-scale chart of parameter counts",
    });

    const detail = LR.el("div", "arch-detail");
    mount.appendChild(detail);

    const YR0 = 1985, YR1 = 2018;
    const TX0 = 60, TX1 = W - 40, TY = 52;
    const xOf = (yr) => TX0 + ((yr - YR0) / (YR1 - YR0)) * (TX1 - TX0);

    LR.drag(cv, W, H, {
      hit: function (p) {
        for (let i = 0; i < A.length; i++) {
          const dx = p.x - xOf(A[i].year);
          if (Math.abs(dx) < 20 && p.y > TY - 26 && p.y < TY + 30) { sel = i; draw(); return false; }
        }
        return false;
      },
    });

    function draw() {
      nBtns.forEach((b, i) => b.classList.toggle("on", i === sel));
      ctx.clearRect(0, 0, W, H);

      // ── timeline band ──
      ctx.strokeStyle = "#cfcfcf"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(TX0 - 10, TY); ctx.lineTo(TX1 + 10, TY); ctx.stroke();
      [1990, 1995, 2000, 2005, 2010, 2015].forEach(function (yr) {
        ctx.strokeStyle = "#dddddd"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(xOf(yr), TY - 5); ctx.lineTo(xOf(yr), TY + 5); ctx.stroke();
        ctx.font = "600 10.5px Inter, sans-serif";
        ctx.fillStyle = C.faint; ctx.textAlign = "center";
        ctx.fillText(String(yr), xOf(yr), TY + 20);
      });
      A.forEach(function (d, i) {
        const x = xOf(d.year);
        LR.dot(ctx, x, TY, i === sel ? 9 : 6.5, i === sel ? C.orange : "#ffffff", i === sel ? "#ffffff" : C.text);
        if (i !== sel) LR.dot(ctx, x, TY, 6.5, "#ffffff", C.text);
        ctx.font = (i === sel ? "800 12.5px" : "600 11.5px") + " Inter, sans-serif";
        ctx.fillStyle = i === sel ? C.orange : C.text;
        ctx.textAlign = "center";
        const lift = i % 2 === 0 ? -14 : -30; // stagger labels
        ctx.fillText(d.name, x, TY + lift);
        ctx.font = "600 10px Inter, sans-serif";
        ctx.fillStyle = C.faint;
        ctx.fillText(String(d.year), x, TY + 34);
      });

      // ── parameter chart ──
      const P = {
        x0: 60, y0: 116, w: W - 100, h: H - 116 - 46,
        xmin: YR0, xmax: YR1, ymin: 3.5, ymax: 8.2,
        xlabel: "year", ylabel: "parameters (log₁₀)",
        xticks: [1990, 1995, 2000, 2005, 2010, 2015],
        yticks: [4, 5, 6, 7, 8],
      };
      const { sx, sy } = LR.plot(ctx, P);

      const withP = A.filter((d) => d.params !== null);
      ctx.strokeStyle = "rgba(0,0,0,0.22)";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      withP.forEach(function (d, i) {
        const x = sx(d.year), y = sy(Math.log10(d.params));
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      withP.forEach(function (d) {
        const i = A.indexOf(d);
        const x = sx(d.year), y = sy(Math.log10(d.params));
        LR.dot(ctx, x, y, i === sel ? 8 : 6, i === sel ? C.orange : C.purple, "#ffffff");
        ctx.font = "700 11.5px 'JetBrains Mono', Menlo, monospace";
        ctx.fillStyle = C.text;
        ctx.textAlign = d.key === "googlenet" ? "left" : "center";
        ctx.fillText(d.paramsTxt, d.key === "googlenet" ? x + 12 : x, y - 12);
        if (d.err) {
          ctx.font = "600 11px Inter, sans-serif";
          ctx.fillStyle = C.green;
          ctx.textAlign = d.key === "googlenet" ? "left" : "center";
          ctx.fillText(d.err, d.key === "googlenet" ? x + 12 : x, y + 24);
        }
      });

      // ResNet: no parameter count in the lecture; mark it on the chart edge
      const rx = sx(2015);
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = C.amber; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(rx, P.y0 + 6); ctx.lineTo(rx, P.y0 + P.h - 6); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillStyle = C.amber;
      ctx.textAlign = "right";
      ctx.fillText("ResNet: 100+ layers,", rx - 8, P.y0 + 24);
      ctx.fillText("params not given in the lecture", rx - 8, P.y0 + 40);

      // ── detail panel ──
      const d = A[sel];
      detail.innerHTML =
        "<div class='ad-name'>" + d.name + " <span class='ad-year'>(" + d.year + ")</span></div>" +
        "<div class='ad-grid'>" +
        item("depth", d.depth) +
        item("parameters", d.paramsTxt) +
        item("error", d.err || "not reported in the lecture") +
        item("categories", d.cats) +
        item("image size", d.img) +
        item("training examples", d.train) +
        "</div>" +
        "<div class='ad-idea'><b>Signature idea:</b> " + d.idea + "</div>";
      function item(k, v) {
        return "<div class='ad-item'><span>" + k + "</span><b>" + v + "</b></div>";
      }
    }
    draw();
  };
})();
