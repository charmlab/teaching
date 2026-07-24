/* ══════════════════════════════════════════════════════════════
   figures-b.js — Autoencoders & VAEs lesson, sections 4–6
   Fig 4.1 gaps (signature) · Fig 5.1 pointdist · Fig 6.1 kltrade
   (signature) · Fig 6.2 reparam (signature) · Fig 6.3 vaecode
   Fig 6.4 generate
   Uses the shared autoencoder core LR.ae from figures-a.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const AE = LR.ae;

  /* shared latent-panel scaffold: draws the axes for a [-4,4]² latent */
  function latentPlot(ctx, x0, y0, size) {
    const Pl = {
      x0: x0, y0: y0, w: size, h: size,
      xmin: -4, xmax: 4, ymin: -4, ymax: 4,
      xlabel: "z₁", ylabel: "z₂",
      xticks: [-4, -2, 0, 2, 4], yticks: [-4, -2, 0, 2, 4],
    };
    return { Pl: Pl, sc: LR.plot(ctx, Pl) };
  }

  /* proximity weights only (no image), for cheap realism statistics */
  function latentWeights(z, layout) {
    const ws = [];
    layout.clusters.forEach(function (c) {
      const rel = AE.sub2(z, c.mu);
      const u = Math.max(-layout.halfLen, Math.min(layout.halfLen, AE.dot2(rel, c.dir)));
      const q = [c.mu[0] + c.dir[0] * u, c.mu[1] + c.dir[1] * u];
      const d = AE.nrm2(AE.sub2(z, q));
      ws.push(Math.exp(-(d * d) / (2 * layout.tau * layout.tau)));
    });
    const wsum = ws.reduce((a, b) => a + b, 0);
    return { wsum: wsum, wn: ws.map((w) => w / (wsum || 1e-12)) };
  }

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — the gappy latent space of a deterministic AE
     ══════════════════════════════════════════════════════════ */
  LR.figs.gaps = function (mount) {
    LR.header(
      mount,
      "Drop a sample into a gap",
      "Left: the 2-D latent codes of a deterministic autoencoder trained on three glyph families. Click or drag anywhere to pick a z and the decoder runs live on the right. Try the space between clusters."
    );

    const layout = AE.layout(0);
    const items = AE.latents(layout);
    let probe = [0.3, 2.55]; // starts in the gap between disc and cross

    const bar = LR.controls(mount);
    LR.button(bar, "Drop into a gap", function () { probe = [0.3, 2.55]; render(); }, "primary");
    LR.button(bar, "Between all three", function () { probe = [0.15, -0.2]; render(); });
    LR.button(bar, "Onto the manifold", function () { probe = items[8].z.slice(); render(); });

    const cvo = LR.canvas(mount, 800, 440, {
      aria: "Left: scatter of latent codes forming three separated clusters with a movable probe point. Right: the decoded 16 by 16 glyph for the probe, with a bar chart of the decoder's blend weights",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;

    const ro = LR.readout(mount, [
      { k: "z", label: "probe z" },
      { k: "dom", label: "dominant family" },
      { k: "real", label: "blend purity" },
    ]);
    const msg = LR.msg(mount);

    const PANEL = { x0: 62, y0: 18, size: 360 };

    LR.drag(cvo.cv, W, H, {
      hit: function (p) {
        return p.x >= PANEL.x0 && p.x <= PANEL.x0 + PANEL.size && p.y >= PANEL.y0 && p.y <= PANEL.y0 + PANEL.size;
      },
      down: function (p) { setProbe(p); },
      move: function (p) { setProbe(p); },
    });
    cvo.cv.tabIndex = 0;
    cvo.cv.addEventListener("keydown", function (e) {
      const step = e.shiftKey ? 0.5 : 0.15;
      if (e.key === "ArrowLeft") { probe[0] -= step; e.preventDefault(); render(); }
      if (e.key === "ArrowRight") { probe[0] += step; e.preventDefault(); render(); }
      if (e.key === "ArrowUp") { probe[1] += step; e.preventDefault(); render(); }
      if (e.key === "ArrowDown") { probe[1] -= step; e.preventDefault(); render(); }
    });

    function setProbe(p) {
      const zx = -4 + ((p.x - PANEL.x0) / PANEL.size) * 8;
      const zy = 4 - ((p.y - PANEL.y0) / PANEL.size) * 8;
      probe = [Math.max(-4, Math.min(4, zx)), Math.max(-4, Math.min(4, zy))];
      render();
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const lp = latentPlot(ctx, PANEL.x0, PANEL.y0, PANEL.size);
      const sx = lp.sc.sx, sy = lp.sc.sy;

      // training latents
      items.forEach(function (it) {
        LR.dot(ctx, sx(it.z[0]), sy(it.z[1]), 3.6, AE.clusterColor(it.c));
      });
      layout.clusters.forEach(function (c, ci) {
        ctx.fillStyle = AE.clusterColor(ci);
        ctx.font = "700 11.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(c.name, sx(c.mu[0]), sy(c.mu[1]) - 22);
      });

      // probe
      LR.dot(ctx, sx(probe[0]), sy(probe[1]), 7, "#ffffff", C.red);
      LR.dot(ctx, sx(probe[0]), sy(probe[1]), 3, C.red);

      // decode
      const dec = AE.decode(probe, layout);
      const gx = 500, gy = 40, gsz = 200;
      ctx.fillStyle = C.text;
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("decoder output gθ(z)", gx + gsz / 2, gy - 10);
      AE.drawGlyph(ctx, dec.img, gx, gy, gsz, { border: "#999999", borderW: 1.4 });

      // blend-weight bars (three families plus the off-manifold component)
      const bx = 490, by = 268, bw = 210, bh = 15;
      const names = layout.clusters.map((c) => c.name).concat(["off-manifold"]);
      const cols = [0, 1, 2].map(AE.clusterColor).concat(["#b4b4b4"]);
      names.forEach(function (nm, ci) {
        const y = by + ci * 28;
        ctx.fillStyle = "#f3f3f3";
        ctx.fillRect(bx + 74, y, bw - 74, bh);
        ctx.fillStyle = cols[ci];
        ctx.fillRect(bx + 74, y, (bw - 74) * dec.wn[ci], bh);
        ctx.font = "600 11px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillStyle = C.text;
        ctx.fillText(nm, bx + 68, y + 12);
        ctx.textAlign = "left";
        ctx.fillStyle = LR.C.muted;
        ctx.fillText(LR.fmtF(dec.wn[ci], 2), bx + bw + 6, y + 12);
      });

      ro.set("z", "[" + LR.fmtF(probe[0], 2) + ", " + LR.fmtF(probe[1], 2) + "]");
      ro.set("dom", layout.clusters[dec.dominant].name, AE.clusterColor(dec.dominant));
      ro.set("real", LR.fmtF(dec.realism, 2) + (dec.realism > 0.85 ? " (clean)" : " (mixture)"), dec.realism > 0.85 ? C.green : C.red);

      if (dec.realism > 0.85) {
        msg.show("On the manifold: one family dominates the blend and the decode is a crisp, plausible glyph. Codes the encoder actually produces decode well.", "good");
      } else if (dec.realism < 0.62) {
        msg.show("In a gap: no cluster claims this z, so the decoder smears families into a glyph that matches nothing in the training set. This is why you cannot generate by sampling a deterministic AE's latent space.", "bad");
      } else {
        msg.show("Edge of a cluster: the decode is already smearing away from the family. The further you drift from the training codes, the less the output means.", "info");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — a point per input vs a distribution per input
     ══════════════════════════════════════════════════════════ */
  LR.figs.pointdist = function (mount) {
    LR.header(
      mount,
      "Make every point wider",
      "The same latent codes, two encoders. A deterministic encoder commits to one point per input; a VAE encoder returns a Gaussian N(μ, σ²I) per input. Widen σ and watch the blobs tile the space between clusters."
    );

    const layout = AE.layout(0);
    const items = AE.latents(layout);
    let mode = 1; // 0 = points, 1 = distributions
    let sigma = 0.45;

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bPts = LR.button(grp, "deterministic: points", function () { mode = 0; render(); });
    const bDist = LR.button(grp, "VAE: distributions", function () { mode = 1; render(); }, "on");
    LR.slider(bar, "encoder σ", 0.15, 1.0, 0.05, sigma, function (v) { sigma = v; render(); }, (v) => LR.fmtF(v, 2));

    const cvo = LR.canvas(mount, 560, 470, {
      aria: "Latent space scatter; in distribution mode each code becomes a translucent Gaussian disc, and wider sigma makes the discs overlap and cover the gaps",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;
    const ro = LR.readout(mount, [
      { k: "cover", label: "latent area covered" },
      { k: "kl", label: "avg KL(qᵢ ‖ N(0, I))" },
    ]);
    const msg = LR.msg(mount);

    function coverage(sig) {
      // fraction of the [-4,4]² grid where the mixture density is non-negligible
      const Ngrid = 46;
      let hit = 0;
      for (let gi = 0; gi < Ngrid; gi++) {
        for (let gj = 0; gj < Ngrid; gj++) {
          const zx = -4 + (8 * (gi + 0.5)) / Ngrid;
          const zy = -4 + (8 * (gj + 0.5)) / Ngrid;
          let dens = 0;
          for (let k = 0; k < items.length; k++) {
            const dx = zx - items[k].z[0], dy = zy - items[k].z[1];
            dens += Math.exp(-(dx * dx + dy * dy) / (2 * sig * sig)) / (2 * Math.PI * sig * sig * items.length);
          }
          if (dens > 0.008) hit += 1;
        }
      }
      return hit / (Ngrid * Ngrid);
    }

    function render() {
      bPts.classList.toggle("on", mode === 0);
      bDist.classList.toggle("on", mode === 1);
      ctx.clearRect(0, 0, W, H);
      const lp = latentPlot(ctx, 62, 18, 400);
      const sx = lp.sc.sx, sy = lp.sc.sy;
      const pxPerUnit = 400 / 8;

      if (mode === 1) {
        items.forEach(function (it) {
          ctx.fillStyle = "rgba(26,26,26,0.10)";
          ctx.beginPath();
          ctx.arc(sx(it.z[0]), sy(it.z[1]), 2 * sigma * pxPerUnit, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "rgba(26,26,26,0.35)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(sx(it.z[0]), sy(it.z[1]), sigma * pxPerUnit, 0, Math.PI * 2);
          ctx.stroke();
        });
      }
      items.forEach(function (it) {
        LR.dot(ctx, sx(it.z[0]), sy(it.z[1]), mode === 0 ? 3.8 : 2.6, AE.clusterColor(it.c));
      });
      layout.clusters.forEach(function (c, ci) {
        ctx.fillStyle = AE.clusterColor(ci);
        ctx.font = "700 11.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(c.name, sx(c.mu[0]), sy(c.mu[1]) - 24);
      });

      if (mode === 0) {
        ro.set("cover", "isolated points (measure zero)", C.red);
        ro.set("kl", "undefined: no distribution to compare");
        msg.show("Each input owns exactly one point. Between the clusters: nothing. Sampling from this space almost surely lands where no training code lives.", "bad");
      } else {
        const cov = coverage(sigma);
        // average KL over items, isotropic sigma (real formula, computed live)
        let kl = 0;
        items.forEach(function (it) {
          kl += AE.klGauss(it.z, [sigma, sigma]);
        });
        kl /= items.length;
        ro.set("cover", Math.round(cov * 100) + "% of the panel", cov > 0.4 ? C.green : C.amber);
        ro.set("kl", LR.fmtF(kl, 2) + " nats", kl < 3 ? C.green : C.amber);
        if (sigma < 0.3) {
          msg.show("Narrow blobs: barely better than points. The decoder still only ever sees a thin shell around each code.", "info");
        } else {
          msg.show("Each input now claims a whole neighbourhood, and neighbourhoods overlap. During training the decoder must handle every z inside a blob, so nearby codes are forced to decode to similar outputs: the space between points gains meaning.", "good");
        }
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.1 — the reconstruction vs KL trade-off (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.kltrade = function (mount) {
    LR.header(
      mount,
      "One dial: reconstruction against the prior",
      "The KL weight β pulls every encoder distribution toward N(0, I). Slide it: at low β the latent is expressive but gappy; at high β it is smooth and sampleable but reconstructions blur. Both errors and the KL are computed live."
    );

    let beta = 0.15;
    let genSeed = 5;

    const bar = LR.controls(mount);
    LR.slider(bar, "KL weight β", 0, 1, 0.01, beta, function (v) { beta = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Resample generations ▸", function () { genSeed += 1; render(); }, "primary");

    const cvo = LR.canvas(mount, 840, 480, {
      aria: "Left: latent space where cluster blobs migrate toward the origin and widen as the KL weight rises, with the standard normal shown as dashed rings. Right: three reconstructions and six random generations, updating live",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;

    const ro = LR.readout(mount, [
      { k: "kl", label: "avg KL(qᵢ ‖ N(0, I))" },
      { k: "rec", label: "reconstruction MSE (pixels)" },
      { k: "gen", label: "prior samples on-manifold" },
    ]);
    const msg = LR.msg(mount);

    // fixed reconstruction probes: one item per family (indices into latents)
    const PROBE_IDX = [4, 20, 40];

    function render() {
      const layout = AE.layout(beta);
      const items = AE.latents(layout);
      const R = AE.renderers();

      ctx.clearRect(0, 0, W, H);
      const lp = latentPlot(ctx, 62, 18, 380);
      const sx = lp.sc.sx, sy = lp.sc.sy;
      const pxPerUnit = 380 / 8;

      // the target prior N(0, I): dashed rings at 1σ and 2σ
      [1, 2].forEach(function (r) {
        ctx.strokeStyle = "#b4b4b4";
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(sx(0), sy(0), r * pxPerUnit, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      });
      ctx.fillStyle = LR.C.faint;
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("N(0, I)", sx(0) + 2 * pxPerUnit + 5, sy(0));

      // encoder blobs
      items.forEach(function (it) {
        ctx.fillStyle = "rgba(26,26,26,0.09)";
        ctx.beginPath();
        ctx.arc(sx(it.z[0]), sy(it.z[1]), layout.sigmaEnc * pxPerUnit, 0, Math.PI * 2);
        ctx.fill();
      });
      items.forEach(function (it) {
        LR.dot(ctx, sx(it.z[0]), sy(it.z[1]), 2.8, AE.clusterColor(it.c));
      });

      // ---- metrics (all real) ----
      let kl = 0;
      items.forEach(function (it) {
        kl += AE.klGauss(it.z, [layout.sigmaEnc, layout.sigmaEnc]);
      });
      kl /= items.length;

      // reconstruction error: decode each item's mean and compare to its true glyph
      let rec = 0, count = 0;
      for (let k = 0; k < items.length; k += 4) {
        const it = items[k];
        const truth = R[it.c](it.s);
        const dec = AE.decode(it.z, layout);
        let e = 0;
        for (let p = 0; p < truth.length; p++) e += (truth[p] - dec.img[p]) ** 2;
        rec += e / truth.length;
        count += 1;
      }
      rec /= count;

      // generation quality: fraction of N(0, I) samples that land on the manifold
      const rand = LR.rng(424);
      let onMan = 0;
      const NS = 220;
      for (let s = 0; s < NS; s++) {
        const z = [LR.gauss(rand), LR.gauss(rand)];
        const lw = latentWeights(z, layout);
        if (lw.wsum > 0.45) onMan += 1;
      }
      const genFrac = onMan / NS;

      // ---- right panel: reconstructions and generations ----
      const rx = 490;
      ctx.fillStyle = C.text;
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("reconstructions  (input vs decode of μ)", rx, 30);
      PROBE_IDX.forEach(function (idx, i) {
        const it = items[idx];
        const truth = R[it.c](it.s);
        const dec = AE.decode(it.z, layout);
        const y = 42 + i * 78;
        AE.drawGlyph(ctx, truth, rx, y, 64);
        LR.arrow(ctx, rx + 74, y + 32, rx + 100, y + 32, "#999999", 1.6);
        AE.drawGlyph(ctx, dec.img, rx + 110, y, 64, { border: "#999999" });
      });

      ctx.fillStyle = C.text;
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillText("generations  (z ~ N(0, I), decoded)", rx, 300);
      const grand = LR.rng(1000 + genSeed);
      for (let g = 0; g < 6; g++) {
        const z = [LR.gauss(grand), LR.gauss(grand)];
        const dec = AE.decode(z, layout);
        const lw = latentWeights(z, layout);
        const gx = rx + (g % 3) * 76;
        const gy = 312 + Math.floor(g / 3) * 76;
        AE.drawGlyph(ctx, dec.img, gx, gy, 64, { border: lw.wsum > 0.45 ? C.green : C.red, borderW: 1.6 });
      }

      ro.set("kl", LR.fmtF(kl, 2) + " nats", kl < 1.5 ? C.green : C.amber);
      ro.set("rec", LR.fmtF(rec, 4), rec < 0.01 ? C.green : C.red);
      ro.set("gen", Math.round(genFrac * 100) + "% of 220 samples", genFrac > 0.8 ? C.green : C.red);

      if (beta < 0.2) {
        msg.show("Reconstruction paradise, generation desert: codes sit far from the origin with tiny σ, KL is huge, and most draws from N(0, I) land in the void between clusters. This is a compressor, not a generator.", "bad");
      } else if (beta > 0.8) {
        msg.show("The latent now matches the prior: nearly every N(0, I) sample decodes to something plausible. The price shows on the left column: clusters overlap, decodes of μ blur across families, reconstruction error is up. That is the trade.", "info");
      } else {
        msg.show("The middle ground: enough KL pressure to close most gaps, enough freedom to keep families distinct. Real VAE training is exactly this negotiation, weighted by the KL term.", "good");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.2 — the reparameterization trick (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.reparam = function (mount) {
    LR.header(
      mount,
      "z = μ + σ ⊙ ε, with the randomness quarantined",
      "μ and σ come from the encoder; ε is pure noise from N(0, 1). Move ε and watch z orbit the blob; move μ or σ and watch the same ε land somewhere new. Gradients flow through μ and σ because, given ε, z is plain arithmetic."
    );

    let mu = [0.5, -1.0];
    let sg = [0.8, 0.3];
    let eps = [1.2, -0.5];

    const bar = LR.controls(mount);
    const m1 = LR.slider(bar, "μ₁", -2, 2, 0.05, mu[0], function (v) { mu[0] = v; render(); }, (v) => LR.fmtF(v, 2));
    const m2 = LR.slider(bar, "μ₂", -2, 2, 0.05, mu[1], function (v) { mu[1] = v; render(); }, (v) => LR.fmtF(v, 2));
    const s1 = LR.slider(bar, "σ₁", 0.1, 1.5, 0.05, sg[0], function (v) { sg[0] = v; render(); }, (v) => LR.fmtF(v, 2));
    const s2 = LR.slider(bar, "σ₂", 0.1, 1.5, 0.05, sg[1], function (v) { sg[1] = v; render(); }, (v) => LR.fmtF(v, 2));
    const bar2 = LR.controls(mount);
    const e1 = LR.slider(bar2, "ε₁", -3, 3, 0.05, eps[0], function (v) { eps[0] = v; render(); }, (v) => LR.fmtF(v, 2));
    const e2 = LR.slider(bar2, "ε₂", -3, 3, 0.05, eps[1], function (v) { eps[1] = v; render(); }, (v) => LR.fmtF(v, 2));
    let epsSeed = 17;
    LR.button(bar2, "Sample ε ~ N(0, 1) ▸", function () {
      const rand = LR.rng(epsSeed += 1);
      eps = [LR.gauss(rand), LR.gauss(rand)];
      e1.set(eps[0]); e2.set(eps[1]);
      render();
    }, "primary");

    const cvo = LR.canvas(mount, 620, 450, {
      aria: "Latent plane showing the encoder's Gaussian blob at mu with axis lengths sigma, a ghost cloud of samples, and the current sample z equal to mu plus sigma times epsilon",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;

    const ro = LR.readout(mount, [
      { k: "eps", label: "ε" },
      { k: "z", label: "z = μ + σ ⊙ ε" },
      { k: "kl", label: "KL(q ‖ N(0, I))" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      ctx.clearRect(0, 0, W, H);
      const lp = latentPlot(ctx, 62, 18, 380);
      const sx = lp.sc.sx, sy = lp.sc.sy;
      const pxPerUnit = 380 / 8;

      // ghost samples of q (seeded, real Box-Muller draws through the same formula)
      const rand = LR.rng(300);
      for (let i = 0; i < 260; i++) {
        const gz = [mu[0] + sg[0] * LR.gauss(rand), mu[1] + sg[1] * LR.gauss(rand)];
        ctx.fillStyle = "rgba(26,26,26,0.16)";
        ctx.fillRect(sx(gz[0]) - 1.4, sy(gz[1]) - 1.4, 2.8, 2.8);
      }

      // 1σ and 2σ ellipses of q
      [1, 2].forEach(function (k) {
        ctx.strokeStyle = k === 1 ? C.orange : "rgba(26,26,26,0.45)";
        ctx.lineWidth = k === 1 ? 2 : 1.2;
        ctx.beginPath();
        ctx.ellipse(sx(mu[0]), sy(mu[1]), k * sg[0] * pxPerUnit, k * sg[1] * pxPerUnit, 0, 0, Math.PI * 2);
        ctx.stroke();
      });

      // prior rings for reference
      ctx.strokeStyle = "#c6c6c6";
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.arc(sx(0), sy(0), pxPerUnit, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = LR.C.faint;
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("N(0, I)", sx(0) + pxPerUnit + 4, sy(0) - 4);

      // mu marker and the arrow sigma*eps
      const z = [mu[0] + sg[0] * eps[0], mu[1] + sg[1] * eps[1]];
      LR.dot(ctx, sx(mu[0]), sy(mu[1]), 5, C.purple, "#ffffff");
      ctx.fillStyle = C.purple;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText("μ", sx(mu[0]) - 9, sy(mu[1]) - 8);
      LR.arrow(ctx, sx(mu[0]), sy(mu[1]), sx(z[0]), sy(z[1]), C.green, 2);
      LR.dot(ctx, sx(z[0]), sy(z[1]), 6.5, "#ffffff", C.red);
      LR.dot(ctx, sx(z[0]), sy(z[1]), 2.8, C.red);
      ctx.fillStyle = C.red;
      ctx.textAlign = "left";
      ctx.fillText("z", sx(z[0]) + 10, sy(z[1]) + 4);
      ctx.fillStyle = C.green;
      ctx.font = "600 11px Inter, sans-serif";
      ctx.fillText("σ ⊙ ε", (sx(mu[0]) + sx(z[0])) / 2 + 8, (sy(mu[1]) + sy(z[1])) / 2);

      ro.set("eps", "[" + LR.fmtF(eps[0], 2) + ", " + LR.fmtF(eps[1], 2) + "]");
      ro.set("z", "[" + LR.fmtF(z[0], 2) + ", " + LR.fmtF(z[1], 2) + "]", C.red);
      ro.set("kl", LR.fmtF(AE.klGauss(mu, sg), 3) + " nats", C.purple);

      const dist = Math.hypot(mu[0], mu[1]);
      if (dist < 0.4 && Math.abs(sg[0] - 1) < 0.2 && Math.abs(sg[1] - 1) < 0.2) {
        msg.show("q now sits on the prior: KL near zero. This is where the KL term wants every encoder output to live, so that sampling ε alone (decoder input z = ε) generates data.", "good");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.3 — reparameterize and vae_loss, executed for real
     ══════════════════════════════════════════════════════════ */
  LR.figs.vaecode = function (mount) {
    LR.header(
      mount,
      "reparameterize and the KL term, executed",
      "Set μ and σ, press Run: five real ε draws go through z = μ + σ·ε, and the KL term of vae_loss is computed from the closed form. The reconstruction term uses the fixed example pair from the text."
    );

    let mu = 0.5, sg = 0.8, runSeed = 20;

    const bar = LR.controls(mount);
    const mCtl = LR.slider(bar, "mu", -2, 2, 0.05, mu, function (v) { mu = v; }, (v) => LR.fmtF(v, 2));
    const sCtl = LR.slider(bar, "sigma", 0.1, 2, 0.05, sg, function (v) { sg = v; }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Reset (0.5, 0.8)", function () { mu = 0.5; sg = 0.8; mCtl.set(mu); sCtl.set(sg); });
    LR.button(bar, "Run ▸", run, "primary");

    const out = LR.el("pre", "run-output attn-runout", "press Run to execute reparameterize(mu, sigma) and vae_loss(...)");
    mount.appendChild(out);

    // fixed reconstruction example (matches the numbers quoted in the text)
    const X = [1.0, -0.5], XH = [0.9, -0.3];

    function run() {
      runSeed += 1;
      const rand = LR.rng(runSeed);
      const L = [];
      L.push(">>> mu, sigma = " + LR.fmtF(mu, 2) + ", " + LR.fmtF(sg, 2));
      L.push(">>> for _ in range(5): print(reparameterize(mu, sigma))");
      for (let i = 0; i < 5; i++) {
        const e = LR.gauss(rand);
        const z = mu + sg * e;
        L.push("  eps = " + pad(LR.fmtF(e, 3), 7) + "   z = " + LR.fmtF(mu, 2) + " + " + LR.fmtF(sg, 2) + " * " + LR.fmtF(e, 3) + " = " + LR.fmtF(z, 3));
      }
      const kl = 0.5 * (mu * mu + sg * sg - 1 - Math.log(sg * sg));
      let recon = 0;
      for (let i = 0; i < X.length; i++) recon += (X[i] - XH[i]) ** 2;
      recon /= X.length;
      L.push(">>> kl_term(mu, sigma)          # 0.5*(mu^2 + sigma^2 - 1 - log sigma^2)");
      L.push("  " + LR.fmtF(kl, 4) + " nats");
      L.push(">>> vae_loss(x, x_hat, mu, sigma)   # x = [1.0, -0.5], x_hat = [0.9, -0.3]");
      L.push("  recon (MSE) = " + LR.fmtF(recon, 4) + "   kl = " + LR.fmtF(kl, 4) + "   total = " + LR.fmtF(recon + kl, 4));
      if (Math.abs(mu) < 0.05 && Math.abs(sg - 1) < 0.05) {
        L.push("# mu = 0, sigma = 1: the KL term is exactly 0. q equals the prior.");
      }
      out.textContent = L.join("\n");
    }
    function pad(s, n) { s = String(s); while (s.length < n) s = " " + s; return s; }
  };

  /* ══════════════════════════════════════════════════════════
     Fig 6.4 — generate by sampling, interpolate by walking
     ══════════════════════════════════════════════════════════ */
  LR.figs.generate = function (mount) {
    LR.header(
      mount,
      "Delete the encoder, keep the decoder",
      "The latent here is the high-β layout from Fig 6.1: smooth and prior-matched. Left: walk the straight line between two codes and decode every step. Right: draw z ~ N(0, I) and decode, which is the whole generative recipe."
    );

    const layout = AE.layout(0.85);
    const items = AE.latents(layout);
    // endpoints: a disc code and a cross code
    const zA = items[6].z, zB = items[24].z;

    let t = 0.5;
    let genSeed = 9;

    const bar = LR.controls(mount);
    LR.slider(bar, "interpolation t", 0, 1, 0.01, t, function (v) { t = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Generate 9 ▸", function () { genSeed += 1; render(); }, "primary");

    const cvo = LR.canvas(mount, 840, 400, {
      aria: "Left: seven decoded glyphs along the straight latent path between a disc code and a cross code, morphing smoothly. Right: a three by three grid of glyphs decoded from standard normal samples",
    });
    const ctx = cvo.ctx, W = cvo.W, H = cvo.H;

    const ro = LR.readout(mount, [
      { k: "z", label: "z(t) = (1−t)·z_A + t·z_B" },
      { k: "fam", label: "dominant family" },
      { k: "ok", label: "grid samples on-manifold" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      ctx.clearRect(0, 0, W, H);

      // interpolation strip
      ctx.fillStyle = C.text;
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("latent interpolation: disc → cross", 24, 26);
      const steps = 7;
      for (let i = 0; i < steps; i++) {
        const tt = i / (steps - 1);
        const z = [(1 - tt) * zA[0] + tt * zB[0], (1 - tt) * zA[1] + tt * zB[1]];
        const dec = AE.decode(z, layout);
        AE.drawGlyph(ctx, dec.img, 24 + i * 62, 40, 54);
        ctx.fillStyle = LR.C.faint;
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("t=" + LR.fmtF(tt, 2), 24 + i * 62 + 27, 108);
      }
      // the big current decode
      const zc = [(1 - t) * zA[0] + t * zB[0], (1 - t) * zA[1] + t * zB[1]];
      const decC = AE.decode(zc, layout);
      AE.drawGlyph(ctx, decC.img, 130, 140, 180, { border: C.orange, borderW: 2 });
      ctx.fillStyle = LR.C.muted;
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("gθ(z(t)) at your slider position", 220, 340);

      // generation grid
      ctx.fillStyle = C.text;
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("z ~ N(0, I) → gθ(z)", 520, 26);
      const rand = LR.rng(700 + genSeed);
      let ok = 0;
      for (let g = 0; g < 9; g++) {
        const z = [LR.gauss(rand), LR.gauss(rand)];
        const dec = AE.decode(z, layout);
        const lw = latentWeights(z, layout);
        if (lw.wsum > 0.45) ok += 1;
        const gx = 520 + (g % 3) * 96;
        const gy = 40 + Math.floor(g / 3) * 96;
        AE.drawGlyph(ctx, dec.img, gx, gy, 84, { border: lw.wsum > 0.45 ? C.green : C.red, borderW: 1.6 });
      }
      ctx.fillStyle = LR.C.muted;
      ctx.font = "600 11.5px Inter, sans-serif";
      ctx.fillText("each press of Generate is nine fresh Gaussian draws", 520, 340);

      ro.set("z", "[" + LR.fmtF(zc[0], 2) + ", " + LR.fmtF(zc[1], 2) + "]");
      ro.set("fam", layout.clusters[decC.dominant].name + " (" + LR.fmtF(decC.realism, 2) + ")", AE.clusterColor(decC.dominant));
      ro.set("ok", ok + " / 9", ok >= 8 ? C.green : C.amber);

      msg.show("This is the payoff of the whole lesson: because the KL term herded every q(z | x) toward N(0, I), a plain Gaussian sample is a legitimate latent code, and the decoder alone is a generative model. No input x anywhere in the loop.", "good");
    }
    render();
  };
})();
