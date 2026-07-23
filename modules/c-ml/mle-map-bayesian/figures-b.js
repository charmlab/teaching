/* ══════════════════════════════════════════════════════════════
   figures-b.js — MLE / MAP / Bayesian lesson, sections 3–5
   Fig 3.1 pouch (signature) · Fig 3.2 zerofreq · Fig 4.1 conjugacy
   Fig 4.2 betaupdate (signature) · Fig 5.1 threeway · Fig 5.2 betacode
   Uses the shared estimation core LR.est from lesson-core.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const EST = LR.est;

  /* shared: density curve on a plot scaffold */
  function curve(ctx, sx, sy, f, xmin, xmax, color, width, dash) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 2.4;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    const STEPS = 240;
    let started = false;
    for (let i = 0; i <= STEPS; i++) {
      const x = xmin + ((xmax - xmin) * i) / STEPS;
      const y = f(x);
      if (!isFinite(y)) { started = false; continue; }
      const px = sx(x), py = sy(y);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    if (dash) ctx.setLineDash([]);
  }

  function vline(ctx, sx, sy, x, ytop, color, label, offset) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(sx(x), sy(0));
    ctx.lineTo(sx(x), sy(ytop));
    ctx.stroke();
    ctx.setLineDash([]);
    if (label) {
      ctx.fillStyle = color;
      ctx.font = "700 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, sx(x) + (offset || 0), sy(ytop) - 6);
    }
  }

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — the coins-in-a-pouch predictor (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.pouch = function (mount) {
    LR.header(
      mount,
      "Which coin type did we draw?",
      "Score every type: s_k = π_k · p(D_new | θ̂_k). Edit any count or frequency; the argmax winner is outlined. Switch estimators and watch the winner flip."
    );

    // deep copy of the shared table so edits stay local to this figure
    const coins = EST.POUCH.map((c) => ({ name: c.name, nh: c.nh, nt: c.nt, pi: c.pi }));
    let newH = EST.POUCH_NEW.h, newT = EST.POUCH_NEW.t;
    let mode = "mle"; // 'mle' | 'map' | 'bayes'
    const A = 2, B = 2; // Beta(2,2) prior for MAP and Bayesian modes

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bMLE = LR.button(grp, "MLE plug-in", function () { mode = "mle"; render(); }, "on");
    const bMAP = LR.button(grp, "MAP plug-in · Beta(2,2)", function () { mode = "map"; render(); });
    const bBAY = LR.button(grp, "fully Bayesian · Beta(2,2)", function () { mode = "bayes"; render(); });

    const bar2 = LR.controls(mount);
    LR.numInput(bar2, "new heads ", newH, 0, 50, function (v) { newH = v; render(); });
    LR.numInput(bar2, "new tails ", newT, 0, 50, function (v) { newT = v; render(); });
    LR.button(bar2, "Reset table ⟲", function () {
      EST.POUCH.forEach(function (c, i) {
        coins[i].nh = c.nh; coins[i].nt = c.nt; coins[i].pi = c.pi;
      });
      newH = EST.POUCH_NEW.h; newT = EST.POUCH_NEW.t;
      render();
    });

    const tblWrap = LR.el("div", "wtable-wrap");
    mount.appendChild(tblWrap);
    const ro = LR.readout(mount, [
      { k: "dnew", label: "D_new" },
      { k: "pisum", label: "Σ π_k" },
      { k: "winner", label: "prediction Ĉ_new" },
    ]);
    const msg = LR.msg(mount);

    function estTheta(c) {
      return mode === "map" ? EST.mapEst(c.nh, c.nt, A, B) : EST.mle(c.nh, c.nt);
    }
    function fit(c) {
      if (mode === "bayes") return EST.predDataset(c.nh, c.nt, A, B, newH, newT);
      return EST.binomPmf(newH + newT, newH, estTheta(c));
    }

    function editCell(td, value, onset, aria) {
      td.textContent = "";
      const inp = document.createElement("input");
      inp.type = "number";
      inp.value = value;
      inp.min = 0;
      inp.step = aria.indexOf("frequency") >= 0 ? 0.05 : 1;
      inp.className = "num-input";
      inp.style.width = "62px";
      inp.setAttribute("aria-label", aria);
      inp.addEventListener("input", function () {
        const v = parseFloat(inp.value);
        if (!isNaN(v) && v >= 0) { onset(v); renderTable(); }
      });
      td.appendChild(inp);
    }

    function renderTable() {
      // scores
      const rows = coins.map(function (c) {
        const th = estTheta(c);
        const f = fit(c);
        return { c, th, f, s: c.pi * f };
      });
      let win = 0;
      rows.forEach(function (r, i) { if (r.s > rows[win].s) win = i; });

      tblWrap.innerHTML = "";
      const tbl = LR.el("table", "mtx");
      const head = LR.el("tr");
      const modeLab = mode === "mle" ? "θ̂_ML" : mode === "map" ? "θ̂_MAP" : "posterior";
      const fitLab = mode === "bayes" ? "p(D_new | D_k)" : "p(D_new | " + modeLab + ")";
      ["type", "history N_H", "history N_T", "π_k", modeLab, fitLab, "score s_k"].forEach(function (h) {
        head.appendChild(LR.el("th", null, h));
      });
      tbl.appendChild(head);

      rows.forEach(function (r, i) {
        const tr = LR.el("tr");
        tr.appendChild(LR.el("th", null, r.c.name));
        const tdNH = LR.el("td"); editCell(tdNH, r.c.nh, (v) => (r.c.nh = Math.round(v)), r.c.name + " historical heads count"); tr.appendChild(tdNH);
        const tdNT = LR.el("td"); editCell(tdNT, r.c.nt, (v) => (r.c.nt = Math.round(v)), r.c.name + " historical tails count"); tr.appendChild(tdNT);
        const tdPi = LR.el("td"); editCell(tdPi, r.c.pi, (v) => (r.c.pi = v), r.c.name + " type frequency"); tr.appendChild(tdPi);
        const thTd = LR.el("td", null, mode === "bayes"
          ? "Beta(" + (r.c.nh + A) + ", " + (r.c.nt + B) + ")"
          : (isFinite(r.th) ? LR.fmtF(r.th, 4) : "–"));
        tr.appendChild(thTd);
        tr.appendChild(LR.el("td", null, isFinite(r.f) ? r.f.toFixed(5) : "–"));
        const sTd = LR.el("td", null, isFinite(r.s) ? r.s.toFixed(5) : "–");
        if (i === win) sTd.classList.add("pouch-win");
        tr.appendChild(sTd);
        tbl.appendChild(tr);
      });
      tblWrap.appendChild(tbl);

      const piSum = coins.reduce((s, c) => s + c.pi, 0);
      ro.set("dnew", "(" + newH + " heads, " + newT + " tails)");
      ro.set("pisum", LR.fmtF(piSum, 2) + (Math.abs(piSum - 1) > 0.005 ? "  (should be 1)" : ""), Math.abs(piSum - 1) > 0.005 ? C.red : C.green);
      ro.set("winner", rows[win].c.name, C.green);

      const modeName = mode === "mle" ? "MLE plug-in" : mode === "map" ? "MAP plug-in" : "fully Bayesian";
      if (mode === "mle") {
        msg.show(modeName + ": each type is judged by its best-fit point estimate. Coin 3's 10-flip history gives it a seductive θ̂ = 0.9 that matches the new data almost perfectly.", "info");
      } else if (mode === "map") {
        msg.show(modeName + ": the Beta(2,2) prior tempers every estimate toward 0.5, and thin histories move the most. Coin 3 deflates to 10/12 ≈ 0.833 while coin 2 barely budges, and the argmax changes hands.", "good");
      } else {
        msg.show(modeName + ": each score integrates p(D_new | θ) over the type's whole posterior via the Beta-function ratio (section 5). Coin 3's wide posterior dilutes its score; the verdict agrees with MAP.", "good");
      }
    }

    function render() {
      bMLE.classList.toggle("on", mode === "mle");
      bMAP.classList.toggle("on", mode === "map");
      bBAY.classList.toggle("on", mode === "bayes");
      renderTable();
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.2 — the zero-frequency trap
     ══════════════════════════════════════════════════════════ */
  LR.figs.zerofreq = function (mount) {
    LR.header(
      mount,
      "Two flips can declare an outcome impossible",
      "Flip the coin (true bias 0.7, hidden from the estimators). While the record is all heads, the MLE assigns p(T) = 0 exactly. The prior refuses."
    );

    const rand = LR.rng(16);
    const TRUE_THETA = 0.7;
    let flips = [];
    let usePrior = false;
    const A = 2, B = 2;

    const bar = LR.controls(mount);
    LR.button(bar, "Flip ▸", function () { doFlips(1); }, "primary");
    LR.button(bar, "Flip 5 ▸▸", function () { doFlips(5); });
    LR.button(bar, "Reset ⟲", function () { flips = []; render(); });
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bOff = LR.button(grp, "MLE only", function () { usePrior = false; render(); }, "on");
    const bOn = LR.button(grp, "add Beta(2,2) prior", function () { usePrior = true; render(); });

    const strip = LR.el("div", "flip-strip");
    strip.setAttribute("aria-label", "sequence of observed flips");
    mount.appendChild(strip);

    const bars = LR.el("div", "attn-weightbar");
    mount.appendChild(bars);

    const ro = LR.readout(mount, [
      { k: "rec", label: "record (N_H, N_T)" },
      { k: "mle", label: "θ̂ MLE" },
      { k: "map", label: "θ̂ MAP" },
      { k: "pt", label: "estimated p(T)" },
    ]);
    const msg = LR.msg(mount);

    function doFlips(n) {
      for (let i = 0; i < n; i++) flips.push(rand() < TRUE_THETA ? "H" : "T");
      render();
    }

    function barRow(label, value, color) {
      const item = LR.el("div", "wb-item");
      item.appendChild(LR.el("div", "wb-lab", label));
      const track = LR.el("div", "wb-track");
      const fill = LR.el("div", "wb-fill");
      fill.style.width = Math.round(value * 100) + "%";
      fill.style.background = color;
      track.appendChild(fill);
      item.appendChild(track);
      item.appendChild(LR.el("div", "wb-val", LR.fmtF(value, 3)));
      return item;
    }

    function render() {
      bOff.classList.toggle("on", !usePrior);
      bOn.classList.toggle("on", usePrior);

      strip.innerHTML = "";
      if (flips.length === 0) {
        strip.appendChild(LR.el("span", "fig-sub", "no flips yet: press Flip"));
      }
      flips.slice(-30).forEach(function (f) {
        strip.appendChild(LR.el("span", "flip-chip " + f.toLowerCase(), f));
      });
      if (flips.length > 30) {
        strip.appendChild(LR.el("span", "fig-sub", "(+" + (flips.length - 30) + " earlier)"));
      }

      const nh = flips.filter((f) => f === "H").length;
      const nt = flips.length - nh;
      const mle = EST.mle(nh, nt);
      const map = EST.mapEst(nh, nt, A, B);

      bars.innerHTML = "";
      if (flips.length > 0) {
        bars.appendChild(barRow("MLE p(H)", mle, C.green));
        bars.appendChild(barRow("MLE p(T)", 1 - mle, nt === 0 ? C.red : C.green));
        if (usePrior) {
          bars.appendChild(barRow("MAP p(H)", map, C.orange));
          bars.appendChild(barRow("MAP p(T)", 1 - map, C.orange));
        }
      }

      ro.set("rec", flips.length ? "(" + nh + ", " + nt + ")" : "–");
      ro.set("mle", isFinite(mle) ? LR.fmtF(mle, 3) : "–", C.green);
      ro.set("map", usePrior && isFinite(map) ? LR.fmtF(map, 3) : "prior off", C.orange);
      ro.set("pt", flips.length
        ? (usePrior ? LR.fmtF(1 - map, 3) + " (MAP)" : LR.fmtF(1 - mle, 3) + " (MLE)")
        : "–", usePrior ? C.orange : (nt === 0 && flips.length ? C.red : C.green));

      if (!flips.length) { msg.hide(); return; }
      if (nt === 0 && !usePrior) {
        msg.show("All " + nh + " flips were heads, so θ̂_ML = " + nh + "/" + nh + " = 1 and the model claims p(T) = 0: tails is impossible. Any future computation that multiplies by p(T) is now zeroed out. Switch the prior on.", "bad");
      } else if (nt === 0 && usePrior) {
        msg.show("Same all-heads record, but the Beta(2,2) prior contributes one phantom flip of each kind: θ̂_MAP = (" + nh + "+1)/(" + nh + "+2) = " + LR.fmtF(map, 3) + ", so p(T) = " + LR.fmtF(1 - map, 3) + " stays alive.", "good");
      } else {
        msg.show("A tail has arrived, so the MLE cliff is behind us. Note how MLE and MAP now track each other more closely as flips accumulate: with real evidence for both outcomes, the prior matters less and less.", "info");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — conjugacy: Beta × Bernoulli = Beta
     ══════════════════════════════════════════════════════════ */
  LR.figs.conjugacy = function (mount) {
    LR.header(
      mount,
      "Prior × likelihood = posterior, in the same family",
      "The posterior's exponents are the prior's plus the counts: Beta(a,b) × θ^N_H (1−θ)^N_T ∝ Beta(N_H + a, N_T + b)."
    );

    let a = 2, b = 2, nh = 6, nt = 2;

    const bar = LR.controls(mount);
    const aS = LR.slider(bar, "prior a", 1, 12, 0.5, a, function (v) { a = v; render(); }, (v) => LR.fmt(v, 1));
    const bS = LR.slider(bar, "prior b", 1, 12, 0.5, b, function (v) { b = v; render(); }, (v) => LR.fmt(v, 1));
    LR.numInput(bar, "N_H ", nh, 0, 100, function (v) { nh = v; render(); });
    LR.numInput(bar, "N_T ", nt, 0, 100, function (v) { nt = v; render(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 320, {
      aria: "Overlaid curves: Beta prior, normalized Bernoulli likelihood, and the resulting Beta posterior",
    });

    const legend = LR.el("div", "legend-row");
    legend.innerHTML =
      '<span class="lg"><span class="sw" style="background:' + C.purple + '"></span> prior Beta(a, b)</span>' +
      '<span class="lg"><span class="sw" style="background:#9aa0a6"></span> likelihood (normalized for display)</span>' +
      '<span class="lg"><span class="sw" style="background:' + C.orange + '"></span> posterior Beta(N_H + a, N_T + b)</span>';
    mount.appendChild(legend);

    const ro = LR.readout(mount, [
      { k: "prior", label: "prior" },
      { k: "post", label: "posterior" },
      { k: "family", label: "family check" },
    ]);

    function render() {
      const pa = nh + a, pb = nt + b;
      // normalized likelihood is itself Beta(nh+1, nt+1); computed, not asserted
      const likA = nh + 1, likB = nt + 1;

      let ymax = 0;
      for (let i = 1; i < 400; i++) {
        const x = i / 400;
        ymax = Math.max(ymax, EST.betaPdf(x, a, b), EST.betaPdf(x, likA, likB), EST.betaPdf(x, pa, pb));
      }
      ymax *= 1.12;

      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 58, y0: 20, w: W - 84, h: H - 66,
        xmin: 0, xmax: 1, ymin: 0, ymax: ymax,
        xlabel: "θ", ylabel: "density",
      };
      const { sx, sy } = LR.plot(ctx, P);

      const clip = (f) => (x) => Math.min(f(x), ymax);
      curve(ctx, sx, sy, clip((x) => EST.betaPdf(x, a, b)), 0, 1, C.purple, 2.2, [6, 4]);
      curve(ctx, sx, sy, clip((x) => EST.betaPdf(x, likA, likB)), 0, 1, "#9aa0a6", 2);
      // posterior filled
      ctx.beginPath();
      ctx.moveTo(sx(0), sy(0));
      for (let i = 0; i <= 240; i++) {
        const x = i / 240;
        ctx.lineTo(sx(x), sy(Math.min(EST.betaPdf(x, pa, pb), ymax)));
      }
      ctx.lineTo(sx(1), sy(0));
      ctx.closePath();
      ctx.fillStyle = "rgba(232, 89, 12, 0.10)";
      ctx.fill();
      curve(ctx, sx, sy, clip((x) => EST.betaPdf(x, pa, pb)), 0, 1, C.orange, 2.8);

      ro.set("prior", "Beta(" + LR.fmt(a, 1) + ", " + LR.fmt(b, 1) + ")", C.purple);
      ro.set("post", "Beta(" + LR.fmt(pa, 1) + ", " + LR.fmt(pb, 1) + ")", C.orange);
      ro.set("family", "Beta in, Beta out: exponents add", C.green);
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.2 — the prior-to-posterior updater (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.betaupdate = function (mount) {
    LR.header(
      mount,
      "Watch a prior wash out",
      "Pick a prior, stream in flips from a hidden coin (true θ = 0.7), and watch the posterior Beta(N_H + a, N_T + b) sharpen while the MLE, MAP, and predictive-mean markers converge."
    );

    const TRUE_THETA = 0.7;
    const rand = LR.rng(779);
    let a = 2, b = 2, nh = 0, nt = 0;

    const bar = LR.controls(mount);
    const aS = LR.slider(bar, "prior a", 1, 20, 1, a, function (v) { a = v; render(); }, (v) => LR.fmt(v, 0));
    const bS = LR.slider(bar, "prior b", 1, 20, 1, b, function (v) { b = v; render(); }, (v) => LR.fmt(v, 0));
    LR.button(bar, "Flip 1 ▸", function () { doFlips(1); }, "primary");
    LR.button(bar, "Flip 10 ▸▸", function () { doFlips(10); });
    LR.button(bar, "Flip 100 ▸▸▸", function () { doFlips(100); });
    LR.button(bar, "Reset data ⟲", function () { nh = 0; nt = 0; render(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 820, 340, {
      aria: "Beta prior and posterior curves over theta, with MLE, MAP, and posterior-mean markers that converge as flips accumulate",
    });

    const legend = LR.el("div", "legend-row");
    legend.innerHTML =
      '<span class="lg"><span class="sw" style="background:#b197fc"></span> prior (dashed)</span>' +
      '<span class="lg"><span class="sw" style="background:' + C.orange + '"></span> posterior</span>' +
      '<span class="lg"><span class="sw" style="background:' + C.green + '"></span> MLE</span>' +
      '<span class="lg"><span class="sw" style="background:' + C.orange + '"></span> MAP (mode)</span>' +
      '<span class="lg"><span class="sw" style="background:' + C.purple + '"></span> predictive mean</span>';
    mount.appendChild(legend);

    const ro = LR.readout(mount, [
      { k: "data", label: "data (N_H, N_T)" },
      { k: "post", label: "posterior" },
      { k: "mle", label: "θ̂ MLE" },
      { k: "map", label: "θ̂ MAP" },
      { k: "mean", label: "predictive mean" },
    ]);
    const msg = LR.msg(mount);

    function doFlips(n) {
      for (let i = 0; i < n; i++) {
        if (rand() < TRUE_THETA) nh += 1; else nt += 1;
      }
      render();
    }

    function render() {
      const pa = nh + a, pb = nt + b;
      const mle = EST.mle(nh, nt);
      const map = EST.mapEst(nh, nt, a, b);
      const mean = EST.postPred(nh, nt, a, b);

      let ymax = 0;
      for (let i = 1; i < 400; i++) {
        const x = i / 400;
        ymax = Math.max(ymax, EST.betaPdf(x, pa, pb), EST.betaPdf(x, a, b));
      }
      ymax *= 1.15;

      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 58, y0: 22, w: W - 84, h: H - 68,
        xmin: 0, xmax: 1, ymin: 0, ymax: ymax,
        xlabel: "θ", ylabel: "density",
      };
      const { sx, sy } = LR.plot(ctx, P);
      const clip = (f) => (x) => Math.min(f(x), ymax);

      // true theta tick
      ctx.strokeStyle = "#c8c8c8";
      ctx.lineWidth = 1.4;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(TRUE_THETA), sy(0));
      ctx.lineTo(sx(TRUE_THETA), P.y0);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.faint;
      ctx.font = "600 10.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("true θ = 0.7", sx(TRUE_THETA), P.y0 + 10);

      // prior (dashed light purple)
      curve(ctx, sx, sy, clip((x) => EST.betaPdf(x, a, b)), 0, 1, "#b197fc", 2, [6, 4]);
      // posterior filled
      ctx.beginPath();
      ctx.moveTo(sx(0), sy(0));
      for (let i = 0; i <= 240; i++) {
        const x = i / 240;
        ctx.lineTo(sx(x), sy(Math.min(EST.betaPdf(x, pa, pb), ymax)));
      }
      ctx.lineTo(sx(1), sy(0));
      ctx.closePath();
      ctx.fillStyle = "rgba(232, 89, 12, 0.10)";
      ctx.fill();
      curve(ctx, sx, sy, clip((x) => EST.betaPdf(x, pa, pb)), 0, 1, C.orange, 2.8);

      // markers on a common axis
      const top = Math.min(EST.betaPdf(isFinite(map) ? map : mean, pa, pb), ymax);
      if (isFinite(mle)) vline(ctx, sx, sy, mle, top * 0.9, C.green, "MLE " + LR.fmtF(mle, 3), -50);
      if (isFinite(map)) vline(ctx, sx, sy, map, top, C.orange, "MAP " + LR.fmtF(map, 3), 0);
      vline(ctx, sx, sy, mean, top * 0.78, C.purple, "mean " + LR.fmtF(mean, 3), 52);

      ro.set("data", "(" + nh + ", " + nt + ")  N = " + (nh + nt));
      ro.set("post", "Beta(" + pa + ", " + pb + ")", C.orange);
      ro.set("mle", isFinite(mle) ? LR.fmtF(mle, 4) : "undefined (no data yet)", C.green);
      ro.set("map", isFinite(map) ? LR.fmtF(map, 4) : "–", C.orange);
      ro.set("mean", LR.fmtF(mean, 4), C.purple);

      const N = nh + nt;
      if (N === 0) {
        msg.show("No data: the posterior IS the prior, Beta(" + a + ", " + b + "). Everything you believe comes from pseudo-counts.", "info");
      } else if (N < 20) {
        const spread = Math.max(Math.abs((isFinite(mle) ? mle : mean) - mean), Math.abs((isFinite(map) ? map : mean) - mean));
        msg.show("N = " + N + ": the prior still carries real weight and the three estimators disagree by up to " + LR.fmtF(spread, 3) + ". This spread is the small-data regime where the choice of estimator matters.", "info");
      } else {
        msg.show("N = " + N + ": the counts dominate the pseudo-counts. The posterior is tightening around the true bias and the three markers are converging; the prior has all but washed out.", "good");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — MLE vs MAP vs Bayesian, converging in N
     ══════════════════════════════════════════════════════════ */
  LR.figs.threeway = function (mount) {
    LR.header(
      mount,
      "Three estimators, one destiny",
      "Left: the posterior at the current N with all three estimates marked. Right: each estimate as a function of N at the fixed heads ratio; slide N and watch them converge."
    );

    let ratio = 1.0;  // heads fraction: start at the dramatic all-heads case
    let N = 2;        // start at the zero-frequency example (2, 0)
    let a = 2, b = 2;

    const bar = LR.controls(mount);
    const nS = LR.slider(bar, "N (flips)", 1, 500, 1, N, function (v) { N = v; render(); }, (v) => LR.fmt(v, 0));
    const rS = LR.slider(bar, "heads ratio", 0, 1, 0.05, ratio, function (v) { ratio = v; render(); }, (v) => LR.fmtF(v, 2));
    const aS = LR.slider(bar, "prior a", 1, 10, 1, a, function (v) { a = v; render(); }, (v) => LR.fmt(v, 0));
    const bS = LR.slider(bar, "prior b", 1, 10, 1, b, function (v) { b = v; render(); }, (v) => LR.fmt(v, 0));

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left);
    grid.appendChild(right);
    left.appendChild(LR.el("div", "pane-label", "posterior at current N"));
    right.appendChild(LR.el("div", "pane-label", "estimates vs N (ratio fixed)"));

    const L = LR.canvas(left, 400, 260, { aria: "Posterior density at the current sample size with MLE, MAP, and mean marked" });
    const R = LR.canvas(right, 400, 260, { aria: "MLE, MAP, and posterior mean plotted against sample size, converging as N grows" });

    const ro = LR.readout(mount, [
      { k: "counts", label: "counts (N_H, N_T)" },
      { k: "mle", label: "θ̂ MLE" },
      { k: "map", label: "θ̂ MAP" },
      { k: "pred", label: "p(x_new = H | D)" },
      { k: "spread", label: "max disagreement" },
    ]);
    const msg = LR.msg(mount);

    function countsAt(n) {
      const h = Math.round(n * ratio);
      return { nh: h, nt: n - h };
    }

    function render() {
      const { nh, nt } = countsAt(N);
      const pa = nh + a, pb = nt + b;
      const mle = EST.mle(nh, nt);
      const map = EST.mapEst(nh, nt, a, b);
      const mean = EST.postPred(nh, nt, a, b);

      /* left: posterior with markers */
      let ymax = 0;
      for (let i = 1; i < 300; i++) ymax = Math.max(ymax, EST.betaPdf(i / 300, pa, pb));
      ymax *= 1.15;
      L.ctx.clearRect(0, 0, L.W, L.H);
      const PL = { x0: 46, y0: 16, w: L.W - 62, h: L.H - 56, xmin: 0, xmax: 1, ymin: 0, ymax: ymax, xlabel: "θ", ylabel: "p(θ | D)" };
      const gl = LR.plot(L.ctx, PL);
      curve(L.ctx, gl.sx, gl.sy, (x) => Math.min(EST.betaPdf(x, pa, pb), ymax), 0, 1, C.orange, 2.4);
      const top = ymax / 1.15;
      if (isFinite(mle)) vline(L.ctx, gl.sx, gl.sy, mle, top * 0.9, C.green, null);
      if (isFinite(map)) vline(L.ctx, gl.sx, gl.sy, map, top, C.orange, null);
      vline(L.ctx, gl.sx, gl.sy, mean, top * 0.78, C.purple, null);

      /* right: estimator trajectories over N */
      R.ctx.clearRect(0, 0, R.W, R.H);
      const PR = { x0: 46, y0: 16, w: R.W - 62, h: R.H - 56, xmin: 1, xmax: 500, ymin: 0, ymax: 1, xlabel: "N (flips at this ratio)", ylabel: "estimate", xticks: [1, 100, 200, 300, 400, 500] };
      const gr = LR.plot(R.ctx, PR);
      const series = [
        { f: (n) => { const c = countsAt(n); return EST.mle(c.nh, c.nt); }, col: C.green },
        { f: (n) => { const c = countsAt(n); return EST.mapEst(c.nh, c.nt, a, b); }, col: C.orange },
        { f: (n) => { const c = countsAt(n); return EST.postPred(c.nh, c.nt, a, b); }, col: C.purple },
      ];
      series.forEach(function (s) {
        R.ctx.strokeStyle = s.col;
        R.ctx.lineWidth = 2.2;
        R.ctx.beginPath();
        let started = false;
        for (let n = 1; n <= 500; n += 1) {
          const v = s.f(n);
          if (!isFinite(v)) { started = false; continue; }
          const px = gr.sx(n), py = gr.sy(v);
          if (!started) { R.ctx.moveTo(px, py); started = true; }
          else R.ctx.lineTo(px, py);
        }
        R.ctx.stroke();
      });
      // current-N cursor
      R.ctx.strokeStyle = "#bbb";
      R.ctx.lineWidth = 1.4;
      R.ctx.setLineDash([3, 3]);
      R.ctx.beginPath();
      R.ctx.moveTo(gr.sx(N), gr.sy(0));
      R.ctx.lineTo(gr.sx(N), gr.sy(1));
      R.ctx.stroke();
      R.ctx.setLineDash([]);
      [{ v: mle, col: C.green }, { v: map, col: C.orange }, { v: mean, col: C.purple }].forEach(function (m) {
        if (isFinite(m.v)) LR.dot(R.ctx, gr.sx(N), gr.sy(m.v), 4.5, m.col, "#fff");
      });

      const vals = [mle, map, mean].filter(isFinite);
      const spread = vals.length > 1 ? Math.max(...vals) - Math.min(...vals) : NaN;
      ro.set("counts", "(" + nh + ", " + nt + ")");
      ro.set("mle", isFinite(mle) ? LR.fmtF(mle, 4) : "–", C.green);
      ro.set("map", isFinite(map) ? LR.fmtF(map, 4) : "–", C.orange);
      ro.set("pred", LR.fmtF(mean, 4), C.purple);
      ro.set("spread", isFinite(spread) ? LR.fmtF(spread, 4) : "–", spread > 0.05 ? C.red : C.green);

      if (nh === 2 && nt === 0 && a === 2 && b === 2) {
        msg.show("The lecture's example: MLE 1.0, MAP 0.75, predictive 0.667. Three answers spread by a third of the whole parameter range, from two coin flips.", "info");
      } else if (isFinite(spread) && spread < 0.01) {
        msg.show("Disagreement below 0.01: at this N the estimators are interchangeable, and the cheap plug-in is the sensible engineering choice.", "good");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.2 — the Beta update as a rerunnable code exercise
     ══════════════════════════════════════════════════════════ */
  LR.figs.betacode = function (mount) {
    LR.header(
      mount,
      "beta_update, executed for real",
      "The pipeline from the Python above runs on whatever counts and prior you set. Presets reproduce the worked examples."
    );

    let nh = 2, nt = 0, a = 2, b = 2;

    const bar = LR.controls(mount);
    const nhIn = LR.numInput(bar, "N_H ", nh, 0, 10000, function (v) { nh = v; });
    const ntIn = LR.numInput(bar, "N_T ", nt, 0, 10000, function (v) { nt = v; });
    const aIn = LR.numInput(bar, "a ", a, 1, 1000, function (v) { a = v; });
    const bIn = LR.numInput(bar, "b ", b, 1, 1000, function (v) { b = v; });
    LR.button(bar, "small data (2, 0)", function () { setVals(2, 0, 2, 2); run(); });
    LR.button(bar, "lecture data (55, 45)", function () { setVals(55, 45, 2, 2); run(); });
    LR.button(bar, "Run ▸", run, "primary");

    const out = LR.el("pre", "run-output attn-runout", "press Run to execute beta_update(nh, nt, a, b)");
    mount.appendChild(out);

    function setVals(h, t, pa, pb) {
      nh = h; nt = t; a = pa; b = pb;
      nhIn.set(h); ntIn.set(t); aIn.set(pa); bIn.set(pb);
    }

    function run() {
      const pa = nh + a, pb = nt + b;
      const L = [];
      L.push(">>> beta_update(" + nh + ", " + nt + ", " + a + ", " + b + ")");
      L.push("posterior      : Beta(" + pa + ", " + pb + ")        # (nh + a, nt + b)");
      if (nh + nt > 0) {
        L.push("mle            : " + LR.fmtF(EST.mle(nh, nt), 4) + "               # nh / (nh + nt)");
      } else {
        L.push("mle            : undefined            # no data: 0/0");
      }
      const mode = EST.mapEst(nh, nt, a, b);
      if (isFinite(mode)) {
        L.push("map (mode)     : " + LR.fmtF(mode, 4) + "               # (nh + a - 1) / (N + a + b - 2)");
      } else {
        L.push("map (mode)     : undefined            # mode needs a, b > 1 or data");
      }
      L.push("pred (mean)    : " + LR.fmtF(EST.postPred(nh, nt, a, b), 4) + "               # (nh + a) / (N + a + b)");
      const nextH = EST.predDataset(nh, nt, a, b, 1, 0);
      L.push("check          : predDataset(h=1, t=0) = " + LR.fmtF(nextH, 4) + "  # equals the mean, as it must");
      out.textContent = L.join("\n");
    }
    run();
  };
})();
