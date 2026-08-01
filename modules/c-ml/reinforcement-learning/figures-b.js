/* ══════════════════════════════════════════════════════════════
   figures-b.js — Reinforcement Learning lesson, second half
   Fig 3.2 carmdp · Fig 4.1 vimaze (signature) · Fig 4.2 bellman
   Fig 4.3 vipi · Fig 4.4 vicode · Fig 5.1 qlearn (signature)
   Uses the shared RL core LR.rl from figures-a.js.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;
  const RL = LR.rl;

  /* ══════════════════════════════════════════════════════════
     Fig 3.2 — the car MDP transition explorer
     ══════════════════════════════════════════════════════════ */
  LR.figs.carmdp = function (mount) {
    LR.header(
      mount,
      "The car MDP: three states, two actions, one bad idea",
      "Choose an action from the current state to see its transitions, then Drive to sample them for real and watch empirical frequencies find the model."
    );

    /* the source states P(Warm|Warm,Slow)=0.5, P(Overheated|Warm,Fast)=1.0,
       R(...,-10), fast pays +2 vs slow +1 while not overheated, optimal
       Slow-in-Warm / Fast-in-Cool. The remaining entries below complete the
       table with the standard version of this example (flagged in caption). */
    const STATES = ["Cool", "Warm", "Overheated"];
    const SCOLOR = [C.green, C.amber, C.red];
    const T = {
      "Cool|Slow": [{ p: 1.0, s2: 0, r: 1 }],
      "Cool|Fast": [{ p: 0.5, s2: 0, r: 2 }, { p: 0.5, s2: 1, r: 2 }],
      "Warm|Slow": [{ p: 0.5, s2: 0, r: 1 }, { p: 0.5, s2: 1, r: 1 }],
      "Warm|Fast": [{ p: 1.0, s2: 2, r: -10 }],
    };
    const OPT = ["Fast", "Slow", "–"]; // optimal action per state (source)

    let cur = 0;           // current state index
    let act = "Slow";      // selected action
    let total = 0, drives = 0;
    let rand = LR.rng(77);
    const tally = {};      // "s|a" -> {count, outcomes: {s2: n}}
    let last = null;

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const bSlow = LR.button(grp, "action: Slow", function () { act = "Slow"; render(); }, "on");
    const bFast = LR.button(grp, "action: Fast", function () { act = "Fast"; render(); });
    const driveBtn = LR.button(bar, "Drive one step ▸", drive, "primary");
    LR.button(bar, "Reset ⟲", function () {
      cur = 0; total = 0; drives = 0; last = null; rand = LR.rng(77);
      Object.keys(tally).forEach((k) => delete tally[k]);
      render();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 700, 290, {
      aria: "Three engine states, Cool, Warm and Overheated, with transition arrows labelled by probability and reward for the selected action",
    });
    const ro = LR.readout(mount, [
      { k: "s", label: "state" },
      { k: "total", label: "total reward" },
      { k: "emp", label: "empirical vs model" },
    ]);
    const msg = LR.msg(mount);

    function key(s, a) { return STATES[s] + "|" + a; }

    function drive() {
      if (cur === 2) { msg.show("Overheated is absorbing: the drive is over. Reset to try again.", "bad"); return; }
      const k = key(cur, act);
      const outs = T[k];
      if (!tally[k]) tally[k] = { count: 0, outcomes: {} };
      tally[k].count++;
      let u = rand(), acc = 0, pick = outs[outs.length - 1];
      for (const o of outs) { acc += o.p; if (u <= acc) { pick = o; break; } }
      tally[k].outcomes[pick.s2] = (tally[k].outcomes[pick.s2] || 0) + 1;
      total += pick.r; drives++;
      last = { from: cur, a: act, to: pick.s2, r: pick.r, k };
      cur = pick.s2;
      render();
    }

    function render() {
      bSlow.classList.toggle("on", act === "Slow");
      bFast.classList.toggle("on", act === "Fast");
      driveBtn.disabled = cur === 2;
      ctx.clearRect(0, 0, W, H);

      const pos = [{ x: 120, y: 110 }, { x: 350, y: 110 }, { x: 580, y: 110 }];
      // transition arrows for the selected action from the current state
      const k = key(cur, act);
      const outs = T[k] || [];
      outs.forEach(function (o) {
        const A = pos[cur], B = pos[o.s2];
        const hot = last && last.k === k && last.to === o.s2;
        ctx.save();
        ctx.globalAlpha = 0.95;
        if (o.s2 === cur) {
          // self-loop
          ctx.strokeStyle = hot ? C.orange : "#888";
          ctx.lineWidth = hot ? 2.6 : 1.8;
          ctx.beginPath();
          ctx.arc(A.x, A.y - 58, 24, 0.85 * Math.PI, 0.15 * Math.PI, false);
          ctx.stroke();
          LR.arrow(ctx, A.x + 20, A.y - 45, A.x + 26, A.y - 38, hot ? C.orange : "#888", hot ? 2.6 : 1.8);
          ctx.fillStyle = hot ? C.orange : C.text;
          ctx.font = "700 12px Inter, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("p = " + o.p + " · R = " + (o.r > 0 ? "+" : "") + o.r, A.x, A.y - 92);
        } else {
          const dir = B.x > A.x ? 1 : -1;
          LR.arrow(ctx, A.x + 42 * dir, A.y - 8, B.x - 42 * dir, B.y - 8, hot ? C.orange : "#888", hot ? 2.6 : 1.8);
          ctx.fillStyle = hot ? C.orange : C.text;
          ctx.font = "700 12px Inter, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("p = " + o.p + " · R = " + (o.r > 0 ? "+" : "") + o.r, (A.x + B.x) / 2, A.y - 22);
        }
        ctx.restore();
      });

      // state circles
      STATES.forEach(function (s, i) {
        const p = pos[i];
        ctx.beginPath();
        ctx.arc(p.x, p.y, 40, 0, Math.PI * 2);
        ctx.fillStyle = i === cur ? SCOLOR[i] : "#fff";
        ctx.fill();
        ctx.strokeStyle = SCOLOR[i];
        ctx.lineWidth = i === cur ? 3 : 2;
        ctx.stroke();
        ctx.fillStyle = i === cur ? "#fff" : C.text;
        ctx.font = "700 13px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(s, p.x, p.y + 4);
        ctx.fillStyle = LR.C.muted;
        ctx.font = "600 11px Inter, sans-serif";
        ctx.fillText(i === 2 ? "terminal" : "optimal: go " + OPT[i], p.x, p.y + 62);
      });

      ctx.fillStyle = LR.C.faint;
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("arrows: P(s′ | s, a) and R(s, a, s′) for the selected action from the current state", 14, H - 10);

      ro.set("s", STATES[cur], SCOLOR[cur]);
      ro.set("total", (total > 0 ? "+" : "") + total + " over " + drives + " steps", total >= 0 ? C.green : C.red);
      const tk = tally["Warm|Slow"];
      if (tk && tk.count > 0) {
        const stay = (tk.outcomes[1] || 0) / tk.count;
        ro.set("emp", "P̂(Warm|Warm,Slow) = " + LR.fmtF(stay, 2) + " after " + tk.count + " tries (model: 0.5)", C.purple);
      } else {
        ro.set("emp", "drive Slow from Warm to estimate P(Warm|Warm,Slow)");
      }

      if (cur === 2) {
        msg.show("Overheated: reward −10 and the drive terminates. Fast from Warm makes this certain, P = 1.0, which is why the optimal policy goes Slow there despite the smaller payout.", "bad");
      } else if (last && last.k === "Cool|Fast") {
        msg.show("Fast from Cool paid +2 and the coin " + (last.to === 0 ? "kept the engine Cool" : "warmed the engine") + ". Risk-free in Cool: the worst case, Warm, is still recoverable.", "good");
      } else if (drives === 0) {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.1 — value iteration on the 3×3 maze (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.vimaze = function (mount) {
    LR.header(
      mount,
      "Value iteration: sweep, and watch value radiate",
      "Every press of Sweep applies one Bellman backup to every state. Then bend the problem: γ, step cost, goal reward, and re-run."
    );

    const cfg = RL.mazeCfg();
    let V = RL.initV(cfg);
    let k = 0, delta = Infinity;
    let timer = null;

    const bar = LR.controls(mount);
    const sweepBtn = LR.button(bar, "Sweep ▸", function () { stopRun(); doSweep(); render(); }, "primary");
    const runBtn = LR.button(bar, "Run to convergence ▸▸", toggleRun);
    LR.button(bar, "Reset V ⟲", function () { stopRun(); resetV(); render(); });

    const bar2 = LR.controls(mount);
    LR.slider(bar2, "γ", 0.5, 0.99, 0.01, cfg.gamma, function (v) { cfg.gamma = v; stopRun(); resetV(); render(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar2, "step reward", -3, 0, 0.25, cfg.step, function (v) { cfg.step = v; stopRun(); resetV(); render(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar2, "goal reward", 2, 20, 1, cfg.goalV, function (v) { cfg.goalV = v; stopRun(); resetV(); render(); }, (v) => "+" + LR.fmt(v, 0));

    const { cv, ctx, W, H } = LR.canvas(mount, 560, 420, {
      aria: "A 3 by 3 maze whose state values fill in from the goal outward with each value-iteration sweep, with greedy policy arrows in each cell",
    });
    const ro = LR.readout(mount, [
      { k: "k", label: "sweep k" },
      { k: "delta", label: "‖V(k+1) − V(k)‖∞" },
      { k: "status", label: "status" },
    ]);
    const msg = LR.msg(mount);

    function resetV() { V = RL.initV(cfg); k = 0; delta = Infinity; }
    function doSweep() {
      const s = RL.viSweep(V, cfg);
      V = s.V; delta = s.delta; k++;
    }
    function toggleRun() {
      if (timer) { stopRun(); return; }
      if (delta <= 1e-4 && k > 0) resetV();
      if (LR.reducedMotion) {
        while (delta > 1e-4 && k < 200) doSweep();
        render();
        return;
      }
      runBtn.textContent = "Pause ❚❚";
      timer = setInterval(function () {
        doSweep(); render();
        if (delta <= 1e-4 || k >= 200) stopRun();
      }, 650);
    }
    function stopRun() {
      if (timer) clearInterval(timer);
      timer = null;
      runBtn.textContent = "Run to convergence ▸▸";
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const cell = 120, x0 = (W - 3 * cell) / 2, y0 = 30;
      let vmax = cfg.goalV;
      V.forEach((row) => row.forEach((v) => { vmax = Math.max(vmax, Math.abs(v)); }));
      RL.drawGrid(ctx, {
        x0, y0, cell, cfg,
        colorOf: function (r, c) { return RL.rampSigned(V[r][c], vmax); },
        label: function (r, c) {
          if (r === cfg.gr && c === cfg.gc) return "+" + LR.fmt(cfg.goalV, 0);
          return LR.fmt(V[r][c], 2);
        },
        arrows: k > 0 ? function (r, c) { return RL.greedy(V, cfg, r, c); } : null,
      });
      ctx.fillStyle = LR.C.muted;
      ctx.font = "11.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("V(s) after " + k + " sweep" + (k === 1 ? "" : "s") + "  ·  " + LR.fmtF(cfg.step, 2) + " per move (walls too, you stay put)  ·  γ = " + LR.fmtF(cfg.gamma, 2), W / 2, y0 + 3 * cell + 24);
      ctx.fillText("arrows: the greedy policy with respect to the current V", W / 2, y0 + 3 * cell + 42);

      ro.set("k", String(k));
      ro.set("delta", k === 0 ? "–" : LR.fmtF(delta, 4), delta <= 1e-4 ? C.green : C.orange);
      ro.set("status", k === 0 ? "V₀ = 0 everywhere" : delta <= 1e-4 ? "converged" : "still moving", delta <= 1e-4 ? C.green : undefined);

      const isDefault = Math.abs(cfg.gamma - 0.9) < 1e-9 && Math.abs(cfg.step + 1) < 1e-9 && cfg.goalV === 10;
      if (k === 0) {
        msg.hide();
      } else if (isDefault && k === 1) {
        msg.show("Sweep 1: only the goal's neighbours learned anything, V₁ = −1 + 0.9·10 = 8. Everyone else backed up a zero and paid the step cost.", "info");
      } else if (isDefault && k === 2) {
        msg.show("Sweep 2: the second ring found the 8s, V₂ = −1 + 0.9·8 = 6.2. The goal's influence has travelled exactly two steps.", "info");
      } else if (isDefault && delta <= 1e-4) {
        msg.show("Converged: values 3.12, 4.58, 6.2, 8 fan out from the goal (final since sweep 4; the last sweep confirms Δ = 0), and every arrow points along a shortest path home.", "good");
      } else if (delta <= 1e-4) {
        msg.show("Converged in " + k + " sweeps with your settings. Compare the value landscape against the defaults: γ and the rewards reshape how far the goal's pull reaches.", "good");
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.2 — the Bellman backup inspector
     ══════════════════════════════════════════════════════════ */
  LR.figs.bellman = function (mount) {
    LR.header(
      mount,
      "Click a state, read its backup",
      "The update V(k+1)(s) = maxₐ Σ P(s′|s,a)[R + γV(k)(s′)] expanded into its four candidates, with live numbers from the chosen sweep."
    );

    const cfg = RL.mazeCfg();
    // precompute V_0 ... V_5 (real sweeps)
    const VS = [RL.initV(cfg)];
    for (let i = 0; i < 5; i++) VS.push(RL.viSweep(VS[i], cfg).V);
    let ksel = 2;
    let sel = { r: 1, c: 1 };

    const bar = LR.controls(mount);
    LR.slider(bar, "use values from sweep k", 0, 5, 1, ksel, function (v) { ksel = Math.max(0, Math.min(5, Math.round(v))); render(); }, (v) => "k = " + Math.round(v));

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left); grid.appendChild(right);

    left.appendChild(LR.el("div", "pane-label", "the maze at sweep k (click a state)"));
    const gwrap = LR.el("div");
    gwrap.style.cssText = "display:grid;grid-template-columns:repeat(3,86px);gap:4px;justify-content:center;margin-top:6px";
    gwrap.setAttribute("role", "group");
    gwrap.setAttribute("aria-label", "3 by 3 maze states; activate one to expand its Bellman backup");
    left.appendChild(gwrap);
    const cells = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      const b = LR.el("button", null, "");
      b.type = "button";
      b.style.cssText = "height:86px;border:1.5px solid #c9c9c9;border-radius:10px;font:700 15px Inter,sans-serif;cursor:pointer";
      (function (rr, cc) {
        b.addEventListener("click", function () { sel = { r: rr, c: cc }; render(); });
      })(r, c);
      gwrap.appendChild(b);
      cells.push(b);
    }

    right.appendChild(LR.el("div", "pane-label", "the backup, term by term"));
    const panel = LR.el("div", "stage-detail");
    panel.style.marginTop = "6px";
    right.appendChild(panel);

    function render() {
      const V = VS[ksel];
      let vmax = cfg.goalV;
      V.forEach((row) => row.forEach((v) => { vmax = Math.max(vmax, Math.abs(v)); }));
      cells.forEach(function (b, i) {
        const r = Math.floor(i / 3), c = i % 3;
        const isGoal = r === cfg.gr && c === cfg.gc;
        const bg = RL.rampSigned(V[r][c], vmax);
        b.style.background = bg;
        b.style.color = RL.txtOn(bg);
        b.textContent = isGoal ? "+10" : LR.fmt(V[r][c], 2);
        b.style.outline = sel.r === r && sel.c === c ? "3px solid " + C.purple : "none";
        b.setAttribute("aria-label", isGoal
          ? "goal state, value pinned at +10"
          : "state row " + r + " column " + c + ", value " + LR.fmtF(V[r][c], 2) + " at sweep " + ksel);
        b.setAttribute("aria-pressed", sel.r === r && sel.c === c ? "true" : "false");
      });

      const { r, c } = sel;
      if (r === cfg.gr && c === cfg.gc) {
        panel.innerHTML = "<b>The goal state.</b> Its value is pinned at the goal reward, +10; it is absorbing, so no backup is computed here. Pick any other state.";
        return;
      }
      const { cands, best } = RL.backup(V, cfg, r, c);
      const rows = cands.map(function (cd) {
        const isBest = cd.a === best.a;
        return "<div class='sd-row' style='font-family:var(--mono);font-size:13px;" +
          (isBest ? "color:" + C.orange + ";font-weight:700" : "") + "'>" +
          cd.name + (cd.wall ? " (wall: stay)" : " → (" + cd.nr + "," + cd.nc + ")") +
          ":  −1 + 0.9 × " + LR.fmt(V[cd.nr][cd.nc], 2) + " = " + LR.fmt(cd.val, 2) +
          (isBest ? "   ← max" : "") + "</div>";
      }).join("");
      panel.innerHTML =
        "<div class='sd-row'><b>State (" + r + "," + c + ") at sweep k = " + ksel + ".</b> Deterministic moves, so each sum over s′ is a single term, R + γV<sub>k</sub>(s′):</div>" +
        rows +
        "<div class='sd-row' style='margin-top:8px'><b>V<sub>" + (ksel + 1) + "</sub>(" + r + "," + c + ") = " +
        LR.fmt(best.val, 2) + "</b>, the max, exactly what the next sweep writes here." +
        (ksel < 5 ? " (Check: move the slider to k = " + (ksel + 1) + " and this cell reads " + LR.fmt(VS[ksel + 1][r][c], 2) + ".)" : "") + "</div>";
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.3 — value iteration vs policy iteration, racing
     ══════════════════════════════════════════════════════════ */
  LR.figs.vipi = function (mount) {
    LR.header(
      mount,
      "Two roads to the same policy",
      "VI: many cheap value sweeps, policy only at the end. PI: an explicit policy, fully evaluated then improved, few but costly rounds."
    );

    const cfg = RL.mazeCfg();
    let vi, pi;
    function reset() {
      vi = { V: RL.initV(cfg), sweeps: 0, delta: Infinity };
      pi = {
        pol: Array.from({ length: 3 }, () => [3, 3, 3]), // start deliberately bad: all West
        V: RL.initV(cfg),
        outer: 0, inner: 0, stable: false,
      };
    }
    reset();

    const bar = LR.controls(mount);
    LR.button(bar, "Step VI (one sweep)", function () { stepVI(); render(); }, "primary");
    LR.button(bar, "Step PI (evaluate + improve)", function () { stepPI(); render(); }, "primary");
    LR.button(bar, "Run both to convergence ▸▸", function () {
      let guard = 0;
      while (vi.delta > 1e-6 && guard++ < 500) stepVI();
      guard = 0;
      while (!pi.stable && guard++ < 50) stepPI();
      render();
    });
    LR.button(bar, "Reset ⟲", function () { reset(); render(); });

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left); grid.appendChild(right);
    left.appendChild(LR.el("div", "pane-label", "value iteration"));
    right.appendChild(LR.el("div", "pane-label", "policy iteration"));
    const cvL = LR.canvas(left, 300, 300, { aria: "Value iteration maze: values update each sweep, arrows show the implied greedy policy" });
    const cvR = LR.canvas(right, 300, 300, { aria: "Policy iteration maze: an explicit policy's arrows, updated after each evaluate-and-improve round" });

    const ro = LR.readout(mount, [
      { k: "vi", label: "VI work" },
      { k: "pi", label: "PI work" },
      { k: "same", label: "policies" },
    ]);
    const msg = LR.msg(mount);

    function stepVI() {
      if (vi.delta <= 1e-6) return;
      const s = RL.viSweep(vi.V, cfg);
      vi.V = s.V; vi.delta = s.delta; vi.sweeps++;
    }

    function stepPI() {
      if (pi.stable) return;
      // full policy evaluation of the current policy (inner sweeps counted)
      let d = Infinity, guard = 0;
      let V = RL.initV(cfg);
      while (d > 1e-4 && guard++ < 400) {
        const s = RL.peSweep(V, cfg, pi.pol);
        V = s.V; d = s.delta; pi.inner++;
      }
      pi.V = V;
      // greedy improvement
      let changed = false;
      const newPol = pi.pol.map((row) => row.slice());
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        if (r === cfg.gr && c === cfg.gc) continue;
        const g = RL.greedy(pi.V, cfg, r, c);
        if (g.indexOf(pi.pol[r][c]) === -1) { newPol[r][c] = g[0]; changed = true; }
      }
      pi.pol = newPol;
      pi.outer++;
      if (!changed) pi.stable = true;
    }

    function drawPane(cvo, V, arrowsFn, title) {
      const ctx = cvo.ctx;
      ctx.clearRect(0, 0, cvo.W, cvo.H);
      const cell = 84, x0 = (cvo.W - 3 * cell) / 2, y0 = 14;
      let vmax = cfg.goalV;
      V.forEach((row) => row.forEach((v) => { vmax = Math.max(vmax, Math.abs(v)); }));
      RL.drawGrid(ctx, {
        x0, y0, cell, cfg,
        colorOf: function (r, c) { return RL.rampSigned(V[r][c], vmax); },
        label: function (r, c) {
          if (r === cfg.gr && c === cfg.gc) return "+10";
          return LR.fmt(V[r][c], 1);
        },
        arrows: arrowsFn,
        goalLabel: false,
      });
      ctx.fillStyle = LR.C.muted;
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(title, cvo.W / 2, y0 + 3 * cell + 20);
    }

    function samePolicy() {
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
        if (r === cfg.gr && c === cfg.gc) continue;
        if (RL.greedy(vi.V, cfg, r, c).indexOf(pi.pol[r][c]) === -1) return false;
      }
      return true;
    }

    function render() {
      drawPane(cvL, vi.V, vi.sweeps > 0 ? function (r, c) { return RL.greedy(vi.V, cfg, r, c); } : null,
        "after " + vi.sweeps + " sweep" + (vi.sweeps === 1 ? "" : "s") + (vi.delta <= 1e-6 ? " · converged" : ""));
      drawPane(cvR, pi.V, function (r, c) { return [pi.pol[r][c]]; },
        pi.outer === 0 ? "π₀: all West, on purpose" : "after " + pi.outer + " round" + (pi.outer === 1 ? "" : "s") + (pi.stable ? " · policy stable" : ""));

      ro.set("vi", vi.sweeps + " cheap sweeps" + (vi.delta <= 1e-6 ? " (done)" : ""), C.orange);
      ro.set("pi", pi.outer + " outer rounds · " + pi.inner + " evaluation sweeps inside", C.purple);
      const same = vi.sweeps > 0 && pi.outer > 0 && samePolicy();
      ro.set("same", same ? "identical (greedy-equivalent)" : "not yet comparable", same ? C.green : undefined);

      if (vi.delta <= 1e-6 && pi.stable) {
        msg.show("Both converged to the same optimal policy, arrow for arrow: VI in " + vi.sweeps + " cheap sweeps, PI in " + pi.outer + " outer rounds that hid " + pi.inner + " evaluation sweeps. Same Bellman core, two schedules.", "good");
      } else if (pi.outer === 1) {
        msg.show("PI's first evaluation priced the all-West policy honestly (walls forever: V → −10), and one greedy improvement already points the goal's neighbours home.", "info");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 4.4 — value_iteration, executed for real
     ══════════════════════════════════════════════════════════ */
  LR.figs.vicode = function (mount) {
    LR.header(
      mount,
      "value_iteration, executed for real",
      "The same computation as the Python above, run on the 3×3 maze at your γ. Every printed number is computed on the spot."
    );

    let gamma = 0.9;
    const bar = LR.controls(mount);
    const gS = LR.slider(bar, "γ", 0.5, 0.99, 0.01, gamma, function (v) { gamma = v; }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Run ▸", run, "primary");

    const out = LR.el("pre", "attn-runout", "press Run to execute value_iteration(P, R, gamma=0.9) on the maze");
    mount.appendChild(out);

    function gridStr(V, d) {
      return V.map(function (row, r) {
        return "  [" + row.map(function (v, c) {
          const s = (r === 0 && c === 2) ? "10.0" : LR.fmtF(v, d);
          return (" " .repeat(Math.max(0, 6 - s.length))) + s;
        }).join(" ") + "]";
      }).join("\n");
    }

    function run() {
      const cfg = RL.mazeCfg();
      cfg.gamma = gamma;
      const L = [];
      L.push(">>> V, policy = value_iteration(P, R, gamma=" + LR.fmtF(gamma, 2) + ")");
      let V = RL.initV(cfg), delta = Infinity, k = 0;
      while (delta > 1e-6 && k < 200) {
        const s = RL.viSweep(V, cfg);
        V = s.V; delta = s.delta; k++;
        if (k <= 5) {
          L.push("sweep " + k + "   ||V_k+1 - V_k||_inf = " + LR.fmtF(delta, 4));
          L.push(gridStr(V, 2));
        } else if (k === 6) {
          L.push("...");
        }
      }
      L.push(">>> converged after " + k + " sweeps (eps = 1e-6)");
      L.push(">>> policy   # argmax_a of R + gamma * V(s'), per state");
      const pol = [];
      for (let r = 0; r < 3; r++) {
        pol.push("  [" + [0, 1, 2].map(function (c) {
          if (r === cfg.gr && c === cfg.gc) return " G ";
          return " " + RL.greedy(V, cfg, r, c).map((a) => RL.DIRS[a].glyph).join("") + " ";
        }).join(" ") + "]");
      }
      L.push(pol.join("\n"));
      L.push("# gamma=0.9 reproduces the lecture: 8 after sweep 1, 6.2 after sweep 2, values final by sweep 4");
      out.textContent = L.join("\n");
    }
  };

  /* ══════════════════════════════════════════════════════════
     Fig 5.1 — the Q-learning agent (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.qlearn = function (mount) {
    LR.header(
      mount,
      "Q-learning, live: a table filling in backward",
      "The agent knows nothing: no P, no R. It acts by softmax on its own Q̂ table, and every number below is written by the update rule as it walks."
    );

    const cfg = RL.qCfg(4, 4);
    const START = { r: 3, c: 0 };
    let st = RL.qInit(cfg);
    let rand = LR.rng(21);
    let agent = { r: START.r, c: START.c };
    let episodes = 0, lastSteps = "–", lastUpd = null;
    let kSoft = 0.03, stochastic = false;
    let timer = null;

    const bar = LR.controls(mount);
    const stepBtn = LR.button(bar, "Step", function () { stopRun(); doStep(); render(); });
    const epBtn = LR.button(bar, "Run episode ▸", function () { stopRun(); runEpisode(); }, "primary");
    LR.button(bar, "Train ×25", function () {
      stopRun();
      for (let i = 0; i < 25; i++) runEpisodeInstant();
      agent = { r: START.r, c: START.c };
      render();
    });
    LR.button(bar, "Reset ⟲", function () {
      stopRun();
      st = RL.qInit(cfg); rand = LR.rng(21);
      agent = { r: START.r, c: START.c };
      episodes = 0; lastSteps = "–"; lastUpd = null;
      render();
    });

    const bar2 = LR.controls(mount);
    LR.slider(bar2, "softmax k (explore ↔ exploit)", 0.005, 0.3, 0.005, kSoft, function (v) { kSoft = v; render(); }, (v) => LR.fmtF(v, 3));
    const stochBtn = LR.button(bar2, "stochastic world (20% slip)", function () {
      stochastic = !stochastic;
      stochBtn.classList.toggle("on", stochastic);
      render();
    });

    const { cv, ctx, W, H } = LR.canvas(mount, 620, 560, {
      aria: "A 4 by 4 gridworld showing all four Q values at each cell's edges, the maximum as the cell's shade, the agent as a purple dot, and the most recent update highlighted",
    });
    const ro = LR.readout(mount, [
      { k: "ep", label: "episodes" },
      { k: "steps", label: "steps, last episode" },
      { k: "fill", label: "non-zero Q̂ entries" },
      { k: "upd", label: "last update" },
    ]);
    const msg = LR.msg(mount);

    function opts() { return { k: kSoft, slip: stochastic ? 0.2 : 0, useAlpha: stochastic, rand }; }

    function doStep() {
      const out = RL.qStep(st, cfg, agent.r, agent.c, opts());
      lastUpd = { r: agent.r, c: agent.c, a: out.a, val: out.newVal, rew: out.rew, alpha: out.alpha, nr: out.nr, nc: out.nc };
      if (out.done) {
        episodes++;
        lastSteps = "(single steps)";
        agent = { r: START.r, c: START.c };
      } else {
        agent = { r: out.nr, c: out.nc };
      }
    }

    function runEpisodeInstant() {
      let r = START.r, c = START.c, steps = 0;
      while (steps < 400) {
        const out = RL.qStep(st, cfg, r, c, opts());
        steps++;
        lastUpd = { r, c, a: out.a, val: out.newVal, rew: out.rew, alpha: out.alpha, nr: out.nr, nc: out.nc };
        r = out.nr; c = out.nc;
        if (out.done) break;
      }
      episodes++;
      lastSteps = steps;
    }

    function runEpisode() {
      if (LR.reducedMotion) { runEpisodeInstant(); agent = { r: START.r, c: START.c }; render(); return; }
      let r = START.r, c = START.c, steps = 0;
      epBtn.disabled = true;
      timer = setInterval(function () {
        const out = RL.qStep(st, cfg, r, c, opts());
        steps++;
        lastUpd = { r, c, a: out.a, val: out.newVal, rew: out.rew, alpha: out.alpha, nr: out.nr, nc: out.nc };
        r = out.nr; c = out.nc;
        agent = { r, c };
        render();
        if (out.done || steps >= 400) {
          episodes++;
          lastSteps = steps;
          agent = { r: START.r, c: START.c };
          stopRun();
          render();
        }
      }, 70);
    }
    function stopRun() {
      if (timer) clearInterval(timer);
      timer = null;
      epBtn.disabled = false;
    }

    function maxQ(r, c) { return Math.max(...st.Q[r][c]); }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const cell = 128, x0 = (W - cfg.cols * cell) / 2, y0 = 26;

      for (let r = 0; r < cfg.rows; r++)
        for (let c = 0; c < cfg.cols; c++) {
          const x = x0 + c * cell, y = y0 + r * cell;
          const isGoal = r === cfg.gr && c === cfg.gc;
          const bg = isGoal ? RL.rampGreen(0.25) : RL.ramp(maxQ(r, c) / 100);
          ctx.fillStyle = bg;
          ctx.fillRect(x, y, cell, cell);
          ctx.strokeStyle = "#c9c9c9";
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, cell, cell);
          if (isGoal) {
            ctx.strokeStyle = C.green;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
            ctx.fillStyle = C.green;
            ctx.font = "800 15px Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("GOAL", x + cell / 2, y + cell / 2 - 2);
            ctx.font = "700 12px Inter, sans-serif";
            ctx.fillText("+100 to enter", x + cell / 2, y + cell / 2 + 16);
            continue;
          }
          const tcol = RL.txtOn(bg);
          // centre: max Q
          const mq = maxQ(r, c);
          ctx.fillStyle = tcol;
          ctx.font = "800 16px Inter, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(mq > 0 ? LR.fmt(mq, 1) : "·", x + cell / 2, y + cell / 2 + 5);
          // the four directional entries
          ctx.font = "600 10.5px Inter, sans-serif";
          const q = st.Q[r][c];
          const hot = function (a) {
            return lastUpd && lastUpd.r === r && lastUpd.c === c && lastUpd.a === a;
          };
          const put = function (a, tx, ty, align) {
            ctx.fillStyle = hot(a) ? C.red : tcol;
            ctx.textAlign = align;
            ctx.fillText(RL.DIRS[a].glyph + " " + (q[a] > 0 ? LR.fmt(q[a], 1) : "0"), tx, ty);
          };
          put(0, x + cell / 2, y + 15, "center");
          put(2, x + cell / 2, y + cell - 7, "center");
          put(1, x + cell - 6, y + cell / 2 + 4, "right");
          put(3, x + 6, y + cell / 2 + 4, "left");
        }
      // frame + agent
      ctx.strokeStyle = C.axis;
      ctx.lineWidth = 1.6;
      ctx.strokeRect(x0, y0, cfg.cols * cell, cfg.rows * cell);
      LR.dot(ctx, x0 + agent.c * cell + cell / 2, y0 + agent.r * cell + cell * 0.72, 9, C.purple, "#fff");

      ctx.fillStyle = LR.C.muted;
      ctx.font = "11.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("each edge: Q̂(s, a) for that direction · centre and shade: maxₐ Q̂ · red: the entry just written · purple dot: agent (start bottom-left)", W / 2, H - 8);

      let nz = 0;
      for (let r = 0; r < cfg.rows; r++) for (let c = 0; c < cfg.cols; c++) {
        if (r === cfg.gr && c === cfg.gc) continue;
        st.Q[r][c].forEach((v) => { if (v > 0) nz++; });
      }
      ro.set("ep", String(episodes));
      ro.set("steps", String(lastSteps), typeof lastSteps === "number" && lastSteps <= 8 ? C.green : undefined);
      ro.set("fill", nz + " / 60");
      if (lastUpd) {
        const u = lastUpd;
        const tgt = u.rew + (u.nr === cfg.gr && u.nc === cfg.gc ? 0 : 0.9 * maxQ(u.nr, u.nc));
        ro.set("upd",
          "Q̂((" + u.r + "," + u.c + ")," + RL.DIRS[u.a].name + ") ← " +
          (u.alpha !== null && u.alpha !== undefined && stochastic
            ? "(1−α)·old + α·[r + γ·max] = " + LR.fmt(u.val, 1) + "  (α = " + LR.fmtF(u.alpha, 2) + ")"
            : "r + γ·max = " + u.rew + " + 0.9·" + LR.fmt(u.rew > 0 ? 0 : maxQ(u.nr, u.nc), 1) + " = " + LR.fmt(u.val, 1)),
          C.orange);
      } else {
        ro.set("upd", "–");
      }

      // narrate the propagation rings
      const rings = [100, 90, 81, 72.9].map(function (v) {
        let n = 0;
        for (let r = 0; r < cfg.rows; r++) for (let c = 0; c < cfg.cols; c++)
          st.Q[r][c].forEach((qv) => { if (Math.abs(qv - v) < 0.51) n++; });
        return n;
      });
      if (episodes === 0) {
        msg.hide();
      } else if (nz === 0) {
        msg.show("A whole episode of zeros: every update wrote 0 + γ·max{0,...} = 0. Nothing can be learned until the goal is touched.", "info");
      } else if (nz <= 3) {
        msg.show("First contact: only the transition into the goal earned a value (100). Everything the agent knows lives one step from the goal.", "info");
      } else if (stochastic) {
        msg.show("Stochastic world: targets vary from visit to visit, so the α_n = 1/(1+visits) rule blends instead of overwriting. Values wobble early, then settle toward the expectation.", "info");
      } else if (rings[1] > 0 && rings[2] === 0) {
        msg.show("The 90s have appeared: 0 + 0.9·100, one step further from the goal than the 100s. Exactly the worked example's arithmetic, discovered by walking.", "good");
      } else if (rings[2] > 0 || rings[3] > 0) {
        msg.show("The rings 100, 90, 81, 72.9 are filling in: each episode drags reward information one more transition backward along the travelled path.", "good");
      }
    }
    render();
  };
})();
