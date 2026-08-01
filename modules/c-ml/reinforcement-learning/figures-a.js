/* ══════════════════════════════════════════════════════════════
   figures-a.js — Reinforcement Learning lesson, first half
   Fig 0.1 hook · Fig 1.1 explore · Fig 1.2 bandit (signature)
   Fig 2.1 ladder · Fig 2.2 loop · Fig 3.1 discount
   Also defines the shared RL core (LR.rl) used by figures-b.js:
   the 3x3 planning maze, Bellman backups, value-iteration sweeps,
   greedy policies, the Q-learning world and update, softmax
   exploration, grid rendering, and colour ramps.
   All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared RL core (also used by figures-b.js) ─────────── */
  const RL = (LR.rl = {
    DIRS: [
      { dr: -1, dc: 0, name: "N", glyph: "↑" },
      { dr: 0, dc: 1, name: "E", glyph: "→" },
      { dr: 1, dc: 0, name: "S", glyph: "↓" },
      { dr: 0, dc: -1, name: "W", glyph: "←" },
    ],

    /* the lecture's 3x3 planning maze: gamma 0.9, -1 per move,
       -1 for a wall (stay put), +10 at the goal (top-right).
       V(goal) is pinned at the goal reward; deterministic moves. */
    mazeCfg: function () {
      return { rows: 3, cols: 3, gr: 0, gc: 2, step: -1, goalV: 10, gamma: 0.9 };
    },

    zeros: function (rows, cols) {
      return Array.from({ length: rows }, () => new Array(cols).fill(0));
    },

    initV: function (cfg) {
      const V = RL.zeros(cfg.rows, cfg.cols);
      V[cfg.gr][cfg.gc] = cfg.goalV;
      return V;
    },

    /* where does action a from (r,c) lead? walls bounce you back */
    next: function (cfg, r, c, a) {
      const d = RL.DIRS[a];
      let nr = r + d.dr, nc = c + d.dc;
      let wall = false;
      if (nr < 0 || nr >= cfg.rows || nc < 0 || nc >= cfg.cols) { nr = r; nc = c; wall = true; }
      return { nr, nc, wall };
    },

    /* the four candidate backups R + gamma * V(s') at one state */
    backup: function (V, cfg, r, c) {
      const cands = RL.DIRS.map(function (d, a) {
        const { nr, nc, wall } = RL.next(cfg, r, c, a);
        return { a, name: d.name, glyph: d.glyph, nr, nc, wall, val: cfg.step + cfg.gamma * V[nr][nc] };
      });
      let best = cands[0];
      cands.forEach((cd) => { if (cd.val > best.val + 1e-12) best = cd; });
      return { cands, best };
    },

    /* one synchronous value-iteration sweep (Bellman optimality) */
    viSweep: function (V, cfg) {
      const V2 = V.map((row) => row.slice());
      let delta = 0;
      for (let r = 0; r < cfg.rows; r++)
        for (let c = 0; c < cfg.cols; c++) {
          if (r === cfg.gr && c === cfg.gc) continue;
          const b = RL.backup(V, cfg, r, c).best.val;
          delta = Math.max(delta, Math.abs(b - V[r][c]));
          V2[r][c] = b;
        }
      return { V: V2, delta };
    },

    runVI: function (cfg, tol, maxSweeps) {
      let V = RL.initV(cfg), sweeps = 0, delta = Infinity;
      while (delta > (tol || 1e-9) && sweeps < (maxSweeps || 500)) {
        const s = RL.viSweep(V, cfg);
        V = s.V; delta = s.delta; sweeps++;
      }
      return { V, sweeps };
    },

    /* greedy action indices at (r,c) under V (ties kept) */
    greedy: function (V, cfg, r, c) {
      const { cands } = RL.backup(V, cfg, r, c);
      const mx = Math.max(...cands.map((cd) => cd.val));
      return cands.filter((cd) => cd.val > mx - 1e-9).map((cd) => cd.a);
    },

    /* one policy-evaluation sweep for a fixed policy pi[r][c] = action */
    peSweep: function (V, cfg, pi) {
      const V2 = V.map((row) => row.slice());
      let delta = 0;
      for (let r = 0; r < cfg.rows; r++)
        for (let c = 0; c < cfg.cols; c++) {
          if (r === cfg.gr && c === cfg.gc) continue;
          const { nr, nc } = RL.next(cfg, r, c, pi[r][c]);
          const v = cfg.step + cfg.gamma * V[nr][nc];
          delta = Math.max(delta, Math.abs(v - V[r][c]));
          V2[r][c] = v;
        }
      return { V: V2, delta };
    },

    /* softmax over action values with sharpness k (the source's
       exploration policy; numerically stabilized) */
    softmax: function (vals, k) {
      const m = Math.max(...vals.map((v) => k * v));
      const e = vals.map((v) => Math.exp(k * v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    },
    sample: function (probs, rand) {
      let u = rand(), acc = 0;
      for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u <= acc) return i; }
      return probs.length - 1;
    },

    /* ── the Q-learning world: reward 0 per move, +100 entering
          the goal (absorbing), gamma 0.9 ─────────────────────── */
    qCfg: function (rows, cols) {
      return { rows: rows, cols: cols, gr: 0, gc: cols - 1, goalR: 100, gamma: 0.9 };
    },
    qInit: function (cfg) {
      return {
        Q: Array.from({ length: cfg.rows }, () =>
          Array.from({ length: cfg.cols }, () => [0, 0, 0, 0])),
        visits: Array.from({ length: cfg.rows }, () =>
          Array.from({ length: cfg.cols }, () => [0, 0, 0, 0])),
      };
    },
    /* one Q-learning step from (r,c). opts: {k (softmax), slip (prob of
       random slip), useAlpha (learning-rate rule), rand}
       returns {a, nr, nc, rew, done, newVal, alpha} */
    qStep: function (st, cfg, r, c, opts) {
      const rand = opts.rand;
      const a = RL.sample(RL.softmax(st.Q[r][c], opts.k), rand);
      let move = a;
      if (opts.slip && rand() < opts.slip) move = Math.floor(rand() * 4); // the world slips, not the choice
      let { nr, nc } = RL.next(cfg, r, c, move);
      const done = nr === cfg.gr && nc === cfg.gc;
      const rew = done ? cfg.goalR : 0;
      const target = rew + (done ? 0 : cfg.gamma * Math.max(...st.Q[nr][nc]));
      let alpha = null;
      if (opts.useAlpha) {
        st.visits[r][c][a] += 1;
        alpha = 1 / (1 + st.visits[r][c][a]);
        st.Q[r][c][a] = (1 - alpha) * st.Q[r][c][a] + alpha * target;
      } else {
        st.Q[r][c][a] = target; // deterministic training rule
      }
      return { a, nr, nc, rew, done, newVal: st.Q[r][c][a], alpha };
    },

    /* ── colour ramps ───────────────────────────────────────── */
    ramp: function (v) {
      // v in [0,1] → white → orange
      const t = Math.max(0, Math.min(1, v));
      const r = Math.round(255 + (232 - 255) * t);
      const g = Math.round(255 + (89 - 255) * t);
      const b = Math.round(255 + (12 - 255) * t);
      return "rgb(" + r + "," + g + "," + b + ")";
    },
    rampGreen: function (v) {
      // v in [0,1] → white → green
      const t = Math.max(0, Math.min(1, v));
      const r = Math.round(255 + (47 - 255) * t);
      const g = Math.round(255 + (158 - 255) * t);
      const b = Math.round(255 + (68 - 255) * t);
      return "rgb(" + r + "," + g + "," + b + ")";
    },
    rampSigned: function (v, vmax) {
      // v in [-vmax, vmax] → purple ← white → orange
      const t = Math.max(-1, Math.min(1, v / (vmax || 1)));
      if (t >= 0) return RL.ramp(t);
      const u = -t;
      const r = Math.round(255 + (112 - 255) * u);
      const g = Math.round(255 + (72 - 255) * u);
      const b = Math.round(255 + (232 - 255) * u);
      return "rgb(" + r + "," + g + "," + b + ")";
    },
    txtOn: function (bg) {
      const m = bg.match(/(\d+),(\d+),(\d+)/);
      if (!m) return "#111";
      const lum = 0.299 * +m[1] + 0.587 * +m[2] + 0.114 * +m[3];
      return lum > 150 ? "#111" : "#fff";
    },

    /* rounded-rect path (avoids ctx.roundRect for older browsers) */
    rrect: function (ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    },

    /* ── generic gridworld renderer ─────────────────────────── */
    // o: {x0, y0, cell, cfg, V (2D or null), colorOf(r,c) -> css,
    //     label(r,c) -> string, arrows(r,c) -> [action idx] or null,
    //     agent: {r, c} or null, goalLabel}
    drawGrid: function (ctx, o) {
      const cfg = o.cfg;
      for (let r = 0; r < cfg.rows; r++)
        for (let c = 0; c < cfg.cols; c++) {
          const x = o.x0 + c * o.cell, y = o.y0 + r * o.cell;
          const isGoal = r === cfg.gr && c === cfg.gc;
          const bg = o.colorOf(r, c);
          ctx.fillStyle = bg;
          ctx.fillRect(x, y, o.cell, o.cell);
          ctx.strokeStyle = "#c9c9c9";
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, o.cell, o.cell);
          if (isGoal) {
            ctx.strokeStyle = C.green;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(x + 1.5, y + 1.5, o.cell - 3, o.cell - 3);
          }
          const lab = o.label ? o.label(r, c) : null;
          if (lab !== null && lab !== undefined) {
            ctx.fillStyle = RL.txtOn(bg);
            ctx.font = "700 " + Math.round(o.cell * 0.22) + "px Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(lab, x + o.cell / 2, y + o.cell / 2 - (o.arrows ? o.cell * 0.1 : 0));
            ctx.textBaseline = "alphabetic";
          }
          if (o.arrows && !isGoal) {
            const as = o.arrows(r, c);
            if (as && as.length) {
              ctx.fillStyle = RL.txtOn(bg);
              ctx.font = "700 " + Math.round(o.cell * 0.24) + "px Inter, sans-serif";
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(as.map((a) => RL.DIRS[a].glyph).join(""), x + o.cell / 2, y + o.cell * 0.72);
              ctx.textBaseline = "alphabetic";
            }
          }
        }
      // frame
      ctx.strokeStyle = C.axis;
      ctx.lineWidth = 1.6;
      ctx.strokeRect(o.x0, o.y0, cfg.cols * o.cell, cfg.rows * o.cell);
      if (o.goalLabel !== false) {
        const gx = o.x0 + cfg.gc * o.cell, gy = o.y0 + cfg.gr * o.cell;
        ctx.fillStyle = C.green;
        ctx.font = "800 " + Math.round(o.cell * 0.16) + "px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("GOAL", gx + o.cell / 2, gy + o.cell * 0.22);
      }
      if (o.agent) {
        const ax = o.x0 + o.agent.c * o.cell + o.cell / 2;
        const ay = o.y0 + o.agent.r * o.cell + o.cell / 2;
        LR.dot(ctx, ax, ay, Math.max(6, o.cell * 0.13), C.purple, "#fff");
      }
    },
  });

  /* ══════════════════════════════════════════════════════════
     Fig 0.1 — the hook: an agent learning from reward alone
     ══════════════════════════════════════════════════════════ */
  LR.figs.hook = function (mount) {
    LR.header(
      mount,
      "An agent learns the way home from reward alone",
      "Press Train. The only feedback in this world is +100 at the goal; every value and arrow below is learned live by the section-5 algorithm."
    );

    const cfg = RL.qCfg(4, 4);
    const START = { r: 3, c: 0 };
    let st = RL.qInit(cfg);
    let rand = LR.rng(657);
    let episodes = 0;
    let lastSteps = "–";
    let agent = { r: START.r, c: START.c };
    let timer = null;

    const { cv, ctx, W, H } = LR.canvas(mount, 460, 400, {
      aria: "A 4 by 4 gridworld. A Q-learning agent explores from the bottom-left toward a goal at the top-right; cells darken as learned values grow and arrows show the learned policy.",
    });

    const ro = LR.readout(mount, [
      { k: "ep", label: "episodes" },
      { k: "steps", label: "steps, last episode" },
      { k: "known", label: "states with value" },
    ]);

    const bar = LR.controls(mount);
    const trainBtn = LR.button(bar, "Train ▸", toggleTrain, "primary");
    LR.button(bar, "One episode", function () { stopTrain(); runEpisode(true); });
    LR.button(bar, "Reset ⟲", function () {
      stopTrain();
      st = RL.qInit(cfg); rand = LR.rng(657);
      episodes = 0; lastSteps = "–";
      agent = { r: START.r, c: START.c };
      render();
    });
    const msg = LR.msg(mount);

    // exploration anneals as episodes accumulate (softmax k grows)
    function kNow() { return 0.03 + 0.012 * episodes; }

    function runEpisode(animate) {
      let r = START.r, c = START.c, steps = 0;
      const path = [];
      while (steps < 300) {
        const out = RL.qStep(st, cfg, r, c, { k: kNow(), rand });
        steps++;
        path.push({ r: out.nr, c: out.nc });
        r = out.nr; c = out.nc;
        if (out.done) break;
      }
      episodes++;
      lastSteps = steps;
      if (animate && !LR.reducedMotion) {
        // replay the walked path quickly (the learning already happened above)
        let i = 0;
        const t = setInterval(function () {
          if (i >= path.length) { clearInterval(t); agent = { r: START.r, c: START.c }; render(); return; }
          agent = path[i++]; render();
        }, Math.max(12, 700 / path.length));
      } else {
        agent = { r: START.r, c: START.c };
        render();
      }
    }

    function toggleTrain() {
      if (timer) { stopTrain(); return; }
      trainBtn.textContent = "Pause ❚❚";
      timer = setInterval(function () {
        runEpisode(false);
        if (episodes >= 60) stopTrain();
      }, LR.reducedMotion ? 600 : 180);
    }
    function stopTrain() {
      if (timer) clearInterval(timer);
      timer = null;
      trainBtn.textContent = "Train ▸";
    }

    function maxQ(r, c) { return Math.max(...st.Q[r][c]); }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const cell = 88, x0 = (W - cfg.cols * cell) / 2, y0 = 26;
      RL.drawGrid(ctx, {
        x0, y0, cell, cfg,
        colorOf: function (r, c) {
          if (r === cfg.gr && c === cfg.gc) return RL.rampGreen(0.25);
          return RL.ramp(maxQ(r, c) / 100);
        },
        label: function (r, c) {
          if (r === cfg.gr && c === cfg.gc) return "+100";
          const v = maxQ(r, c);
          return v > 0 ? LR.fmt(v, 1) : "";
        },
        arrows: function (r, c) {
          const v = maxQ(r, c);
          if (v <= 0) return null;
          const mx = Math.max(...st.Q[r][c]);
          return [0, 1, 2, 3].filter((a) => st.Q[r][c][a] > mx - 1e-9);
        },
        agent,
      });
      ctx.fillStyle = LR.C.muted;
      ctx.font = "11.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("cell shade and number: learned value  ·  arrows: learned policy  ·  purple dot: the agent", W / 2, H - 8);

      let known = 0;
      for (let r = 0; r < cfg.rows; r++) for (let c = 0; c < cfg.cols; c++)
        if (!(r === cfg.gr && c === cfg.gc) && maxQ(r, c) > 0) known++;
      ro.set("ep", String(episodes));
      ro.set("steps", String(lastSteps), typeof lastSteps === "number" && lastSteps < 10 ? C.green : undefined);
      ro.set("known", known + " / 15");

      if (episodes === 0) msg.hide();
      else if (known <= 3) msg.show("Early days: the agent has stumbled into the goal, and only the states next to it know anything yet.", "info");
      else if (known < 15) msg.show("Value is radiating backward from the goal, episode by episode. Sections 4 and 5 explain exactly why it flows in that direction.", "info");
      else msg.show("Every state now knows its worth, and the arrows walk straight home. Learned from reward alone: no labels, no correct answers, ever.", "good");
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — exploration vs exploitation, everyday clothes
     ══════════════════════════════════════════════════════════ */
  LR.figs.explore = function (mount) {
    LR.header(
      mount,
      "The same dilemma, four costumes",
      "Every card is one exploit-or-explore choice. Then play the restaurant version for real: 30 nights, your epsilon, genuine random draws."
    );

    const CARDS = [
      { e: "🍜", q: "Dinner tonight", ex: "Exploit: the favourite you know is good.", xp: "Explore: the new place that might be better." },
      { e: "🛢️", q: "Where to drill", ex: "Exploit: the site that already pays.", xp: "Explore: survey an unproven spot." },
      { e: "♟️", q: "Next move", ex: "Exploit: the line you trust.", xp: "Explore: the experimental sacrifice." },
      { e: "📢", q: "Which ad to run", ex: "Exploit: the best click-through so far.", xp: "Explore: a variant you know nothing about." },
    ];
    const grid = LR.el("div", "hook-grid");
    CARDS.forEach(function (cd) {
      const el = LR.el("div", "hook-card");
      el.appendChild(LR.el("div", "hook-emoji", cd.e));
      el.appendChild(LR.el("div", "hook-q", cd.q));
      el.appendChild(LR.el("div", "hook-a", "<b style='color:" + C.orange + "'>" + cd.ex + "</b><br>" + cd.xp));
      grid.appendChild(el);
    });
    mount.appendChild(grid);

    // the restaurant bandit: favourite has a known average of 6.5/10;
    // the new place's true average is hidden until you simulate.
    const FAV_MEAN = 6.5;
    let eps = 0.2;
    let seedBump = 0;

    const bar = LR.controls(mount);
    LR.slider(bar, "ε (explore rate)", 0, 1, 0.05, eps, function (v) { eps = v; }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Simulate 30 nights ▸", simulate, "primary");

    const ro = LR.readout(mount, [
      { k: "total", label: "total enjoyment" },
      { k: "base", label: "always-favourite baseline" },
      { k: "truth", label: "the new place, truly" },
    ]);
    const msg = LR.msg(mount);
    ro.set("total", "–"); ro.set("base", LR.fmtF(30 * FAV_MEAN, 0) + " expected"); ro.set("truth", "hidden until you simulate");

    function simulate() {
      const rand = LR.rng(902 + seedBump++);
      const newMean = 4 + 5 * rand(); // hidden truth, drawn fresh each run
      let Q = [FAV_MEAN, 0], N = [1, 0], total = 0, triedNew = 0;
      for (let night = 0; night < 30; night++) {
        let a;
        if (N[1] === 0 && rand() < eps) a = 1;
        else a = rand() < eps ? (rand() < 0.5 ? 0 : 1) : (Q[0] >= Q[1] ? 0 : 1);
        const mean = a === 0 ? FAV_MEAN : newMean;
        const rwd = Math.max(0, Math.min(10, mean + LR.gauss(rand) * 1.2));
        N[a]++; Q[a] += (rwd - Q[a]) / N[a];
        total += rwd;
        if (a === 1) triedNew++;
      }
      ro.set("total", LR.fmtF(total, 1) + " over 30 nights", C.orange);
      ro.set("truth", LR.fmtF(newMean, 1) + " / 10" + (newMean > FAV_MEAN ? "  (better than the favourite!)" : "  (worse than the favourite)"), newMean > FAV_MEAN ? C.green : C.red);
      const base = 30 * FAV_MEAN;
      if (triedNew === 0) {
        msg.show("You never tried the new place, so you will never know what it was worth. Zero exploration means zero information, forever.", "info");
      } else if (total > base + 5) {
        msg.show("Exploring paid: " + triedNew + " visits to the new place discovered its quality and beat the always-favourite baseline of " + LR.fmtF(base, 0) + ".", "good");
      } else if (total < base - 5) {
        msg.show("Exploring cost you this time: " + triedNew + " experimental nights, and the baseline of " + LR.fmtF(base, 0) + " would have eaten better. Information has a price.", "bad");
      } else {
        msg.show("About even with the baseline of " + LR.fmtF(base, 0) + ". Run it again: the draws are real, and the verdict varies.", "info");
      }
    }
  };

  /* ══════════════════════════════════════════════════════════
     Fig 1.2 — the bandit explorer (signature figure)
     ══════════════════════════════════════════════════════════ */
  LR.figs.bandit = function (mount) {
    LR.header(
      mount,
      "Four machines, hidden payouts, your move",
      "Pull arms by hand, or hand control to a strategy. Every payout is a real draw from a hidden distribution; every bar is the live sample average."
    );

    const K = 4, SD = 1.0;
    let seedBump = 0;
    let means, rand, Q, N, t, total, regret, lastArm, lastPay, revealed;

    function reset() {
      rand = LR.rng(1123 + 977 * seedBump++);
      means = Array.from({ length: K }, () => 1 + 2.5 * rand());
      Q = new Array(K).fill(0);
      N = new Array(K).fill(0);
      t = 0; total = 0; regret = 0; lastArm = -1; lastPay = null; revealed = false;
    }
    reset();

    let strategy = "manual"; // manual | greedy | eps | ucb
    let eps = 0.1, cUCB = 1.0;
    let timer = null;

    // arm buttons
    const armBar = LR.el("div", "fig-controls");
    armBar.setAttribute("role", "group");
    armBar.setAttribute("aria-label", "Slot machines: activate one to pull its arm");
    mount.appendChild(armBar);
    const armBtns = [];
    for (let a = 0; a < K; a++) {
      armBtns.push(LR.button(armBar, "🎰 Pull arm " + (a + 1), (function (aa) {
        return function () { setStrategy("manual"); pull(aa); render(); };
      })(a)));
    }

    // strategy toggles + dials
    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const sBtns = {
      manual: LR.button(grp, "manual", function () { setStrategy("manual"); }, "on"),
      greedy: LR.button(grp, "greedy", function () { setStrategy("greedy"); }),
      eps: LR.button(grp, "ε-greedy", function () { setStrategy("eps"); }),
      ucb: LR.button(grp, "UCB", function () { setStrategy("ucb"); }),
    };
    const playBtn = LR.button(bar, "Auto-play ▸", togglePlay, "primary");
    LR.button(bar, "Step", function () { autoPull(); render(); });

    const bar2 = LR.controls(mount);
    LR.slider(bar2, "ε", 0, 0.5, 0.01, eps, function (v) { eps = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar2, "c (UCB)", 0, 2, 0.1, cUCB, function (v) { cUCB = v; render(); }, (v) => LR.fmtF(v, 1));
    const revBtn = LR.button(bar2, "Reveal true values", function () { revealed = !revealed; render(); });
    LR.button(bar2, "Reset (new machines) ⟲", function () { stopPlay(); reset(); render(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 700, 330, {
      aria: "Bar chart of the four arms' sample-average estimates with play counts, UCB exploration bonuses as whiskers, and true means revealed as green ticks on request",
    });

    const ro = LR.readout(mount, [
      { k: "t", label: "rounds t" },
      { k: "total", label: "total reward" },
      { k: "regret", label: "expected regret" },
      { k: "last", label: "last pull" },
    ]);
    const msg = LR.msg(mount);

    function setStrategy(s) {
      strategy = s;
      Object.keys(sBtns).forEach((k2) => sBtns[k2].classList.toggle("on", k2 === s));
      if (s === "manual") stopPlay();
      render();
    }

    function pull(a) {
      const pay = means[a] + LR.gauss(rand) * SD;
      t++; N[a]++;
      Q[a] += (pay - Q[a]) / N[a]; // incremental sample average
      total += pay;
      regret += Math.max(...means) - means[a]; // expected regret of this choice
      lastArm = a; lastPay = pay;
    }

    function chooseAuto() {
      if (strategy === "greedy") {
        let best = 0;
        for (let a = 1; a < K; a++) if (Q[a] > Q[best]) best = a;
        return best;
      }
      if (strategy === "eps") {
        if (rand() < eps) return Math.floor(rand() * K);
        let best = 0;
        for (let a = 1; a < K; a++) if (Q[a] > Q[best]) best = a;
        return best;
      }
      // UCB: play every arm once first, then argmax of Q + bonus
      for (let a = 0; a < K; a++) if (N[a] === 0) return a;
      let best = 0, bv = -Infinity;
      for (let a = 0; a < K; a++) {
        const v = Q[a] + cUCB * Math.sqrt(Math.log(Math.max(t, 2)) / N[a]);
        if (v > bv) { bv = v; best = a; }
      }
      return best;
    }

    function autoPull() {
      if (strategy === "manual") return;
      pull(chooseAuto());
    }

    function togglePlay() {
      if (timer) { stopPlay(); return; }
      if (strategy === "manual") setStrategy("eps");
      playBtn.textContent = "Pause ❚❚";
      timer = setInterval(function () { autoPull(); render(); }, LR.reducedMotion ? 700 : 160);
    }
    function stopPlay() {
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = "Auto-play ▸";
    }

    function render() {
      revBtn.classList.toggle("on", revealed);
      ctx.clearRect(0, 0, W, H);
      const P = {
        x0: 56, y0: 18, w: W - 80, h: H - 70,
        xmin: -0.5, xmax: K - 0.5, ymin: 0, ymax: 4.5,
        xlabel: "arm a", ylabel: "estimated value Q_t(a)",
        xticks: [0, 1, 2, 3], yticks: [0, 1, 2, 3, 4],
      };
      const { sx, sy } = LR.plot(ctx, P);
      // relabel x ticks as arm numbers
      ctx.fillStyle = "#fff";
      ctx.fillRect(P.x0, P.y0 + P.h + 5, P.w, 14);
      ctx.fillStyle = LR.C.faint;
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "center";
      for (let a = 0; a < K; a++) ctx.fillText("arm " + (a + 1), sx(a), P.y0 + P.h + 16);

      const bw = 64;
      for (let a = 0; a < K; a++) {
        const x = sx(a) - bw / 2;
        const qv = Math.max(0, Math.min(4.5, Q[a]));
        // estimate bar
        ctx.fillStyle = a === lastArm ? C.orange : C.orangeSoft;
        ctx.fillRect(x, sy(qv), bw, sy(0) - sy(qv));
        ctx.strokeStyle = C.orange;
        ctx.lineWidth = 1.4;
        ctx.strokeRect(x, sy(qv), bw, sy(0) - sy(qv));
        // UCB whisker
        if (strategy === "ucb" && N[a] > 0 && t > 1) {
          const bonus = cUCB * Math.sqrt(Math.log(t) / N[a]);
          const topV = Math.min(4.5, Q[a] + bonus);
          ctx.strokeStyle = C.purple;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          ctx.moveTo(sx(a), sy(qv));
          ctx.lineTo(sx(a), sy(Math.max(0, topV)));
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(sx(a) - 9, sy(Math.max(0, topV)));
          ctx.lineTo(sx(a) + 9, sy(Math.max(0, topV)));
          ctx.stroke();
          ctx.fillStyle = C.purple;
          ctx.font = "600 10px Inter, sans-serif";
          ctx.fillText("+" + LR.fmtF(bonus, 2), sx(a) + 24, sy(topV) + 3);
        }
        // true mean tick
        if (revealed) {
          ctx.strokeStyle = C.green;
          ctx.lineWidth = 2.6;
          ctx.beginPath();
          ctx.moveTo(x - 6, sy(means[a]));
          ctx.lineTo(x + bw + 6, sy(means[a]));
          ctx.stroke();
          ctx.fillStyle = C.green;
          ctx.font = "700 10.5px Inter, sans-serif";
          ctx.textAlign = "left";
          ctx.fillText("q* = " + LR.fmtF(means[a], 2), x + bw + 8, sy(means[a]) + 3.5);
          ctx.textAlign = "center";
        }
        // labels on the bar
        ctx.fillStyle = C.text;
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillText(N[a] ? LR.fmtF(Q[a], 2) : "–", sx(a), sy(qv) - 6);
        ctx.fillStyle = LR.C.muted;
        ctx.font = "10.5px Inter, sans-serif";
        ctx.fillText("N = " + N[a], sx(a), sy(0) + 30);
      }

      ro.set("t", String(t));
      ro.set("total", LR.fmtF(total, 1));
      ro.set("regret", LR.fmtF(regret, 1), regret > 8 ? C.red : C.green);
      ro.set("last", lastArm >= 0 ? "arm " + (lastArm + 1) + " paid " + LR.fmtF(lastPay, 2) : "–", C.orange);

      if (t === 0) { msg.hide(); return; }
      const bestArm = means.indexOf(Math.max(...means));
      const bestEst = Q.indexOf(Math.max(...Q));
      if (revealed && bestEst !== bestArm && t > 3) {
        msg.show("Your current best estimate (arm " + (bestEst + 1) + ") is not the true best arm (arm " + (bestArm + 1) + "): a lucky streak is fooling the averages. More exploration would expose it.", "bad");
      } else if (strategy === "greedy" && t > 12) {
        msg.show("Greedy has stopped exploring: whichever arm looked best early is absorbing every play, and any better arm with an unlucky start stays buried.", "info");
      } else if (strategy === "ucb" && t > K) {
        msg.show("Purple whiskers are UCB's uncertainty bonuses: they shrink as an arm's count grows, so trials flow toward whichever plausible best case is highest.", "info");
      } else {
        msg.hide();
      }
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — the bandit → contextual → MDP ladder
     ══════════════════════════════════════════════════════════ */
  LR.figs.ladder = function (mount) {
    LR.header(
      mount,
      "Where does state enter?",
      "Three settings, one diagram growing. Watch what the reward is allowed to depend on, and what the action is allowed to change."
    );

    const RUNGS = [
      {
        name: "Bandit", cond: "r ~ p(· | a)",
        text: "Reward depends on the <b>action only</b>. Playing a machine affects this round's payout and nothing else: next round the casino is exactly as before.",
      },
      {
        name: "Contextual bandit", cond: "r ~ p(· | s, a)",
        text: "A <b>state</b> arrives from outside (which user sees the ad), and the reward depends on state and action. But your action still does not shape the next state.",
      },
      {
        name: "MDP", cond: "(r, s′) ~ p(·, · | s, a)",
        text: "The action now also produces the <b>next state</b>. The loop closes: today's choice shapes tomorrow's situation, and decisions become sequential.",
      },
    ];
    let rung = 0;
    let phase = 0; // animation phase along the arrows
    let timer = null;

    const bar = LR.controls(mount);
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const btns = RUNGS.map(function (rg, i) {
      return LR.button(grp, rg.name, function () { rung = i; phase = 0; render(); });
    });
    const pulseBtn = LR.button(bar, LR.reducedMotion ? "Step" : "Pulse ▸", function () {
      if (LR.reducedMotion) { phase = (phase + 1) % 4; render(); return; }
      if (timer) { clearInterval(timer); timer = null; pulseBtn.textContent = "Pulse ▸"; return; }
      pulseBtn.textContent = "Stop";
      timer = setInterval(function () { phase = (phase + 1) % 4; render(); }, 550);
    }, "primary");

    const narrate = LR.el("div", "stepper-narrate");
    mount.appendChild(narrate);

    const { cv, ctx, W, H } = LR.canvas(mount, 700, 240, {
      aria: "Diagram of agent and environment. In the bandit only an action arrow and a reward arrow exist; the contextual bandit adds a state input; the MDP adds a next-state arrow looping back.",
    });

    function box(x, y, w, h, label, hot) {
      ctx.fillStyle = hot ? C.orange : "#fff";
      ctx.strokeStyle = hot ? C.orange : "#999";
      ctx.lineWidth = 1.8;
      RL.rrect(ctx, x, y, w, h, 10);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = hot ? "#fff" : C.text;
      ctx.font = "700 14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x + w / 2, y + h / 2 + 5);
    }

    function labelArrow(x1, y1, x2, y2, text, color, hot) {
      ctx.save();
      ctx.globalAlpha = hot ? 1 : 0.45;
      LR.arrow(ctx, x1, y1, x2, y2, color, hot ? 2.6 : 1.8);
      if (text) {
        ctx.fillStyle = color;
        ctx.font = (hot ? "800" : "600") + " 12.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(text, (x1 + x2) / 2, Math.min(y1, y2) - 9);
      }
      ctx.restore();
    }

    function render() {
      btns.forEach((b, i) => b.classList.toggle("on", i === rung));
      const rg = RUNGS[rung];
      narrate.innerHTML = "<b>" + rg.name + "</b> · <span style='font-family:var(--mono)'>" + rg.cond + "</span>. " + rg.text;

      ctx.clearRect(0, 0, W, H);
      const AY = 78, AH = 56, AX = 90, AW = 150, EX = W - 90 - 150;
      // phases: 0 idle/state-in, 1 action, 2 reward, 3 next-state (MDP)
      box(AX, AY, AW, AH, "Agent", phase === 1);
      box(EX, AY, AW, AH, "Environment", phase === 2 || phase === 3);

      // action arrow (all rungs)
      labelArrow(AX + AW + 6, AY + 16, EX - 6, AY + 16, "action aₜ", C.orange, phase === 1);
      // reward arrow (all rungs)
      ctx.save();
      ctx.globalAlpha = phase === 2 ? 1 : 0.45;
      LR.arrow(ctx, EX - 6, AY + AH - 12, AX + AW + 6, AY + AH - 12, C.green, phase === 2 ? 2.6 : 1.8);
      ctx.fillStyle = C.green;
      ctx.font = (phase === 2 ? "800" : "600") + " 12.5px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("reward rₜ₊₁", (AX + AW + EX) / 2, AY + AH + 4);
      ctx.restore();
      // state input (contextual + MDP)
      if (rung >= 1) {
        ctx.save();
        ctx.globalAlpha = phase === 0 ? 1 : 0.45;
        LR.arrow(ctx, AX + AW / 2, 26, AX + AW / 2, AY - 6, C.purple, phase === 0 ? 2.6 : 1.8);
        ctx.fillStyle = C.purple;
        ctx.font = (phase === 0 ? "800" : "600") + " 12.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(rung === 1 ? "state sₜ (from outside)" : "state sₜ", AX + AW / 2, 18);
        ctx.restore();
      }
      // next-state loop (MDP only): down from Environment, across, up into Agent
      if (rung === 2) {
        const hot = phase === 3;
        ctx.save();
        ctx.globalAlpha = hot ? 1 : 0.4;
        ctx.strokeStyle = C.purple;
        ctx.lineWidth = hot ? 2.4 : 1.6;
        const yb = AY + AH + 40;
        ctx.beginPath();
        ctx.moveTo(EX + AW / 2, AY + AH + 4);
        ctx.lineTo(EX + AW / 2, yb);
        ctx.lineTo(AX + AW / 2, yb);
        ctx.stroke();
        LR.arrow(ctx, AX + AW / 2, yb, AX + AW / 2, AY + AH + 8, C.purple, hot ? 2.4 : 1.6);
        ctx.fillStyle = C.purple;
        ctx.font = (hot ? "800" : "600") + " 12.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("next state sₜ₊₁: the action changed the world", W / 2, yb + 18);
        ctx.restore();
      }
      // rung annotation
      ctx.fillStyle = LR.C.faint;
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "left";
      const deps = ["reward ← action", "reward ← state + action", "reward AND next state ← state + action"];
      ctx.fillText(deps[rung], 14, H - 10);
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.2 — the agent-environment loop, on a real environment
     ══════════════════════════════════════════════════════════ */
  LR.figs.loop = function (mount) {
    LR.header(
      mount,
      "The RL interaction loop, running",
      "Each press of Step executes one genuine transition in the 3×3 maze (random policy): action out, reward and next state back."
    );

    const cfg = RL.mazeCfg();
    let rand = LR.rng(41);
    let s = { r: 2, c: 0 };
    let t = 0, lastA = -1, lastR = null, flash = 0;

    const grid = LR.el("div", "dual-pane");
    mount.appendChild(grid);
    const left = LR.el("div");
    const right = LR.el("div");
    grid.appendChild(left); grid.appendChild(right);

    const { cv, ctx, W, H } = LR.canvas(left, 380, 210, {
      aria: "Agent and environment boxes with an action arrow flowing right and a reward-and-next-state arrow flowing left",
    });
    right.appendChild(LR.el("div", "pane-label", "live trajectory (real transitions)"));
    const log = LR.el("pre", "attn-runout");
    log.style.minHeight = "150px";
    log.textContent = "press Step to interact\n";
    right.appendChild(log);

    const bar = LR.controls(mount);
    LR.button(bar, "Step ▸", step, "primary");
    LR.button(bar, "Reset ⟲", function () {
      rand = LR.rng(41); s = { r: 2, c: 0 }; t = 0; lastA = -1; lastR = null; flash = 0;
      log.textContent = "press Step to interact\n";
      render();
    });
    const ro = LR.readout(mount, [
      { k: "t", label: "t" },
      { k: "s", label: "state sₜ" },
      { k: "r", label: "last reward" },
    ]);

    function stName(p) { return "(" + p.r + "," + p.c + ")"; }

    function step() {
      const a = Math.floor(rand() * 4); // random policy: the loop is the lesson here
      const { nr, nc } = RL.next(cfg, s.r, s.c, a);
      const done = nr === cfg.gr && nc === cfg.gc;
      const rew = done ? cfg.goalV : cfg.step;
      const line = "t=" + t + "  s=" + stName(s) + "  a=" + RL.DIRS[a].name +
        "  →  r=" + (rew > 0 ? "+" : "") + rew + "  s'=" + stName({ r: nr, c: nc }) + (done ? "  GOAL, restart" : "");
      log.textContent += line + "\n";
      log.scrollTop = log.scrollHeight;
      lastA = a; lastR = rew; t++;
      s = done ? { r: 2, c: 0 } : { r: nr, c: nc };
      if (!LR.reducedMotion) {
        flash = 1; render();
        setTimeout(function () { flash = 2; render(); }, 240);
        setTimeout(function () { flash = 0; render(); }, 480);
      } else {
        render();
      }
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const AY = 62, AH = 52;
      const bx = function (x, label, hot) {
        ctx.fillStyle = hot ? C.orange : "#fff";
        ctx.strokeStyle = hot ? C.orange : "#999";
        ctx.lineWidth = 1.8;
        RL.rrect(ctx, x, AY, 128, AH, 10);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = hot ? "#fff" : C.text;
        ctx.font = "700 13.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(label, x + 64, AY + AH / 2 + 5);
      };
      bx(22, "Agent", flash === 1);
      bx(W - 150, "Environment", flash === 2);
      // arrows
      ctx.globalAlpha = flash === 1 ? 1 : 0.55;
      LR.arrow(ctx, 154, AY + 12, W - 154, AY + 12, C.orange, flash === 1 ? 2.6 : 1.8);
      ctx.fillStyle = C.orange;
      ctx.font = "700 12px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("action " + (lastA >= 0 ? RL.DIRS[lastA].name : "aₜ"), W / 2, AY + 2);
      ctx.globalAlpha = flash === 2 ? 1 : 0.55;
      LR.arrow(ctx, W - 154, AY + AH - 8, 154, AY + AH - 8, C.green, flash === 2 ? 2.6 : 1.8);
      ctx.fillStyle = C.green;
      ctx.fillText("reward " + (lastR === null ? "rₜ₊₁" : (lastR > 0 ? "+" : "") + lastR) + "  ·  next state sₜ₊₁", W / 2, AY + AH + 18);
      ctx.globalAlpha = 1;
      ctx.fillStyle = LR.C.muted;
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText("the loop: sₜ → aₜ → rₜ₊₁, sₜ₊₁ → repeat", W / 2, H - 14);

      ro.set("t", String(t));
      ro.set("s", stName(s));
      ro.set("r", lastR === null ? "–" : (lastR > 0 ? "+" : "") + lastR, lastR > 0 ? C.green : undefined);
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — the discount factor as a planning horizon
     ══════════════════════════════════════════════════════════ */
  LR.figs.discount = function (mount) {
    LR.header(
      mount,
      "γ decides how far ahead the agent cares",
      "Bars are the weights γᵏ on rewards k steps away. Two concrete reward streams are evaluated live as you slide γ."
    );

    let gamma = 0.9;
    let stream = 0; // 0: +1 forever, 1: +10 at k=3

    const bar = LR.controls(mount);
    const gS = LR.slider(bar, "discount γ", 0.1, 0.99, 0.01, gamma, function (v) { gamma = v; render(); }, (v) => LR.fmtF(v, 2));
    const grp = LR.el("div", "toggle-group");
    bar.appendChild(grp);
    const b0 = LR.button(grp, "stream A: +1 every step", function () { stream = 0; render(); }, "on");
    const b1 = LR.button(grp, "stream B: +10, three steps away", function () { stream = 1; render(); });

    const { cv, ctx, W, H } = LR.canvas(mount, 700, 300, {
      aria: "Bar chart of discount weights gamma to the power k for k from 0 to 15, with the selected reward stream's contributions highlighted",
    });
    const ro = LR.readout(mount, [
      { k: "ret", label: "discounted return" },
      { k: "horizon", label: "effective horizon 1/(1−γ)" },
      { k: "w3", label: "weight at k = 3" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      b0.classList.toggle("on", stream === 0);
      b1.classList.toggle("on", stream === 1);
      ctx.clearRect(0, 0, W, H);
      const KMAX = 15;
      const P = {
        x0: 56, y0: 18, w: W - 80, h: H - 74,
        xmin: -0.6, xmax: KMAX + 0.6, ymin: 0, ymax: 1.0,
        xlabel: "steps into the future k", ylabel: "weight γᵏ",
        xticks: [0, 3, 6, 9, 12, 15], yticks: [0, 0.25, 0.5, 0.75, 1],
      };
      const { sx, sy } = LR.plot(ctx, P);
      const bw = (sx(1) - sx(0)) * 0.66;
      for (let k = 0; k <= KMAX; k++) {
        const w = Math.pow(gamma, k);
        const active = stream === 0 || k === 3;
        ctx.fillStyle = active ? C.orange : "#e9ecef";
        ctx.globalAlpha = active ? 0.9 : 1;
        ctx.fillRect(sx(k) - bw / 2, sy(w), bw, sy(0) - sy(w));
        ctx.globalAlpha = 1;
        if (k <= 4 || k === KMAX) {
          ctx.fillStyle = LR.C.muted;
          ctx.font = "10px Inter, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(LR.fmtF(w, 2), sx(k), sy(w) - 5);
        }
      }
      if (stream === 1) {
        ctx.fillStyle = C.green;
        ctx.font = "800 11.5px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("the +10 lives here", sx(3), sy(Math.pow(gamma, 3)) - 20);
      }

      // live returns
      let ret;
      if (stream === 0) {
        ret = 0;
        for (let k = 0; k < 4000; k++) ret += Math.pow(gamma, k);
      } else {
        ret = 10 * Math.pow(gamma, 3);
      }
      ro.set("ret", LR.fmtF(ret, 2) + (stream === 0 ? "  (→ 1/(1−γ) = " + LR.fmtF(1 / (1 - gamma), 2) + ")" : "  (= 10·γ³)"), C.orange);
      ro.set("horizon", LR.fmtF(1 / (1 - gamma), 1) + " steps");
      ro.set("w3", LR.fmtF(Math.pow(gamma, 3), 3));

      if (gamma >= 0.85) {
        msg.show("Far-sighted: at γ = " + LR.fmtF(gamma, 2) + " a reward three steps away keeps " + Math.round(Math.pow(gamma, 3) * 100) + "% of its value. Patience pays, and the maze's long detours can still be worth it.", "good");
      } else if (gamma <= 0.5) {
        msg.show("Near-sighted: at γ = " + LR.fmtF(gamma, 2) + " the same +10 is worth only " + LR.fmtF(10 * Math.pow(gamma, 3), 2) + " today. This agent grabs immediate reward and ignores the future.", "info");
      } else {
        msg.hide();
      }
    }
    render();
  };
})();
