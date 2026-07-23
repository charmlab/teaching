/* ══════════════════════════════════════════════════════════════
   figures-a.js — PCA lesson, foundations half
   Fig 1.1 redundant · Fig 2.1 eigvec · Fig 2.2 covellipse
   Fig 3.1 rotate (signature) · Fig 3.2 projrec (signature)
   Also defines the shared PCA core (LR.pca) used by figures-b.js:
   2x2 closed-form eigendecomposition, general 2x2 eigenpairs,
   sample covariance, cloud sampling, projection/reconstruction,
   and the shared demo cloud. All numbers computed live; nothing
   staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared PCA core (also used by figures-b.js) ────────── */
  const P = (LR.pca = {
    mean2: function (pts) {
      let mx = 0, my = 0;
      pts.forEach((p) => { mx += p[0]; my += p[1]; });
      return [mx / pts.length, my / pts.length];
    },
    center2: function (pts) {
      const m = P.mean2(pts);
      return { m, Xc: pts.map((p) => [p[0] - m[0], p[1] - m[1]]) };
    },
    // sample covariance of centered 2-D points, 1/(N-1) convention (source)
    cov2c: function (Xc) {
      let a = 0, b = 0, d = 0;
      Xc.forEach((p) => { a += p[0] * p[0]; b += p[0] * p[1]; d += p[1] * p[1]; });
      const f = 1 / (Xc.length - 1);
      return [[a * f, b * f], [b * f, d * f]];
    },
    cov2: function (pts) {
      return P.cov2c(P.center2(pts).Xc);
    },
    /* closed-form eigendecomposition of a symmetric 2x2 matrix.
       Returns {l1, l2, u1, u2} with l1 >= l2 and unit, orthogonal u1 ⟂ u2. */
    eig2: function (S) {
      const a = S[0][0], b = S[0][1], d = S[1][1];
      const tr = a + d, det = a * d - b * b;
      const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
      let u1;
      if (Math.abs(b) > 1e-12) u1 = [b, l1 - a];
      else u1 = a >= d ? [1, 0] : [0, 1];
      const n = Math.hypot(u1[0], u1[1]);
      u1 = [u1[0] / n, u1[1] / n];
      return { l1, l2, u1, u2: [-u1[1], u1[0]] };
    },
    /* eigenpairs of a general (possibly non-symmetric) 2x2 matrix.
       Returns null when the eigenvalues are complex (e.g. a rotation). */
    eigGeneral2: function (A) {
      const a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1];
      const tr = a + d, det = a * d - b * c;
      const disc = tr * tr - 4 * det;
      if (disc < -1e-12) return null;
      const s = Math.sqrt(Math.max(0, disc));
      const ls = [(tr + s) / 2, (tr - s) / 2];
      const vecs = ls.map(function (l) {
        let v;
        if (Math.abs(b) > 1e-9) v = [b, l - a];
        else if (Math.abs(c) > 1e-9) v = [l - d, c];
        else v = Math.abs(a - l) < Math.abs(d - l) ? [1, 0] : [0, 1];
        const n = Math.hypot(v[0], v[1]) || 1;
        return [v[0] / n, v[1] / n];
      });
      return { ls, vecs, repeated: s < 1e-9 };
    },
    matvec2: function (A, v) {
      return [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]];
    },
    /* sample n points from a 2-D Gaussian with covariance Strue (mean 0) */
    cloud: function (n, seed, Strue) {
      const rand = LR.rng(seed);
      const e = P.eig2(Strue);
      const s1 = Math.sqrt(Math.max(0, e.l1)), s2 = Math.sqrt(Math.max(0, e.l2));
      const pts = [];
      for (let i = 0; i < n; i++) {
        const g1 = LR.gauss(rand) * s1, g2 = LR.gauss(rand) * s2;
        pts.push([g1 * e.u1[0] + g2 * e.u2[0], g1 * e.u1[1] + g2 * e.u2[1]]);
      }
      return pts;
    },
    /* the shared demo cloud: same covariance shape as the worked example */
    cloud0: function () {
      if (!P.__c0) P.__c0 = P.cloud(90, 20250718, [[2.5, 1.5], [1.5, 1.0]]);
      return P.__c0;
    },
    /* project centered points onto K of the eigenvectors and reconstruct */
    projectRec: function (Xc, eig, K) {
      const dirs = [eig.u1, eig.u2].slice(0, K);
      return Xc.map(function (p) {
        let rx = 0, ry = 0;
        dirs.forEach(function (u) {
          const z = p[0] * u[0] + p[1] * u[1];
          rx += z * u[0]; ry += z * u[1];
        });
        return [rx, ry];
      });
    },
    /* mean squared residual, Frobenius²/(N-1), between points and recs */
    recErr: function (Xc, rec) {
      let s = 0;
      Xc.forEach(function (p, i) {
        s += (p[0] - rec[i][0]) ** 2 + (p[1] - rec[i][1]) ** 2;
      });
      return s / (Xc.length - 1);
    },
  });

  /* ══════════════════════════════════════════════════════════
     Fig 1.1 — two features, one measurement (°C vs °F)
     ══════════════════════════════════════════════════════════ */
  LR.figs.redundant = function (mount) {
    LR.header(
      mount,
      "A 2-D dataset with 1-D content",
      "City temperatures recorded twice: in °C and in °F. Add sensor noise and watch how little the verdict changes."
    );

    const bar = LR.controls(mount);
    let noise = 0;
    const N = 26;
    const baseRand = LR.rng(657);
    const cs = [], gs = [];
    for (let i = 0; i < N; i++) {
      cs.push(-5 + 35 * baseRand());
      gs.push(LR.gauss(baseRand));
    }
    LR.slider(bar, "sensor noise (°F)", 0, 4, 0.1, 0, function (v) { noise = v; render(); }, (v) => LR.fmtF(v, 1));

    const { ctx, W, H } = LR.canvas(mount, 700, 380, {
      aria: "Scatter plot of temperatures in Celsius against Fahrenheit, lying on a line, with adjustable sensor noise",
    });

    const ro = LR.readout(mount, [
      { k: "l1", label: "λ1" },
      { k: "l2", label: "λ2" },
      { k: "evr", label: "variance on 1st direction" },
    ]);

    function data() {
      return cs.map((c, i) => [c, 1.8 * c + 32 + gs[i] * noise]);
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const pts = data();
      const { m, Xc } = P.center2(pts);
      const S = P.cov2c(Xc);
      const e = P.eig2(S);

      const Pl = { x0: 58, y0: 20, w: W - 84, h: H - 66, xmin: -10, xmax: 34, ymin: 10, ymax: 96, xlabel: "temperature (°C)", ylabel: "temperature (°F)" };
      const { sx, sy } = LR.plot(ctx, Pl);

      // principal direction through the mean
      const t = 30;
      ctx.save();
      ctx.beginPath(); ctx.rect(Pl.x0, Pl.y0, Pl.w, Pl.h); ctx.clip();
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(sx(m[0] - t * e.u1[0]), sy(m[1] - t * e.u1[1]));
      ctx.lineTo(sx(m[0] + t * e.u1[0]), sy(m[1] + t * e.u1[1]));
      ctx.stroke();
      ctx.restore();

      pts.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 4, C.text));

      ctx.font = "600 12px Inter, sans-serif";
      ctx.fillStyle = C.orange;
      ctx.textAlign = "left";
      ctx.fillText("principal direction u1", Pl.x0 + 10, Pl.y0 + 18);

      const tot = e.l1 + e.l2;
      ro.set("l1", LR.fmtF(e.l1, 2));
      ro.set("l2", LR.fmtF(e.l2, 3));
      ro.set("evr", LR.fmtF((100 * e.l1) / tot, 2) + "%", C.orange);
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.1 — eigenvectors of a transformation (drag v, see Av)
     ══════════════════════════════════════════════════════════ */
  LR.figs.eigvec = function (mount) {
    LR.header(
      mount,
      "Drag v. Find the directions A cannot turn.",
      "The orange vector is Av, computed live. Along an eigenvector, Av lines up with v and only stretches: Av = λv."
    );

    const PRESETS = [
      { name: "Symmetric stretch", A: [[2, 1], [1, 2]] },
      { name: "Pure scale", A: [[1.6, 0], [0, 0.6]] },
      { name: "Shear", A: [[1, 1], [0, 1]] },
      { name: "Rotation 30°", A: [[Math.cos(Math.PI / 6), -Math.sin(Math.PI / 6)], [Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)]] },
    ];
    let A = PRESETS[0].A;
    let theta = 0.35; // direction of v
    let morph = 1;    // grid morph t: (1-t)I + tA

    const bar = LR.controls(mount);
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = PRESETS.map(function (p, i) {
      const b = LR.button(group, p.name, function () {
        A = p.A;
        btns.forEach((x, j) => x.classList.toggle("on", j === i));
        render();
      });
      if (i === 0) b.classList.add("on");
      b.setAttribute("aria-label", "matrix preset: " + p.name);
      return b;
    });
    LR.slider(bar, "grid morph  I → A", 0, 1, 0.01, 1, function (v) { morph = v; render(); }, (v) => LR.fmtF(v, 2));

    const { cv, ctx, W, H } = LR.canvas(mount, 700, 430, {
      aria: "Interactive plane showing a matrix transforming a grid; drag the vector to find eigenvector directions",
    });
    cv.tabIndex = 0;
    cv.setAttribute("aria-label", "Drag the vector, or use left and right arrow keys to rotate it");

    const msg = LR.msg(mount);
    const ro = LR.readout(mount, [
      { k: "v", label: "v" },
      { k: "av", label: "Av" },
      { k: "ang", label: "angle(v, Av)" },
    ]);

    const SC = { x0: 40, y0: 12, w: W - 66, h: H - 52, xmin: -3.2, xmax: 3.2, ymin: -2.6, ymax: 2.6 };

    function scales() {
      const sx = (x) => SC.x0 + ((x - SC.xmin) / (SC.xmax - SC.xmin)) * SC.w;
      const sy = (y) => SC.y0 + SC.h - ((y - SC.ymin) / (SC.ymax - SC.ymin)) * SC.h;
      return { sx, sy };
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const { sx, sy } = scales();
      const M = [
        [1 + (A[0][0] - 1) * morph, A[0][1] * morph],
        [A[1][0] * morph, 1 + (A[1][1] - 1) * morph],
      ];

      ctx.save();
      ctx.beginPath(); ctx.rect(SC.x0, SC.y0, SC.w, SC.h); ctx.clip();

      // morphed grid
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      for (let g = -4; g <= 4; g++) {
        // vertical line x=g and horizontal line y=g, transformed by M
        [[[g, -4.5], [g, 4.5]], [[-4.5, g], [4.5, g]]].forEach(function (seg) {
          ctx.beginPath();
          seg.forEach(function (pt, i) {
            const q = P.matvec2(M, pt);
            if (i === 0) ctx.moveTo(sx(q[0]), sy(q[1]));
            else ctx.lineTo(sx(q[0]), sy(q[1]));
          });
          ctx.stroke();
        });
      }

      // axes
      ctx.strokeStyle = "#cfcfcf"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(SC.xmin), sy(0)); ctx.lineTo(sx(SC.xmax), sy(0)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(0), sy(SC.ymin)); ctx.lineTo(sx(0), sy(SC.ymax)); ctx.stroke();

      // eigen-directions of the full A (dashed)
      const eg = P.eigGeneral2(A);
      if (eg) {
        ctx.setLineDash([6, 5]);
        eg.vecs.forEach(function (u, i) {
          if (eg.repeated && i === 1) return;
          ctx.strokeStyle = i === 0 ? C.green : C.purple;
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(sx(-4 * u[0]), sy(-4 * u[1]));
          ctx.lineTo(sx(4 * u[0]), sy(4 * u[1]));
          ctx.stroke();
          ctx.font = "700 11.5px Inter, sans-serif";
          ctx.fillStyle = i === 0 ? C.green : C.purple;
          ctx.textAlign = "left";
          ctx.fillText("λ = " + LR.fmt(eg.ls[i], 2), sx(1.9 * u[0]) + 6, sy(1.9 * u[1]) - 6);
        });
        ctx.setLineDash([]);
      }

      // v and Av
      const v = [Math.cos(theta), Math.sin(theta)];
      const av = P.matvec2(A, v);
      const vd = [1.35 * v[0], 1.35 * v[1]];
      const avd = [1.35 * av[0], 1.35 * av[1]];
      LR.arrow(ctx, sx(0), sy(0), sx(avd[0]), sy(avd[1]), C.orange, 3);
      LR.arrow(ctx, sx(0), sy(0), sx(vd[0]), sy(vd[1]), C.text, 3);
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillStyle = C.text; ctx.textAlign = "center";
      ctx.fillText("v", sx(vd[0] * 1.14), sy(vd[1] * 1.14));
      ctx.fillStyle = C.orange;
      ctx.fillText("Av", sx(avd[0] * 1.12), sy(avd[1] * 1.12));
      ctx.restore();

      // alignment check
      const cross = v[0] * av[1] - v[1] * av[0];
      const dot = v[0] * av[0] + v[1] * av[1];
      const nAv = Math.hypot(av[0], av[1]);
      const sinAng = nAv < 1e-9 ? 0 : Math.abs(cross) / nAv;
      const ang = (Math.atan2(Math.abs(cross), dot) * 180) / Math.PI;
      if (sinAng < 0.035 && nAv > 1e-6) {
        const lam = dot >= 0 ? nAv : -nAv;
        msg.show("Eigenvector direction found: Av = " + LR.fmtF(lam, 2) + " · v. The matrix only stretches here, λ ≈ " + LR.fmtF(lam, 2) + ".", "good");
      } else if (!eg) {
        msg.show("This rotation has no real eigenvectors: every direction turns, so you will never line the arrows up.", "info");
      } else {
        msg.hide();
      }

      ro.set("v", "[" + LR.fmtF(v[0], 2) + ", " + LR.fmtF(v[1], 2) + "]");
      ro.set("av", "[" + LR.fmtF(av[0], 2) + ", " + LR.fmtF(av[1], 2) + "]");
      ro.set("ang", LR.fmtF(ang, 1) + "°", ang < 2 ? C.green : C.text);
    }

    LR.drag(cv, W, H, {
      move: function (p) {
        // invert: find data coords of pointer
        const x = SC.xmin + ((p.x - SC.x0) / SC.w) * (SC.xmax - SC.xmin);
        const y = SC.ymin + ((SC.y0 + SC.h - p.y) / SC.h) * (SC.ymax - SC.ymin);
        if (Math.hypot(x, y) < 0.15) return;
        theta = Math.atan2(y, x);
        render();
      },
    });
    cv.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { theta += 0.045; render(); e.preventDefault(); }
      if (e.key === "ArrowRight") { theta -= 0.045; render(); e.preventDefault(); }
    });

    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 2.2 — covariance ellipse + eigenvectors (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.covellipse = function (mount) {
    LR.header(
      mount,
      "The principal directions emerge from the spread",
      "Shape the cloud; the sample covariance, its ellipse, and its eigenvectors are recomputed live."
    );

    let sx1 = 1.6, sx2 = 0.8, rho = 0.6, seed = 11;
    const bar = LR.controls(mount);
    LR.slider(bar, "spread of x1 (σ1)", 0.4, 2.2, 0.05, sx1, function (v) { sx1 = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar, "spread of x2 (σ2)", 0.4, 2.2, 0.05, sx2, function (v) { sx2 = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.slider(bar, "correlation ρ", -0.95, 0.95, 0.05, rho, function (v) { rho = v; render(); }, (v) => LR.fmtF(v, 2));
    LR.button(bar, "Resample ⟲", function () { seed += 1; render(); });

    const { ctx, W, H } = LR.canvas(mount, 700, 430, {
      aria: "Point cloud with live covariance ellipse and eigenvector arrows scaled by the square roots of the eigenvalues",
    });

    const ro = LR.readout(mount, [
      { k: "S", label: "Σ (sample)" },
      { k: "l", label: "λ1, λ2" },
      { k: "ang", label: "u1 angle" },
    ]);

    function sample() {
      const rand = LR.rng(seed);
      const pts = [];
      for (let i = 0; i < 250; i++) {
        const g1 = LR.gauss(rand), g2 = LR.gauss(rand);
        pts.push([sx1 * g1, sx2 * (rho * g1 + Math.sqrt(1 - rho * rho) * g2)]);
      }
      return pts;
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      const pts = sample();
      const { Xc } = P.center2(pts);
      const S = P.cov2c(Xc);
      const e = P.eig2(S);

      const Pl = { x0: 52, y0: 16, w: W - 76, h: H - 62, xmin: -5, xmax: 5, ymin: -4, ymax: 4, xlabel: "feature x1", ylabel: "feature x2" };
      const { sx, sy } = LR.plot(ctx, Pl);

      ctx.save();
      ctx.beginPath(); ctx.rect(Pl.x0, Pl.y0, Pl.w, Pl.h); ctx.clip();

      // points
      Xc.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 2.6, "rgba(17,17,17,0.55)"));

      // 1σ and 2σ ellipses from the eigendecomposition
      const ang = Math.atan2(e.u1[1], e.u1[0]);
      [1, 2].forEach(function (kk) {
        ctx.strokeStyle = kk === 1 ? C.orange : "rgba(26,26,26,0.45)";
        ctx.lineWidth = kk === 1 ? 2.4 : 1.6;
        ctx.beginPath();
        for (let i = 0; i <= 72; i++) {
          const t = (i / 72) * 2 * Math.PI;
          const ex = kk * Math.sqrt(Math.max(0, e.l1)) * Math.cos(t);
          const ey = kk * Math.sqrt(Math.max(0, e.l2)) * Math.sin(t);
          const rx = ex * Math.cos(ang) - ey * Math.sin(ang);
          const ry = ex * Math.sin(ang) + ey * Math.cos(ang);
          if (i === 0) ctx.moveTo(sx(rx), sy(ry));
          else ctx.lineTo(sx(rx), sy(ry));
        }
        ctx.stroke();
      });

      // eigenvector arrows scaled by sqrt(lambda)
      const a1 = Math.sqrt(Math.max(0, e.l1)), a2 = Math.sqrt(Math.max(0, e.l2));
      LR.arrow(ctx, sx(0), sy(0), sx(a1 * e.u1[0]), sy(a1 * e.u1[1]), C.green, 3);
      LR.arrow(ctx, sx(0), sy(0), sx(a2 * e.u2[0]), sy(a2 * e.u2[1]), C.purple, 3);
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "center";
      ctx.fillText("u1·√λ1", sx(a1 * e.u1[0] * 1.28), sy(a1 * e.u1[1] * 1.28) - 4);
      ctx.fillStyle = C.purple;
      ctx.fillText("u2·√λ2", sx(a2 * e.u2[0] * 1.45), sy(a2 * e.u2[1] * 1.45) - 4);
      ctx.restore();

      ro.set("S", "[[" + LR.fmtF(S[0][0], 2) + ", " + LR.fmtF(S[0][1], 2) + "], [" + LR.fmtF(S[1][0], 2) + ", " + LR.fmtF(S[1][1], 2) + "]]");
      ro.set("l", LR.fmtF(e.l1, 2) + ", " + LR.fmtF(e.l2, 2), C.orange);
      ro.set("ang", LR.fmtF((Math.atan2(e.u1[1], e.u1[0]) * 180) / Math.PI, 1) + "°");
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.1 — rotate into the principal axes (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.rotate = function (mount) {
    LR.header(
      mount,
      "Turning correlations off",
      "The cloud rotates until the principal components are the axes; the covariance of the rotated points is recomputed at every frame."
    );

    const Xc = P.center2(P.cloud0()).Xc;
    const e = P.eig2(P.cov2c(Xc));
    const alpha = Math.atan2(e.u1[1], e.u1[0]); // rotate by -alpha to align u1 with x-axis

    let t = 0;        // 0 → original frame, 1 → principal frame
    let timer = null;

    const bar = LR.controls(mount);
    const playBtn = LR.button(bar, LR.reducedMotion ? "Animate (steps) ▸" : "Animate ▸", function () {
      if (timer) { stop(); return; }
      playBtn.textContent = "Pause ❚❚";
      if (LR.reducedMotion) {
        timer = setInterval(function () {
          t = Math.min(1, t + 0.25);
          render();
          if (t >= 1) stop();
        }, 700);
      } else {
        timer = setInterval(function () {
          t = Math.min(1, t + 0.012);
          render();
          if (t >= 1) stop();
        }, 16);
      }
    }, "primary");
    LR.button(bar, "Step ¼", function () { stop(); t = Math.min(1, t + 0.25); render(); });
    LR.button(bar, "Reset ⟲", function () { stop(); t = 0; render(); });
    const sl = LR.slider(bar, "rotation progress", 0, 1, 0.01, 0, function (v) { stop(); t = v; render(); }, (v) => LR.fmtF(v, 2));

    function stop() {
      if (timer) clearInterval(timer);
      timer = null;
      playBtn.textContent = LR.reducedMotion ? "Animate (steps) ▸" : "Animate ▸";
    }

    const { ctx, W, H } = LR.canvas(mount, 700, 420, {
      aria: "Centered point cloud rotating so that its principal components become the coordinate axes",
    });

    const ro = LR.readout(mount, [
      { k: "cov", label: "covariance of rotated points" },
      { k: "off", label: "off-diagonal" },
    ]);

    function render() {
      sl.set(t);
      ctx.clearRect(0, 0, W, H);
      const th = -alpha * t;
      const co = Math.cos(th), si = Math.sin(th);
      const R = Xc.map((p) => [co * p[0] - si * p[1], si * p[0] + co * p[1]]);
      const S = P.cov2c(R);

      const done = t >= 0.999;
      const Pl = {
        x0: 52, y0: 16, w: W - 76, h: H - 62, xmin: -5, xmax: 5, ymin: -3.6, ymax: 3.6,
        xlabel: done ? "z1 (1st principal coordinate)" : "x1",
        ylabel: done ? "z2 (2nd principal coordinate)" : "x2",
      };
      const { sx, sy } = LR.plot(ctx, Pl);

      ctx.save();
      ctx.beginPath(); ctx.rect(Pl.x0, Pl.y0, Pl.w, Pl.h); ctx.clip();

      R.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 3, "rgba(17,17,17,0.6)"));

      // rotating principal axes
      const u1r = [co * e.u1[0] - si * e.u1[1], si * e.u1[0] + co * e.u1[1]];
      const u2r = [-u1r[1], u1r[0]];
      const a1 = Math.sqrt(e.l1), a2 = Math.sqrt(e.l2);
      LR.arrow(ctx, sx(0), sy(0), sx(2 * a1 * u1r[0]), sy(2 * a1 * u1r[1]), C.green, 3);
      LR.arrow(ctx, sx(0), sy(0), sx(2 * a2 * u2r[0]), sy(2 * a2 * u2r[1]), C.purple, 3);
      ctx.font = "700 12.5px Inter, sans-serif";
      ctx.fillStyle = C.green; ctx.textAlign = "center";
      ctx.fillText("u1", sx(2 * a1 * u1r[0] * 1.12), sy(2 * a1 * u1r[1] * 1.12));
      ctx.fillStyle = C.purple;
      ctx.fillText("u2", sx(2 * a2 * u2r[0] * 1.3), sy(2 * a2 * u2r[1] * 1.3));
      ctx.restore();

      ro.set("cov", "[[" + LR.fmtF(S[0][0], 3) + ", " + LR.fmtF(S[0][1], 3) + "], [" + LR.fmtF(S[1][0], 3) + ", " + LR.fmtF(S[1][1], 3) + "]]");
      ro.set("off", LR.fmtF(S[0][1], 4) + (done ? "  → decorrelated; diagonal = (λ1, λ2)" : ""), done ? C.green : C.orange);
    }
    render();
  };

  /* ══════════════════════════════════════════════════════════
     Fig 3.2 — project and reconstruct with a K slider (signature)
     ══════════════════════════════════════════════════════════ */
  LR.figs.projrec = function (mount) {
    LR.header(
      mount,
      "Project, reconstruct, and price the damage",
      "Slide K. The residual segments are what reconstruction loses, and their mean square equals the sum of the discarded eigenvalues."
    );

    const Xc = P.center2(P.cloud0()).Xc;
    const S = P.cov2c(Xc);
    const e = P.eig2(S);
    let K = 1;

    const bar = LR.controls(mount);
    LR.slider(bar, "components kept K", 0, 2, 1, 1, function (v) { K = Math.round(v); render(); }, (v) => String(Math.round(v)));

    const { ctx, W, H } = LR.canvas(mount, 700, 420, {
      aria: "Point cloud projected onto its top K principal components, with residual segments from each point to its reconstruction",
    });

    const ro = LR.readout(mount, [
      { k: "kept", label: "variance kept (Σ kept λ)" },
      { k: "err1", label: "err from residuals" },
      { k: "err2", label: "err = Σ discarded λ" },
      { k: "tot", label: "total (λ1+λ2)" },
    ]);
    const msg = LR.msg(mount);

    function render() {
      ctx.clearRect(0, 0, W, H);
      const rec = P.projectRec(Xc, e, K);
      const errRes = P.recErr(Xc, rec);
      const lams = [e.l1, e.l2];
      const kept = lams.slice(0, K).reduce((a, b) => a + b, 0);
      const errSpec = lams.slice(K).reduce((a, b) => a + b, 0);

      const Pl = { x0: 52, y0: 16, w: W - 76, h: H - 62, xmin: -5, xmax: 5, ymin: -3.6, ymax: 3.6, xlabel: "x1 (centered)", ylabel: "x2 (centered)" };
      const { sx, sy } = LR.plot(ctx, Pl);

      ctx.save();
      ctx.beginPath(); ctx.rect(Pl.x0, Pl.y0, Pl.w, Pl.h); ctx.clip();

      // the kept subspace
      if (K === 1) {
        ctx.strokeStyle = "rgba(102,102,102,0.85)"; ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(sx(-6 * e.u1[0]), sy(-6 * e.u1[1]));
        ctx.lineTo(sx(6 * e.u1[0]), sy(6 * e.u1[1]));
        ctx.stroke();
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillStyle = C.green; ctx.textAlign = "left";
        ctx.fillText("kept subspace: span(u1)", Pl.x0 + 10, Pl.y0 + 18);
      } else if (K === 0) {
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillStyle = C.muted; ctx.textAlign = "left";
        ctx.fillText("kept subspace: just the mean point", Pl.x0 + 10, Pl.y0 + 18);
      } else {
        ctx.font = "700 12px Inter, sans-serif";
        ctx.fillStyle = C.green; ctx.textAlign = "left";
        ctx.fillText("kept subspace: the whole plane (lossless)", Pl.x0 + 10, Pl.y0 + 18);
      }

      // residuals
      ctx.strokeStyle = C.red; ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      Xc.forEach(function (p, i) {
        ctx.beginPath();
        ctx.moveTo(sx(p[0]), sy(p[1]));
        ctx.lineTo(sx(rec[i][0]), sy(rec[i][1]));
        ctx.stroke();
      });
      ctx.setLineDash([]);

      // originals and reconstructions
      Xc.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 3, "rgba(17,17,17,0.55)"));
      rec.forEach((p) => LR.dot(ctx, sx(p[0]), sy(p[1]), 3.4, C.orange));
      ctx.restore();

      const tot = e.l1 + e.l2;
      ro.set("kept", LR.fmtF(kept, 3) + "  (" + LR.fmtF((100 * kept) / tot, 1) + "%)", C.green);
      ro.set("err1", LR.fmtF(errRes, 4), C.red);
      ro.set("err2", LR.fmtF(errSpec, 4), C.red);
      ro.set("tot", LR.fmtF(tot, 3));

      if (K < 2) {
        msg.show("The two error readouts agree: mean squared residual " + LR.fmtF(errRes, 4) + " = sum of discarded eigenvalues " + LR.fmtF(errSpec, 4) + ". Section 4 explains why this is not a coincidence.", "good");
      } else {
        msg.show("K = D: nothing discarded, zero error, and the encoding is just a rotation of the data.", "info");
      }
    }
    render();
  };
})();
