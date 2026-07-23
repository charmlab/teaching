/* ══════════════════════════════════════════════════════════════
   figures-a.js — Language Models lesson figures, sections 0–3
   Fig 0.1 seqprob · Fig 2.1 ngramtable · Fig 2.2 ngramgen
   Fig 2.3 ngramcode · Fig 3.1 tokenizer · Fig 3.2 onehot
   Also defines the shared language core (LR.lm) used by
   figures-b.js: tokenizer, n-gram counter, softmax, sampling,
   cosine similarity. All numbers computed live; nothing staged.
   ══════════════════════════════════════════════════════════════ */
(function () {
  "use strict";
  const LR = window.LR;
  const C = LR.C;

  /* ── shared language core (also used by figures-b.js) ───── */
  const LM = (LR.lm = {
    BOS: "<s>",
    EOS: "</s>",

    // word-level tokenizer used by every counting figure
    tokenize: function (line) {
      return line.toLowerCase().match(/[a-z0-9']+/g) || [];
    },

    // one sentence per line -> array of token arrays
    sentences: function (text) {
      return text.split(/\n+/).map(LM.tokenize).filter(function (s) { return s.length; });
    },

    vocab: function (sents) {
      const set = new Set();
      sents.forEach(function (s) { s.forEach(function (w) { set.add(w); }); });
      return Array.from(set).sort();
    },

    // n-gram counting table: Map(contextKey -> {total, next: Map(word -> count)})
    // Sentences are padded with n-1 BOS tokens and closed with one EOS token,
    // which is what makes p(Sam|am) = 1/2 on the lecture's corpus.
    KEY: "\u0001", // non-printing separator keeps ("a","bc") distinct from ("ab","c")
    ctxKey: function (arr) { return arr.join(LM.KEY); },
    counts: function (sents, n) {
      const table = new Map();
      sents.forEach(function (s) {
        const seq = new Array(n - 1).fill(LM.BOS).concat(s, [LM.EOS]);
        for (let i = n - 1; i < seq.length; i++) {
          const key = LM.ctxKey(seq.slice(i - n + 1, i));
          let row = table.get(key);
          if (!row) { row = { total: 0, next: new Map() }; table.set(key, row); }
          row.total += 1;
          row.next.set(seq[i], (row.next.get(seq[i]) || 0) + 1);
        }
      });
      return table;
    },

    rowOf: function (table, ctxArr) {
      return table.get(LM.ctxKey(ctxArr)) || null;
    },

    // p(w | context); alpha > 0 gives add-alpha smoothing over V outcomes.
    // (Smoothing is not in the lecture slides; it is the standard fix used
    // here only so a never-seen pair gets a tiny nonzero probability instead
    // of zeroing out an entire chain-rule product. Flagged addition.)
    pNext: function (table, ctxArr, w, alpha, V) {
      alpha = alpha || 0;
      const row = LM.rowOf(table, ctxArr);
      const c = row ? row.next.get(w) || 0 : 0;
      const tot = row ? row.total : 0;
      if (!alpha) return tot ? c / tot : 0;
      return (c + alpha) / (tot + alpha * V);
    },

    softmax: function (z, tau) {
      tau = tau || 1;
      let m = -Infinity;
      for (let i = 0; i < z.length; i++) m = Math.max(m, z[i] / tau);
      const ex = z.map(function (v) { return Math.exp(v / tau - m); });
      const s = ex.reduce(function (a, b) { return a + b; }, 0);
      return ex.map(function (v) { return v / s; });
    },

    // inverse-CDF sampling from a probability array
    sample: function (probs, rand) {
      let r = rand();
      for (let i = 0; i < probs.length; i++) {
        r -= probs[i];
        if (r <= 0) return i;
      }
      return probs.length - 1;
    },

    cosine: function (a, b) {
      let d = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
      const den = Math.sqrt(na) * Math.sqrt(nb);
      return den ? d / den : 0;
    },

    /* shared corpora (short synthetic text written for this lesson; the Sam
       corpus is the lecture's own worked example) */
    SAM: "I am Sam\nSam I am\nI do not like green eggs and ham",
    PICNIC: [
      "the sun was warm",
      "the park was quiet",
      "we walked to the park",
      "we ate bread and cheese",
      "the dog ran across the grass",
      "the dog ate the cheese",
      "we laughed at the dog",
      "the rain came fast",
      "we ran home in the rain",
      "the bread was gone",
      "the grass was wet",
      "we sat under a tree",
      "the tree kept us dry",
      "the dog sat with us",
      "the sun came back",
      "we walked home happy",
    ].join("\n"),

    // display helper: pretty context for table headers
    showTok: function (w) { return w === LM.BOS ? "&lt;s&gt;" : w === LM.EOS ? "&lt;/s&gt;" : w; },

    /* fixed-label probability bar list.
       rows: [{key, label}] -> returns {set(key, p, opts), row(key)} */
    barList: function (parent, rows, maxWidthP) {
      const box = LR.el("div", "pbar-list");
      const map = {};
      rows.forEach(function (r) {
        const el = LR.el("div", "pbar-row");
        el.appendChild(LR.el("span", "pbar-label", r.label));
        const track = LR.el("div", "pbar-track");
        const fill = LR.el("div", "pbar-fill");
        track.appendChild(fill);
        el.appendChild(track);
        const val = LR.el("span", "pbar-val", "–");
        el.appendChild(val);
        box.appendChild(el);
        map[r.key] = { el, fill, val };
      });
      parent.appendChild(box);
      return {
        set: function (key, p, opts) {
          opts = opts || {};
          const r = map[key];
          if (!r) return;
          r.fill.style.width = Math.max(0, Math.min(1, p / (maxWidthP || 1))) * 100 + "%";
          r.val.textContent = opts.text !== undefined ? opts.text : p.toFixed(3);
          r.el.classList.toggle("cut", !!opts.cut);
          r.el.classList.toggle("win", !!opts.win);
        },
        row: function (key) { return map[key] ? map[key].el : null; },
      };
    },

    // <select> matching the fig-controls styling
    select: function (parent, label, options, value, onchange) {
      const lab = LR.el("label");
      lab.appendChild(document.createTextNode(label));
      const sel = document.createElement("select");
      options.forEach(function (o) {
        const opt = document.createElement("option");
        opt.value = o; opt.textContent = o;
        sel.appendChild(opt);
      });
      sel.value = value;
      sel.setAttribute("aria-label", label);
      sel.addEventListener("change", function () { onchange(sel.value); });
      lab.appendChild(sel);
      parent.appendChild(lab);
      return sel;
    },
  });

  /* ════════════════════════════════════════════════════════════
     Fig 0.1 — the sequence scorer
     A bigram model scores two candidate sentences via the chain rule.
     ════════════════════════════════════════════════════════════ */

  // small synthetic corpus written for this figure (no copyrighted text)
  const HOOK_CORPUS = [
    "a smart person reads every day",
    "a smart person asks questions",
    "she is a smart person",
    "a kind person listens well",
    "a person can learn anything",
    "that person is smart",
    "he is smart",
    "smart ideas come from questions",
  ];

  LR.figs.seqprob = function (mount) {
    LR.header(
      mount,
      "Score two sentences with a bigram model",
      "The model below was “trained” by counting word pairs in this eight-sentence corpus. Each candidate's probability is the chain-rule product of per-word conditionals."
    );

    const corpusBox = LR.el("div", "");
    corpusBox.style.cssText = "font-family:var(--mono);font-size:12.5px;line-height:1.8;background:#fafafa;border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin-bottom:12px;columns:2;column-gap:26px;";
    corpusBox.innerHTML = HOOK_CORPUS.map(function (s) { return "&ldquo;" + s + "&rdquo;"; }).join("<br>");
    mount.appendChild(corpusBox);

    const sents = LM.sentences(HOOK_CORPUS.join("\n"));
    const table = LM.counts(sents, 2);
    const V = LM.vocab(sents).length + 1; // + EOS as a possible next token
    const ALPHA = 0.1; // add-alpha smoothing so unseen pairs are tiny, not zero

    function scoreSeq(words) {
      const seq = [LM.BOS].concat(words, [LM.EOS]);
      let p = 1;
      const factors = [];
      for (let i = 1; i < seq.length; i++) {
        const raw = LM.pNext(table, [seq[i - 1]], seq[i], 0, V);
        const pi = LM.pNext(table, [seq[i - 1]], seq[i], ALPHA, V);
        factors.push({ ctx: seq[i - 1], w: seq[i], p: pi, unseen: raw === 0 });
        p *= pi;
      }
      return { p, factors };
    }

    const holder = LR.el("div");
    mount.appendChild(holder);
    const msg = LR.msg(mount);

    function fmtP(p) {
      if (p >= 0.01) return p.toFixed(3);
      const e = Math.floor(Math.log10(p));
      return (p / Math.pow(10, e)).toFixed(2) + "×10" + supExp(e);
    }
    function supExp(e) {
      const sup = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
      return String(e).split("").map(function (ch) { return sup[ch]; }).join("");
    }

    function renderSeq(title, words) {
      const res = scoreSeq(words);
      const block = LR.el("div", "seq-block");
      block.appendChild(LR.el("div", "seq-title", "“" + words.join(" ") + "”"));
      const strip = LR.el("div", "factor-strip");
      res.factors.forEach(function (f, i) {
        if (i > 0) strip.appendChild(LR.el("span", "factor-times", "×"));
        const chip = LR.el("div", "factor" + (f.unseen ? " tiny" : ""));
        chip.innerHTML =
          "p(" + LM.showTok(f.w) + " | " + LM.showTok(f.ctx) + ")" +
          '<span class="f-p">' + fmtP(f.p) + (f.unseen ? " ⚠" : "") + "</span>";
        if (f.unseen) chip.title = "this pair never occurs in the corpus; the value is the smoothed floor";
        strip.appendChild(chip);
      });
      block.appendChild(strip);
      block.appendChild(LR.el("div", "seq-score", "product:&ensp;p(sequence) = <b>" + fmtP(res.p) + "</b>"));
      return { block, p: res.p };
    }

    let custom = null;
    function renderAll() {
      holder.innerHTML = "";
      const a = renderSeq("A", ["a", "smart", "person"]);
      const b = renderSeq("B", ["a", "person", "smart"]);
      (a.p >= b.p ? a : b).block.classList.add("winner");
      holder.appendChild(a.block);
      holder.appendChild(b.block);
      if (custom) holder.appendChild(custom.block);
      const ratio = a.p / b.p;
      msg.show(
        "“a smart person” comes out about " + LR.fmt(ratio, 0) +
        "× more likely than “a person smart”. Same three words; the order carries the probability.",
        "good"
      );
    }

    // score-your-own input
    const bar = LR.controls(mount);
    const lab = LR.el("label");
    lab.appendChild(document.createTextNode("your own sentence"));
    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = "a kind person is smart";
    inp.setAttribute("aria-label", "sentence to score with the bigram model");
    inp.style.cssText = "font-family:var(--mono);font-size:13px;padding:4px 8px;border:1px solid #cccccc;border-radius:6px;width:min(280px,60vw)";
    lab.appendChild(inp);
    bar.appendChild(lab);
    const vocabSet = new Set(LM.vocab(sents));
    LR.button(bar, "Score it", function () {
      const words = LM.tokenize(inp.value);
      if (!words.length) return;
      const bad = words.filter(function (w) { return !vocabSet.has(w); });
      if (bad.length) {
        msg.show("“" + bad[0] + "” is not in this tiny corpus vocabulary, so the model cannot score it. Stick to words from the corpus above.", "bad");
        return;
      }
      custom = renderSeq("C", words);
      renderAll();
      custom = null; // one-shot: the block stays until the next re-render
    }, "primary");
    inp.addEventListener("keydown", function (e) { if (e.key === "Enter") bar.querySelector(".btn").click(); });

    renderAll();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.1 — the n-gram counting table (signature figure)
     ════════════════════════════════════════════════════════════ */
  LR.figs.ngramtable = function (mount) {
    LR.header(
      mount,
      "Build the counting table from a corpus you control",
      "Edit the corpus (one sentence per line): the table, the sparsity readout, and the query recompute live. Click a row to see it as a next-word distribution."
    );

    const ta = LR.el("textarea", "corpus-box");
    ta.value = LM.SAM;
    ta.setAttribute("aria-label", "training corpus, one sentence per line");
    ta.rows = 3;
    mount.appendChild(ta);

    const bar = LR.controls(mount);
    let n = 2;
    let selKey = null;
    const nSlider = LR.slider(bar, "n (context + 1)", 1, 4, 1, 2, function (v) {
      n = Math.round(v);
      rebuild(true);
    }, function (v) { return String(Math.round(v)); });
    LR.button(bar, "Load Sam corpus", function () { ta.value = LM.SAM; rebuild(true); });
    LR.button(bar, "Load picnic corpus", function () { ta.value = LM.PICNIC; rebuild(true); });

    const wrap = LR.el("div", "ngram-wrap");
    mount.appendChild(wrap);

    const ro = LR.readout(mount, [
      { k: "V", label: "|V| (incl. specials)" },
      { k: "cells", label: "possible cells |V|ⁿ" },
      { k: "nz", label: "nonzero cells" },
      { k: "pct", label: "table filled" },
    ]);

    // distribution panel
    const distTitle = LR.el("div", "fig-title", "");
    distTitle.style.marginTop = "14px";
    mount.appendChild(distTitle);
    const distBox = LR.el("div");
    mount.appendChild(distBox);

    // query row
    const qbar = LR.controls(mount);
    qbar.appendChild(LR.el("span", "", "<b>query</b>&ensp;p( next | context ):"));
    const qCtx = document.createElement("select");
    const qW = document.createElement("select");
    [qCtx, qW].forEach(function (s) {
      s.setAttribute("aria-label", s === qCtx ? "query context" : "query next word");
      qbar.appendChild(s);
    });
    const qOut = LR.el("span", "val", "–");
    qbar.appendChild(qOut);

    let table, sents, vlist;

    function rebuild(resetSel) {
      sents = LM.sentences(ta.value);
      if (!sents.length) { wrap.innerHTML = ""; return; }
      table = LM.counts(sents, n);
      vlist = LM.vocab(sents);
      const Vfull = vlist.length + 2; // + BOS + EOS
      const cols = vlist.concat([LM.EOS]); // possible next tokens
      if (resetSel) selKey = null;

      // rows sorted by total count, capped for display
      let keys = Array.from(table.keys());
      keys.sort(function (a, b) {
        return table.get(b).total - table.get(a).total || (a < b ? -1 : 1);
      });
      const ROWCAP = 40, COLCAP = 18;
      const shownKeys = keys.slice(0, ROWCAP);
      const shownCols = cols.slice(0, COLCAP);

      let html = '<table class="ngram"><thead><tr><th class="rowhead">context \\ next</th>';
      shownCols.forEach(function (c) { html += "<th>" + LM.showTok(c) + "</th>"; });
      if (cols.length > COLCAP) html += "<th>…+" + (cols.length - COLCAP) + "</th>";
      html += '<th>total</th></tr></thead><tbody>';
      shownKeys.forEach(function (key) {
        const row = table.get(key);
        const label = key.split(LM.KEY).map(LM.showTok).join(" ") || "(no context)";
        html += '<tr data-key="' + encodeURIComponent(key) + '"' + (key === selKey ? ' class="sel"' : "") + '>';
        html += '<td class="rowhead">' + label + "</td>";
        shownCols.forEach(function (c) {
          const v = row.next.get(c) || 0;
          html += v ? '<td class="hit" title="p=' + (v / row.total).toFixed(3) + '">' + v + "</td>" : '<td class="zero">·</td>';
        });
        if (cols.length > COLCAP) html += '<td class="zero">…</td>';
        html += "<td><b>" + row.total + "</b></td></tr>";
      });
      html += "</tbody></table>";
      wrap.innerHTML = html;
      if (keys.length > ROWCAP) {
        wrap.insertAdjacentHTML("beforeend",
          '<div style="font-family:var(--sans);font-size:12px;color:var(--faint);padding:6px 10px">showing the ' + ROWCAP + " most frequent of " + keys.length + " observed contexts</div>");
      }
      wrap.querySelectorAll("tbody tr").forEach(function (tr) {
        tr.addEventListener("click", function () {
          selKey = decodeURIComponent(tr.getAttribute("data-key"));
          rebuild(false);
        });
      });

      // sparsity readout: possible vs observed
      let nz = 0;
      table.forEach(function (row) { nz += row.next.size; });
      const cells = Math.pow(Vfull, n);
      ro.set("V", String(Vfull));
      ro.set("cells", cells >= 1e6 ? cells.toExponential(2) : String(Math.round(cells)));
      ro.set("nz", String(nz));
      const pct = (100 * nz) / cells;
      ro.set("pct", (pct < 0.01 ? pct.toExponential(1) : LR.fmt(pct, 2)) + "%", pct < 1 ? C.red : C.green);

      // distribution panel for selected (or default) row
      if (!selKey || !table.has(selKey)) {
        selKey = table.has(LM.ctxKey(["i"])) && n === 2 ? LM.ctxKey(["i"]) : keys[0];
      }
      const selRow = table.get(selKey);
      const label = selKey.split(LM.KEY).map(LM.showTok).join(" ") || "(no context)";
      distTitle.innerHTML = "row “" + label + "” as a distribution &middot; count(context) = " + selRow.total;
      distBox.innerHTML = "";
      const entries = Array.from(selRow.next.entries()).sort(function (a, b) { return b[1] - a[1]; });
      const bl = LM.barList(distBox, entries.map(function (e) { return { key: e[0], label: LM.showTok(e[0]) }; }));
      entries.forEach(function (e) {
        bl.set(e[0], e[1] / selRow.total, { text: e[1] + "/" + selRow.total + " = " + (e[1] / selRow.total).toFixed(3) });
      });

      // query selects: contexts observed, next tokens
      const prevCtx = qCtx.value, prevW = qW.value;
      qCtx.innerHTML = "";
      keys.forEach(function (key) {
        const o = document.createElement("option");
        o.value = key;
        o.textContent = key.split(LM.KEY).map(function (w) { return w === LM.BOS ? "<s>" : w; }).join(" ") || "(none)";
        qCtx.appendChild(o);
      });
      qW.innerHTML = "";
      cols.forEach(function (c) {
        const o = document.createElement("option");
        o.value = c;
        o.textContent = c === LM.EOS ? "</s>" : c;
        qW.appendChild(o);
      });
      if (Array.from(qCtx.options).some(function (o) { return o.value === prevCtx; })) qCtx.value = prevCtx;
      else if (n === 2 && table.has(LM.ctxKey(["i"]))) qCtx.value = LM.ctxKey(["i"]);
      if (Array.from(qW.options).some(function (o) { return o.value === prevW; })) qW.value = prevW;
      else if (qCtx.value === LM.ctxKey(["i"]) && vlist.indexOf("am") >= 0) qW.value = "am";
      updateQuery();
    }

    function updateQuery() {
      const row = table.get(qCtx.value);
      if (!row) { qOut.textContent = "context unseen → 0/0"; return; }
      const c = row.next.get(qW.value) || 0;
      qOut.textContent = c + "/" + row.total + " = " + (c / row.total).toFixed(3);
      qOut.style.color = c === 0 ? C.red : C.text;
    }
    qCtx.addEventListener("change", updateQuery);
    qW.addEventListener("change", updateQuery);

    let deb;
    ta.addEventListener("input", function () {
      clearTimeout(deb);
      deb = setTimeout(function () { rebuild(true); }, 250);
    });

    rebuild(true);
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.2 — the n-gram text generator
     ════════════════════════════════════════════════════════════ */
  LR.figs.ngramgen = function (mount) {
    LR.header(
      mount,
      "Generate text by sampling from the table",
      "Each step: read the last n−1 words, look up their row, sample the next word from that distribution, append, repeat."
    );

    const bar = LR.controls(mount);
    let corpusName = "picnic", n = 2, seedCounter = 1;
    LM.select(bar, "corpus", ["picnic", "Sam"], "picnic", function (v) { corpusName = v; reset(); });
    LR.slider(bar, "n", 1, 4, 1, 2, function (v) { n = Math.round(v); reset(); },
      function (v) { return String(Math.round(v)); });
    const stepBtn = LR.button(bar, "Step ▸", step, "primary");
    let playing = null;
    const playBtn = LR.button(bar, "Play ▸▸", function () {
      if (playing) { stopPlay(); return; }
      playBtn.textContent = "Pause ❚❚";
      playing = setInterval(step, LR.reducedMotion ? 1100 : 450);
    });
    LR.button(bar, "Reset ⟲", reset);

    const out = LR.el("div", "gen-out");
    out.setAttribute("aria-live", "polite");
    mount.appendChild(out);

    const distTitle = LR.el("div", "fig-sub", "");
    mount.appendChild(distTitle);
    const distBox = LR.el("div");
    mount.appendChild(distBox);

    const ro = LR.readout(mount, [
      { k: "steps", label: "tokens sampled" },
      { k: "sents", label: "sentences finished" },
      { k: "choices", label: "options in current row" },
    ]);

    let table, tokens, done, sentCount, rand;
    const MAXTOK = 60;

    function corpusText() { return corpusName === "Sam" ? LM.SAM : LM.PICNIC; }

    function reset() {
      stopPlay();
      table = LM.counts(LM.sentences(corpusText()), n);
      tokens = [];
      done = false;
      sentCount = 0;
      rand = LR.rng(1000 + 97 * seedCounter++);
      render(null, null);
      ro.set("steps", "0");
      ro.set("sents", "0");
      showRow();
    }

    function context() {
      // last n-1 tokens of the current sentence, padded with BOS
      const cur = [];
      for (let i = tokens.length - 1; i >= 0 && cur.length < n - 1; i--) {
        if (tokens[i] === "·") break; // sentence separator
        cur.unshift(tokens[i]);
      }
      while (cur.length < n - 1) cur.unshift(LM.BOS);
      return cur;
    }

    function showRow() {
      const ctx = context();
      const row = LM.rowOf(table, ctx);
      distBox.innerHTML = "";
      const label = ctx.map(LM.showTok).join(" ") || "(no context)";
      if (!row) {
        distTitle.innerHTML = "next-word distribution for context “" + label + "”: <b>context never observed</b>";
        ro.set("choices", "0", C.red);
        return null;
      }
      distTitle.innerHTML = "next-word distribution for context “<b>" + label + "</b>”:";
      const entries = Array.from(row.next.entries()).sort(function (a, b) { return b[1] - a[1]; }).slice(0, 8);
      const bl = LM.barList(distBox, entries.map(function (e) { return { key: e[0], label: LM.showTok(e[0]) }; }));
      entries.forEach(function (e) { bl.set(e[0], e[1] / row.total, { text: (e[1] / row.total).toFixed(3) }); });
      ro.set("choices", String(row.next.size), row.next.size === 1 ? C.red : C.text);
      return { row, bl };
    }

    function step() {
      if (done) { stopPlay(); return; }
      const ctx = context();
      const row = LM.rowOf(table, ctx);
      if (!row) { done = true; stopPlay(); return; }
      const entries = Array.from(row.next.entries());
      const probs = entries.map(function (e) { return e[1] / row.total; });
      const pick = entries[LM.sample(probs, rand)][0];
      if (pick === LM.EOS) {
        tokens.push("·");
        sentCount += 1;
        ro.set("sents", String(sentCount));
      } else {
        tokens.push(pick);
      }
      const stepsSoFar = tokens.filter(function (t) { return t !== "·"; }).length;
      ro.set("steps", String(stepsSoFar));
      if (tokens.length >= MAXTOK) { done = true; stopPlay(); }
      render(pick === LM.EOS ? "·" : pick, context());
      showRow();
    }

    function render(newTok, ctxNow) {
      const ctxSet = ctxNow ? ctxNow.filter(function (w) { return w !== LM.BOS; }) : [];
      let html = "";
      const ctxStart = tokens.length - ctxSet.length;
      tokens.forEach(function (t, i) {
        const isNew = newTok !== null && i === tokens.length - 1;
        const isCtx = !isNew && i >= ctxStart;
        if (isNew) html += '<span class="g-new">' + t + "</span> ";
        else if (isCtx) html += '<span class="g-ctx">' + t + "</span> ";
        else html += t + " ";
      });
      out.innerHTML = html || '<span style="color:var(--faint)">press Step to sample the first word</span>';
      if (done) out.innerHTML += ' <span style="color:var(--faint)">(stopped)</span>';
    }

    function stopPlay() {
      if (playing) clearInterval(playing);
      playing = null;
      playBtn.textContent = "Play ▸▸";
    }

    reset();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 2.3 — run ngram_counts / p_next yourself
     ════════════════════════════════════════════════════════════ */
  LR.figs.ngramcode = function (mount) {
    LR.header(
      mount,
      "Run ngram_counts and p_next yourself",
      "The two functions from the lesson execute against this corpus and print the real count row and distribution for your context."
    );

    const wrap = LR.el("div", "code-exercise");
    mount.appendChild(wrap);

    const controls = LR.el("div", "code-controls");
    wrap.appendChild(controls);

    let n = 2;
    LM.select(controls, "n", ["1", "2", "3"], "2", function (v) { n = parseInt(v, 10); });
    const ctxLab = LR.el("label", "", "context ");
    const ctxInp = document.createElement("input");
    ctxInp.type = "text";
    ctxInp.value = "i";
    ctxInp.setAttribute("aria-label", "query context, n minus 1 words");
    ctxInp.style.cssText = "font-family:var(--mono);font-size:13px;padding:3px 8px;border:1px solid #cccccc;border-radius:6px;width:130px";
    ctxLab.appendChild(ctxInp);
    controls.appendChild(ctxLab);
    LR.button(controls, "Run ▸", run, "primary small");

    const ta = LR.el("textarea", "corpus-box");
    ta.value = LM.SAM;
    ta.rows = 3;
    ta.style.cssText = "border-radius:0;border-left:none;border-right:none;margin:0";
    ta.setAttribute("aria-label", "editable corpus for the code exercise");
    wrap.appendChild(ta);

    const out = LR.el("div", "run-output", "&gt;&gt;&gt; p_next(ngram_counts(corpus, 2), ('i',))\n(press Run)");
    out.style.whiteSpace = "pre-wrap";
    wrap.appendChild(out);

    function pyTuple(arr) {
      return "(" + arr.map(function (w) { return "'" + w + "'"; }).join(", ") + (arr.length === 1 ? "," : "") + ")";
    }

    function run() {
      const sents = LM.sentences(ta.value);
      if (!sents.length) { out.textContent = "corpus is empty"; return; }
      const ctx = LM.tokenize(ctxInp.value.replace(/<s>/g, " bosmarker ")).map(function (w) {
        return w === "bosmarker" ? LM.BOS : w;
      });
      let text = "&gt;&gt;&gt; counts = ngram_counts(corpus, " + n + ")\n";
      if (ctx.length !== n - 1) {
        text += "&gt;&gt;&gt; p_next(counts, " + pyTuple(ctx) + ")\n";
        text += "ValueError: an n=" + n + " model conditions on exactly " + (n - 1) +
          " word" + (n === 2 ? "" : "s") + "; got " + ctx.length;
        out.innerHTML = text;
        return;
      }
      const table = LM.counts(sents, n);
      const row = LM.rowOf(table, ctx);
      text += "&gt;&gt;&gt; counts[" + pyTuple(ctx) + "]\n";
      if (!row) {
        text += "KeyError: " + pyTuple(ctx) + "   # this context never occurs in the corpus\n";
        text += "&gt;&gt;&gt; p_next(counts, " + pyTuple(ctx) + ")\n{}   # no counts, no distribution: the sparsity problem in one line";
        out.innerHTML = text;
        return;
      }
      const entries = Array.from(row.next.entries()).sort(function (a, b) { return b[1] - a[1]; });
      text += "{" + entries.map(function (e) { return "'" + e[0] + "': " + e[1]; }).join(", ") + "}   # total " + row.total + "\n";
      text += "&gt;&gt;&gt; p_next(counts, " + pyTuple(ctx) + ")\n";
      text += "{" + entries.map(function (e) {
        return "'" + e[0] + "': " + (e[1] / row.total).toFixed(4).replace(/0+$/, "").replace(/\.$/, ".0");
      }).join(", ") + "}";
      out.innerHTML = text;
    }
    ctxInp.addEventListener("keydown", function (e) { if (e.key === "Enter") run(); });
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.1 — the tokenizer playground
     ════════════════════════════════════════════════════════════ */

  // toy subword vocabulary, curated for the default text (flagged addition:
  // real subword tokenizers learn their merge table from data; this fixed
  // list stands in so the greedy longest-match behaviour is inspectable)
  const SUBWORDS = [
    "un", "believ", "ably", "able", "token", "iz", "ation", "ing", "ed",
    "the", "at", "of", "to", "in", "student", "s", "love", "water", "loo",
    "univers", "ity", "new", "york", "study", "land", "model", "lang", "uage",
    "er", "es", "a", "and", "we", "read", "chat", "bot", "smart", "person",
  ];
  const ENTITIES = ["university of waterloo", "new york", "machine learning", "language model"];

  LR.figs.tokenizer = function (mount) {
    LR.header(
      mount,
      "One string, three tokenizations",
      "Type anything. Word-level splits on boundaries, subword greedily matches pieces from a small vocabulary, and named-entity keeps known names whole."
    );

    const bar = LR.controls(mount);
    const lab = LR.el("label");
    lab.appendChild(document.createTextNode("text"));
    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = "Unbelievably, the students at the University of Waterloo love tokenization.";
    inp.setAttribute("aria-label", "text to tokenize");
    inp.style.cssText = "font-family:var(--mono);font-size:13px;padding:4px 8px;border:1px solid #cccccc;border-radius:6px;width:min(420px,64vw)";
    lab.appendChild(inp);
    bar.appendChild(lab);

    let mode = "word";
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const modeBtns = {};
    [["word", "word-level"], ["subword", "subword"], ["entity", "named-entity"]].forEach(function (m) {
      modeBtns[m[0]] = LR.button(group, m[1], function () {
        mode = m[0];
        Object.keys(modeBtns).forEach(function (k) { modeBtns[k].classList.toggle("on", k === mode); });
        render();
      });
    });
    modeBtns.word.classList.add("on");

    const chips = LR.el("div", "tok-chips");
    mount.appendChild(chips);
    const ro = LR.readout(mount, [
      { k: "count", label: "tokens" },
      { k: "vocab", label: "vocab size (this text)" },
    ]);
    const msg = LR.msg(mount);

    function wordTokens(text) {
      return text.toLowerCase().match(/[a-z0-9']+|[^\sa-z0-9']/g) || [];
    }

    function entityTokens(text) {
      let t = text.toLowerCase();
      // merge known entities into single tokens with underscores
      ENTITIES.forEach(function (e) {
        t = t.split(e).join(e.replace(/[\s.]+/g, "_"));
      });
      return t.match(/[a-z0-9'_]+|[^\sa-z0-9'_]/g) || [];
    }

    function subwordTokens(text) {
      const words = wordTokens(text);
      const pieces = [];
      words.forEach(function (w) {
        if (!/^[a-z0-9']+$/.test(w)) { pieces.push({ t: w, cont: false }); return; }
        let i = 0, first = true;
        while (i < w.length) {
          let best = null;
          SUBWORDS.forEach(function (sw) {
            if (w.startsWith(sw, i) && (!best || sw.length > best.length)) best = sw;
          });
          if (!best) best = w[i]; // fall back to a single character
          pieces.push({ t: best, cont: !first });
          i += best.length;
          first = false;
        }
      });
      return pieces;
    }

    function render() {
      const raw = mode === "subword"
        ? subwordTokens(inp.value)
        : (mode === "entity" ? entityTokens(inp.value) : wordTokens(inp.value)).map(function (t) {
          return { t: t, cont: false };
        });
      // token IDs: alphabetical index into this text's vocabulary (1-based)
      const types = Array.from(new Set(raw.map(function (p) { return (p.cont ? "##" : "") + p.t; }))).sort();
      const idOf = new Map(types.map(function (t, i) { return [t, i + 1]; }));
      chips.innerHTML = "";
      raw.forEach(function (p) {
        const shown = (p.cont ? "##" : "") + p.t;
        const chip = LR.el("div", "tok-chip" + (/^[a-z0-9'#_]+$/.test(shown) ? "" : " special"));
        chip.innerHTML = shown.replace(/</g, "&lt;") + '<span class="tid">id ' + idOf.get(shown) + "</span>";
        chips.appendChild(chip);
      });
      ro.set("count", String(raw.length));
      ro.set("vocab", String(types.length));
      if (mode === "subword") {
        msg.show("## marks a piece that continues the previous one. Rare words become several known pieces, so nothing is ever fully “out of vocabulary”.", "info");
      } else if (mode === "entity") {
        msg.show("Known names collapse into single tokens (underscored), so “University of Waterloo” is one unit of meaning instead of three words.", "info");
      } else {
        msg.hide();
      }
    }

    let deb;
    inp.addEventListener("input", function () {
      clearTimeout(deb);
      deb = setTimeout(render, 150);
    });
    render();
  };

  /* ════════════════════════════════════════════════════════════
     Fig 3.2 — token IDs on a number line vs one-hot vectors
     ════════════════════════════════════════════════════════════ */
  LR.figs.onehot = function (mount) {
    LR.header(
      mount,
      "ID distance is not meaning distance",
      "Three words, IDs assigned by vocabulary order. The number line invents a similarity that is not there; one-hot vectors erase it, along with all similarity."
    );

    // the lecture's |251-255| < |251-999| example, instantiated with words
    // whose meaning-similarity deliberately mismatches their ID proximity
    const WORDS = [
      { w: "pizza", id: 251, color: C.orange },
      { w: "pillow", id: 255, color: C.purple },
      { w: "burger", id: 999, color: C.green },
    ];
    const VSIZE = 1000;

    const bar = LR.controls(mount);
    let view = "ids";
    const group = LR.el("div", "toggle-group");
    bar.appendChild(group);
    const btns = {};
    [["ids", "token IDs"], ["onehot", "one-hot vectors"]].forEach(function (m) {
      btns[m[0]] = LR.button(group, m[1], function () {
        view = m[0];
        Object.keys(btns).forEach(function (k) { btns[k].classList.toggle("on", k === view); });
        render();
      });
    });
    btns.ids.classList.add("on");

    const { ctx, W, H } = LR.canvas(mount, 820, 300, {
      aria: "Token IDs on a number line, or the same words as one-hot vectors with all pairwise dot products zero",
    });
    const msg = LR.msg(mount);

    // real one-hot vectors, so the dot products below are actually computed
    const onehots = WORDS.map(function (d) {
      const v = new Float64Array(VSIZE);
      v[d.id - 1] = 1;
      return v;
    });
    function dot(a, b) {
      let s = 0;
      for (let i = 0; i < a.length; i++) s += a[i] * b[i];
      return s;
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      ctx.font = "600 12px Inter, sans-serif";
      if (view === "ids") {
        const P = { x0: 60, y0: 40, w: W - 120, h: 150, xmin: 0, xmax: VSIZE + 50, ymin: 0, ymax: 1, xlabel: "token ID", xticks: [0, 200, 400, 600, 800, 1000], yticks: [] };
        const { sx } = LR.plot(ctx, P);
        const yLine = P.y0 + 96;
        ctx.strokeStyle = C.axis;
        ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(sx(0), yLine); ctx.lineTo(sx(VSIZE + 50), yLine); ctx.stroke();
        WORDS.forEach(function (d, i) {
          LR.dot(ctx, sx(d.id), yLine, 6.5, d.color);
          ctx.fillStyle = d.color;
          ctx.textAlign = "center";
          ctx.font = "700 13px Inter, sans-serif";
          const dy = i === 1 ? 30 : -16; // stagger pizza/pillow labels
          ctx.fillText(d.w + " (" + d.id + ")", sx(d.id), yLine + dy);
        });
        // distance braces (computed)
        const d01 = Math.abs(WORDS[0].id - WORDS[1].id);
        const d02 = Math.abs(WORDS[0].id - WORDS[2].id);
        ctx.strokeStyle = C.red;
        ctx.fillStyle = C.red;
        ctx.lineWidth = 1.4;
        const y1 = yLine - 44;
        ctx.beginPath(); ctx.moveTo(sx(WORDS[0].id), y1); ctx.lineTo(sx(WORDS[1].id), y1); ctx.stroke();
        ctx.font = "600 12px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("|251−255| = " + d01 + "  “close”?", (sx(WORDS[0].id) + sx(WORDS[1].id)) / 2 + 40, y1 - 8);
        const y2 = yLine + 52;
        ctx.beginPath(); ctx.moveTo(sx(WORDS[0].id), y2); ctx.lineTo(sx(WORDS[2].id), y2); ctx.stroke();
        ctx.fillText("|251−999| = " + d02 + "  “far”?", (sx(WORDS[0].id) + sx(WORDS[2].id)) / 2, y2 + 16);
        ctx.fillStyle = C.text;
        ctx.font = "600 13px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("The line says pizza ≈ pillow and pizza ≠ burger. IDs are labels, not measurements.", 60, H - 30);
        msg.show("ID distances are real numbers about nothing: vocabulary position (here alphabetical-ish) has no relation to meaning.", "bad");
      } else {
        // one-hot strips
        const x0 = 130, sw = W - 210, y0 = 46, rh = 26, gap = 52;
        ctx.textAlign = "left";
        ctx.fillStyle = C.muted;
        ctx.font = "600 12px Inter, sans-serif";
        ctx.fillText("each row is a vector of length |V| = " + VSIZE + " with a single 1:", x0, 24);
        WORDS.forEach(function (d, i) {
          const y = y0 + i * gap;
          ctx.fillStyle = C.text;
          ctx.font = "700 13px Inter, sans-serif";
          ctx.textAlign = "right";
          ctx.fillText(d.w, x0 - 14, y + rh / 2 + 4);
          ctx.fillStyle = "#f2f2f2";
          ctx.fillRect(x0, y, sw, rh);
          ctx.strokeStyle = "#dcdcdc";
          ctx.strokeRect(x0, y, sw, rh);
          const px = x0 + ((d.id - 1) / VSIZE) * sw;
          ctx.fillStyle = d.color;
          ctx.fillRect(px, y, Math.max(3, sw / VSIZE), rh);
          ctx.font = "600 10.5px Inter, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("1 at " + d.id, px, y - 4);
          ctx.textAlign = "left";
          ctx.fillStyle = C.faint;
          ctx.fillText("0 everywhere else", x0 + sw + 8 - 90, y + rh + 12);
        });
        // dot products, actually computed from the arrays above
        const yD = y0 + 3 * gap + 8;
        ctx.fillStyle = C.text;
        ctx.font = "700 13.5px JetBrains Mono, Menlo, monospace";
        ctx.textAlign = "left";
        const pairs = [[0, 1], [0, 2], [1, 2]];
        pairs.forEach(function (pr, i) {
          const v = dot(onehots[pr[0]], onehots[pr[1]]);
          ctx.fillText(
            "o_" + WORDS[pr[0]].w + " · o_" + WORDS[pr[1]].w + " = " + v,
            60 + i * 250, yD
          );
        });
        ctx.fillStyle = C.muted;
        ctx.font = "600 12.5px Inter, sans-serif";
        ctx.fillText("every pair is orthogonal: no fake ordering, and no similarity either. All pairwise distances are √2.", 60, yD + 26);
        msg.show("The fake geometry is gone, but pizza is now exactly as far from burger as from pillow. One-hot fixes the lie and installs a blindness.", "info");
      }
    }
    render();
  };
})();
