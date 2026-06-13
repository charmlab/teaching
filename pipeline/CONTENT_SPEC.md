# content.md — module content specification (v0.1)

One `content.md` per module is the **single source of truth**: the backbone of
what is taught. Everything else is compiled from it — class LaTeX/slides, web
course notes (KaTeX), interactive Play components, and Build notebooks.

## File layout per module

```
modules/
  c-ml/
    content.md        ← the backbone (this spec)
    sources/          ← original lecture-note PDFs/LaTeX it was extracted from
    widgets/          ← Play components (built from widget blocks)
    notebook.ipynb    ← Build notebook (generated from build blocks)
```

## Frontmatter

```yaml
---
module: c-ml                  # id matching data.js
title: Classical ML
version: 0.1
sources:                      # lecture notes that feed this module
  - lecture03-knn.pdf
  - lecture04-trees-ensembles.pdf
outcomes:                     # what a learner can DO afterwards
  - Choose between KNN, trees, and ensembles for a tabular problem
  - Explain the bias-variance tradeoff in plain language
---
```

## The three tiers

| Tier | Markup | Who must learn it | Compiles into |
|------|--------|-------------------|----------------|
| **core**    | untagged prose (the default) | everyone — executives included | every output |
| **student** | `::: student` … `:::`        | students yes, executives no    | student notes/LaTeX + web (student view) |
| **deep**    | `::: deep` … `:::`           | nobody is required to — "for the interested reader" | full web notes only |

Example:

```markdown
## 2. K-Nearest Neighbours {#sec-knn}

To classify a new point, look at the $k$ closest labelled points and take a
majority vote. Distance is doing all the work here: "closest" presumes a
meaningful metric on the features.

::: student
**Why scaling matters — a derivation.** With Euclidean distance
$d(x, x') = \sqrt{\sum_j (x_j - x'_j)^2}$, a feature measured in grams rather
than kilograms contributes $10^6\times$ more to the sum …
:::

::: deep
**For the interested reader.** In high dimensions, distances concentrate:
the ratio of nearest to farthest neighbour distance tends to 1, which is one
face of the curse of dimensionality …
:::

Choosing $k$ trades off noise against blur: small $k$ memorizes, large $k$
oversmooths.
```

## THE FLOW INVARIANT (the rule that makes tiers work)

> Delete every `student` and `deep` block. The remaining core text must read as
> a complete, self-sufficient story — no dangling references, no missing
> notation, no broken transitions.

Operational rules that guarantee it:

1. **Tiered blocks are asides, not plot points.** They open with a signpost
   (`**Why this works.**`, `**Proof.**`, `**For the interested reader.**`) and
   are fully self-contained.
2. **Notation flows downhill only.** Anything defined inside a `student` block
   may be used in `student` and `deep` blocks, never in core text. Anything
   defined in `deep` stays in `deep`. Core may be referenced by everyone.
3. **Transitions live in core.** The sentence that carries the reader from
   section to section is never inside a tiered block.
4. **No forward pointers from core into tiers.** Core text never says "as shown
   in the derivation above" when the derivation is tier-tagged. (Tiered text
   pointing at core is fine.)

Compilation check: render the file three times (core; core+student;
core+student+deep) and read each top to bottom. All three must flow.

## Other block types (orthogonal to tiers; tag with a tier when not core)

Attributes go on the first lines INSIDE the block (`id:`, `goal:`, …) — the
fence itself carries only the class name.

```markdown
::: example                       # worked example; default tier = core
…
:::

::: widget                        # → spec for an interactive Play component
id: knn-decision-boundary
goal: feel how k changes the decision boundary
controls: k (slider 1–50), dataset (two-moons | blobs), metric (L2 | L1)
shows: scatter + shaded decision regions, live as controls move
:::

::: animation                     # → spec for a Manim animation (3b1b-style);
id: knn-voting                    #   rendered video embeds in notes/recordings
goal: animate the k nearest neighbours being found and voting
style: manim, 20–30 seconds
:::

::: build                         # → becomes a notebook section with asserts
id: knn-from-scratch
task: implement KNN predict() with NumPy broadcasting; no sklearn
given: train/test split of the penguins dataset, distance fn signature
check: accuracy within 1% of sklearn reference; runtime < 2s
:::

::: figure
src: bias-variance.svg
caption: Bias–variance as a dartboard.
:::
```

## How each ingredient consumes content.md

- **Listen** — slides are generated section-by-section from core (+student for
  the student track) via pandoc → beamer using `style.tex`; the recording/
  podcast narrates the same backbone.
- **Play** — each `widget` block is a build ticket for one interactive
  component; the surrounding prose becomes the explanatory text around it.
- **Build** — `build` blocks compile to notebook scaffolds: markdown cell
  (task), code cell with TODOs (given), hidden test cell (check).

## Math & conventions

- Math in `$...$` / `$$...$$` only (works in both KaTeX and LaTeX).
- Stable section ids (`{#sec-knn}`) — they become URL anchors and LaTeX labels.
- Macros shared between web and LaTeX live in `macros.tex` (KaTeX accepts a
  macro dictionary; pandoc includes the same file).
- Anything uncertain during extraction is marked `<!-- TODO[extract]: ... -->`,
  never guessed.
