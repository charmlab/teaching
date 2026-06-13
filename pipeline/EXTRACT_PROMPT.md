# Reusable prompt: lecture PDFs → content.md

Paste everything below the line into a fresh conversation, attach the lecture
PDFs for ONE module, and fill in the two ALL-CAPS slots. Attach
`CONTENT_SPEC.md` alongside the PDFs.

---

You are extracting the teaching backbone of a university module from my lecture
notes, with **high fidelity**. The attached PDFs are the complete source
material for the module **MODULE TITLE** (module id: `MODULE-ID`). The attached
`CONTENT_SPEC.md` defines the exact output format — follow it precisely.

Produce a single `content.md`.

**Fidelity rules (non-negotiable):**
1. Extract, don't author. Every claim, definition, derivation, and example must
   come from the PDFs. Do not add material, do not "improve" the mathematics,
   do not substitute your preferred notation — preserve mine exactly.
2. Where the PDFs are ambiguous, illegible, or appear to have an error, do NOT
   silently fix it. Insert `<!-- TODO[extract]: describe the issue -->` at that
   spot and move on.
3. Preserve the pedagogical ordering of the lectures unless two lectures
   overlap, in which case merge and note the merge in a TODO comment.
4. Keep my running threads and recurring intuitions (e.g. a thread like "linear
   separability and where it breaks") — these are the spine of the story, they
   are always core tier.

**Tier assignment (three tiers, per the spec):**
- **core** (untagged): what EVERYONE including executives must learn —
  definitions, the central intuition, what the method is for, when it fails,
  the headline example.
- **`::: student`**: required for students but skipped for executives —
  derivations, proofs, algorithmic details, complexity analysis, exam-level
  mechanics.
- **`::: deep`**: "for the interested reader" — even students aren't required
  to know it: research-level asides, pathological cases, historical notes,
  connections to advanced material.

When unsure between two tiers, choose the LOWER (more visible) tier and add
`<!-- TODO[tier]: maybe student/deep? -->`.

**The flow invariant — check it before you finish:** mentally delete all
`student` and `deep` blocks and re-read your output top to bottom. The core
text alone must be a complete, flowing story: no undefined notation, no
"as derived above" pointing into a deleted block, no orphaned transitions.
Then re-read with `student` blocks restored — same requirement. If a passage
fails, restructure: move the load-bearing sentence into core, leave the detail
in the tier.

**Also mark, where the source material suggests them:**
- `::: widget` blocks: wherever the notes contain a figure or argument that
  would be better as an interactive manipulation (decision boundary, loss
  surface, etc.). Spec goal/controls/shows from what the figure conveys.
- `::: build` blocks: wherever the notes contain an exercise, or where
  implementing something small would cement the section. Spec task/given/check.
- `::: example` blocks around worked examples from the notes.

**Output:** the complete `content.md` (frontmatter included: module id, title,
sources = the PDF filenames, outcomes distilled from the notes' stated
objectives), followed by a short **extraction report**: lecture → section
mapping, every TODO you inserted, and anything in the PDFs you deliberately
left out (e.g. admin slides, course logistics) so I can verify nothing of
substance was dropped.
