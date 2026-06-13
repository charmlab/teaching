#!/usr/bin/env python3
"""md_to_ipynb.py — generate a Build notebook from content.md

Collects every `::: build` block and emits a Jupyter notebook:
section context (markdown) + task cell (markdown) + scaffold cell
(code with TODOs) + a check cell. Stdlib only.

Usage: python3 md_to_ipynb.py modules/c-ml/content.md out/c-ml.ipynb
"""
import json
import re
import sys


def parse(src: str):
    """Return (title, [ {section, body} ]) for every build block."""
    title = "Build notebook"
    m = re.search(r"^---\n.*?^title:\s*(.+?)\s*$.*?^---", src, re.S | re.M)
    if m:
        title = m.group(1).strip("\"'")

    blocks = []
    section = ""
    in_build, depth, buf = False, 0, []
    for line in src.splitlines():
        h = re.match(r"^##\s+(.*?)(?:\s*\{#.*\})?\s*$", line)
        if h:
            section = h.group(1)
        fence = re.match(r"^(:::+)\s*(.*)$", line)
        if fence:
            label = fence.group(2).strip()
            if not in_build and ("build" in label.split() or ".build" in label):
                in_build, depth, buf = True, 1, []
                continue
            if in_build:
                depth += 1 if label else -1
                if depth == 0:
                    blocks.append({"section": section, "body": "\n".join(buf).strip()})
                    in_build = False
                    continue
        if in_build:
            buf.append(line)
    return title, blocks


def field(body: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.+?)(?=^\w+:|\Z)", body, re.S | re.M)
    return m.group(1).strip() if m else ""


def cell_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def cell_code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def main():
    src_path, out_path = sys.argv[1], sys.argv[2]
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    title, blocks = parse(src)

    cells = [cell_md(f"# {title} — Build\n\nImplement it yourself. Each exercise has a"
                     " scaffold and a check; make the checks pass.")]
    if not blocks:
        cells.append(cell_md("_No `::: build` blocks found in content.md yet._"))

    for i, b in enumerate(blocks, 1):
        task = field(b["body"], "task") or b["body"]
        given = field(b["body"], "given")
        check = field(b["body"], "check")
        ident = field(b["body"], "id") or f"exercise-{i}"
        cells.append(cell_md(f"## Exercise {i}: {ident}\n\n_Section: {b['section']}_\n\n**Task.** {task}"
                             + (f"\n\n**Given.** {given}" if given else "")))
        cells.append(cell_code(f"# Exercise {i}: {ident}\n# TODO: your implementation here\n"))
        if check:
            cells.append(cell_code(f"# Check — make this pass\n# {check}\n"
                                   "# TODO: turn the check above into asserts\n"))

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
          "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"  {len(blocks)} build block(s) → {out_path}")


if __name__ == "__main__":
    main()
