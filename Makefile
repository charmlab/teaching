# ============================================================
# The AI Learning Map — content pipeline
#
#   1. Dump lecture PDFs / .tex / images into  modules/<id>/sources/
#   2. make extract MODULE=<id>     → assembles the Claude extraction job
#      (run it with Claude, save the result as modules/<id>/content.md)
#   3. make listen  MODULE=<id> TIER=core|student|full   → PDF
#      make play    MODULE=<id>                          → interactive notes (HTML)
#      make build   MODULE=<id>                          → Jupyter notebook
#      make all     MODULE=<id>                          → all three
#
# Requirements (macOS):  brew install pandoc  &&  brew install --cask basictex
# ============================================================

MODULE ?= _example
TIER   ?= full        # core = executives | student | full = + interested reader
DIR    := modules/$(MODULE)
SRC    := $(DIR)/content.md
OUT    := $(DIR)/out

PANDOC_FLAGS := --from markdown+fenced_divs --lua-filter=pipeline/filters/tiers.lua

.PHONY: help extract listen play build all serve clean

help:
	@echo "Targets:"
	@echo "  make extract MODULE=c-ml              assemble extraction job from sources/"
	@echo "  make listen  MODULE=c-ml TIER=student PDF lecture notes (core|student|full)"
	@echo "  make play    MODULE=c-ml              interactive HTML notes (tier toggle built in)"
	@echo "  make build   MODULE=c-ml              Jupyter notebook from build blocks"
	@echo "  make all     MODULE=c-ml              listen (all tiers) + play + build"
	@echo "  make serve                            preview the site at http://localhost:8080"

extract:
	@mkdir -p $(DIR)/sources
	@if [ -z "$$(ls -A $(DIR)/sources 2>/dev/null)" ]; then \
	  echo "⚠  $(DIR)/sources/ is empty — dump your PDFs/.tex/.png there first"; exit 1; fi
	@{ sed -e 's/MODULE TITLE/$(MODULE)/g' -e 's/MODULE-ID/$(MODULE)/g' pipeline/EXTRACT_PROMPT.md; \
	   echo ""; echo "## Source files provided (contents of $(DIR)/sources/)"; \
	   ls -1 $(DIR)/sources | sed 's/^/- /'; } > $(DIR)/EXTRACTION_JOB.md
	@echo "✓ Wrote $(DIR)/EXTRACTION_JOB.md"
	@echo "  → Open a Claude session, attach everything in $(DIR)/sources/ plus"
	@echo "    pipeline/CONTENT_SPEC.md, paste the job, save the answer as $(SRC)"

$(SRC):
	@echo "✗ $(SRC) not found — run 'make extract MODULE=$(MODULE)' first"; exit 1

listen: $(SRC)
	@mkdir -p $(OUT)
	TIER=$(TIER) pandoc $(SRC) $(PANDOC_FLAGS) \
	  --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=11pt \
	  -H pipeline/templates/listen-header.tex \
	  -o $(OUT)/$(MODULE)-$(TIER).pdf
	@echo "✓ $(OUT)/$(MODULE)-$(TIER).pdf"

play: $(SRC)
	@mkdir -p $(OUT)
	TIER=full pandoc $(SRC) $(PANDOC_FLAGS) \
	  --standalone --katex --table-of-contents \
	  --template=pipeline/templates/notes.html \
	  -o $(OUT)/notes.html
	@echo "✓ $(OUT)/notes.html"

build: $(SRC)
	@mkdir -p $(OUT)
	python3 pipeline/scripts/md_to_ipynb.py $(SRC) $(OUT)/$(MODULE).ipynb
	@echo "✓ $(OUT)/$(MODULE).ipynb"

all: play build
	@$(MAKE) listen MODULE=$(MODULE) TIER=core
	@$(MAKE) listen MODULE=$(MODULE) TIER=student
	@$(MAKE) listen MODULE=$(MODULE) TIER=full

serve:
	@echo "Serving on http://localhost:8080 (Ctrl-C to stop)"
	@python3 -m http.server 8080

clean:
	rm -rf modules/*/out
