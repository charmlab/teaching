# ============================================================
# The AI Learning Map — content pipeline
#
#   1. Dump lecture PDFs / .tex / images into  modules/<id>/sources/<lang>/
#      (LANG defaults to 'en'; set LANG=fa for Farsi materials)
#   2. make extract MODULE=<id> [LANG=en|fa]   → assembles the Claude extraction job
#      (run it with Claude, save the result as modules/<id>/content.md or content-fa.md)
#   3. make listen  MODULE=<id> [LANG=en|fa] [TIER=core|student|full]  → PDF
#      make play    MODULE=<id>                                         → interactive notes (HTML)
#      make build   MODULE=<id>                                         → Jupyter notebook
#      make all     MODULE=<id> [LANG=en|fa]                           → all three
#
# Requirements (macOS):  brew install pandoc  &&  brew install --cask basictex
# ============================================================

MODULE ?= _example
TIER   ?= full        # core = executives | student | full = + interested reader
LANG   ?= en          # en = English | fa = Farsi/Persian

DIR    := modules/$(MODULE)
SRCDIR := $(DIR)/sources/$(LANG)
SRC    := $(DIR)/content$(if $(filter fa,$(LANG)),-fa,).md
ASSETS := $(DIR)/assets

PANDOC_FLAGS := --from markdown+fenced_divs --lua-filter=pipeline/filters/tiers.lua

.PHONY: help extract listen play build all serve clean

help:
	@echo "Targets:"
	@echo "  make extract MODULE=c-ml [LANG=fa]               assemble extraction job from sources/<lang>/"
	@echo "  make listen  MODULE=c-ml [LANG=fa] [TIER=student] PDF lecture notes (core|student|full)"
	@echo "  make play    MODULE=c-ml                          interactive HTML notes (tier toggle built in)"
	@echo "  make build   MODULE=c-ml                          Jupyter notebook from build blocks"
	@echo "  make all     MODULE=c-ml [LANG=fa]                listen (all tiers) + play + build"
	@echo "  make serve                                         preview the site at http://localhost:8080"

extract:
	@if [ ! -d "$(SRCDIR)" ]; then \
	  echo "✗ $(SRCDIR)/ not found — create it and drop your PDFs/.tex/.png there"; exit 1; fi
	@if [ -z "$$(ls -A $(SRCDIR) 2>/dev/null | grep -v '.gitkeep')" ]; then \
	  echo "⚠  $(SRCDIR)/ is empty — dump your PDFs/.tex/.png there first"; exit 1; fi
	@{ sed -e 's/MODULE TITLE/$(MODULE)/g' -e 's/MODULE-ID/$(MODULE)/g' \
	       -e 's/LANG/$(LANG)/g' pipeline/EXTRACT_PROMPT.md; \
	   echo ""; echo "## Source files provided (contents of $(SRCDIR)/)"; \
	   ls -1 $(SRCDIR) | grep -v '.gitkeep' | sed 's/^/- /'; } > $(DIR)/EXTRACTION_JOB.md
	@echo "✓ Wrote $(DIR)/EXTRACTION_JOB.md"
	@echo "  → Open a Claude session, attach everything in $(SRCDIR)/ plus"
	@echo "    pipeline/CONTENT_SPEC.md, paste the job, save the answer as $(SRC)"

$(SRC):
	@echo "✗ $(SRC) not found — run 'make extract MODULE=$(MODULE) LANG=$(LANG)' first"; exit 1

listen: $(SRC)
	@mkdir -p $(ASSETS)
	@echo "→ Generating LaTeX source…"
	TIER=$(TIER) pandoc $(SRC) $(PANDOC_FLAGS) \
	  --standalone -V geometry:margin=1in -V fontsize=11pt \
	  -H pipeline/templates/listen-header.tex \
	  -o $(ASSETS)/$(MODULE)-$(LANG)-$(TIER).tex
	@echo "✓ LaTeX: $(ASSETS)/$(MODULE)-$(LANG)-$(TIER).tex"
	@echo "→ Compiling PDF…"
	cd $(ASSETS) && xelatex -interaction=nonstopmode $(MODULE)-$(LANG)-$(TIER).tex 2>&1 | tail -5
	@cp $(ASSETS)/$(MODULE)-$(LANG)-$(TIER).pdf $(ASSETS)/generated.pdf
	@echo "✓ PDF:   $(ASSETS)/generated.pdf"

play: $(SRC)
	@mkdir -p $(ASSETS)
	TIER=full pandoc $(SRC) $(PANDOC_FLAGS) \
	  --standalone --katex --table-of-contents \
	  --template=pipeline/templates/notes.html \
	  -o $(ASSETS)/notes-$(LANG).html
	@echo "✓ $(ASSETS)/notes-$(LANG).html"

build: $(SRC)
	@mkdir -p $(ASSETS)
	python3 pipeline/scripts/md_to_ipynb.py $(SRC) $(ASSETS)/$(MODULE).ipynb
	@echo "✓ $(ASSETS)/$(MODULE).ipynb"

all: play build
	@$(MAKE) listen MODULE=$(MODULE) LANG=$(LANG) TIER=core
	@$(MAKE) listen MODULE=$(MODULE) LANG=$(LANG) TIER=student
	@$(MAKE) listen MODULE=$(MODULE) LANG=$(LANG) TIER=full

serve:
	@echo "Serving on http://localhost:8080 (Ctrl-C to stop)"
	@python3 -m http.server 8080

clean:
	rm -rf modules/*/assets/generated.pdf modules/*/assets/generated.tex \
	       modules/*/assets/*.aux modules/*/assets/*.log modules/*/assets/*.out
