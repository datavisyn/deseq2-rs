# R DESeq2 reference outputs are generated in this (pinned) Bioconductor image
BIOCONDUCTOR_IMAGE ?= bioconductor/bioconductor_docker:RELEASE_3_23-R-4.6.1

.DEFAULT_GOAL := help

.PHONY: help  ## Show this help
help:
	@grep -E '^\.PHONY: [a-z0-9-]+  ## ' $(MAKEFILE_LIST) | sed -E 's/\.PHONY: ([a-z0-9-]+)  ## (.*)/  \1\t\2/' | expand -t 20

.PHONY: develop  ## Build the extension into .venv, with the test dependencies
develop:
	uv sync --extra test --reinstall-package deseq2-rs

.PHONY: test  ## Run the Rust and Python tests
test: develop
	cargo test
	uv run --no-sync pytest

.PHONY: lint  ## Check formatting and run clippy
lint:
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

.PHONY: format  ## Format the Rust code
format:
	cargo fmt

.PHONY: wheel  ## Build a release wheel into dist/
wheel:
	uvx maturin build --release --out dist

.PHONY: licenses  ## Regenerate THIRD_PARTY_LICENSES.md (needs `cargo install cargo-about --features cli`)
licenses:
	cargo about generate about.hbs -o THIRD_PARTY_LICENSES.md

.PHONY: update-deseq2  ## Update rust_deseq2 to the latest upstream commit, regenerate licenses and test
update-deseq2:
	cargo update -p rust_deseq2
	$(MAKE) licenses test

.PHONY: fixtures  ## Regenerate the golden test inputs (count matrices) in tests/fixtures
fixtures:
	uv run --no-sync python scripts/make_fixtures.py tests/fixtures

.PHONY: goldens  ## Regenerate the R DESeq2 reference outputs in tests/fixtures (needs Docker)
goldens: fixtures
	@# DESeq2 is installed on first use into .r-lib (git-ignored) and reused afterwards
	mkdir -p .r-lib
	docker run --rm --user "$$(id -u):$$(id -g)" -e HOME=/tmp -e R_LIBS_USER=/work/.r-lib \
		-v "$(CURDIR):/work" -w /work $(BIOCONDUCTOR_IMAGE) \
		Rscript scripts/deseq2_reference.R tests/fixtures
