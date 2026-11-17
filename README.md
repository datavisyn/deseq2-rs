# deseq2-rs

Python bindings for [rust_deseq2](https://github.com/necoli1822/rust_deseq2), a Rust implementation of
DESeq2 that is much faster and needs far less memory than pydeseq2.

```bash
pip install deseq2-rs   # or: uv add deseq2-rs
```

Prebuilt wheels (no Rust toolchain needed) for CPython ≥ 3.12 on Linux x86_64/aarch64 (manylinux_2_28) and
macOS arm64. Other platforms build from the sdist, which needs Rust ≥ 1.89.

```python
import deseq2_rs

results = deseq2_rs.deseq2(
    counts,                            # numpy array, samples x genes, raw counts (any int/float dtype)
    condition=group_labels,            # one level per sample
    numerator="treated",
    denominator="control",
    gene_ids=gene_ids,                 # optional, added as `gene_id` column with its dtype
    covariates={"batch": batches},     # optional: numeric -> continuous, otherwise factor
)
# polars DataFrame: gene_id, base_mean, log2_fold_change, lfc_se, stat, pvalue, padj, dispersion
```

This runs the equivalent of R's `DESeq(dds)` + `results(dds, contrast=c("condition", numerator, denominator))`
for the design `~ covariates + condition`: median-of-ratios size factors, parametric dispersion trend, Wald test,
Cook's distance outlier replacement + refit (`refit_cooks`, `min_replicates`), Cook's filtering (`cooks_filter`,
`cooks_cutoff`) and independent filtering (`independent_filter`, `alpha`). The results match R DESeq2 to ~1e-6
relative or better, checked by the golden tests on every change.

## Memory and threads

- The counts array is only read (no Python-side conversion) and copied once into the float64 genes x samples
  matrix rust_deseq2 works on. Delete your own reference after the call if you don't need it anymore.
- The GIL is released during the computation, which runs on its own rayon thread pool of `n_threads`
  (default: all CPUs available to the process, respecting cgroup limits).
- Results are moved into numpy arrays without copying and wrapped in a polars DataFrame.

## How it is built

- `src/pipeline.rs` ports the Wald test path of rust_deseq2's CLI (`src/main.rs::run_analysis`, the orchestration
  is not part of its library) on top of the library functions. Check it against upstream when updating.
- `src/lib.rs` contains the pyo3 bindings (`deseq2_rs._core`, abi3 for Python ≥ 3.12),
  `python/deseq2_rs/__init__.py` the Python API.

## Development

Needs Rust ≥ 1.89 and [uv](https://docs.astral.sh/uv/).

```bash
make develop    # build the extension into .venv with the test dependencies
make test       # cargo test + pytest
make lint       # cargo fmt --check + clippy
make wheel      # release wheel into dist/
python benchmarks/compare_pydeseq2.py   # speed/memory/results vs pydeseq2
```

### Tests

- `tests/test_deseq2_rs.py`: behaviour of the Python API (dtypes, layouts, contrasts, covariates, errors) and a
  cross-check against pydeseq2.
- `tests/test_golden.py` and `tests/golden.rs`: results and size factors vs **R DESeq2** on the fixtures in
  `tests/fixtures/` (two groups, n = 2, all-zero genes, outlier replacement + refit, Cook's filtering, multi-factor
  design, three levels, filters disabled). Tolerances are ~100x the differences measured when the fixtures were
  created, so numerical drift in an update fails the tests.
- The fixtures are regenerated with `make goldens`: `scripts/make_fixtures.py` writes the inputs,
  `scripts/deseq2_reference.R` runs R DESeq2 in a pinned Bioconductor Docker image (versions in
  `tests/fixtures/VERSIONS`). CI only compares, it never needs R.

## Updating rust_deseq2

The crate is pulled from GitHub (`branch = "main"` in `Cargo.toml`), `Cargo.lock` pins the commit.

- **Automatically:** `update-rust-deseq2.yml` runs weekly (or manually from the Actions tab). If upstream has
  new commits it opens/refreshes a PR with the old → new commit, a compare link and the upstream commit list.
  CI builds and tests the wheels on every platform; merge only when it's green.
- **Locally:** `make update-deseq2` (`cargo update -p rust_deseq2`, regenerates `THIRD_PARTY_LICENSES.md`, runs
  the tests).

Either way, check whether upstream's `src/main.rs::run_analysis` changed and port that to `src/pipeline.rs`
(the update PR says so when `src/main.rs` was touched).

## Releasing

Versions come from git tags (`vX.Y.Z`); the version in `Cargo.toml` is only the base for the first release.

- Merging a change to `Cargo.lock` into `main` (e.g. an update PR) makes `release.yml` push the next patch tag.
  For other releases run `release` manually from the Actions tab and pick patch/minor/major.
- The tag triggers `publish.yml`: it stamps the version into `Cargo.toml`, runs the full CI (build + tests on
  every platform), publishes wheels + sdist to PyPI via trusted publishing and creates a GitHub release that names
  the pinned rust_deseq2 commit.

## License

MIT, see `LICENSE`. The wheels bundle Rust crates whose licenses are listed in `THIRD_PARTY_LICENSES.md`
(regenerate with `make licenses`, needs `cargo install cargo-about --features cli`).
