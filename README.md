# deseq2-rs

Python bindings for [rust_deseq2](https://github.com/necoli1822/rust_deseq2), a Rust implementation of
DESeq2 that is much faster and needs far less memory than pydeseq2.

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
`cooks_cutoff`) and independent filtering (`independent_filter`, `alpha`).

## Memory and threads

- The counts array is only read (no Python-side conversion) and copied once into the float64 genes x samples
  matrix rust_deseq2 works on. Delete your own reference after the call if you don't need it anymore.
- The GIL is released during the computation, which runs on its own rayon thread pool of `n_threads`
  (default: all CPUs available to the process, respecting cgroup limits).
- Results are moved into numpy arrays without copying and wrapped in a polars DataFrame.

## How it is built

- `src/pipeline.rs` ports the Wald test path of rust_deseq2's CLI (`src/main.rs::run_analysis`, the orchestration
  is not part of its library) on top of the library functions. Check it against upstream when updating.
- `src/lib.rs` contains the pyo3 bindings (`deseq2_rs._core`), `deseq2_rs/__init__.py` the Python API.

## Updating rust_deseq2

The crate is pulled from GitHub (`branch = "main"` in `Cargo.toml`), `Cargo.lock` pins the commit. To update:

```bash
make update-deseq2   # cargo update -p rust_deseq2 + uv sync
```

Then review the upstream changes of `src/main.rs::run_analysis` and port them to `src/pipeline.rs` if needed.
To only follow releases, use `tag = "vX.Y.Z"` instead of `branch = "main"`.

## Development

```bash
cargo test --manifest-path visyn_pro/libs/deseq2_rs/Cargo.toml
uv sync --all-extras && pytest visyn_pro/libs/deseq2_rs
python visyn_pro/libs/deseq2_rs/benchmarks/compare_pydeseq2.py   # speed/memory/results vs pydeseq2
```

Building requires a Rust toolchain (>= 1.89).
