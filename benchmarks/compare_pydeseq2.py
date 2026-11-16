"""
Compares runtime, peak memory and results of deseq2_rs and pydeseq2 on synthetic counts.

    python visyn_pro/libs/deseq2_rs/benchmarks/compare_pydeseq2.py --samples 50 200 --genes 20000

Every engine runs in its own subprocess, so its peak RSS (minus the RSS after creating the counts) is measured in isolation.
pydeseq2 is configured like visyn_pro's diff_exp_live task (low_memory, condition as 0/1, joblib workers).
"""

import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np


def make_counts(n_samples: int, n_genes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = rng.lognormal(4, 2, size=n_genes)
    dispersion = 0.05 + 3 / means
    is_a = np.arange(n_samples) < n_samples // 2
    fold = np.where(rng.random(n_genes) < 0.1, rng.choice([0.25, 4.0], n_genes), 1.0)
    mu = np.where(is_a[:, None], means * fold, means)
    counts = rng.negative_binomial(1 / dispersion, 1 / (1 + mu * dispersion)).astype(np.int32)
    return counts, is_a


def max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def run_engine(engine: str, n_samples: int, n_genes: int, n_cpus: int | None) -> dict:
    counts, is_a = make_counts(n_samples, n_genes)
    baseline = max_rss_mb()
    start = time.perf_counter()
    if engine == "deseq2_rs":
        import deseq2_rs

        results = deseq2_rs.deseq2(counts, is_a, numerator=True, denominator=False, n_threads=n_cpus)
        lfc, padj = results["log2_fold_change"].to_numpy(), results["padj"].to_numpy()
    else:
        import anndata as ad
        import pandas as pd
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats

        inference = DefaultInference(n_cpus=n_cpus)
        adata = ad.AnnData(
            X=counts,
            obs=pd.DataFrame({"condition": is_a.astype(int)}, index=pd.Index([str(i) for i in range(n_samples)])),
            var=pd.DataFrame(index=pd.Index([str(i) for i in range(n_genes)])),
        )
        dds = DeseqDataSet(adata=adata, design_factors="condition", refit_cooks=True, inference=inference, low_memory=True, quiet=True)
        del adata
        dds.deseq2()
        stats = DeseqStats(dds, contrast=["condition", 1, 0], inference=inference, quiet=True)  # pyright: ignore[reportArgumentType]
        stats.summary()
        lfc, padj = stats.results_df["log2FoldChange"].to_numpy(), stats.results_df["padj"].to_numpy()
    seconds = time.perf_counter() - start
    return {"seconds": seconds, "peak_mb": max_rss_mb() - baseline, "lfc": lfc.tolist(), "padj": padj.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, nargs="+", default=[20, 100, 500])
    parser.add_argument("--genes", type=int, default=20000)
    parser.add_argument("--cpus", type=int, default=None, help="threads (deseq2_rs) / worker processes (pydeseq2)")
    parser.add_argument("--engines", nargs="+", default=["deseq2_rs", "pydeseq2"], choices=["deseq2_rs", "pydeseq2"])
    parser.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        engine, n_samples, n_genes = args.child
        print(json.dumps(run_engine(engine, int(n_samples), int(n_genes), args.cpus)))
        return

    print(f"{'samples':>8} {'genes':>7} {'engine':>10} {'time [s]':>9} {'peak [MB]':>10}  agreement")
    for n_samples in args.samples:
        outputs = {}
        for engine in args.engines:
            command = [sys.executable, __file__, "--child", engine, str(n_samples), str(args.genes)]
            if args.cpus:
                command += ["--cpus", str(args.cpus)]
            out = subprocess.run(command, check=True, capture_output=True, text=True).stdout
            outputs[engine] = result = json.loads(out.strip().splitlines()[-1])
            print(f"{n_samples:>8} {args.genes:>7} {engine:>10} {result['seconds']:>9.2f} {result['peak_mb']:>10.0f}")
        if len(outputs) == 2:
            rs, py = (outputs[e] for e in ("deseq2_rs", "pydeseq2"))
            lfc_rs, lfc_py = np.array(rs["lfc"], dtype=float), np.array(py["lfc"], dtype=float)
            sig_rs = set(np.flatnonzero(np.array(rs["padj"], dtype=float) < 0.05))
            sig_py = set(np.flatnonzero(np.array(py["padj"], dtype=float) < 0.05))
            finite = np.isfinite(lfc_rs) & np.isfinite(lfc_py)
            print(
                f"{'':>38}  max |Δlog2FC| {np.abs(lfc_rs - lfc_py)[finite].max():.2e}, "
                f"significant (padj<0.05): {len(sig_rs)} vs {len(sig_py)}, differing {len(sig_rs ^ sig_py)}"
            )


if __name__ == "__main__":
    main()
