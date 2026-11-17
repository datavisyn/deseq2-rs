"""
Benchmarks the DESeq2 step of both diff exp tasks (pydeseq2 vs deseq2_rs) under container resource limits.

Each run is its own Docker container with `--cpus` / `--memory` (no swap), like a pod on an EC2 node, so joblib,
rayon and visyn_pro's `get_available_memory` see the limits exactly as in a deployment. The container reuses this
repo's `.venv` (mounted read-only at the same path), so the image only needs the same system Python as the host
(Ubuntu 24.04 -> python3.12).

    # counts: a samples x genes .npy of raw counts (e.g. a subset of the DepMap zarr store)
    python visyn_pro/libs/deseq2_rs/benchmarks/constrained_benchmark.py --counts counts.npy \\
        --profiles 1:2g 2:4g 2:8g 4:16g --samples 45 200 1000

The first half of the samples is group A, the rest group B. Genes are pre-filtered like the task (min_count=10).
Reported: wall time of the DESeq2 step, peak memory of the whole container (cgroup memory.peak, including
pydeseq2's worker processes) and whether it was OOM killed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

IMAGE = "deseq2-bench:ubuntu24.04-v2"
DOCKERFILE = "FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y --no-install-recommends python3.12 && rm -rf /var/lib/apt/lists/* \\\n    && ln -sf python3.12 /usr/bin/python3\n"
REPO = Path(__file__).resolve().parents[4]
CGROUP = Path("/sys/fs/cgroup")


def _cgroup_int(name: str) -> int | None:
    try:
        value = (CGROUP / name).read_text().strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def child(engine: str, counts_path: str, n_samples: int, min_count: int) -> dict:
    import numpy as np

    counts = np.load(counts_path, mmap_mode="r")[:n_samples]
    counts = np.ascontiguousarray(counts)
    is_a = np.arange(n_samples) < n_samples // 2
    # Task pre-filter: at least min_count counts in as many samples as the smaller group has
    keep = (counts >= min_count).sum(axis=0) >= min(is_a.sum(), (~is_a).sum())
    counts = np.ascontiguousarray(counts[:, keep]).astype(np.int32, copy=False)
    gene_ids = np.arange(counts.shape[1])
    info = {"genes": int(counts.shape[1]), "mem_before_mb": (_cgroup_int("memory.current") or 0) / 1024**2}

    # Exactly what the diff exp task runs: its engine sizing (CPUs, workers, memory check) and runners
    from visyn_pro.bio.utils.resources import get_memory_budget
    from visyn_pro.bio.workflows.differential_expression import engines

    try:
        plan = engines.plan_deseq2(n_samples, counts.shape[1], engine, get_memory_budget())  # pyright: ignore[reportArgumentType]
    except engines.InsufficientMemoryError as e:
        return info | {"error": f"rejected: {e}"}
    info["threads"] = plan.n_cpus
    info["estimated_mb"] = plan.estimated_memory / 1024**2
    start = time.perf_counter()
    if plan.engine == "rust_deseq2":
        results = engines.run_rust_deseq2(counts, is_a, gene_ids, plan.n_cpus)
    else:
        sample_ids = np.array([str(i) for i in range(n_samples)])
        results = engines.run_pydeseq2(counts, is_a, sample_ids, gene_ids, plan.n_cpus)
    lfc, padj = results["log2_fold_change"].to_numpy(), results["padj"].to_numpy()

    info["seconds"] = time.perf_counter() - start
    info["peak_mb"] = (_cgroup_int("memory.peak") or 0) / 1024**2
    info["lfc"] = [float(x) for x in lfc]
    info["padj"] = [float(x) for x in padj]
    return info


def ensure_image() -> None:
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        subprocess.run(["docker", "build", "-t", IMAGE, "-"], input=DOCKERFILE.encode(), check=True)


def run_container(engine: str, cpus: str, memory: str, counts: Path, n_samples: int, min_count: int, timeout: int) -> dict:
    command = [
        "docker", "run", "--rm", f"--cpus={cpus}", f"--memory={memory}", f"--memory-swap={memory}",
        "--user", f"{os.getuid()}:{os.getgid()}", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "HOME=/tmp",
        "-v", f"{REPO}:{REPO}:ro", "-v", f"{counts.parent}:{counts.parent}:ro", "-w", str(REPO),
        IMAGE, str(REPO / ".venv/bin/python"), str(Path(__file__).resolve()),
        "--child", engine, str(counts), str(n_samples), "--min-count", str(min_count),
    ]  # fmt: skip
    start = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    if process.returncode == 137:
        return {"error": "OOM killed", "seconds": time.perf_counter() - start}
    if process.returncode != 0:
        return {"error": f"exit {process.returncode}: {process.stderr.strip().splitlines()[-1:]}"}
    return json.loads(process.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=Path, help="samples x genes raw counts (.npy)")
    parser.add_argument("--profiles", nargs="+", default=["1:2g", "2:4g", "2:8g", "4:16g"], help="cpus:memory per container")
    parser.add_argument("--samples", type=int, nargs="+", default=[45, 200, 1000])
    parser.add_argument("--engines", nargs="+", default=["rust_deseq2", "pydeseq2"], choices=["rust_deseq2", "pydeseq2", "auto"])
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--json-out", type=Path, help="write all results (without the per-gene values) to this file")
    parser.add_argument("--child", nargs=3, metavar=("ENGINE", "COUNTS", "N_SAMPLES"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        engine, counts, n_samples = args.child
        print(json.dumps(child(engine, counts, int(n_samples), args.min_count)))
        return

    import numpy as np

    ensure_image()
    rows = []
    print(f"{'profile':>8} {'samples':>7} {'genes':>6} {'engine':>8} {'threads':>7} {'time [s]':>9} {'peak [MB]':>9}  notes", flush=True)
    for profile in args.profiles:
        cpus, memory = profile.split(":")
        for n_samples in args.samples:
            outputs = {}
            for engine in args.engines:
                result = run_container(engine, cpus, memory, args.counts.resolve(), n_samples, args.min_count, args.timeout)
                outputs[engine] = result
                notes = result.get("error", "")
                print(
                    f"{profile:>8} {n_samples:>7} {result.get('genes', ''):>6} {engine:>8} {result.get('threads', ''):>7} "
                    f"{result.get('seconds', float('nan')):>9.1f} {result.get('peak_mb', float('nan')):>9.0f}  {notes}",
                    flush=True,
                )
                rows.append(
                    {"profile": profile, "samples": n_samples, "engine": engine}
                    | {k: v for k, v in result.items() if k not in ("lfc", "padj")}
                )
            if all("lfc" in outputs.get(e, {}) for e in ("rust_deseq2", "pydeseq2")):
                rs, py = outputs["rust_deseq2"], outputs["pydeseq2"]
                lfc_rs, lfc_py = np.array(rs["lfc"]), np.array(py["lfc"])
                finite = np.isfinite(lfc_rs) & np.isfinite(lfc_py)
                sig_rs = set(np.flatnonzero(np.nan_to_num(rs["padj"], nan=1) < 0.05))
                sig_py = set(np.flatnonzero(np.nan_to_num(py["padj"], nan=1) < 0.05))
                diff = np.abs(lfc_rs - lfc_py)[finite]
                print(
                    f"{'':>8} |dLFC| p99 {np.quantile(diff, 0.99):.1e} max {diff.max():.1e}, padj<0.05: {len(sig_rs)} vs {len(sig_py)} "
                    f"({len(sig_rs ^ sig_py)} differ)",
                    flush=True,
                )
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    sys.exit(main())
