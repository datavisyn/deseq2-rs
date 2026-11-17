"""Writes the inputs of the golden tests: one directory per case in `tests/fixtures/` with

- `counts.csv`: raw counts, genes x samples (first column `gene_id`), the layout R's DESeq2 expects
- `coldata.csv`: sample metadata (`sample`, `condition` and the covariates of the case)
- `options.json`: contrast and options, read by `scripts/deseq2_reference.R` and `tests/test_golden.py`

The reference outputs (`expected.csv`, `size_factors.csv`) are written by `make goldens` (R DESeq2 in Docker).
The data is deterministic (seeded), so rerunning this script only changes the inputs when this script changes.

Usage: python scripts/make_fixtures.py tests/fixtures
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Case:
    name: str
    description: str
    n_per_group: int
    n_genes: int = 300
    seed: int = 0
    levels: tuple[str, ...] = ("a", "b")
    numerator: str = "a"
    denominator: str = "b"
    # genes (indices) that are all zero
    zero_genes: int = 0
    # (gene, sample) pairs whose count is multiplied by 50 to create Cook's distance outliers
    outliers: list[tuple[int, int]] = field(default_factory=list)
    batch: bool = False
    age: bool = False
    alpha: float = 0.1
    independent_filter: bool = True
    cooks_filter: bool = True
    refit_cooks: bool = True


CASES = [
    Case("basic_n4", "Two groups, 4 replicates each", n_per_group=4, n_genes=500),
    Case("small_n2", "Two groups, 2 replicates each (minimal design)", n_per_group=2, seed=1),
    Case("all_zero_genes", "20 all-zero genes and genes with zeros in one group", n_per_group=4, seed=2, zero_genes=20),
    Case(
        "outlier_refit_n7",
        "7 replicates per group: outliers are replaced and the genes refitted (minReplicatesForReplace = 7)",
        n_per_group=7,
        seed=3,
        outliers=[(45, 0), (60, 3), (75, 9), (90, 12), (300 - 1, 6)],
    ),
    Case(
        "outlier_filter_n3",
        "3 replicates per group: too few to replace outliers, their p-values are removed by Cook's filtering",
        n_per_group=3,
        seed=4,
        outliers=[(45, 0), (60, 4), (75, 2)],
    ),
    Case("multi_factor", "Design ~ batch + age + condition", n_per_group=6, seed=5, batch=True, age=True),
    Case(
        "three_levels",
        "Condition with three levels, contrast c vs a",
        n_per_group=4,
        seed=6,
        levels=("a", "b", "c"),
        numerator="c",
        denominator="a",
    ),
    Case(
        "no_filters",
        "Independent filtering, Cook's filtering and outlier replacement disabled",
        n_per_group=7,
        seed=7,
        outliers=[(45, 0)],
        independent_filter=False,
        cooks_filter=False,
        refit_cooks=False,
    ),
]


def simulate(case: Case) -> tuple[np.ndarray, dict[str, list]]:
    """Negative binomial counts (genes x samples) with DESeq2's parametric dispersion trend.

    In the first level 20 genes are up- and 20 down-regulated (4x), in the third level (if any) 20 others are up.
    """
    rng = np.random.default_rng(case.seed)
    n_samples = case.n_per_group * len(case.levels)
    condition = [level for level in case.levels for _ in range(case.n_per_group)]

    means = rng.lognormal(5, 1.5, size=case.n_genes)
    fold = np.ones((case.n_genes, n_samples))
    first = np.array(condition) == case.levels[0]
    fold[:20, first], fold[20:40, first] = 4.0, 0.25
    if len(case.levels) > 2:
        fold[40:60, np.array(condition) == case.levels[2]] = 4.0

    coldata: dict[str, list] = {"sample": [f"s{i}" for i in range(n_samples)], "condition": condition}
    if case.batch:
        batch = np.array(["x", "y"] * (n_samples // 2) + ["x"] * (n_samples % 2))
        coldata["batch"] = batch.tolist()
        fold[:, batch == "y"] *= rng.lognormal(0, 0.3, size=(case.n_genes, 1))  # gene-specific batch effect
    if case.age:
        # centered and scaled (decades around 45 years), as DESeq2 recommends for numeric design variables
        age = np.round(rng.uniform(-2.5, 2.5, size=n_samples), 2)
        coldata["age"] = age.tolist()
        fold[60:80] *= np.exp(0.2 * age)  # 20 genes depend on age

    mu = means[:, None] * fold
    dispersion = (0.05 + 3 / means)[:, None]
    counts = rng.negative_binomial(1 / dispersion, 1 / (1 + mu * dispersion)).astype(np.int64)

    if case.zero_genes:
        counts[-case.zero_genes :] = 0
        counts[-2 * case.zero_genes : -case.zero_genes, first] = 0  # zero in one group only
    for gene, sample in case.outliers:
        counts[gene, sample] = max(counts[gene].max(), 10) * 50
    return counts, coldata


def write_case(case: Case, directory: Path) -> None:
    counts, coldata = simulate(case)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "counts.csv").open("w") as f:
        f.write(",".join(["gene_id", *coldata["sample"]]) + "\n")
        for i, row in enumerate(counts):
            f.write(",".join([f"g{i}", *map(str, row)]) + "\n")
    with (directory / "coldata.csv").open("w") as f:
        columns = list(coldata)
        f.write(",".join(columns) + "\n")
        for values in zip(*(coldata[c] for c in columns), strict=True):
            f.write(",".join(map(str, values)) + "\n")
    options = {
        "description": case.description,
        "numerator": case.numerator,
        "denominator": case.denominator,
        "levels": list(case.levels),
        "factors": ["batch"] if case.batch else [],
        "continuous": ["age"] if case.age else [],
        "alpha": case.alpha,
        "independent_filter": case.independent_filter,
        "cooks_filter": case.cooks_filter,
        "refit_cooks": case.refit_cooks,
    }
    (directory / "options.json").write_text(json.dumps(options, indent=2) + "\n")


def main(root: Path) -> None:
    for case in CASES:
        write_case(case, root / case.name)
        print(f"wrote {root / case.name}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"))
