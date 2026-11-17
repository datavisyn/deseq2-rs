"""Golden tests: deseq2_rs vs R DESeq2 reference outputs on the fixtures in `tests/fixtures`.

These catch numerical drift when updating rust_deseq2, not just crashes. Inputs come from `scripts/make_fixtures.py`,
reference outputs from R DESeq2 (`make goldens`, see `tests/fixtures/VERSIONS`). Size factors are checked by the
Rust test `tests/golden.rs`, they are not part of the Python API.
"""

import json
from pathlib import Path

import deseq2_rs
import numpy as np
import polars as pl
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CASES = sorted(p.name for p in FIXTURES.iterdir() if (p / "expected.csv").exists())

# deseq2_rs column -> R results() column
COLUMNS = {
    "base_mean": "baseMean",
    "log2_fold_change": "log2FoldChange",
    "lfc_se": "lfcSE",
    "stat": "stat",
    "pvalue": "pvalue",
    "padj": "padj",
    "dispersion": "dispersion",
}

# (rtol, atol) per column, about 100x the largest difference to R DESeq2 1.52 measured on x86_64 when the fixtures
# were created (base_mean ~5e-15, LFC/lfcSE/stat/dispersion <= 3e-7, pvalue/padj <= 9e-7 relative). Tight enough
# to catch numerical drift in a rust_deseq2 update, loose enough for platform differences (FMA, libm).
# base_mean only depends on the size factors and must match R up to rounding.
TOLERANCES = {
    "base_mean": (1e-12, 1e-12),
    "dispersion": (1e-5, 1e-12),
    "log2_fold_change": (1e-5, 1e-10),
    "lfc_se": (1e-5, 1e-12),
    "stat": (1e-5, 1e-10),
    "pvalue": (1e-4, 1e-15),
    "padj": (1e-4, 1e-15),
}


def load_case(name: str) -> tuple[pl.DataFrame, pl.DataFrame, dict, pl.DataFrame]:
    directory = FIXTURES / name
    counts = pl.read_csv(directory / "counts.csv")
    coldata = pl.read_csv(directory / "coldata.csv")
    options = json.loads((directory / "options.json").read_text())
    expected = pl.read_csv(directory / "expected.csv", null_values=["NA"], infer_schema_length=None)
    return counts, coldata, options, expected


def run_deseq2_rs(counts: pl.DataFrame, coldata: pl.DataFrame, options: dict) -> pl.DataFrame:
    samples = coldata["sample"].to_list()
    matrix = counts.select(samples).to_numpy().T  # samples x genes
    covariates = {name: coldata[name].to_numpy() for name in options["factors"] + options["continuous"]}
    return deseq2_rs.deseq2(
        matrix,
        coldata["condition"].to_numpy(),
        numerator=options["numerator"],
        denominator=options["denominator"],
        gene_ids=counts["gene_id"],
        covariates=covariates or None,
        alpha=options["alpha"],
        independent_filter=options["independent_filter"],
        cooks_filter=options["cooks_filter"],
        refit_cooks=options["refit_cooks"],
        n_threads=1,
    )


@pytest.fixture(scope="module", params=CASES)
def case(request):
    counts, coldata, options, expected = load_case(request.param)
    return run_deseq2_rs(counts, coldata, options), expected


def test_fixtures_exist():
    assert len(CASES) >= 8, "run `make goldens` to generate the R reference outputs"


def test_genes_in_order(case):
    actual, expected = case
    assert actual["gene_id"].to_list() == expected["gene_id"].to_list()


@pytest.mark.parametrize("column", list(COLUMNS))
def test_matches_r_deseq2(case, column):
    actual, expected = case
    a = actual[column].to_numpy().astype(np.float64)
    e = expected[COLUMNS[column]].fill_null(np.nan).to_numpy().astype(np.float64)

    # R's NA (filtered / not testable) must be NaN here and vice versa
    np.testing.assert_array_equal(np.isnan(a), np.isnan(e), err_msg=f"{column}: NA pattern differs from R")
    rtol, atol = TOLERANCES[column]
    finite = ~np.isnan(e)
    np.testing.assert_allclose(a[finite], e[finite], rtol=rtol, atol=atol, err_msg=column)
