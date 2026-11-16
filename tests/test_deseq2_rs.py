import importlib.util

import deseq2_rs
import numpy as np
import polars as pl
import pytest

has_pydeseq2 = importlib.util.find_spec("pydeseq2") is not None


def make_counts(n_per_group=4, n_genes=200, seed=0):
    """Negative binomial counts (samples x genes) with 20 up- and 20 down-regulated genes in group "a"."""
    rng = np.random.default_rng(seed)
    means = rng.lognormal(5, 1.5, size=n_genes)
    fold = np.ones(n_genes)
    fold[:20], fold[20:40] = 4.0, 0.25
    mu = np.vstack([np.tile(means * fold, (n_per_group, 1)), np.tile(means, (n_per_group, 1))])
    dispersion = 0.05 + 3 / means  # DESeq2's parametric trend: asymptotic dispersion + extra Poisson
    counts = rng.negative_binomial(1 / dispersion, 1 / (1 + mu * dispersion))
    condition = ["a"] * n_per_group + ["b"] * n_per_group
    return counts.astype(np.int64), condition


COLUMNS = ["base_mean", "log2_fold_change", "lfc_se", "stat", "pvalue", "padj", "dispersion"]


def test_detects_regulated_genes():
    counts, condition = make_counts()
    results = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b")

    assert results.columns == COLUMNS
    assert results.height == counts.shape[1]
    # true log2 fold changes are 2 / -2 / 0
    assert np.median(results["log2_fold_change"][:20].to_numpy()) > 1.5
    assert np.median(results["log2_fold_change"][20:40].to_numpy()) < -1.5
    assert abs(np.median(results["log2_fold_change"][40:].to_numpy())) < 0.2
    significant = set(results.with_row_index().filter(pl.col("padj") < 0.05)["index"].to_list())
    assert len(significant & set(range(40))) >= 30
    assert len(significant - set(range(40))) <= 5


@pytest.mark.parametrize(
    "convert",
    [
        lambda c: c.astype(np.int32),
        lambda c: c.astype(np.uint16),
        lambda c: c.astype(np.float64),
        lambda c: c.astype(np.float32),
        np.asfortranarray,
        lambda c: c.astype(">i4"),
        lambda c: np.ascontiguousarray(c.T).T,
    ],
    ids=["int32", "uint16", "float64", "float32", "fortran", "big-endian", "transposed-view"],
)
def test_dtypes_and_layouts_give_identical_results(convert):
    counts, condition = make_counts(n_genes=50)
    expected = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b")
    actual = deseq2_rs.deseq2(convert(counts), condition, numerator="a", denominator="b")
    assert actual.equals(expected)


def test_swapped_contrast_negates_fold_changes():
    counts, condition = make_counts(n_genes=50)
    ab = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b")
    ba = deseq2_rs.deseq2(counts, condition, numerator="b", denominator="a")
    np.testing.assert_allclose(ab["log2_fold_change"], -ba["log2_fold_change"], atol=1e-8)
    np.testing.assert_allclose(ab["pvalue"], ba["pvalue"], rtol=1e-8)


def test_gene_ids_keep_their_dtype():
    counts, condition = make_counts(n_genes=30)
    entrez = np.arange(30, dtype=np.int64) * 7
    results = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b", gene_ids=entrez)
    assert results.columns[0] == "gene_id"
    assert results["gene_id"].dtype == pl.Int64
    assert results["gene_id"].to_list() == entrez.tolist()

    names = pl.Series([f"g{i}" for i in range(30)])
    results = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b", gene_ids=names)
    assert results["gene_id"].to_list() == names.to_list()


def test_bool_and_int_conditions():
    counts, condition = make_counts(n_genes=30)
    is_a = np.array(condition) == "a"
    expected = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b")
    by_bool = deseq2_rs.deseq2(counts, is_a, numerator=1, denominator=0)
    by_int = deseq2_rs.deseq2(counts, is_a.astype(int), numerator=1, denominator=0)
    by_bool_levels = deseq2_rs.deseq2(counts, is_a, numerator=True, denominator=False)
    assert by_bool_levels.equals(by_bool)
    # the levels sort differently ("0" < "1" vs "a" < "b"), so the design is coded differently: equal up to numerics
    np.testing.assert_allclose(by_bool["log2_fold_change"], expected["log2_fold_change"], rtol=1e-4)
    assert by_int.equals(by_bool)


def test_covariates():
    counts, condition = make_counts(n_genes=50)
    batch = ["x", "y"] * 4
    age = np.linspace(20, 60, 8)
    results = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b", covariates={"batch": batch, "age": age})
    assert np.median(results["log2_fold_change"][:20].to_numpy()) > 1.5


def test_options_disable_filtering():
    counts, condition = make_counts(n_genes=50)
    results = deseq2_rs.deseq2(
        counts, condition, numerator="a", denominator="b", independent_filter=False, cooks_filter=False, refit_cooks=False
    )
    assert results["padj"].is_nan().sum() == 0
    single_thread = deseq2_rs.deseq2(
        counts, condition, numerator="a", denominator="b", independent_filter=False, cooks_filter=False, refit_cooks=False, n_threads=1
    )
    assert single_thread.equals(results)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"counts": np.zeros(5)}, "2D numpy array"),
        ({"counts": np.array([["1"]])}, "integer or float dtype"),
        ({"condition": ["a", "b"]}, "one value per sample"),
        ({"numerator": "c"}, "not a level"),
        ({"denominator": "a"}, "must be different"),
        ({"gene_ids": [1, 2]}, "one value per gene"),
        ({"covariates": {"condition": [1] * 8}}, "reserved"),
        ({"counts": -np.ones((8, 5))}, "non-negative"),
        ({"counts": np.full((8, 5), np.nan)}, "non-negative"),
    ],
)
def test_invalid_input(kwargs, error):
    counts, condition = make_counts(n_genes=5)
    arguments = {"counts": counts, "condition": condition, "numerator": "a", "denominator": "b"} | kwargs
    with pytest.raises((ValueError, TypeError), match=error):
        deseq2_rs.deseq2(**arguments)


@pytest.mark.skipif(not has_pydeseq2, reason="pydeseq2 not installed")
@pytest.mark.parametrize("n_per_group", [5, 8])
def test_agrees_with_pydeseq2(n_per_group):
    import pandas as pd
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts, condition = make_counts(n_per_group=n_per_group, n_genes=300, seed=1)
    if n_per_group >= 7:
        counts[0, 50] = counts[:, 50].max() * 20  # an outlier to exercise the replacement + refit

    samples = pd.Index([f"s{i}" for i in range(len(condition))])
    genes = pd.Index([f"g{i}" for i in range(counts.shape[1])])
    dds = DeseqDataSet(
        counts=pd.DataFrame(counts, index=samples, columns=genes),
        metadata=pd.DataFrame({"condition": condition}, index=samples),
        design="~condition",
        quiet=True,
        n_cpus=1,
    )
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", "a", "b"], quiet=True, n_cpus=1)
    stats.summary()
    expected = stats.results_df

    actual = deseq2_rs.deseq2(counts, condition, numerator="a", denominator="b", gene_ids=genes)

    np.testing.assert_allclose(actual["base_mean"], expected["baseMean"], rtol=1e-6)
    # pydeseq2 itself is not identical to R DESeq2 (rust_deseq2's reference), allow small differences
    np.testing.assert_allclose(actual["log2_fold_change"], expected["log2FoldChange"], atol=1e-2)
    np.testing.assert_allclose(actual["pvalue"], expected["pvalue"], rtol=0.1, atol=1e-4)
    significant = set(actual.filter(pl.col("padj") < 0.1)["gene_id"])
    expected_significant = set(expected.index[expected["padj"] < 0.1])
    assert len(significant ^ expected_significant) <= 0.02 * len(expected_significant)
