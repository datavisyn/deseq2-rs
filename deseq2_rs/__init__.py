"""DESeq2 differential expression analysis backed by `rust_deseq2 <https://github.com/necoli1822/rust_deseq2>`_.

>>> import numpy as np
>>> import deseq2_rs
>>> results = deseq2_rs.deseq2(
...     counts,                                   # samples x genes raw counts, any int/float dtype
...     condition=["treated", "treated", "control", "control"],
...     numerator="treated",
...     denominator="control",
...     gene_ids=gene_ids,
... )
>>> results.columns
['gene_id', 'base_mean', 'log2_fold_change', 'lfc_se', 'stat', 'pvalue', 'padj', 'dispersion']

The design is ``~ covariates + condition``, tested with a Wald test for ``numerator`` vs ``denominator``,
followed by DESeq2's default outlier replacement/refit, Cook's distance filtering and independent filtering,
like ``DESeq(dds)`` + ``results(dds, contrast=c("condition", numerator, denominator))`` in R.

Memory: the counts are read without copying on the Python side and converted once into the float64
genes x samples matrix rust_deseq2 works on. The GIL is released during the computation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from deseq2_rs._core import deseq2_wald

__all__ = ["deseq2"]

CONDITION = "condition"


def _labels(values: npt.ArrayLike, name: str, n_samples: int) -> list[str]:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError(f"{name} must have one value per sample ({n_samples}), got shape {labels.shape}")
    if labels.dtype.kind == "b":
        labels = labels.astype(np.int8)  # True/False -> "1"/"0", for arrays and scalar levels alike
    return [str(label) for label in labels.tolist()]


def deseq2(
    counts: np.ndarray,
    condition: npt.ArrayLike,
    *,
    numerator: Any,
    denominator: Any,
    gene_ids: npt.ArrayLike | pl.Series | None = None,
    covariates: Mapping[str, npt.ArrayLike] | None = None,
    alpha: float = 0.1,
    refit_cooks: bool = True,
    min_replicates: int = 7,
    cooks_filter: bool = True,
    cooks_cutoff: float | None = None,
    independent_filter: bool = True,
    n_threads: int | None = None,
) -> pl.DataFrame:
    """
    Runs a DESeq2 differential expression analysis of ``numerator`` vs ``denominator``.

    Args:
        counts: Raw counts as a 2D numpy array of shape (samples, genes), any integer or float dtype. Float counts must be
            whole numbers. It is only read, so the caller can ``del`` its reference right after the call returns.
        condition: Level of the tested factor for each sample (str, int or bool values; compared as strings).
        numerator: Level of ``condition`` in the numerator of the fold change (e.g. "treated").
        denominator: Level of ``condition`` in the denominator of the fold change, the reference (e.g. "control").
        gene_ids: Optional gene ids, one per counts column. Added as first column ``gene_id`` with their dtype.
        covariates: Optional additional design variables, e.g. ``{"batch": batches}``. Numeric arrays are continuous,
            any other dtype (str, bool, categorical, ...) is a factor. The design is ``~ covariates + condition``.
        alpha: Significance level used by independent filtering (R: ``results(alpha=)``).
        refit_cooks: Replace count outliers (Cook's distance) and refit those genes (R: ``DESeq(minReplicatesForReplace=)``).
        min_replicates: Minimum samples in a design cell for its outliers to be replaced.
        cooks_filter: Set p-values of genes with a Cook's distance outlier to NaN (R: ``results(cooksCutoff=)``).
        cooks_cutoff: Cook's distance cutoff, defaults to the 99% quantile of the F(p, m - p) distribution like DESeq2.
        independent_filter: Filter out genes with low mean counts before the BH adjustment (R: ``results(independentFiltering=)``).
        n_threads: Number of threads, defaults to all CPUs available to the process.

    Returns:
        One row per gene (in the order of the counts columns) with the columns ``gene_id`` (if ``gene_ids`` is given),
        ``base_mean``, ``log2_fold_change``, ``lfc_se``, ``stat``, ``pvalue``, ``padj`` and ``dispersion``.
        Like DESeq2, ``pvalue``/``padj`` are NaN for genes filtered as outliers or by independent filtering.
    """
    if not isinstance(counts, np.ndarray) or counts.ndim != 2:
        raise ValueError("counts must be a 2D numpy array of shape (samples, genes)")
    n_samples, n_genes = counts.shape
    if counts.dtype.kind not in "iuf":
        raise TypeError(f"counts must have an integer or float dtype, got {counts.dtype}")
    if counts.dtype.byteorder == ">" or (counts.dtype.kind == "f" and counts.dtype.itemsize < 4):
        counts = counts.astype(np.float64)  # big endian / float16 are not supported by the binding

    levels = _labels(condition, CONDITION, n_samples)
    numerator, denominator = (_labels([level], "level", 1)[0] for level in (numerator, denominator))
    for level in (numerator, denominator):
        if level not in levels:
            raise ValueError(f"{level!r} is not a level of condition, levels: {sorted(set(levels))}")
    if numerator == denominator:
        raise ValueError("numerator and denominator must be different levels")

    factors: list[tuple[str, list[str]]] = []
    continuous: list[tuple[str, list[float]]] = []
    for name, values in (covariates or {}).items():
        if name == CONDITION:
            raise ValueError(f"{CONDITION!r} is reserved for the tested factor, rename the covariate")
        array = np.asarray(values)
        if array.dtype.kind in "iuf":
            if array.shape != (n_samples,):
                raise ValueError(f"covariate {name!r} must have one value per sample ({n_samples}), got shape {array.shape}")
            continuous.append((name, array.astype(np.float64).tolist()))
        else:
            factors.append((name, _labels(array, f"covariate {name!r}", n_samples)))

    if gene_ids is not None and len(gene_ids) != n_genes:  # pyright: ignore[reportArgumentType]
        raise ValueError(f"gene_ids must have one value per gene ({n_genes}), got {len(gene_ids)}")  # pyright: ignore[reportArgumentType]

    columns = deseq2_wald(
        counts,
        levels,
        numerator,
        denominator,
        factors=factors,
        continuous=continuous,
        alpha=alpha,
        min_replicates_for_replace=min_replicates if refit_cooks else None,
        cooks_filter=cooks_filter,
        cooks_cutoff=cooks_cutoff,
        independent_filtering=independent_filter,
        n_threads=n_threads,
    )
    gene_index = columns.pop("gene_index")
    results = pl.DataFrame(columns)
    if gene_ids is not None:
        ids = gene_ids if isinstance(gene_ids, pl.Series) else pl.Series(np.asarray(gene_ids))
        results = results.insert_column(0, ids.gather(gene_index).alias("gene_id"))
    return results
