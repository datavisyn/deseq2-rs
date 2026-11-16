//! Python bindings for `rust_deseq2`, see `deseq2_rs/__init__.py` for the public API.

mod pipeline;

use ndarray::{Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub use pipeline::{Design, Options, run_wald};

/// Copies samples x genes counts of any numeric dtype into the genes x samples f64 matrix
/// `rust_deseq2` works on, in a single (parallel) pass and without intermediate copies.
macro_rules! transpose_to_f64 {
    ($view:expr) => {{
        let view = $view;
        let mut out = Array2::<f64>::zeros((view.ncols(), view.nrows()));
        Zip::from(out.rows_mut()).and(view.columns()).par_for_each(|mut gene, column| {
            gene.zip_mut_with(&column, |o, &c| *o = c as f64);
        });
        out
    }};
}

/// Supported count dtypes, tried in order of how common they are for count matrices.
macro_rules! with_counts_view {
    ($counts:expr, |$view:ident| $body:expr) => {{
        let counts = $counts;
        with_counts_view!(@try counts, $view, $body, i32, i64, f64, f32, u32, u64, i16, u16, i8, u8)
    }};
    (@try $counts:ident, $view:ident, $body:expr, $($t:ty),+) => {{
        $(
            if let Ok(array) = $counts.extract::<PyReadonlyArray2<$t>>() {
                let $view: ArrayView2<$t> = array.as_array();
                return $body;
            }
        )+
        Err(PyTypeError::new_err(
            "counts must be a 2D numpy array of an integer or float dtype",
        ))
    }};
}

fn to_py_err(err: rust_deseq2::error::DeseqError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Runs DESeq2 (Wald test) and returns the result columns as numpy arrays, see `deseq2_rs.deseq2`.
#[pyfunction]
#[pyo3(signature = (
    counts, condition, numerator, denominator, *,
    factors=Vec::new(), continuous=Vec::new(), alpha=0.1, min_replicates_for_replace=Some(7),
    cooks_filter=true, cooks_cutoff=None, independent_filtering=true, n_threads=None,
))]
#[allow(clippy::too_many_arguments)]
fn deseq2_wald<'py>(
    py: Python<'py>,
    counts: &Bound<'py, PyAny>,
    condition: Vec<String>,
    numerator: String,
    denominator: String,
    factors: Vec<(String, Vec<String>)>,
    continuous: Vec<(String, Vec<f64>)>,
    alpha: f64,
    min_replicates_for_replace: Option<usize>,
    cooks_filter: bool,
    cooks_cutoff: Option<f64>,
    independent_filtering: bool,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let design = Design {
        condition,
        numerator,
        denominator,
        factors,
        continuous,
    };
    let options = Options {
        alpha,
        min_replicates_for_replace,
        cooks_cutoff: cooks_filter.then_some(cooks_cutoff),
        independent_filtering,
    };
    // A local pool instead of rayon's global one: the thread count is per call, and the global pool can only be configured once
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads.unwrap_or(0))
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    with_counts_view!(counts, |view| {
        if view.nrows() != design.condition.len() {
            return Err(PyValueError::new_err(format!(
                "counts has {} samples (rows) but condition has {} values",
                view.nrows(),
                design.condition.len()
            )));
        }
        // Release the GIL for the whole computation, the counts stay borrowed (read-only) meanwhile
        py.detach(|| {
            pool.install(|| {
                let matrix = transpose_to_f64!(view);
                run_wald(matrix, design, &options)
            })
        })
        .map_err(to_py_err)
        .and_then(|results| results_to_dict(py, results))
    })
}

fn results_to_dict<'py>(py: Python<'py>, results: rust_deseq2::io::DESeqResults) -> PyResult<Bound<'py, PyDict>> {
    // Gene ids passed to rust_deseq2 are the column positions, return them as an index to map the real ids back
    let gene_index = results
        .gene_ids
        .iter()
        .map(|id| id.parse::<u64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(format!("unexpected gene id in results: {e}")))?;
    let dict = PyDict::new(py);
    dict.set_item("gene_index", gene_index.into_pyarray(py))?;
    // Vec -> numpy without copying
    for (name, column) in [
        ("base_mean", results.base_means),
        ("log2_fold_change", results.log2_fold_changes),
        ("lfc_se", results.lfc_se),
        ("stat", results.stat),
        ("pvalue", results.pvalues),
        ("padj", results.padj),
        ("dispersion", results.dispersions),
    ] {
        dict.set_item(name, column.into_pyarray(py))?;
    }
    Ok(dict)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Forward rust_deseq2's `log` messages (e.g. warnings about the input) to Python's logging
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(deseq2_wald, m)?)?;
    Ok(())
}
