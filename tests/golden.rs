//! Golden test for the size factors (median of ratios) against R DESeq2, on the fixtures in `tests/fixtures`
//! (see `tests/test_golden.py` for the results). Size factors don't depend on the design, so only the condition is used.

use std::fs;
use std::path::{Path, PathBuf};

use _core::{Design, build_dataset};
use ndarray::Array2;
use rust_deseq2::prelude::{SizeFactorMethod, estimate_size_factors};

fn fixtures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut dirs: Vec<_> = fs::read_dir(root)
        .expect("tests/fixtures missing")
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("size_factors.csv").exists())
        .collect();
    dirs.sort();
    dirs
}

/// Rows of a simple CSV (no quoting) without the header.
fn read_csv(path: &Path) -> Vec<Vec<String>> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect()
}

/// genes x samples counts from counts.csv (first column gene_id).
fn read_counts(path: &Path) -> Array2<f64> {
    let rows = read_csv(path);
    let n_samples = rows[0].len() - 1;
    let values = rows.iter().flat_map(|r| r[1..].iter().map(|v| v.parse::<f64>().unwrap()));
    Array2::from_shape_vec((rows.len(), n_samples), values.collect()).unwrap()
}

#[test]
fn size_factors_match_r_deseq2() {
    let cases = fixtures();
    assert!(cases.len() >= 8, "run `make goldens` to generate the R reference outputs");
    for dir in cases {
        let counts = read_counts(&dir.join("counts.csv"));
        let condition: Vec<String> = read_csv(&dir.join("coldata.csv")).into_iter().map(|r| r[1].clone()).collect();
        let expected: Vec<f64> =
            read_csv(&dir.join("size_factors.csv")).iter().map(|r| r[1].parse().unwrap()).collect();

        let design = Design {
            numerator: condition[0].clone(),
            denominator: condition[condition.len() - 1].clone(),
            condition,
            factors: vec![],
            continuous: vec![],
        };
        let mut dds = build_dataset(counts, &design).unwrap();
        estimate_size_factors(&mut dds, SizeFactorMethod::Ratio).unwrap();
        let actual = dds.size_factors().expect("size factors not set");

        assert_eq!(actual.len(), expected.len(), "{}", dir.display());
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - e).abs() <= 1e-10 * e.abs(), "{}: size factor of sample {i}: {a} != {e} (R)", dir.display());
        }
    }
}
