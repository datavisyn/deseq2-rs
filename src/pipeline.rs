//! DESeq2 Wald test pipeline on top of the `rust_deseq2` library.
//!
//! `rust_deseq2` exposes every step as a library function, but the orchestration of R's
//! `DESeq()` + `results()` (outlier replacement, refit, Cook's and independent filtering)
//! only lives in its CLI (`src/main.rs::run_analysis`). This module ports the Wald test path
//! of that function, with the CLI's defaults. Keep it in sync when updating `rust_deseq2`.

use ndarray::Array2;
use rust_deseq2::filter::{
    calculate_cooks_distance, default_cooks_cutoff, filter_by_cooks_with_counts, n_or_more_in_cell, replace_outliers,
};
use rust_deseq2::prelude::*;
use rust_deseq2::testing::wald_test_extended;

/// Name of the tested factor in the sample metadata.
pub const CONDITION: &str = "condition";

/// R DESeq2 / rust_deseq2 CLI default for the trimmed mean used to replace outliers.
const OUTLIER_TRIM: f64 = 0.2;

pub struct Design {
    /// Level of each sample for the tested factor.
    pub condition: Vec<String>,
    pub numerator: String,
    pub denominator: String,
    /// Categorical covariates (e.g. batch), added to the design before the condition.
    pub factors: Vec<(String, Vec<String>)>,
    /// Continuous covariates, added to the design before the condition.
    pub continuous: Vec<(String, Vec<f64>)>,
}

pub struct Options {
    pub alpha: f64,
    /// R: `DESeq(minReplicatesForReplace=)`, `None` disables outlier replacement + refit.
    pub min_replicates_for_replace: Option<usize>,
    /// R: `results(cooksCutoff=)`, `Some(None)` uses DESeq2's default cutoff, `None` disables Cook's filtering.
    pub cooks_cutoff: Option<Option<f64>>,
    pub independent_filtering: bool,
}

/// Builds the dataset for the design `~ factors + continuous + condition`.
///
/// `counts` is genes x samples. Gene and sample ids are positional (`"0"`, `"1"`, ...).
pub fn build_dataset(counts: Array2<f64>, design: &Design) -> Result<DESeqDataSet> {
    let (n_genes, n_samples) = counts.dim();
    let gene_ids = (0..n_genes).map(|i| i.to_string()).collect();
    let sample_ids: Vec<String> = (0..n_samples).map(|i| i.to_string()).collect();
    let counts = CountMatrix::new(counts, gene_ids, sample_ids.clone())?;

    let mut metadata = SampleMetadata::new(sample_ids);
    metadata.add_condition(CONDITION, design.condition.clone())?;
    for (name, values) in &design.factors {
        metadata.add_condition(name, values.clone())?;
    }
    for (name, values) in &design.continuous {
        metadata.add_continuous(name, values.clone())?;
    }

    let dds = if design.factors.is_empty() && design.continuous.is_empty() {
        DESeqDataSet::new(counts, metadata, CONDITION)?
    } else {
        let mut builder = DESeqDataSet::builder().counts(counts).metadata(metadata).main_effect(CONDITION);
        for (name, _) in &design.factors {
            builder = builder.factor(name);
        }
        for (name, _) in &design.continuous {
            builder = builder.continuous(name);
        }
        builder.build()?
    };
    Ok(dds)
}

/// Runs size factors, dispersions, GLM fit, Wald test, outlier handling and filtering.
///
/// `counts` is genes x samples. Gene and sample ids are positional (`"0"`, `"1"`, ...), the caller maps them back.
pub fn run_wald(counts: Array2<f64>, design: Design, options: &Options) -> Result<DESeqResults> {
    let n_samples = counts.ncols();
    let mut dds = build_dataset(counts, &design)?;

    estimate_size_factors(&mut dds, SizeFactorMethod::Ratio)?;
    estimate_dispersions(&mut dds, TrendFitMethod::Parametric, &DispersionParams::default())?;
    let design_info = fit_glm(&mut dds, &GlmFitParams::default())?;

    let contrast = ContrastSpec::Simple {
        variable: CONDITION.to_string(),
        numerator: design.numerator,
        denominator: design.denominator,
    };
    let wald = |dds: &DESeqDataSet| wald_test_extended(dds, &design_info, contrast.clone(), options.alpha, false, 0.0);
    let mut results = wald(&dds)?;

    // --- Cook's outlier replacement + refit (R: replaceOutliers + refitWithoutOutliers, after the first test) ---
    let n_coefs = dds.design_matrix().map(|m| m.ncols()).unwrap_or(2);
    let cooks_cutoff = options.cooks_cutoff.flatten().unwrap_or_else(|| default_cooks_cutoff(n_samples, n_coefs));
    let mut replaceable_samples = vec![false; n_samples];
    let mut now_zero: Vec<usize> = Vec::new();
    // Cook's distances before any replacement, used for the Cook's filter afterwards
    let mut original_cooks: Option<Array2<f64>> = None;
    // Counts before replacement (refitWithoutOutliers restores them in R), only kept if anything was replaced
    let mut original_counts: Option<Array2<f64>> = None;

    if let Some(min_replicates) = options.min_replicates_for_replace {
        let cooks = calculate_cooks_distance(&dds)?;
        if let Some(design_matrix) = dds.design_matrix() {
            replaceable_samples = n_or_more_in_cell(design_matrix, min_replicates);
        }
        // replace_outliers is a no-op without replaceable samples, skip the copy of the counts then
        let counts_before = replaceable_samples.iter().any(|&r| r).then(|| dds.counts().counts().to_owned());
        let (modified_genes, flagged_genes) = match counts_before {
            Some(_) => replace_outliers(&mut dds, &cooks, cooks_cutoff, min_replicates, OUTLIER_TRIM)?,
            None => (Vec::new(), Vec::new()),
        };
        if !modified_genes.is_empty() {
            log::info!(
                "Replaced outlier counts in {} genes, refitting {} genes",
                modified_genes.len(),
                flagged_genes.len()
            );
            let counts = dds.counts().counts();
            now_zero =
                flagged_genes.iter().copied().filter(|&gene| counts.row(gene).iter().all(|&c| c == 0.0)).collect();
            rust_deseq2::dispersion::refit_without_outliers(&mut dds, &flagged_genes)?;
            results = wald(&dds)?;
            original_counts = counts_before;
        }
        original_cooks = Some(cooks);
    }
    let refit_done = original_counts.is_some();

    // --- Cook's filtering (R: results(cooksCutoff=)) on the pre-replacement Cook's distances ---
    if options.cooks_cutoff.is_some() {
        // R recordMaxCooks: max over samples in cells with 3+ replicates, minus samples already replaced
        let mut max_cooks_samples = match dds.design_matrix() {
            Some(design_matrix) => n_or_more_in_cell(design_matrix, 3),
            None => vec![false; n_samples],
        };
        if refit_done {
            for (eligible, &replaceable) in max_cooks_samples.iter_mut().zip(&replaceable_samples) {
                *eligible &= !replaceable;
            }
        }
        if max_cooks_samples.iter().any(|&s| s) {
            let cooks = match original_cooks.take() {
                Some(cooks) => cooks,
                None => calculate_cooks_distance(&dds)?,
            };
            // Without a refit the counts are unchanged, take them out of the (no longer needed) dataset instead of copying
            let counts = match original_counts.take() {
                Some(counts) => counts,
                None => std::mem::take(dds.counts_mut().counts_mut()),
            };
            // R results(): the low-count heuristic applies to a single two-level factor design
            let is_two_level_design = match design_info.design_vars.as_slice() {
                [v] => {
                    design_info.reference_levels.contains_key(v)
                        && *v == design_info.design_variable
                        && design_info.levels.len() == 2
                }
                _ => false,
            };
            filter_by_cooks_with_counts(
                &mut results,
                &cooks,
                cooks_cutoff,
                Some(&counts),
                is_two_level_design,
                &max_cooks_samples,
            );
        }
    }
    drop((original_cooks, original_counts, dds));

    // R: all-zero genes have no dispersion estimate (`dispersions(dds)` is NA), rust_deseq2 reports a value
    for (dispersion, &base_mean) in results.dispersions.iter_mut().zip(&results.base_means) {
        if base_mean == 0.0 {
            *dispersion = f64::NAN;
        }
    }

    // R results(): genes that are all zero after outlier replacement are reported as not changed
    for &gene in &now_zero {
        results.log2_fold_changes[gene] = 0.0;
        results.lfc_se[gene] = 0.0;
        results.stat[gene] = 0.0;
        results.pvalues[gene] = 1.0;
    }

    if options.independent_filtering {
        independent_filtering(&mut results, options.alpha);
    } else {
        results.padj = benjamini_hochberg(&results.pvalues);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(condition: &[&str]) -> Design {
        Design {
            condition: condition.iter().map(|s| s.to_string()).collect(),
            numerator: "B".into(),
            denominator: "A".into(),
            factors: vec![],
            continuous: vec![],
        }
    }

    fn options() -> Options {
        Options {
            alpha: 0.1,
            min_replicates_for_replace: Some(7),
            cooks_cutoff: Some(None),
            independent_filtering: true,
        }
    }

    fn counts() -> Array2<f64> {
        // 6 genes x 6 samples (3 A, 3 B), gene 0 up in B, gene 1 down in B
        ndarray::array![
            [100.0, 110.0, 90.0, 400.0, 420.0, 380.0],
            [300.0, 310.0, 290.0, 75.0, 80.0, 70.0],
            [500.0, 520.0, 480.0, 500.0, 510.0, 490.0],
            [50.0, 55.0, 45.0, 50.0, 52.0, 48.0],
            [200.0, 220.0, 180.0, 200.0, 210.0, 190.0],
            [600.0, 620.0, 580.0, 600.0, 610.0, 590.0],
        ]
    }

    #[test]
    fn detects_up_and_down_regulated_genes() {
        let results = run_wald(counts(), design(&["A", "A", "A", "B", "B", "B"]), &options()).unwrap();
        assert_eq!(results.gene_ids, (0..6).map(|i| i.to_string()).collect::<Vec<_>>());
        assert!(results.log2_fold_changes[0] > 1.5);
        assert!(results.log2_fold_changes[1] < -1.5);
        assert!(results.padj[0] < 0.01 && results.padj[1] < 0.01);
        assert!(results.log2_fold_changes[2].abs() < 0.2);
    }

    #[test]
    fn swapping_the_contrast_flips_the_fold_change() {
        let up = run_wald(counts(), design(&["A", "A", "A", "B", "B", "B"]), &options()).unwrap();
        let mut d = design(&["A", "A", "A", "B", "B", "B"]);
        std::mem::swap(&mut d.numerator, &mut d.denominator);
        let down = run_wald(counts(), d, &options()).unwrap();
        for (a, b) in up.log2_fold_changes.iter().zip(&down.log2_fold_changes) {
            assert!((a + b).abs() < 1e-9);
        }
    }

    #[test]
    fn supports_categorical_covariates() {
        let mut d = design(&["A", "A", "A", "B", "B", "B"]);
        d.factors.push(("batch".into(), ["x", "y", "x", "y", "x", "y"].iter().map(|s| s.to_string()).collect()));
        let results = run_wald(counts(), d, &options()).unwrap();
        assert!(results.log2_fold_changes[0] > 1.5);
    }
}
