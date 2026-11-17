# Writes the R DESeq2 reference outputs of the golden tests, for every case directory in the given root:
#   expected.csv      gene_id, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, dispersion
#   size_factors.csv  sample, size_factor
# and <root>/VERSIONS with the R and DESeq2 versions used.
#
# Run via `make goldens` (Bioconductor Docker image, see Makefile). The inputs come from scripts/make_fixtures.py.
# Each case runs the equivalent of deseq2_rs.deseq2(): DESeq(dds) + results(dds, contrast = ...).

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[[1]] else "tests/fixtures"

for (pkg in c("DESeq2", "jsonlite")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
    BiocManager::install(pkg, update = FALSE, ask = FALSE, lib = Sys.getenv("R_LIBS_USER"))
  }
}
suppressPackageStartupMessages(library(DESeq2))

run_case <- function(dir) {
  options <- jsonlite::fromJSON(file.path(dir, "options.json"))
  counts <- as.matrix(read.csv(file.path(dir, "counts.csv"), row.names = 1, check.names = FALSE))
  coldata <- read.csv(file.path(dir, "coldata.csv"), row.names = 1, stringsAsFactors = FALSE)
  # deseq2_rs sorts the levels alphabetically, the contrast makes the reference level irrelevant for the results
  coldata$condition <- factor(coldata$condition, levels = sort(unique(coldata$condition)))
  for (f in options$factors) coldata[[f]] <- factor(coldata[[f]])
  design <- as.formula(paste("~", paste(c(options$factors, options$continuous, "condition"), collapse = " + ")))

  dds <- DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = design)
  dds <- DESeq(
    dds,
    minReplicatesForReplace = if (options$refit_cooks) 7 else Inf,
    quiet = TRUE
  )
  res <- results(
    dds,
    contrast = c("condition", options$numerator, options$denominator),
    alpha = options$alpha,
    independentFiltering = options$independent_filter,
    cooksCutoff = if (options$cooks_filter) TRUE else FALSE
  )

  expected <- data.frame(
    gene_id = rownames(res),
    baseMean = res$baseMean,
    log2FoldChange = res$log2FoldChange,
    lfcSE = res$lfcSE,
    stat = res$stat,
    pvalue = res$pvalue,
    padj = res$padj,
    dispersion = dispersions(dds)
  )
  write.csv(expected, file.path(dir, "expected.csv"), row.names = FALSE, quote = FALSE)
  size_factors <- data.frame(sample = colnames(dds), size_factor = sizeFactors(dds))
  write.csv(size_factors, file.path(dir, "size_factors.csv"), row.names = FALSE, quote = FALSE)
  message(sprintf("%s: %d genes, %d with padj < %.2f", basename(dir), nrow(res), sum(res$padj < options$alpha, na.rm = TRUE), options$alpha))
}

for (dir in list.dirs(root, recursive = FALSE)) {
  if (file.exists(file.path(dir, "options.json"))) run_case(dir)
}
writeLines(
  c(R.version.string, paste("DESeq2", packageVersion("DESeq2")), paste("Bioconductor", BiocManager::version())),
  file.path(root, "VERSIONS")
)
