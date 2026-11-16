import numpy as np
import numpy.typing as npt

def deseq2_wald(
    counts: np.ndarray,
    condition: list[str],
    numerator: str,
    denominator: str,
    *,
    factors: list[tuple[str, list[str]]] = ...,
    continuous: list[tuple[str, list[float]]] = ...,
    alpha: float = 0.1,
    min_replicates_for_replace: int | None = 7,
    cooks_filter: bool = True,
    cooks_cutoff: float | None = None,
    independent_filtering: bool = True,
    n_threads: int | None = None,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint64]]: ...
