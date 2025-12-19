import numpy as np
import numpy.typing as npt


def weighted_quantile(
    x: npt.NDArray[np.float64], q: float, weights: npt.NDArray[np.float64]
) -> float:
    if np.sum(weights) >= 1 - q:
        x_ordered = np.argsort(x)
        index_threshold = np.min(np.where(np.cumsum(weights[x_ordered]) >= 1 - q))
        quantile = np.sort(x)[index_threshold]

    else:
        quantile = float("inf")

    return float(quantile)
