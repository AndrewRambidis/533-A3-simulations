import numpy as np
import numpy.typing as npt

from sklearn.utils import shuffle



def data_generation(
        n: np.int32,
        p: np.int32,
        true_betas: npt.NDArray[np.float32],
        std: np.float32
            ) -> list[npt.NDArray[np.float32]]:
    # predictors generated via a Unif[-1,1];
    # error distributed via n[0, sigma^2].
    # in paper: n = 1000, p = 10, sigma^2 in [0.1, 0.5, 1.0, 2.25, 4.0]
    X = np.random.uniform(-1, 1, size=(n, p))
    e = np.random.normal(0, std, n).reshape(-1, 1)
    Y = X @ true_betas + e

    return [X, Y]

def data_split(
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float32],
        permute: bool = False
            ) -> dict[dict[str, npt.NDArray[np.float32]]]:
    n = X.shape[0]
    mid = np.floor(n / 2).astype(np.int32)

    if permute:
        X, Y = shuffle(X, Y, random_state=42)

    D1_X = X[0:mid, :]
    D1_Y = Y[0:mid, :]

    D2_X = X[mid:n, :]
    D2_Y = Y[mid:n, :]

    return {
        "D1": {"D1_X": D1_X, "D1_Y": D1_Y},
        "D2": {"D2_X": D2_X, "D2_Y": D2_Y}
        }


def var_est(
        Y_true: npt.NDArray[np.float32],
        Y_pred: npt.NDArray[np.float32],
        n: np.int32,
        p: np.int32
            ) -> np.float32:
    return (1 / (n - p)) * np.linalg.norm(Y_true - Y_pred)**2
