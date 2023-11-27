import numpy as np
import numpy.typing as npt


def MSE(
    Y_true: npt.NDArray[np.float32],
    Y_pred: npt.NDArray[np.float32]
        ) -> np.float32:
    return (1 / Y_true.shape[0]) * np.linalg.norm(Y_true - Y_pred)**2


def OLS_AIC(
    Y_true: npt.NDArray[np.float32],
    Y_pred: npt.NDArray[np.float32],
    n: int,
    p: int
        ) -> list[float, float]:
    return n * np.log(MSE(Y_true, Y_pred)) + 2 * p


def OLS_BIC(
    Y_true: npt.NDArray[np.float32],
    Y_pred: npt.NDArray[np.float32],
    n: int,
    p: int
        ) -> list[float, float]:
    return n * np.log(MSE(Y_true, Y_pred)) + p * np.log(n)
