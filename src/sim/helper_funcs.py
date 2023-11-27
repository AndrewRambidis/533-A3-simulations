import numpy as np
import numpy.typing as npt

from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression


def data_generation(
        n: np.int32,
        p: np.int32,
        true_betas: npt.NDArray[np.float32],
        intercept: np.float32,
        std: np.float32
            ) -> list[npt.NDArray[np.float32]]:
    # predictors generated via a Unif[-1,1];
    # error distributed via n[0, sigma^2].
    # in paper: n = 1000, p = 10, sigma^2 in [0.1, 0.5, 1.0, 2.25, 4.0]
    X = np.random.uniform(-1, 1, size=(n, p))
    e = np.random.normal(0, std, n).reshape(-1, 1)
    Y = X @ true_betas + np.full(shape=(n, 1), fill_value=intercept) + e

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


def build_covariates(X_full, Y_true, permute: bool = False):
    Data = data_split(X_full, Y_true, permute)
    X_D1 = Data["D1"]["D1_X"]
    Y_D1 = Data["D1"]["D1_Y"]
    X_D2 = Data["D2"]["D2_X"]
    Y_D2 = Data["D2"]["D2_Y"]

    X0_D1 = np.ones(shape=(X_D1.shape))
    X1_D1 = X_D1[:, :1]
    X2_D1 = X_D1[:, :2]
    X3_D1 = X_D1[:, :3]
    X4_D1 = X_D1[:, :4]
    X5_D1 = X_D1[:, :5]
    X6_D1 = X_D1[:, :6]
    X7_D1 = X_D1[:, :7]
    X8_D1 = X_D1[:, :8]
    X9_D1 = X_D1[:, :9]
    X10_D1 = X_D1[:, :10]

    X0_D2 = np.ones(shape=(100, 1))
    X1_D2 = X_D2[:, :1]
    X2_D2 = X_D2[:, :2]
    X3_D2 = X_D2[:, :3]
    X4_D2 = X_D2[:, :4]
    X5_D2 = X_D2[:, :5]
    X6_D2 = X_D2[:, :6]
    X7_D2 = X_D2[:, :7]
    X8_D2 = X_D2[:, :8]
    X9_D2 = X_D2[:, :9]
    X10_D2 = X_D2[:, :10]

    covariates_D1 = [
            X0_D1, X1_D1, X2_D1, X3_D1, X4_D1, X5_D1,
            X6_D1, X7_D1, X8_D1, X9_D1, X10_D1
            ]
    covariates_D2 = [
            X0_D2, X1_D2, X2_D2, X3_D2, X4_D2, X5_D2,
            X6_D2, X7_D2, X8_D2, X9_D2, X10_D2
            ]
    candidate_models = [
            [0, 0, LinearRegression(), 0] for Xj in covariates_D1
                        ]

    return [covariates_D1, Y_D1], [covariates_D2, Y_D2], candidate_models


def full_covariates(X_full):
    X0 = np.ones(shape=(X_full.shape))
    X1 = X_full[:, :1]
    X2 = X_full[:, :2]
    X3 = X_full[:, :3]
    X4 = X_full[:, :4]
    X5 = X_full[:, :5]
    X6 = X_full[:, :6]
    X7 = X_full[:, :7]
    X8 = X_full[:, :8]
    X9 = X_full[:, :9]
    X10 = X_full[:, :10]

    full_covariates = [
        X0, X1, X2, X3, X4, X5,
        X6, X7, X8, X9, X10
    ]

    return full_covariates


def retrain_models(X_new, Y_new, chosen_models):
    for model in chosen_models:
        Xj = X_new[model[2]]
        n_j = Xj.shape[0]
        p_j = Xj.shape[1]
        model[0] = model[0].fit(Xj, Y_new)
        model[1] = var_est(Y_new, model[0].predict(Xj), n_j, p_j)

    return chosen_models
