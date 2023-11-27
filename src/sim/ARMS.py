import numpy as np
import numpy.typing as npt

from .helper_funcs import data_split, var_est
from .model_selection import OLS_AIC, OLS_BIC

from sklearn.linear_model import LinearRegression


def ARMS_init(
        X: npt.NDArray[np.float32],
        Y: list[npt.NDArray[np.float32]],
        candidate_models: list[list],
        m: np.int32 
        ) -> list:
    # candidate_models: [[AIC_1, BIC_1, M1], ..., [AIC_K, BIC_K, MK]]
    # find estimates for each candidate models
    for j in range(len(candidate_models)):
        # betas -> aic -> bic
        candidate_models[j][2] = candidate_models[j][2].fit(X, Y[j])
        candidate_models[j][0] = OLS_AIC(
            Y[j], candidate_models[j][2].predict(X), X.shape[0], X.shape[1]
            )
        candidate_models[j][1] = OLS_BIC(
            Y[j], candidate_models[j][2].predict(X), X.shape[0], X.shape[1]
            )

    # get top 'm' models of both AIC and BIC and return
    AIC_ordered_models = sorted(
        candidate_models, key=lambda x: int(x[0])
        )[::-1][:m]
    AIC_ordered_models = [item[2] for item in AIC_ordered_models]

    BIC_ordered_models = sorted(
        candidate_models, key=lambda x: int(x[1])
        )[::-1][:m]
    BIC_ordered_models = [item[2] for item in BIC_ordered_models]

    chosen_models = [*AIC_ordered_models, *BIC_ordered_models]
    chosen_models = [
        [mj, var_est(Y, mj.predict(X), X.shape[0], X.shape[1])]
        for mj in chosen_models
        ]

    # return chosen candidates
    return chosen_models


def ARMS_iter(
        X2: npt.NDArray[np.float32],
        Y2: npt.NDArray[np.float32],
        chosen_models: list,
        n: np.int32 # this is total n
        ) -> list:
    # calculating discrepancy for each model j
    discrepancy = []
    for model_j in chosen_models:
        Y2_preds_j = model_j[0].predict(X2)

        D_j = np.linalg.norm(Y2 - Y2_preds_j)**2
        discrepancy.append(D_j)

    # calculating weight for each model j
    model_weights = []

    # calculate the denominator
    var_mod1 = np.array([var[1]**(-n / 2) for var in chosen_models])
    var_mod2 = np.array([var[1]**(-1) for var in chosen_models])
    D_mod = np.array([(Dj / 2) for Dj in discrepancy])
    weight_denom = np.sum(var_mod1 * np.exp(-var_mod2 * D_mod))

    # calculate weights
    for j, mod_j in enumerate(chosen_models):
        W_j = mod_j[1]**(-n / 2) * np.exp(-mod_j[1]**(-1) * (discrepancy[j])) \
            / weight_denom
        model_weights.append([mod_j[0], W_j])

    return model_weights



def ARMS(X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32], m: np.int32, N: np.int32 = 50):
    
    for rep in range(N - 1):
        # split the data
        data_dict = data_split(X, Y)
