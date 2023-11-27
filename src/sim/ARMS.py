import numpy as np
import numpy.typing as npt

from .helper_funcs import build_covariates, full_covariates, retrain_models
from .helper_funcs import var_est
from .model_selection import OLS_AIC, OLS_BIC

# from sklearn.linear_model import LinearRegression


def ARMS_screening(
        X: list[npt.NDArray[np.float32]],
        Y: npt.NDArray[np.float32],
        candidate_models: list[list],
        m: np.int32
        ) -> list:
    # candidate_models: [[AIC_1, BIC_1, M1, var_est_1],
    # ..., [AIC_K, BIC_K, MK, var_est_K]]
    # find estimates for each candidate models
    for j in range(len(candidate_models)):
        n_j = X[j].shape[0]
        p_j = X[j].shape[1]
        # betas -> aic -> bic
        candidate_models[j][2] = candidate_models[j][2].fit(X[j], Y)
        candidate_models[j][0] = OLS_AIC(
            Y, candidate_models[j][2].predict(X[j]), n_j, p_j
            )
        candidate_models[j][1] = OLS_BIC(
            Y, candidate_models[j][2].predict(X[j]), n_j, p_j
            )

    # calculating estimated error variance
    for j, model in enumerate(candidate_models):
        n_j = X[j].shape[0]
        p_j = X[j].shape[1]
        model[3] = var_est(Y, model[2].predict(X[j]), n_j, p_j)
        model.append(j)

    # get top 'm' models of both AIC and BIC and return
    AIC_ordered_models = sorted(
        candidate_models, key=lambda x: int(x[0])
        )[:m]
    AIC_ordered_models = [[item[2], item[3], item[4]]
                          for item in AIC_ordered_models]

    BIC_ordered_models = sorted(
        candidate_models, key=lambda x: int(x[1])
        )[:m+1]
    BIC_ordered_models = [[item[2], item[3], item[4]]
                          for item in BIC_ordered_models]

    chosen_models = [*AIC_ordered_models, *BIC_ordered_models]
    best_AIC = AIC_ordered_models[0]
    best_BIC = BIC_ordered_models[0]

    # return chosen candidates
    return chosen_models, best_AIC, best_BIC


def ARMS_iter(
        X2: list[npt.NDArray[np.float32]],
        Y2: npt.NDArray[np.float32],
        chosen_models: list,
        n_total: np.int32  # this is total n
        ) -> list:
    # calculating discrepancy for each model j
    discrepancy = []
    for model_j in chosen_models:
        Y2_preds_j = model_j[0].predict(X2[model_j[2]])

        D_j = np.linalg.norm(Y2 - Y2_preds_j)**2
        discrepancy.append(D_j)

    # calculating weight for each model j
    model_weights = []

    # calculate the denominator
    var_mod1 = np.array(
        [np.sqrt(var[1])**(-n_total / 2) for var in chosen_models]
        )
    var_mod2 = np.array([np.sqrt(var[1])**(-2) for var in chosen_models])
    D_mod = np.array([(Dj / 2) for Dj in discrepancy])
    weight_denom = np.sum(var_mod1 * np.exp(-var_mod2 * D_mod))

    # calculate weights
    for j, mod_j in enumerate(chosen_models):
        W_j = np.sqrt(mod_j[1])**(-n_total / 2) \
            * np.exp(-np.sqrt(mod_j[1])**(-2) * (discrepancy[j] / 2)) \
            / weight_denom
        model_weights.append(W_j)

    return model_weights


def ARMS(
        X_full: npt.NDArray[np.float32],
        Y_true: npt.NDArray[np.float32],
        m: np.int32, N: np.int32 = 50
        ):
    aggregate_model_weights = []
    # iteration 0
    # split the data
    D1, D2, candidate_models = build_covariates(
        X_full, Y_true, permute=False
        )
    # perform screening stage
    chosen_models, best_AIC, best_BIC = ARMS_screening(
        D1[0], D1[1], candidate_models, m
        )

    # aggregate the model weights to get final weight for model j
    aggregate_model_weights = [[mod[2], 0] for mod in chosen_models]
    for j in range(N-1):
        weights_j = ARMS_iter(D2[0], D2[1], chosen_models, X_full.shape[0])
        for i, agg_weight in enumerate(aggregate_model_weights):
            agg_weight[1] = agg_weight[1] + weights_j[i]

        D1, D2, _ = build_covariates(
            X_full, Y_true, permute=True
        )

        chosen_models = retrain_models(D1[0], D1[1], chosen_models)

    # divide summed weights by N
    aggregate_model_weights = [[w[0], w[1]/N] for w in aggregate_model_weights]

    # compute final model
    covariates = full_covariates(X_full)
    final_model = retrain_models(covariates, Y_true, chosen_models)
    final_model = [[model[2], model[0]] for model in chosen_models]
    for j, model in enumerate(final_model):
        Wj_hat = aggregate_model_weights[j][1]
        model.append(Wj_hat)

    return final_model, covariates, best_AIC, best_BIC
