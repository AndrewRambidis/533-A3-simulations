import unittest
import numpy as np

from src.sim.ARMS import ARMS
from src.sim.helper_funcs import data_generation
from src.sim.model_selection import MSE
from sklearn.linear_model import LinearRegression


class testARMS(unittest.TestCase):
    def test_base(self):
        X_full, Y_true = data_generation(
            1000, 10,
            np.array(
                [1.5, 1.6, 1.7, 1.5, 0.4, 0.3, 0.2, 0.1, 0, 0]
                ).reshape(-1, 1), 0.9, np.sqrt(4.0))

        # chosen_models = ARMS_screening(D1[0], D1[1], candidate_models, 4)
        # weights = ARMS_iter(D2[0], D2[1], chosen_models, 100)
        # print(weights)
        # print(sum([weight[1] for weight in weights]))

        final_model, covariates, best_AIC, best_BIC = ARMS(X_full, Y_true, 5, 100)

        Y_preds = [
            mod[2] * mod[1].predict(covariates[mod[0]]) for mod in final_model
            ]
        final_model_preds = np.zeros(Y_true.shape)
        for preds_j in Y_preds:
            final_model_preds += preds_j

        # print(final_model_preds.shape)
        # print(np.concatenate((final_model_preds, Y_true), axis=1))
        print("ARMS MSE:")
        print(MSE(Y_true, final_model_preds))
        print("AIC MSE:")
        print(MSE(Y_true, best_AIC[0].predict(covariates[best_AIC[2]])))
        print("BIC MSE:")
        print(MSE(Y_true, best_AIC[0].predict(covariates[best_AIC[2]])))
