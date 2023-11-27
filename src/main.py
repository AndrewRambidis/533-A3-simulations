import numpy as np

from sim.ARMS import ARMS
from sim.model_selection import MSE
from sim.helper_funcs import data_generation, final_model_preds


def full_run(betas, intercept, variance, rep_range=100):
    over_all = {"ARMS": 0, "AIC": 0, "BIC": 0}
    for rep in range(rep_range):
        X, Y = data_generation(
            1000, 10,
            betas, intercept, np.sqrt(variance))

        ARMS_f, covariates, AIC, BIC = ARMS(X, Y, 5, 100)
        ARMS_preds = final_model_preds(ARMS_f, covariates, Y)
        AIC_preds = AIC[0].predict(covariates[AIC[2]])
        BIC_preds = BIC[0].predict(covariates[BIC[2]])

        over_all["ARMS"] += MSE(Y, ARMS_preds)
        over_all["AIC"] += MSE(Y, AIC_preds)
        over_all["BIC"] += MSE(Y, BIC_preds)

    over_all["ARMS"] = over_all["ARMS"] / 100
    over_all["AIC"] = over_all["AIC"] / 100
    over_all["BIC"] = over_all["BIC"] / 100

    return over_all


print("Case 3:")
betas = np.array([0.8, 0.9, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1,1)
intercept = 1.0

print("sigma^2 = 0.1")
var_01 = 0.1
print(full_run(betas, intercept, var_01))
print()

print("sigma^2 = 0.5")
var_05 = 0.5
print(full_run(betas, intercept, var_05))
print()

print("sigma^2 = 1.0")
var_1 = 1.0
print(full_run(betas, intercept, var_1))
print()

print("sigma^2 = 2.25")
var_225 = 2.25
print(full_run(betas, intercept, var_225))
print()

print("sigma^2 = 4.0")
var_4 = 4.0
print(full_run(betas, intercept, var_4))
print()
