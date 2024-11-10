import numpy as np

from agdesa import AGDESA
from gdesa import GDESA
from mgdesa import MGDESA


def sphere(x: np.ndarray):
    return np.sum((x + 50) ** 2)


def rastrigin(x: np.ndarray):
    return 10 * x.shape[0] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


"""
Examples of the usage of the GDESA, MGDESA, and AGDESA
"""
print("========= SPHERE =========")

opt_res_gdesa_params = GDESA().get_default_parameters()
opt_res_gdesa_params["func"] = sphere
opt_res_gdesa_params["bounds"] = [(-100, 100)] * 30
opt_res_gdesa_params["seed"] = 1

opt_res_gdesa = GDESA().optimize(**opt_res_gdesa_params)

opt_res_mgdesa_params = MGDESA().get_default_parameters()
opt_res_mgdesa_params["func"] = sphere
opt_res_mgdesa_params["bounds"] = [(-100, 100)] * 30
opt_res_mgdesa_params["seed"] = 1

opt_res_mgdesa = MGDESA().optimize(**opt_res_mgdesa_params)

opt_res_agdesa_params = AGDESA().get_default_parameters()
opt_res_agdesa_params["func"] = sphere
opt_res_agdesa_params["bounds"] = [(-100, 100)] * 30
opt_res_agdesa_params["seed"] = 1

opt_res_agdesa = AGDESA().optimize(**opt_res_agdesa_params)

print(f"Optimal value found using GDESA: {opt_res_gdesa[1]}")
print(f"Optimal value found using MGDESA: {opt_res_mgdesa[1]}")
print(f"Optimal value found using AGDESA: {opt_res_agdesa[1]}")

print("========= RASTRIGIN =========")

opt_res_gdesa_params = GDESA().get_default_parameters()
opt_res_gdesa_params["func"] = rastrigin
opt_res_gdesa_params["bounds"] = [(-5.12, 5.12)] * 30
opt_res_gdesa_params["seed"] = 1
opt_res_gdesa_params["crossover_prob"] = 0.01

opt_res_gdesa = GDESA().optimize(**opt_res_gdesa_params)

opt_res_mgdesa_params = MGDESA().get_default_parameters()
opt_res_mgdesa_params["func"] = rastrigin
opt_res_mgdesa_params["bounds"] = [(-5.12, 5.12)] * 30
opt_res_mgdesa_params["seed"] = 1
opt_res_mgdesa_params["crossover_prob"] = 0.01

opt_res_mgdesa = MGDESA().optimize(**opt_res_mgdesa_params)

opt_res_agdesa_params = AGDESA().get_default_parameters()
opt_res_agdesa_params["func"] = rastrigin
opt_res_agdesa_params["bounds"] = [(-5.12, 5.12)] * 30
opt_res_agdesa_params["seed"] = 1
opt_res_agdesa_params["crossover_prob"] = 0.01

opt_res_agdesa = AGDESA().optimize(**opt_res_agdesa_params)

print(f"Optimal value found using GDESA: {opt_res_gdesa[1]}")
print(f"Optimal value found using MGDESA: {opt_res_mgdesa[1]}")
print(f"Optimal value found using AGDESA: {opt_res_agdesa[1]}")
