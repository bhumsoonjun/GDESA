from __future__ import annotations

import math

import numpy as np
from typing import *
import copy
import scipy.optimize as opt
import scipy.stats as stats
from abc import ABC, abstractmethod

from utils import latin


class BaseGDESA(ABC):

    def __init__(self):
        self.func = None
        self.bounds = None
        self.lb = None
        self.ub = None

        self.popsize = None
        self.mutation_factor = None
        self.crossover_prob = None
        self.T = None
        self.gradient_prob = None
        self.max_gen = None

        self.population = None
        self.population_fitness = None
        self.best_solution = None
        self.best_fitness = None
        self.cooldown_list = None

        self.seed = 0
        self.nfev = 0
        pass

    def apply_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        opt_res = opt.minimize(
            self.func_wrapper,
            x,
            method="L-BFGS-B",
            bounds=self.bounds,
            jac="2-point",
        )
        return opt_res.x, opt_res.fun

    def func_wrapper(self, x: np.ndarray) -> float:
        self.nfev += 1
        fitness = self.func(x)
        if self.best_fitness is None or fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = x
        return fitness

    def validate_params(self, **kwargs):
        default_param_keys = self.get_default_parameters().keys()
        for key in kwargs.keys():
            if key not in default_param_keys:
                raise ValueError(f"Parameter: {key} not found in given parameters")

    @abstractmethod
    def optimize(self, **kwargs):
        pass

    @abstractmethod
    def get_default_parameters(self):
        pass

    @abstractmethod
    def setup_params(self, *args, **kwargs):
        pass

    def setup(self, **kwargs):
        self.setup_params(**kwargs)

        np.random.seed(self.seed)

        self.lb = [b[0] for b in self.bounds]
        self.ub = [b[1] for b in self.bounds]

        self.population = latin(self.popsize, self.bounds, self.seed)
        self.population_fitness = np.apply_along_axis(
            self.func_wrapper, 1, self.population
        )

        min_idx = np.argmin(self.population_fitness)
        self.best_solution = self.population[min_idx]
        self.best_fitness = self.population_fitness[min_idx]

        self.cooldown_list = np.zeros_like(self.population_fitness)
