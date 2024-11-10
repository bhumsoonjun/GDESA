from __future__ import annotations
from base_gdesa import BaseGDESA
import numpy as np
from typing import *
import math

from utils import *


class AGDESA(BaseGDESA):

    def __init__(self):
        super().__init__()
        self.S = None
        self.L = None
        self.T_init = None

    def get_default_parameters(self) -> Dict:
        return {
            "func": None,
            "bounds": None,
            "popsize": 40,
            "mutation_factor": 0.5,
            "crossover_prob": 0.1,
            "T": 5000,
            "nl": 100,
            "gradient_prob": 0.01,
            "max_gen": 5000,
            "seed": 0,
        }

    def setup_params(self, **kwargs):
        self.validate_params(**kwargs)

        self.func = kwargs["func"]
        self.bounds = kwargs["bounds"]
        self.popsize = kwargs["popsize"]
        self.mutation_factor = kwargs["mutation_factor"]
        self.crossover_prob = kwargs["crossover_prob"]
        self.T_init = kwargs["T"]
        self.nl = kwargs["nl"]
        self.gradient_prob = kwargs["gradient_prob"]
        self.max_gen = kwargs["max_gen"]
        self.seed = kwargs["seed"]

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.S = np.zeros_like(self.population_fitness)
        self.L = np.zeros_like(self.population_fitness)

    def pre_process(self, gen: int):
        for j in range(self.popsize):
            if gen - self.L[j] > self.nl:
                self.S[j] = 1
            else:
                self.S[j] = 0

    def post_process(self, gen: int):
        pass

    def step(self, gen: int):
        T = self.T_init / gen

        for j in range(self.popsize):
            target_vector = self.population[j]

            if (
                self.gradient_prob > 0
                and np.random.rand() < self.gradient_prob / self.popsize
                and self.cooldown_list[j] <= gen
            ):
                trial_vector, trial_val = self.apply_gradient(target_vector)
                self.cooldown_list[j] = math.ceil(1 / self.gradient_prob) + gen
            else:
                mutant_vector = dual_mutation(self.population, self.mutation_factor, j)
                trial_vector = binomial_crossover(target_vector, mutant_vector, self.crossover_prob)
                trial_vector = check_bounds(trial_vector, self.bounds)
                trial_val = self.func_wrapper(trial_vector)

            delta = trial_val - self.population_fitness[j]

            if delta < 0:
                self.population[j] = trial_vector
                self.population_fitness[j] = trial_val
                self.L[j] = gen
            elif (
                T > 0
                and self.S[j] == 1
                and (delta / T < 10e-10 or np.random.rand() < np.exp(-delta / T))
            ):
                self.population[j] = trial_vector
                self.population_fitness[j] = trial_val
                self.cooldown_list[j] = (
                    math.ceil(1 / self.gradient_prob) + gen
                    if self.gradient_prob > 0
                    else 0
                )

        return self.best_solution, self.best_fitness
