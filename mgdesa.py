from __future__ import annotations

from base_gdesa import BaseGDESA
import numpy as np
from typing import *
import math
from utils import *

class MGDESA(BaseGDESA):

    def __init__(self):
        super().__init__()
        self.alpha = None
        self.beta = None
        self.elitism_ratio = None
        self.elitism_amount = None
        self.TL = None
        self.p_cnt = 1

    def get_default_parameters(self) -> Dict:
        return {
            "func": None,
            "bounds": None,
            "popsize": 40,
            "mutation_factor": 0.5,
            "crossover_prob": 0.1,
            "TL": [(200000, 0.8, 0), (20000, 0.2, 0.5), (1000, 0.8, 0.8)],
            "alpha": 0.95,
            "beta": 10,
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
        self.TL = kwargs["TL"]
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.gradient_prob = kwargs["gradient_prob"]
        self.max_gen = kwargs["max_gen"]
        self.seed = kwargs["seed"]

        self.T = self.TL[0][0]
        self.elitism_ratio = self.TL[0][1]
        self.elitism_amount = int(self.popsize * self.elitism_ratio)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.elitism_amount = int(self.popsize * self.elitism_ratio)

    def pre_process(self, gen: int):
        if self.p_cnt < len(self.TL) and gen >= self.max_gen * self.TL[self.p_cnt][2]:
            self.T = self.TL[self.p_cnt][0]
            self.elitism_amount = int(self.popsize * self.TL[self.p_cnt][1])
            indices = np.arange(self.popsize)
            np.random.shuffle(indices)

            self.population = self.population[indices]
            self.population_fitness = self.population_fitness[indices]
            best = np.argmin(self.population_fitness)

            self.population[0], self.population[best] = (
                self.population[best],
                self.population[0],
            )
            self.population_fitness[0], self.population_fitness[best] = (
                self.population_fitness[best],
                self.population_fitness[0],
            )
            self.p_cnt += 1

        if gen % self.beta == 0:
            self.T = self.alpha * self.T

    def post_process(self, gen: int):
        pass

    def step(self, gen: int):

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
            elif (
                self.T > 0
                and j >= self.elitism_amount
                and (
                    delta / self.T < 10e-10
                    or np.random.rand() < np.exp(-delta / self.T)
                )
            ):
                self.population[j] = trial_vector
                self.population_fitness[j] = trial_val
                self.cooldown_list[j] = (
                    math.ceil(1 / self.gradient_prob) + gen
                    if self.gradient_prob > 0
                    else 0
                )

        return self.best_solution, self.best_fitness
