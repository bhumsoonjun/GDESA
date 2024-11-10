import copy
from typing import *
import numpy as np
from scipy import stats


def rand_1_bin_mutator(
    population: np.ndarray, differential_factor: float, i: int
) -> np.ndarray:
    j, k, l = 0, 0, 0
    while j == i:
        j = np.random.randint(0, population.shape[0])
    while k == i or k == j:
        k = np.random.randint(0, population.shape[0])
    while l == i or l == j or l == k:
        l = np.random.randint(0, population.shape[0])
    mutant_vector = population[j] + differential_factor * (
        population[k] - population[l]
    )
    return mutant_vector


def rand_2_bin_mutator(
    population: np.ndarray, differential_factor: float, i: int
) -> np.ndarray:
    j, k, l, m, n = 0, 0, 0, 0, 0
    while j == i:
        j = np.random.randint(0, population.shape[0])
    while k == i or k == j:
        k = np.random.randint(0, population.shape[0])
    while l == i or l == j or l == k:
        l = np.random.randint(0, population.shape[0])
    while m == i or m == j or m == k or m == l:
        m = np.random.randint(0, population.shape[0])
    while n == i or n == j or n == k or n == l or n == m:
        n = np.random.randint(0, population.shape[0])
    mutant_vector = population[j] + differential_factor * (
        population[k] - population[l] + population[m] - population[n]
    )
    return mutant_vector


def dual_mutation(population: np.ndarray, mutation_factor: float, i: int) -> np.ndarray:
    return (
        rand_1_bin_mutator(population, mutation_factor, i)
        if np.random.rand() < 0.5
        else rand_2_bin_mutator(population, mutation_factor, i)
    )

def binomial_crossover(
    target_vector: np.ndarray,
    mutant_vector: np.ndarray,
    crossover_prob: float
) -> np.ndarray:
    trial = np.copy(target_vector)
    fill_point = np.random.randint(0, target_vector.shape[0])
    crossovers = np.random.uniform(size=target_vector.shape)
    crossovers = crossovers < crossover_prob
    crossovers[fill_point] = True
    trial = np.where(crossovers, mutant_vector, trial)
    return trial

def check_bounds(x: np.ndarray, bounds: List[Tuple]) -> np.ndarray:
    x_copy = copy.deepcopy(x)
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    x_copy = np.clip(x_copy, lb, ub)
    return x_copy

def latin(n: int, bounds: List[Tuple], seed: int) -> np.ndarray:
    sample = stats.qmc.LatinHypercube(d=len(bounds), seed=seed).random(n=n)
    bounded = np.apply_along_axis(lambda x: check_bounds(x, bounds), 1, sample)
    return np.array(bounded, dtype=np.float64)
