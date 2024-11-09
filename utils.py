import numpy as np

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
    if np.random.rand() < 0.5:
        mutant_vector = rand_1_bin_mutator(
            population, mutation_factor, i
        )
    else:
        mutant_vector = rand_2_bin_mutator(
            population, mutation_factor, i
        )
    return mutant_vector