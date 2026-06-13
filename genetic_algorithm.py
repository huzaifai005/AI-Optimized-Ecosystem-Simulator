"""
Genetic Algorithm made from scratch

It evolves EcosystemParams to maximise fitness (survival + biodiversity + stability).
"""

import random, time
import numpy as np
from simulation import EcosystemParams, run_simulation

class GeneticAlgorithm:
    """
    Standard GA with:
      - Tournament selection
      - Uniform crossover
      - Gaussian mutation
      - Elitism (top-1 always survives)
    """

    def __init__(self, population_size: int = 40, mutation_rate: float = 0.10,
                 crossover_rate: float = 0.80, tournament_k: int = 3,
                 grid_size: int = 20, hemisphere: str = 'N',
                 max_months: int = 120, callback=None):

        self.pop_size       = population_size
        self.mutation_rate  = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_k   = tournament_k
        self.grid_size      = grid_size
        self.hemisphere     = hemisphere
        self.max_months     = max_months
        self.callback       = callback   # called each generation with progress dict

        self.population : list[EcosystemParams] = []
        self.fitness    : list[float]            = []

        # History tracking
        self.best_fitness_history : list[float] = []
        self.avg_fitness_history  : list[float] = []
        self.diversity_history    : list[float] = []
        self.best_params          : EcosystemParams | None = None
        self.generation           = 0
        self.running              = False

    # Initialise 
    def initialise(self):
        self.population = [EcosystemParams.random_params(self.grid_size)
                           for _ in range(self.pop_size)]
        self.fitness    = [0.0] * self.pop_size
        self.best_fitness_history.clear()
        self.avg_fitness_history.clear()
        self.diversity_history.clear()
        self.generation = 0

    # Evaluate one individual 
    def _evaluate(self, params: EcosystemParams) -> float:
        _, fitness, _ = run_simulation(params, self.grid_size,
                                       self.hemisphere, self.max_months)
        return fitness

    # Evaluate whole population 
    def _evaluate_population(self):
        for i, ind in enumerate(self.population):
            self.fitness[i] = self._evaluate(ind)

    # Tournament selection
        candidates = random.sample(range(self.pop_size), self.tournament_k)
        winner     = max(candidates, key=lambda i: self.fitness[i])
        return self.population[winner].clone()

    # Uniform crossover 
    def _crossover(self, p1: EcosystemParams,
                   p2: EcosystemParams) -> tuple:
        v1, v2 = p1.to_vector(), p2.to_vector()
        if random.random() > self.crossover_rate:
            return p1.clone(), p2.clone()
        mask  = np.random.rand(len(v1)) < 0.5
        c1    = np.where(mask, v1, v2)
        c2    = np.where(mask, v2, v1)
        return EcosystemParams.from_vector(c1), EcosystemParams.from_vector(c2)

    # Gaussian mutation 
    def _mutate(self, ind: EcosystemParams) -> EcosystemParams:
        vec    = ind.to_vector()
        bounds = EcosystemParams.BOUNDS
        for i in range(len(vec)):
            if random.random() < self.mutation_rate:
                sigma      = (bounds[i, 1] - bounds[i, 0]) * 0.08
                vec[i]    += random.gauss(0, sigma)
                vec[i]     = float(np.clip(vec[i], bounds[i, 0], bounds[i, 1]))
        return EcosystemParams.from_vector(vec)

    # Diversity metric 
    def _diversity(self) -> float:
        mat = np.array([ind.to_vector() for ind in self.population])
        rng = EcosystemParams.BOUNDS[:, 1] - EcosystemParams.BOUNDS[:, 0]
        rng = np.where(rng == 0, 1, rng)
        return float(np.mean(np.std(mat, axis=0) / rng))

    # Single generation 
    def step_generation(self):
        self._evaluate_population()

        best_idx  = int(np.argmax(self.fitness))
        best_fit  = self.fitness[best_idx]
        avg_fit   = float(np.mean(self.fitness))
        div       = self._diversity()

        self.best_fitness_history.append(best_fit)
        self.avg_fitness_history.append(avg_fit)
        self.diversity_history.append(div)

        if self.best_params is None or best_fit > (self.best_fitness_history[-2] if len(self.best_fitness_history) > 1 else -1):
            self.best_params = self.population[best_idx].clone()

        # Build next generation
        new_pop = [self.population[best_idx].clone()]   # elitism
        while len(new_pop) < self.pop_size:
            parent1 = self._select()
            parent2 = self._select()
            child1, child2 = self._crossover(parent1, parent2)
            new_pop.append(self._mutate(child1))
            if len(new_pop) < self.pop_size:
                new_pop.append(self._mutate(child2))

        self.population = new_pop
        self.generation += 1

        info = dict(generation=self.generation,
                    best_fitness=best_fit,
                    avg_fitness=avg_fit,
                    diversity=div,
                    best_params=self.best_params)
        if self.callback:
            self.callback(info)
        return info

    # Full run
    def run(self, generations: int = 30):
        self.running = True
        self.initialise()
        results = []
        for _ in range(generations):
            if not self.running:
                break
            info = self.step_generation()
            results.append(info)
        self.running = False
        return results

    def stop(self):
        self.running = False

    # Summary string
        if not self.best_params:
            return "GA has not been run yet."
        lines = [
            f"=== Genetic Algorithm Summary ===",
            f"Generations completed : {self.generation}",
            f"Best fitness achieved : {max(self.best_fitness_history):.4f}",
            f"Final diversity       : {self.diversity_history[-1]:.4f}" if self.diversity_history else "",
            "",
            "Best Parameters Found:",
            self.best_params.describe()
        ]
        return '\n'.join(lines)
