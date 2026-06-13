"""
Particle Swarm Optimisation (from scratch)

Each particle is a vector of EcosystemParams values.
Velocity update:  v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
"""

import random
import numpy as np
from simulation import EcosystemParams, run_simulation

class Particle:
    def __init__(self, grid_size: int = 20):
        self.position  = EcosystemParams.random_params(grid_size).to_vector()
        rng            = EcosystemParams.BOUNDS[:, 1] - EcosystemParams.BOUNDS[:, 0]
        self.velocity  = np.array([random.uniform(-r * 0.05, r * 0.05) for r in rng])
        self.pbest_pos = self.position.copy()
        self.pbest_fit = -1.0
        self.fitness   = -1.0

class ParticleSwarmOptimisation:
    """
    Standard PSO with:
      w  = inertia weight (linearly decayed)
      c1 = cognitive coefficient
      c2 = social coefficient
    """

    def __init__(self, n_particles: int = 30,
                 w_start: float = 0.9, w_end: float = 0.4,
                 c1: float = 1.5, c2: float = 1.5,
                 grid_size: int = 20, hemisphere: str = 'N',
                 max_months: int = 120, callback=None):

        self.n_particles = n_particles
        self.w_start     = w_start
        self.w_end       = w_end
        self.c1          = c1
        self.c2          = c2
        self.grid_size   = grid_size
        self.hemisphere  = hemisphere
        self.max_months  = max_months
        self.callback    = callback

        self.particles  : list[Particle]      = []
        self.gbest_pos  : np.ndarray | None   = None
        self.gbest_fit  : float               = -1.0
        self.best_params: EcosystemParams | None = None

        # History
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history : list[float] = []
        self.diversity_history   : list[float] = []
        self.iteration           = 0
        self.running             = False
        self._total_iterations   = 30

    # Helpers
    def _evaluate(self, vec: np.ndarray) -> float:
        params = EcosystemParams.from_vector(vec)
        _, fit, _ = run_simulation(params, self.grid_size,
                                   self.hemisphere, self.max_months)
        return fit

    def _clip(self, vec: np.ndarray) -> np.ndarray:
        return np.clip(vec, EcosystemParams.BOUNDS[:, 0],
                            EcosystemParams.BOUNDS[:, 1])

    def _diversity(self) -> float:
        mat = np.array([p.position for p in self.particles])
        rng = EcosystemParams.BOUNDS[:, 1] - EcosystemParams.BOUNDS[:, 0]
        rng = np.where(rng == 0, 1, rng)
        return float(np.mean(np.std(mat, axis=0) / rng))

    # Initialise
    def initialise(self):
        self.particles = [Particle(self.grid_size) for _ in range(self.n_particles)]
        self.gbest_pos = None
        self.gbest_fit = -1.0
        self.best_fitness_history.clear()
        self.avg_fitness_history.clear()
        self.diversity_history.clear()
        self.iteration = 0

        # First evaluation
        for p in self.particles:
            p.fitness   = self._evaluate(p.position)
            p.pbest_fit = p.fitness
            if p.fitness > self.gbest_fit:
                self.gbest_fit = p.fitness
                self.gbest_pos = p.position.copy()

        self.best_params = EcosystemParams.from_vector(self.gbest_pos)

    # Single iteration
    def step_iteration(self):
        """Advance one PSO iteration.  Call initialise() first."""
        progress  = self.iteration / max(self._total_iterations - 1, 1)
        w         = self.w_start - progress * (self.w_start - self.w_end)

        for p in self.particles:
            r1      = np.random.rand(len(p.position))
            r2      = np.random.rand(len(p.position))
            cog     = self.c1 * r1 * (p.pbest_pos - p.position)
            soc     = self.c2 * r2 * (self.gbest_pos - p.position)
            p.velocity = w * p.velocity + cog + soc

            # Clamp velocity to +-20 % of range
            rng         = EcosystemParams.BOUNDS[:, 1] - EcosystemParams.BOUNDS[:, 0]
            max_vel     = rng * 0.20
            p.velocity  = np.clip(p.velocity, -max_vel, max_vel)

            p.position  = self._clip(p.position + p.velocity)
            p.fitness   = self._evaluate(p.position)

            if p.fitness > p.pbest_fit:
                p.pbest_fit = p.fitness
                p.pbest_pos = p.position.copy()

            if p.fitness > self.gbest_fit:
                self.gbest_fit = p.fitness
                self.gbest_pos = p.position.copy()
                self.best_params = EcosystemParams.from_vector(self.gbest_pos)

        fits = [p.fitness for p in self.particles]
        self.best_fitness_history.append(self.gbest_fit)
        self.avg_fitness_history.append(float(np.mean(fits)))
        self.diversity_history.append(self._diversity())
        self.iteration += 1

        info = dict(iteration=self.iteration,
                    best_fitness=self.gbest_fit,
                    avg_fitness=float(np.mean(fits)),
                    diversity=self._diversity(),
                    best_params=self.best_params)
        if self.callback:
            self.callback(info)
        return info

    # Full run 
    def run(self, iterations: int = 30):
        self._total_iterations = iterations
        self.running = True
        self.initialise()
        results = []
        for _ in range(iterations):
            if not self.running:
                break
            info = self.step_iteration()
            results.append(info)
        self.running = False
        return results

    def stop(self):
        self.running = False

    # Summary
    def summary(self) -> str:
        if not self.best_params:
            return "PSO has not been run yet."
        lines = [
            "=== Particle Swarm Optimisation Summary ===",
            f"Iterations completed  : {self.iteration}",
            f"Best fitness achieved : {self.gbest_fit:.4f}",
            f"Final diversity       : {self.diversity_history[-1]:.4f}" if self.diversity_history else "",
            "",
            "Best Parameters Found:",
            self.best_params.describe()
        ]
        return '\n'.join(lines)
