"""
Core Ecosystem Simulation Engine
"""

import random, math
import numpy as np

# Seasons
_NORTH = {1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring',
          6:'Summer', 7:'Summer', 8:'Summer', 9:'Autumn', 10:'Autumn',
          11:'Autumn', 12:'Winter'}
_SOUTH = {1:'Summer', 2:'Summer', 3:'Autumn', 4:'Autumn', 5:'Autumn',
          6:'Winter', 7:'Winter', 8:'Winter', 9:'Spring', 10:'Spring',
          11:'Spring',  12:'Summer'}

def get_season(month: int, hemisphere: str = 'N') -> str:
    m = ((month - 1) % 12) + 1
    return _NORTH[m] if hemisphere == 'N' else _SOUTH[m]

SEASON_COLORS = {'Spring': '#A5D6A7', 'Summer': '#FFF59D',
                 'Autumn': '#FFCC80', 'Winter': '#B0BEC5'}

# Ecosystem Parameters  (24 total)
class EcosystemParams:

    PARAM_NAMES = [
        'initial_plants', 'initial_herbivores', 'initial_carnivores',
        'plant_spread_spring', 'plant_spread_summer',
        'plant_spread_autumn', 'plant_spread_winter',
        'plant_max_age', 'plant_winter_death',
        'energy_from_plant', 'herb_move_cost', 'herb_monthly_loss',
        'herb_repro_thresh', 'herb_vision', 'herb_max_age', 'herb_start_energy',
        'energy_from_herb', 'carn_move_cost', 'carn_monthly_loss',
        'carn_repro_thresh', 'carn_vision', 'carn_max_age',
        'carn_start_energy', 'hunt_rate'
    ]

    # (min, max) for each parameter
    BOUNDS = np.array([
        (30,  800), (5,   300), (2,   50),
        (0.15,0.55),(0.25,0.65),(0.08,0.35),(0.02,0.18),
        (18,  60),  (0.02,0.12),
        (15,  55),  (1.0, 5.5), (2.0, 9.0),
        (55,  135), (2,   6),   (28,  75),  (65,  135),
        (38,  95),  (1.5, 6.5), (3.5, 11.0),
        (75,  155), (3,   8),   (45,  100), (75,  135), (0.38,0.92)
    ], dtype=float)

    INT_IDX = {0, 1, 2, 7, 13, 14, 20, 21}   # indices of integer-valued params

    DEFAULTS = np.array([
        250, 100, 10,
        0.35, 0.45, 0.20, 0.08,
        36,   0.05,
        30.0, 2.0,  5.0,
        80.0, 3,    48,   100.0,
        60.0, 3.0,  7.0,
        100.0,5,    72,   100.0, 0.70
    ], dtype=float)

    # Human-readable labels for GUI
    LABELS = [
        'Initial Plants', 'Initial Herbivores', 'Initial Carnivores',
        'Plant Spread (Spring)', 'Plant Spread (Summer)',
        'Plant Spread (Autumn)', 'Plant Spread (Winter)',
        'Plant Max Age', 'Plant Winter Death Rate',
        'Energy from Plant', 'Herb Move Cost', 'Herb Monthly Loss',
        'Herb Repro Threshold', 'Herb Vision', 'Herb Max Age', 'Herb Start Energy',
        'Energy from Herb', 'Carn Move Cost', 'Carn Monthly Loss',
        'Carn Repro Threshold', 'Carn Vision', 'Carn Max Age',
        'Carn Start Energy', 'Hunt Success Rate'
    ]

    def __init__(self, vec=None):
        v = self.DEFAULTS.copy() if vec is None else np.asarray(vec, dtype=float)
        self._vec = np.clip(v, self.BOUNDS[:, 0], self.BOUNDS[:, 1])
        self._sync()

    def _sync(self):
        for i, name in enumerate(self.PARAM_NAMES):
            val = float(self._vec[i])
            setattr(self, name, int(round(val)) if i in self.INT_IDX else val)

    def to_vector(self) -> np.ndarray:
        return self._vec.copy()

    @classmethod
    def from_vector(cls, vec):
        return cls(vec)

    @classmethod
    def random_params(cls, grid_size: int = 20):
        mc = grid_size * grid_size
        vec = np.zeros(len(cls.PARAM_NAMES))
        vec[0] = random.randint(max(30,  int(mc * 0.28)), min(800, int(mc * 0.58)))
        vec[1] = random.randint(max(5,   int(mc * 0.05)), min(300, int(mc * 0.17)))
        vec[2] = random.randint(2, min(50, max(2, int(mc * 0.03))))
        total  = vec[0] + vec[1] + vec[2]
        if total > mc * 0.85:
            f = mc * 0.80 / total
            vec[0] = max(30, int(vec[0] * f))
            vec[1] = max(5,  int(vec[1] * f))
            vec[2] = max(2,  int(vec[2] * f))
        for i in range(3, len(cls.PARAM_NAMES)):
            lo, hi = cls.BOUNDS[i]
            vec[i] = (random.randint(int(lo), int(hi))
                      if i in cls.INT_IDX else random.uniform(lo, hi))
        return cls(vec)

    def clone(self):
        return EcosystemParams(self._vec.copy())

    def describe(self) -> str:
        lines = []
        for name, label in zip(self.PARAM_NAMES, self.LABELS):
            val = getattr(self, name)
            lines.append(f"  {label:<28}: {val}")
        return '\n'.join(lines)


# Entity Classes
class Plant:
    __slots__ = ('r', 'c', 'alive', 'age', 'max_age')
    def __init__(self, r, c, max_age=36):
        self.r = r;  self.c = c;  self.alive = True
        self.age = 0
        self.max_age = max(6, max_age + random.randint(-8, 8))


class Herbivore:
    __slots__ = ('r','c','alive','gender','age','energy',
                 'pregnant','preg_timer','nursing','nurse_timer')
    def __init__(self, r, c, start_energy=100.0, gender=None):
        self.r = r;  self.c = c;  self.alive = True
        self.gender = gender or random.choice(('M','F'))
        self.age = 0
        self.energy = float(start_energy) + random.uniform(-8, 8)
        self.pregnant = False;  self.preg_timer  = 0
        self.nursing  = False;  self.nurse_timer = 0


class Carnivore:
    __slots__ = ('r','c','alive','gender','age','energy',
                 'pregnant','preg_timer','nursing','nurse_timer')
    def __init__(self, r, c, start_energy=100.0, gender=None):
        self.r = r;  self.c = c;  self.alive = True
        self.gender = gender or random.choice(('M','F'))
        self.age = 0
        self.energy = float(start_energy) + random.uniform(-8, 8)
        self.pregnant = False;  self.preg_timer  = 0
        self.nursing  = False;  self.nurse_timer = 0


# Ecosystem: Main simulation class
class Ecosystem:

    _SPREAD_KEY = {'Spring': 'plant_spread_spring', 'Summer': 'plant_spread_summer',
                   'Autumn': 'plant_spread_autumn', 'Winter': 'plant_spread_winter'}

    def __init__(self, params=None, grid_size: int = 20, hemisphere: str = 'N'):
        self.params      = params or EcosystemParams()
        self.grid_size   = grid_size
        self.hemisphere  = hemisphere
        self.month       = 0
        self.plants      : list = []
        self.herbivores  : list = []
        self.carnivores  : list = []
        self.grid        = [[None] * grid_size for _ in range(grid_size)]
        self.stats_history: list = []
        self._populate()

    # Initial population placement
    def _populate(self):
        p  = self.params
        N  = self.grid_size
        cells = [(r, c) for r in range(N) for c in range(N)]
        random.shuffle(cells)
        idx = 0

        for kind, count, cls, se, gender_alt in [
            ('plant',     p.initial_plants,     Plant,     None,              None),
            ('herbivore', p.initial_herbivores,  Herbivore, p.herb_start_energy, True),
            ('carnivore', p.initial_carnivores,  Carnivore, p.carn_start_energy, True),
        ]:
            n = min(count, N * N - idx - 2)
            for i in range(n):
                if idx >= len(cells): break
                r, c = cells[idx]; idx += 1
                if kind == 'plant':
                    e = Plant(r, c, p.plant_max_age)
                else:
                    g = ('M' if i % 2 == 0 else 'F') if gender_alt else None
                    e = cls(r, c, se, g)
                getattr(self, kind + 's').append(e)
                self.grid[r][c] = e

    # Grid helpers
    def _nbrs(self, r, c, radius=1):
        N = self.grid_size
        return [(r + dr, c + dc)
                for dr in range(-radius, radius + 1)
                for dc in range(-radius, radius + 1)
                if (dr or dc) and 0 <= r + dr < N and 0 <= c + dc < N]

    def _nearby_type(self, r, c, radius, cls):
        return [self.grid[nr][nc]
                for nr, nc in self._nbrs(r, c, radius)
                if self.grid[nr][nc] is not None
                and self.grid[nr][nc].alive
                and isinstance(self.grid[nr][nc], cls)]

    def _empty_near(self, r, c, radius=1):
        return [(nr, nc) for nr, nc in self._nbrs(r, c, radius)
                if self.grid[nr][nc] is None]

    def _move(self, e, nr, nc):
        self.grid[e.r][e.c] = None
        e.r = nr;  e.c = nc
        self.grid[nr][nc] = e

    def _kill(self, e):
        if self.grid[e.r][e.c] is e:
            self.grid[e.r][e.c] = None
        e.alive = False

    def _step_toward(self, e, target):
        dr = (0 if target.r == e.r else (1 if target.r > e.r else -1))
        dc = (0 if target.c == e.c else (1 if target.c > e.c else -1))
        nr, nc = e.r + dr, e.c + dc
        N = self.grid_size
        return (nr, nc) if 0 <= nr < N and 0 <= nc < N else (None, None)

    # Main simulation step (one month)
    def step(self) -> dict:
        self.month += 1
        season   = get_season(self.month, self.hemisphere)
        p        = self.params
        spread   = getattr(p, self._SPREAD_KEY[season])

        stats = dict(month=self.month, season=season,
                     plants=0, herbivores=0, carnivores=0,
                     plant_births=0, herb_births=0, carn_births=0,
                     plant_deaths=0, herb_deaths=0, carn_deaths=0)

        # Plants
        new_plants = []
        for pl in self.plants:
            if not pl.alive: continue
            pl.age += 1
            if pl.age >= pl.max_age:
                self._kill(pl); stats['plant_deaths'] += 1; continue
            if season == 'Winter' and random.random() < p.plant_winter_death:
                self._kill(pl); stats['plant_deaths'] += 1; continue
            if random.random() < spread:
                emp = self._empty_near(pl.r, pl.c, 2)
                if emp:
                    nr, nc = random.choice(emp)
                    baby = Plant(nr, nc, p.plant_max_age)
                    self.grid[nr][nc] = baby
                    new_plants.append(baby)
                    stats['plant_births'] += 1

        self.plants = [pl for pl in self.plants if pl.alive] + new_plants

        # Herbivores
        new_herbs = []
        random.shuffle(self.herbivores)
        for h in self.herbivores:
            if not h.alive: continue
            h.age    += 1
            h.energy -= p.herb_monthly_loss
            if h.age >= p.herb_max_age or h.energy <= 0:
                self._kill(h); stats['herb_deaths'] += 1; continue

            # Flee if carnivore is close
            if self._nearby_type(h.r, h.c, 2, Carnivore):
                emp = self._empty_near(h.r, h.c, 1)
                if emp:
                    nr, nc = random.choice(emp)
                    h.energy -= p.herb_move_cost * 0.5
                    self._move(h, nr, nc)
            else:
                near_plants = self._nearby_type(h.r, h.c, p.herb_vision, Plant)
                if near_plants:
                    tgt = min(near_plants, key=lambda x: abs(x.r-h.r)+abs(x.c-h.c))
                    if abs(tgt.r-h.r) <= 1 and abs(tgt.c-h.c) <= 1 and tgt.alive:
                        h.energy = min(h.energy + p.energy_from_plant, 200.0)
                        self._kill(tgt)
                        self.plants = [pl for pl in self.plants if pl.alive]
                    else:
                        nr, nc = self._step_toward(h, tgt)
                        if nr is not None and self.grid[nr][nc] is None:
                            h.energy -= p.herb_move_cost
                            self._move(h, nr, nc)
                else:
                    emp = self._empty_near(h.r, h.c, 1)
                    if emp:
                        nr, nc = random.choice(emp)
                        h.energy -= p.herb_move_cost * 0.25
                        self._move(h, nr, nc)

            if not h.alive: continue

            # Pregnancy / birth
            if h.pregnant:
                h.preg_timer += 1
                if h.preg_timer >= 3:
                    h.pregnant = False;  h.preg_timer = 0
                    h.nursing  = True;   h.nurse_timer = 0
                    emp = self._empty_near(h.r, h.c, 2)
                    if emp and h.energy > 45:
                        nr, nc = random.choice(emp)
                        baby = Herbivore(nr, nc, p.herb_start_energy * 0.55,
                                         random.choice(('M','F')))
                        self.grid[nr][nc] = baby
                        new_herbs.append(baby)
                        stats['herb_births'] += 1
                        h.energy -= 22
                continue

            # Reproduce
            if not h.nursing and h.gender == 'F' and h.energy >= p.herb_repro_thresh:
                mates = [m for m in self._nearby_type(h.r, h.c, 2, Herbivore)
                         if m.gender == 'M' and not m.pregnant and m.energy >= 55]
                if mates:
                    h.pregnant = True;  h.preg_timer = 0

            if h.nursing:
                h.nurse_timer += 1
                if h.nurse_timer >= 2:
                    h.nursing = False

        self.herbivores = [h for h in self.herbivores if h.alive] + new_herbs

        # Carnivores
        new_carns = []
        random.shuffle(self.carnivores)
        for cv in self.carnivores:
            if not cv.alive: continue
            cv.age    += 1
            cv.energy -= p.carn_monthly_loss
            if cv.age >= p.carn_max_age or cv.energy <= 0:
                self._kill(cv); stats['carn_deaths'] += 1; continue

            near_herbs = self._nearby_type(cv.r, cv.c, p.carn_vision, Herbivore)
            if near_herbs:
                tgt = min(near_herbs, key=lambda x: abs(x.r-cv.r)+abs(x.c-cv.c))
                if abs(tgt.r-cv.r) <= 1 and abs(tgt.c-cv.c) <= 1 and tgt.alive:
                    if random.random() < p.hunt_rate:
                        cv.energy = min(cv.energy + p.energy_from_herb, 250.0)
                        self._kill(tgt)
                        self.herbivores = [h for h in self.herbivores if h.alive]
                else:
                    nr, nc = self._step_toward(cv, tgt)
                    if nr is not None and self.grid[nr][nc] is None:
                        cv.energy -= p.carn_move_cost
                        self._move(cv, nr, nc)
            else:
                emp = self._empty_near(cv.r, cv.c, 1)
                if emp:
                    nr, nc = random.choice(emp)
                    cv.energy -= p.carn_move_cost * 0.25
                    self._move(cv, nr, nc)

            if not cv.alive: continue

            # Pregnancy / birth
            if cv.pregnant:
                cv.preg_timer += 1
                if cv.preg_timer >= 4:
                    cv.pregnant = False;  cv.preg_timer = 0
                    cv.nursing  = True;   cv.nurse_timer = 0
                    emp = self._empty_near(cv.r, cv.c, 2)
                    if emp and cv.energy > 55:
                        nr, nc = random.choice(emp)
                        baby = Carnivore(nr, nc, p.carn_start_energy * 0.55,
                                          random.choice(('M','F')))
                        self.grid[nr][nc] = baby
                        new_carns.append(baby)
                        stats['carn_births'] += 1
                        cv.energy -= 30
                continue

            if not cv.nursing and cv.gender == 'F' and cv.energy >= p.carn_repro_thresh:
                mates = [m for m in self._nearby_type(cv.r, cv.c, 3, Carnivore)
                         if m.gender == 'M' and not m.pregnant and m.energy >= 75]
                if mates:
                    cv.pregnant = True;  cv.preg_timer = 0

            if cv.nursing:
                cv.nurse_timer += 1
                if cv.nurse_timer >= 3:
                    cv.nursing = False

        self.carnivores = [cv for cv in self.carnivores if cv.alive] + new_carns

        stats.update(plants=len(self.plants),
                     herbivores=len(self.herbivores),
                     carnivores=len(self.carnivores))
        self.stats_history.append(stats)
        return stats

    # Accessors
    def is_alive(self) -> bool:
        return (len(self.plants) > 0 and
                len(self.herbivores) > 0 and
                len(self.carnivores) > 0)

    def get_fitness(self) -> float:
        if not self.stats_history:
            return 0.0
        survival = self.month / 120.0

        total = len(self.herbivores) + len(self.carnivores)
        if total > 0:
            ph = len(self.herbivores) / total
            pc = len(self.carnivores) / total
            bio = 0.0
            if ph > 0: bio -= ph * math.log2(ph + 1e-9)
            if pc > 0: bio -= pc * math.log2(pc + 1e-9)
            bio = min(bio, 1.0)
        else:
            bio = 0.0

        if len(self.stats_history) >= 10:
            hp = [s['herbivores'] for s in self.stats_history[-12:] if s['herbivores'] > 0]
            stability = (1.0 / (1.0 + np.std(hp) / max(np.mean(hp), 1))
                         if hp else 0.0)
        else:
            stability = 0.5

        return float(0.5 * survival + 0.3 * bio + 0.2 * stability)

    def get_snapshot(self):
        snap = []
        for pl in self.plants:
            if pl.alive: snap.append((pl.r, pl.c, 'plant', ''))
        for h in self.herbivores:
            if h.alive:  snap.append((h.r,  h.c,  'herbivore', h.gender))
        for cv in self.carnivores:
            if cv.alive: snap.append((cv.r, cv.c, 'carnivore', cv.gender))
        return snap


# runner; used by GA, PSO, data-collection
def run_simulation(params, grid_size: int = 20, hemisphere: str = 'N',
                   max_months: int = 120):
    eco = Ecosystem(params, grid_size, hemisphere)
    for _ in range(max_months):
        eco.step()
        if not eco.is_alive():
            break
    return eco.month, eco.get_fitness(), eco.stats_history
