"""
Research Experiments & Analysis
Implements all 6 experiments from the project proposal:
  1. Algorithm comparison (GA vs PSO)
  2. Parameter sensitivity analysis
  3. Neural Network prediction accuracy
  4. Hemisphere comparison
  5. Evolutionary dynamics
  6. Decision Tree rule extraction
"""

import numpy as np
from simulation import EcosystemParams, run_simulation

# Experiment 1: GA vs PSO Comparison
def compare_algorithms(ga_history: list, pso_history: list) -> dict:
    ga  = np.array(ga_history,  dtype=float)
    pso = np.array(pso_history, dtype=float)

    def convergence_iter(arr, threshold=0.90):
        if len(arr) == 0: return len(arr)
        target = arr.max() * threshold
        for i, v in enumerate(arr):
            if v >= target: return i + 1
        return len(arr)

    result = dict(
        ga_best=float(ga.max())  if len(ga)  else 0,
        pso_best=float(pso.max()) if len(pso) else 0,
        ga_final=float(ga[-1])   if len(ga)  else 0,
        pso_final=float(pso[-1]) if len(pso) else 0,
        ga_convergence=convergence_iter(ga),
        pso_convergence=convergence_iter(pso),
        ga_history=ga.tolist(),
        pso_history=pso.tolist(),
        winner="GA" if (ga.max() if len(ga) else 0) >= (pso.max() if len(pso) else 0) else "PSO"
    )
    return result



# Experiment 2: Parameter Sensitivity Analysis
def parameter_sensitivity(base_params: EcosystemParams,
                           grid_size: int = 20, hemisphere: str = 'N',
                           n_steps: int = 6, callback=None) -> dict:
    bounds  = EcosystemParams.BOUNDS
    names   = EcosystemParams.PARAM_NAMES
    labels  = EcosystemParams.LABELS
    results = {}
    total   = len(names) * n_steps
    done    = 0

    for idx, (name, label) in enumerate(zip(names, labels)):
        lo, hi   = bounds[idx]
        test_vals= np.linspace(lo, hi, n_steps)
        survivals= []

        for v in test_vals:
            vec          = base_params.to_vector().copy()
            vec[idx]     = v
            p            = EcosystemParams.from_vector(vec)
            months, _, _ = run_simulation(p, grid_size, hemisphere, 60)
            survivals.append(float(months))
            done += 1
            if callback: callback(done, total)

        sa = np.array(survivals)
        results[name] = dict(
            label       = label,
            values      = test_vals.tolist(),
            survivals   = sa.tolist(),
            sensitivity = float(sa.max() - sa.min()),
            mean        = float(sa.mean()),
        )

    # Rank
    ranked = sorted(results.items(), key=lambda x: x[1]['sensitivity'], reverse=True)
    return dict(results=results, ranked=ranked)



# Experiment 3: NN Prediction Accuracy  (computed outside but stored here)
def nn_accuracy_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    rmse  = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res= float(np.sum((y_true - y_pred) ** 2))
    ss_tot= float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    r2    = 1.0 - ss_res / ss_tot
    return dict(mae=mae, rmse=rmse, r2=r2,
                y_true=y_true.tolist(), y_pred=y_pred.tolist())

# Experiment 4: Hemisphere Comparison
def hemisphere_comparison(n_trials: int = 15,
                           grid_size: int = 20,
                           max_months: int = 120,
                           callback=None) -> dict:
    
   # Run n trials random simulations for each hemisphere and compare.
    results = {'N': [], 'S': []}
    total   = n_trials * 2
    done    = 0

    for hemi in ('N', 'S'):
        for _ in range(n_trials):
            params = EcosystemParams.random_params(grid_size)
            months, fit, _ = run_simulation(params, grid_size, hemi, max_months)
            results[hemi].append(dict(months=months, fitness=fit))
            done += 1
            if callback: callback(done, total)

    n_months = np.array([r['months']  for r in results['N']])
    s_months = np.array([r['months']  for r in results['S']])

    return dict(
        north_mean   = float(n_months.mean()),
        south_mean   = float(s_months.mean()),
        north_std    = float(n_months.std()),
        south_std    = float(s_months.std()),
        north_max    = float(n_months.max()),
        south_max    = float(s_months.max()),
        north_months = n_months.tolist(),
        south_months = s_months.tolist(),
    )

# Experiment 5: Evolutionary Dynamics
def evolutionary_dynamics_report(ga_best_hist: list,
                                  ga_avg_hist:  list,
                                  ga_div_hist:  list) -> dict:
    
# Derives insights from GA history arrays already collected during a GA run.
    
    best = np.array(ga_best_hist)
    avg  = np.array(ga_avg_hist)
    div  = np.array(ga_div_hist)

    if len(best) == 0:
        return {}

    # Detect phase boundary where improvement slows
    improvements = np.diff(best, prepend=best[0])
    threshold    = improvements.max() * 0.05
    exploit_start= next((i for i, v in enumerate(improvements) if i > 2 and v < threshold),
                        len(best) // 2)

    return dict(
        best_history       = best.tolist(),
        avg_history        = avg.tolist(),
        diversity_history  = div.tolist(),
        total_improvement  = float(best[-1] - best[0]),
        exploitation_start = exploit_start,
        final_diversity    = float(div[-1]) if len(div) else 0.0,
        converged          = bool(div[-1] < 0.05) if len(div) else False,
    )



# Experiment 6: Decision Tree Insights  (the tree is in ml_models)
def dt_insights_report(tree, X_test: np.ndarray,
                        y_test: np.ndarray) -> dict:
    
# Produce a full report dict.
    
    metrics    = tree.evaluate(X_test, y_test)
    top_feats  = tree.top_features(8)
    return dict(
        accuracy     = metrics['accuracy'],
        top_features = top_feats,
        rules        = tree.rules[:12],    # cap at 12 for display
        predictions  = metrics['predictions'].tolist(),
        y_true       = y_test.tolist(),
    )

# Grid-size Scalability
GRID_SIZES = [10, 20, 30, 40]

def scalability_test(params: EcosystemParams,
                     hemisphere: str = 'N',
                     max_months: int = 60,
                     callback=None) -> dict:
    
# Run the same parameter set on all four grid sizes.
# Returns survival and fitness for each.
    results = {}
    for i, gs in enumerate(GRID_SIZES):
        scaled = EcosystemParams.random_params(gs)
        months, fit, _ = run_simulation(scaled, gs, hemisphere, max_months)
        results[gs] = dict(months=months, fitness=fit, grid_size=gs,
                           area=gs*gs, label=f"{gs}×{gs}")
        if callback: callback(i + 1, len(GRID_SIZES))
    return results
