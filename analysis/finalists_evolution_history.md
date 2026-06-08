# Finalist Evolution History

This note summarizes the evolution history of the two presentation finalists:

- `xXxrmKQvo8ZECIkfTW0ZxO0TKNi`: best overall individual
- `xXxc55dwASBVXnywni3E4zDh0sF`: best practical low-control-cost individual

All metrics below use the saved stats files:

- reward: maximize
- distance: maximize
- control cost: minimize

## 1. Best Overall Individual

### Model

- Gene ID: `xXxrmKQvo8ZECIkfTW0ZxO0TKNi`
- First seen in population: generation `12`
- Final metrics: reward `1461.4393`, distance `92.7004`, control cost `0.3926`

### Artifact Locations

- Model source: `sota/MujocoRL/models/network_xXxrmKQvo8ZECIkfTW0ZxO0TKNi.py`
- Final weights: `sota/MujocoRL/trained_models/xXxrmKQvo8ZECIkfTW0ZxO0TKNi.zip`
- Stats: `sota/MujocoRL/stats/xXxrmKQvo8ZECIkfTW0ZxO0TKNi_stats.json`

### Structural Lineage

The overall-best model came from a mixed lineage involving both mutation and crossover:

1. `xXxjOH38bl0QwZU7tCi3JJAEICx`
2. `xXxAl9pHUTa6ku9ZoiZ8dvLiWjz`
3. `xXxqu7httWQYkWWCbK0u5xY4biH`
4. crossover with `xXxaSluXhlBWfUCI4kS42N4MTrE` to produce `xXxc55dwASBVXnywni3E4zDh0sF`
5. crossover with `xXxeAlUMVM9e9r6AnGWboVQmA2l` to produce `xXxg2z4dICuGMrLrYi9ANUZE92Y`
6. mutation from `xXxg2z4dICuGMrLrYi9ANUZE92Y` to `xXxrmKQvo8ZECIkfTW0ZxO0TKNi`

Note: the intermediate crossover child `xXxg2z4dICuGMrLrYi9ANUZE92Y` appears in the run log ancestry but does not have a saved stats JSON in the current workspace, so the quantitative table below tracks the evaluated milestones around that crossover.

### Quantitative Progress

| Stage | Gene ID | Role in lineage | Reward | Distance | Control Cost | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `xXxjOH38bl0QwZU7tCi3JJAEICx` | baseline ancestor | 755.0857 | 49.8181 | 0.2413 | strong seed-derived baseline |
| 2 | `xXxAl9pHUTa6ku9ZoiZ8dvLiWjz` | mutation | 808.0707 | 53.1346 | 0.2546 | improved reward and distance |
| 3 | `xXxqu7httWQYkWWCbK0u5xY4biH` | mutation | 744.3399 | 51.4049 | 0.2838 | slight regression, higher cost |
| 4 | `xXxaSluXhlBWfUCI4kS42N4MTrE` | crossover parent | 492.7960 | 36.9355 | 0.2459 | weaker but lower-cost alternate branch |
| 5 | `xXxeAlUMVM9e9r6AnGWboVQmA2l` | later crossover parent | 314.8798 | 26.9282 | 0.2237 | low-performance but efficient donor branch |
| 6 | `xXxrmKQvo8ZECIkfTW0ZxO0TKNi` | final winner | 1461.4393 | 92.7004 | 0.3926 | major breakthrough in reward and distance |

### Main Story

- The best overall policy did **not** come from a simple monotonic mutation path.
- It emerged after combining a strong reward-distance branch with lower-cost alternate branches through crossover.
- The final mutation after the second crossover produced a very large jump in both reward and forward progress.
- This is a strong example of why mixed search operators mattered: crossover introduced diversity, and mutation then refined it into the top-performing policy.

## 2. Best Practical Low-Control-Cost Individual

### Model

- Gene ID: `xXxc55dwASBVXnywni3E4zDh0sF`
- First seen in population: generation `8`
- Final metrics: reward `876.8275`, distance `55.2520`, control cost `0.2282`

### Why this model matters

This is the best "practical efficiency" model: it maintains strong reward and distance while keeping control cost substantially lower than the highest-performing policies.

### Artifact Locations

- Model source: `sota/MujocoRL/models/network_xXxc55dwASBVXnywni3E4zDh0sF.py`
- Final weights: `sota/MujocoRL/trained_models/xXxc55dwASBVXnywni3E4zDh0sF.zip`
- Stats: `sota/MujocoRL/stats/xXxc55dwASBVXnywni3E4zDh0sF_stats.json`

### Structural Lineage

This model came from a strong mutation branch that was then combined with a lower-cost branch via crossover:

1. `xXxjOH38bl0QwZU7tCi3JJAEICx`
2. `xXxAl9pHUTa6ku9ZoiZ8dvLiWjz`
3. `xXxqu7httWQYkWWCbK0u5xY4biH`
4. crossover with `xXxaSluXhlBWfUCI4kS42N4MTrE`
5. `xXxc55dwASBVXnywni3E4zDh0sF`

### Quantitative Progress

| Stage | Gene ID | Role in lineage | Reward | Distance | Control Cost | Interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | `xXxjOH38bl0QwZU7tCi3JJAEICx` | baseline ancestor | 755.0857 | 49.8181 | 0.2413 | strong corrected baseline |
| 2 | `xXxAl9pHUTa6ku9ZoiZ8dvLiWjz` | mutation | 808.0707 | 53.1346 | 0.2546 | reward and distance improved |
| 3 | `xXxqu7httWQYkWWCbK0u5xY4biH` | mutation | 744.3399 | 51.4049 | 0.2838 | reward regressed and cost increased |
| 4 | `xXxaSluXhlBWfUCI4kS42N4MTrE` | crossover parent | 492.7960 | 36.9355 | 0.2459 | weaker but lower-cost branch |
| 5 | `xXxc55dwASBVXnywni3E4zDh0sF` | final efficient model | 876.8275 | 55.2520 | 0.2282 | strong reward with improved efficiency |

### Main Story

- The mutation path alone did not steadily improve; one branch regressed in both reward and efficiency.
- Crossover with a lower-cost branch appears to have helped recover efficiency while still preserving good task performance.
- The result is a much better tradeoff point than the pure high-performance winner: lower reward than the overall best, but meaningfully lower control cost with still-solid distance.
- This makes `xXxc55dwASBVXnywni3E4zDh0sF` a strong representative of the "efficient but competitive" part of the Pareto frontier.

## Slide-Friendly Summary

### Best overall: `xXxrmKQvo8ZECIkfTW0ZxO0TKNi`

- Highest reward and highest distance in the run
- Emerged from a mixed mutation + crossover lineage
- Demonstrates that diverse search operators were useful

### Best practical low-cost: `xXxc55dwASBVXnywni3E4zDh0sF`

- Strong reward and distance with lower control cost
- Represents a useful efficient operating point on the Pareto frontier
- Better presentation example for "efficiency tradeoff" than the degenerate raw minimum-control-cost policy
