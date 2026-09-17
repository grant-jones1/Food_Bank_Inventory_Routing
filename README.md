# Multi-Item Food Bank Inventory Routing with Endogenous Information Acquisition

Computational study of multi-item food bank inventory routing under partial
observability, based on my own unpublished POMDP formulation. Includes a
full-information MILP baseline, a stochastic simulator with belief tracking, and
heuristic-to-ADP policies benchmarked on generated instances to quantify the value of
inspection.

## Problem

A food bank distributes several item types to a set of partner agencies. Agency
inventory is *hidden*: the food bank only learns an agency's true stock when it pays
to inspect it. Each period the food bank chooses

1. which agencies to visit (a vehicle routing decision),
2. how much of each item to deliver to each visited agency, and
3. which agencies to inspect.

Costs are routing, holding, shortage (unmet demand), and inspection. The objective is
to minimise expected discounted cost over an infinite horizon. Because the
inspection decision controls what the decision maker gets to observe, the problem
is a POMDP with *endogenous* information acquisition.

## Approach

- **Full-information MILP** (`deterministic.py`): a finite-horizon routing +
  multi-item replenishment model with known inventories and demand, solved with
  HiGHS. Serves as a perfect-hindsight lower bound.
- **Stochastic simulator** (`simulator.py`): partially observable environment
  maintaining a factored belief over agency inventories; beliefs are propagated
  through the inventory balance and collapsed on inspection.
- **Policies** (`policies/`): periodic inspection, belief-based reorder-point with
  uncertainty-triggered inspection, and a rollout (one-step lookahead ADP) policy.
- **Evaluation** (`evaluate.py`): Monte Carlo evaluation with common random
  numbers, cost breakdowns, and value-of-information estimates.

## Results

_To be added._

## Repo structure

```
src/foodbank_ir/         package source (see CLAUDE.md for module map)
tests/                   pytest suite
experiments/configs/     YAML experiment configurations
.github/workflows/       CI (pytest)
```

## Setup

Requires Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

## Citation

_To be added once the paper is available._
