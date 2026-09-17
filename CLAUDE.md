# CLAUDE.md

Guidance for AI assistants (Claude Code) working in this repository.

## Project

Computational study for the paper *"Multi-Item Food Bank Inventory Routing with
Endogenous Information Acquisition."* A food bank delivers multiple item types to
agencies; agency inventory is hidden and only observed when the food bank pays to
inspect it. Each period the decision is which agencies to visit, what to deliver,
and whether to inspect. Formulated as an infinite-horizon POMDP. This repo turns the
formulation into experiments: a full-information MILP baseline, a stochastic
simulator with belief tracking, and heuristic-to-ADP policies benchmarked on
generated instances to quantify the value of inspection.

## Implementation boundary (READ FIRST)

**The repository owner writes all core logic.** That means everything in `src/`
that actually computes something: the optimization model, the simulator and belief
updates, the policies, evaluation, and plotting.

The assistant's role is limited to:

- scaffolding (module/function/class skeletons: signatures, type hints,
  docstrings, and `raise NotImplementedError` bodies),
- tests,
- code review,
- CI and tooling,
- documentation.

Concretely: **do not implement anything in `src/` beyond function signatures, type
hints, docstrings, and `raise NotImplementedError`.** Do not "fill in" a stub, even
a trivial one, even if asked casually. If the owner asks you to implement core
logic, remind them of this rule first and proceed only if they explicitly
reaffirm the request after the reminder.

Things that are fine to do in `src/` without asking: fix a typo in a docstring,
adjust a type hint, add a missing import needed by an annotation, add a new stub
with a signature and docstring.

## Repo layout

```
src/foodbank_ir/           installable package (src layout)
  instance.py              FoodBankInstance dataclass + random instance generator
  deterministic.py         full-information MILP baseline (pyomo + HiGHS)
  simulator.py             POMDP environment: TrueState, BeliefState, Action, Simulator
  evaluate.py              Monte Carlo policy evaluation, CRN comparisons, VoI
  plots.py                 comparison plots (matplotlib)
  policies/
    base.py                abstract Policy
    periodic_inspection.py fixed inspection schedule baseline
    reorder_point.py       belief-based (s, S) with uncertainty-triggered inspection
    rollout.py             one-step lookahead / rollout ADP
tests/                     pytest suite; placeholders are xfail until implemented
  test_properties.py       commented sketches of structural-property checks from the paper
experiments/configs/       YAML experiment configs (results/ is gitignored)
.github/workflows/         CI: pytest on push and pull request
```

## Conventions

- Python 3.13, `src/` layout, package name `foodbank_ir`, installed editable via
  `pip install -e ".[dev]"`.
- Dependencies are declared in `pyproject.toml` only (no requirements.txt).
- LP/MIP solver is HiGHS via `highspy`, driven through pyomo's `appsi_highs`
  interface. Routing subproblems may use OR-Tools.
- Policies act on `BeliefState` only, never on the simulator's hidden `TrueState`
  (which is exposed only when `Simulator(expose_true_state=True)`, for tests and
  oracle bounds).
- Arrays are `numpy` with shape `(num_agencies, num_items)` for per-period
  quantities and `(horizon, num_agencies, num_items)` for trajectories. The depot is
  node id `-1`; agencies are `0..num_agencies-1`.
- Cost breakdown keys are exactly `routing`, `holding`, `shortage`, `inspection`.
- Randomness goes through `numpy.random.Generator` seeded explicitly; episode `i`
  of an evaluation uses `base_seed + i` so policies can be compared with common
  random numbers.
- Docstrings: Google style. Default to no inline comments; add one only when the
  *why* is non-obvious.
- Tests: `pytest`. A placeholder test for unimplemented logic is marked
  `@pytest.mark.xfail(raises=NotImplementedError, reason=...)`; remove the marker
  when the logic lands so the test becomes a real check.
- Run locally: `source .venv/bin/activate && python -m pytest`.
- Commit messages: imperative mood, one line summary; note AI assistance where it
  applies (e.g. "(AI-assisted scaffolding)").
- Do not add new top-level dependencies, CI jobs, or restructure the package
  without asking.
