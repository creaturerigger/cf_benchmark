# CF-Benchmark: Perturbation-Aware Counterfactual Robustness Framework

[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE.md)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey)](https://github.com/creaturerigger/cf_benchmark)
[![status](https://img.shields.io/badge/status-preprint-blue)](https://github.com/creaturerigger/cf_benchmark)

The open-source implementation of the paper:

> **Robust Counterfactual Explanations via Stochastic Multi-Objective Optimisation**
> Volkan Bakir, Polat Goktas, Süreyya Akyüz — *Preprint, under review, 2026*

This framework formulates counterfactual explanation selection as a **stochastic multi-objective optimisation problem**, evaluating candidate explanations under Gaussian input perturbations across three objectives: proximity, geometric instability, and intervention instability. Pareto-efficient solutions are extracted without scalarisation, and instability profiles are summarised via area-under-the-curve scores.

## Overview

Counterfactual explanations identify minimal changes to an input that flip a classifier's prediction, providing actionable recourse. However, most generation methods optimise only validity and proximity, neglecting robustness: small perturbations to the query instance can yield drastically different explanations.

This framework addresses this gap by:

1. **Building candidate counterfactual pools** — R independent generation runs per query, each producing up to N candidates, followed by deduplication (default: R=50, N=5 → up to 250 raw candidates)
2. **Perturbing query instances** — M stochastic Gaussian-perturbed copies per σ level, with feature-type-aware treatment (continuous: additive Gaussian noise; categorical: uniform resampling with probability σ)
3. **Evaluating three objectives** for each candidate:
   - **(i) Proximity** — L2 distance to the original query
   - **(ii) Geometric instability** — mean spatial displacement of the matched CF under perturbations (L2 default; L1, L∞, Cosine, Mahalanobis also supported)
   - **(iii) Intervention instability** — mean Jaccard dissimilarity of feature-change sets under perturbations
4. **Extracting the Pareto front** — non-dominated candidates across all three objectives, avoiding arbitrary scalarisation
5. **Generating instability profiles** — curves of mean instability vs. σ, summarised via trapezoidal AUC scores

## Architecture

```text
YAML Configs
    │
    ├── Dataset Configs ─→ Data Loaders ─→ PYTDataset (preprocessing + train/test split)
    │                                            │
    ├── Model Config ────→ PYTModel ◄────── Trainer (BCELoss, Adam)
    │                          │
    ├── CF Method Config ─→ CF Method Registry ───→ CFPoolBuilder ─→ Deduplicator ─→ CSV Pool
    │                          ├── DiCE (OPT)       │
    │                          ├── NICE (IB)        │
    │                          ├── GS  (HSS) *      │
    │                          ├── MOC (EVO)        │
    │                          └── LORE (RULE)      │
    │                                               │
    └── Experiment Config                      Perturbations (Gaussian / Uniform)
                                                    ├── Continuous: N(0, σ²) clamped to [0,1]
                                                    └── Categorical: Uniform(Vⱼ) with prob σ
                                                    │
                                               Robustness Evaluation
                                                    ├── Nearest-Neighbour Matching (L2)
                                                    ├── Geometric Instability (f₂)
                                                    └── Intervention Instability (f₃, Jaccard)
                                                    │
                                               Pareto Front Extraction (O(n²))
                                                    │
                                               Selection Strategies
                                                    ├── Min-Proximity
                                                    ├── Weighted-Sum (equal / prox-heavy)
                                                    ├── Pareto-Knee
                                                    ├── Pareto-Lex (robustness-first)
                                                    └── Random (baseline)
                                                    │
                                               Instability Profiles & AUC
```

> \* GrowingSpheres is registered and available but was excluded from the full benchmark due to its iterative hypersphere expansion being computationally infeasible at the chosen scale (R=50, n=10 queries). See Section 6.7 of the paper.

## Project Structure

```text
cf_benchmark/
├── main.py                          # Quick-start entry point (single combo)
├── pyproject.toml                   # Project metadata and dependencies
├── configs/                         # YAML configuration files
│   ├── cf_method/
│   │   ├── dice.yaml                # DiCE: gradient search, 500 iterations, proximity 0.5 / diversity 1.0
│   │   ├── nice.yaml                # NICE: sparsity optimisation, HEOM distance, justified_cf=true
│   │   ├── gs.yaml                  # GrowingSpheres: n_in_layer=2000, first_radius=0.1
│   │   ├── moc.yaml                 # MOC: NSGA-II, pop=20, n_gen=175, p_cross=0.57, p_mut=0.79
│   │   └── lore.yaml                # LORE: genetic neighbourhood, num_instances=1000
│   ├── dataset/
│   │   ├── adult.yaml               # Adult Income (UCI, 32,561 instances, 8 features)
│   │   ├── compas.yaml              # COMPAS Recidivism (6,172 instances)
│   │   ├── german.yaml              # German Credit (UCI, 1,000 instances, 20 features)
│   │   ├── lending.yaml             # Lending Club (Kaggle)
│   │   ├── heloc.yaml               # HELOC (FICO)
│   │   └── diabetes.yaml            # Diabetes 130-US Hospitals (768 instances, 8 features)
│   ├── experiment/
│   │   └── robustness_experiment.yaml
│   └── model/
│       └── pytorch_classifier.yaml  # MLP: 1 hidden layer, 20 ReLU units, sigmoid output
├── scripts/
│   ├── run_parallel.py              # Multi-core experiment runner (all dataset × method combos)
│   ├── timing_benchmark.py          # Wall-clock timing estimation with scenario extrapolation
│   └── selection_analysis.py        # Post-hoc selection strategy comparison and figure generation
├── notebooks/
│   └── analysis.ipynb               # Results analysis notebook
├── results/                         # Method-scoped outputs (gitignored except tables)
│   ├── models/                      # Trained model checkpoints (shared across methods)
│   ├── <method>/                    # Per-method results: dice | nice | gs | moc | lore
│   │   ├── figures/                 # Instability profiles, Pareto fronts, distributions
│   │   ├── logs/                    # Prefect pipeline logs (JSONL per dataset)
│   │   ├── pools/                   # CF pool CSVs (original + perturbed, per dataset)
│   │   ├── raw/                     # Raw evaluation records (*_records.csv)
│   │   └── tables/                  # Per-method summary tables
│   ├── tables/                      # Shared tables (timing, pool convergence)
│   └── selection_analysis/          # Cross-method selection strategy outputs
│       ├── tables/                  # Aggregated, per-method, per-dataset CSVs
│       └── figures/                 # Stability curves, trade-off scatter, win-rate bar
└── src/
    ├── cf_methods/                   # Counterfactual generation
    │   ├── base_cf_method.py         # Abstract base class
    │   ├── dice_method.py            # DiCE-X wrapper (gradient-based optimisation)
    │   ├── nice_method.py            # NICE wrapper (nearest-instance, deterministic)
    │   ├── gs_method.py              # GrowingSpheres wrapper (hypersphere expansion)
    │   ├── moc_method.py             # MOC: pymoo NSGA-II, four-objective problem
    │   ├── lore_method.py            # LORE wrapper (local decision tree + rule extraction)
    │   └── registry.py              # @register_method decorator and create_method()
    ├── data/
    │   ├── data_module.py            # load_dataset() entry point
    │   ├── registry.py               # @register_dataset decorator
    │   ├── datasets/                 # Dataset loaders (auto-download on first use)
    │   │   ├── adult.py              # registered as "adult-income"
    │   │   ├── compas.py             # registered as "compas-recidivism"
    │   │   ├── german.py             # registered as "german-credit"
    │   │   ├── lending.py            # registered as "lending-club"
    │   │   ├── heloc.py              # registered as "heloc"
    │   │   └── diabetes.py           # registered as "diabetes"
    │   └── preprocessing/
    │       ├── py_dataset.py         # PyTorch Dataset with MinMaxScaler + OHE preprocessing
    │       └── transform.py          # Transformer: encode / inverse-transform for CF methods
    ├── evaluation/
    │   ├── aggregator.py             # ResultsAggregator: by-sigma and by-dataset rollups
    │   ├── selectors.py              # Five selection strategies + SELECTOR_REGISTRY
    │   ├── plotting.py               # generate_all_figures(): stability curves, Pareto fronts
    │   └── stability_curve.py        # StabilityCurveBuilder + trapezoidal AUC
    ├── models/
    │   ├── base_model.py             # Abstract model base
    │   ├── pytorch_classifier.py     # Configurable MLP (BCELoss, Adam)
    │   └── trainer.py                # Training loop with checkpointing
    ├── orchestration/
    │   ├── prefect_flow.py           # @flow run_pipeline(): full 8-stage pipeline
    │   └── tasks.py                  # @task definitions (data, model, pool, eval, agg)
    ├── perturbations/
    │   ├── base_perturbation.py      # Abstract base + shared categorical resampling
    │   ├── gaussian.py               # Gaussian noise: N(0, σ²), clamped to [0,1]
    │   └── uniform.py                # Uniform noise: U(-ε, ε)
    ├── pool/
    │   ├── pool_builder.py           # CFPoolBuilder: R runs × N CFs, CSV persistence, reuse
    │   └── deduplicator.py           # Exact-duplicate removal via feature-value hashing
    ├── robustness/
    │   ├── geometric.py              # Geometric instability: L1/L2/L∞/Cosine/Mahalanobis
    │   ├── intervention.py           # Intervention instability: Jaccard (default) / DSC
    │   ├── matcher.py                # Greedy nearest-neighbour matching across pools
    │   └── score.py                  # CandidateObjectives, pareto_front(), normalize_objectives()
    └── utils/
        ├── constants.py              # DefaultPaths, distance type enums
        ├── config_loader.py          # YAML config loader with override merging
        ├── logger.py                 # ExperimentLogger (JSONL)
        └── seed.py                   # set_seed(): Python / NumPy / PyTorch / CUDA
```

## Installation

### Prerequisites

[![DiCE-X](https://img.shields.io/badge/DiCE--X-source-orange)](https://github.com/Dice-Extended/dice-x)
[![NICEx](https://img.shields.io/badge/NICEx-PyPI-blue)](https://pypi.org/project/NICEx/)
[![GrowingSpheres](https://img.shields.io/badge/GrowingSpheres-source-orange)](https://github.com/thibaultlaugel/growingspheres)
[![LORE-SA](https://img.shields.io/badge/LORE--SA-source-orange)](https://github.com/kdd-lab/LORE_sa)
[![pymoo](https://img.shields.io/badge/pymoo-%E2%89%A5%200.6.1-blue)](https://pymoo.org/)

### Setup

```bash
# Clone the repository
git clone https://github.com/creaturerigger/cf_benchmark.git
cd cf_benchmark

# Create venv and install core dependencies (includes pymoo for MOC)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Clone and install DiCE-X (custom DiCE fork)
cd ..
git clone https://github.com/Dice-Extended/dice-x.git
cd DiCE-X
pip install -e .
cd ../cf_benchmark

# Install NICE (no-deps to avoid pinned-version conflicts)
pip install NICEx --no-deps

# Install GrowingSpheres (from source)
cd ..
git clone https://github.com/thibaultlaugel/growingspheres.git
cd growingspheres
pip install -e .
cd ../cf_benchmark

# Install LORE-SA (from source)
cd ..
git clone https://github.com/kdd-lab/LORE_sa.git
cd LORE_sa
pip install -e .
cd ../cf_benchmark
```

> **Note:** Python ≥ 3.10 is required. The project was developed and tested on Python 3.12 with PyTorch 2.x and scikit-learn ≥ 1.7.

## Datasets

All six datasets target binary classification tasks in socially sensitive domains. Datasets are auto-downloaded and preprocessed on first use.

| Dataset | Registry Name | Target | Instances | Features | Source |
|---------|--------------|--------|-----------|----------|--------|
| **Adult Income** | `adult-income` | `income` (>50K) | 32,561 | 8 | UCI Repository |
| **COMPAS Recidivism** | `compas-recidivism` | `twoyearrecid` | 6,172 | 10 | ProPublica / OpenML |
| **German Credit** | `german-credit` | `credit_risk` | 1,000 | 20 | UCI Repository |
| **Lending Club** | `lending-club` | `loan_status` | 39,715 | 8 | Kaggle |
| **HELOC** | `heloc` | `RiskPerformance` | 10,459 | 23 | FICO Challenge |
| **Diabetes 130-US** | `diabetes` | `readmitted` (<30 days) | 768 | 8 | UCI Repository |

All datasets except Diabetes contain both continuous and categorical features, requiring the mixed-type perturbation model. The Diabetes dataset uses 8 continuous features (glucose, BMI, age, and related measurements) after preprocessing (Strack et al., 2014).

## Counterfactual Methods

| Method | Strategy | Package | Paper Config | Benchmark Status |
|--------|----------|---------|-------------|-----------------|
| **DiCE** | Optimisation-based | [DiCE-X](https://github.com/Dice-Extended/dice-x) | gradient, 500 iters, prox=0.5, div=1.0 | ✓ All 6 datasets |
| **NICE** | Instance-based | [NICEx](https://pypi.org/project/NICEx/) | sparsity, HEOM, justified_cf=true | ✓ All 6 datasets |
| **MOC** | Evolutionary (NSGA-II) | [pymoo](https://pymoo.org/) | pop=20, 175 gen, SBX+PM | ✓ All 6 datasets |
| **LORE** | Rule-based (DT surrogate) | [LORE-SA](https://github.com/kdd-lab/LORE_sa) | genetic neigh., 1000 instances | ✓ All 6 datasets |
| **GrowingSpheres** | Heuristic search | [GrowingSpheres](https://github.com/thibaultlaugel/growingspheres) | n_in_layer=2000, ball | ✗ Excluded (too slow at R=50) |

MOC is a custom re-implementation of [Dandl et al. (2020)](https://doi.org/10.1007/978-3-030-58112-1_31) using pymoo's NSGA-II, reproducing the four-objective problem (validity, proximity, sparsity, plausibility) with the R `ecr` package defaults.

Methods are registered via `@register_method` and instantiated through `create_method(cfg, model, dataframe, target_column, continuous_features)`. To add a new method, subclass `BaseCounterfactualGenerationMethod`, implement `generate()`, and decorate with `@register_method(name="yourmethod")`.

## Configuration

All parameters are externalised in YAML files under `configs/`.

**Model** (`configs/model/pytorch_classifier.yaml`):
```yaml
model:
  name: pytorch-classifier
  layers:
    - out_features: 20
      activation: relu
      batch_norm: false
      dropout: 0.0
  output_activation: sigmoid
training:
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  batch_size: 64
  test_size: 0.2
```

**CF Method — DiCE** (`configs/cf_method/dice.yaml`):
```yaml
generation:
  total_cfs: 5
  desired_class: opposite
  proximity_weight: 0.5
  diversity_weight: 1.0
  init_near_query_instance: true
search:
  algorithm: gradient
  max_iterations: 500
  stopping_threshold: 0.01
```

**CF Method — NICE** (`configs/cf_method/nice.yaml`):
```yaml
nice:
  optimization: sparsity   # sparsity | proximity | plausibility | none
  justified_cf: true
  distance_metric: HEOM
  num_normalization: minmax
```

**CF Method — MOC** (`configs/cf_method/moc.yaml`):
```yaml
moc:
  pop_size: 20
  n_gen: 175
  p_cross: 0.57
  p_mut: 0.79
  eta_cross: 15
  eta_mut: 20
  k: 1               # nearest neighbours for plausibility objective
```

**Experiment** (`configs/experiment/robustness_experiment.yaml`):
```yaml
pool:
  runs: 15           # overridden to 50 by run_parallel.py scenarios
  per_run: 5
perturbation:
  type: gaussian
  sigma: 0.05        # base σ; overridden per scenario
  M: 20              # number of perturbed copies per query per σ
robustness:
  geometric_metric: l2
  intervention_metric: jaccard_index
```

## Key Concepts

### Candidate Pool Construction

For each query instance, the `CFPoolBuilder` runs the CF generator R times, each requesting N candidates. Results are deduplicated by exact feature-value matching. Pools are persisted as CSV files (`<dataset>_original_cfs.csv`, `<dataset>_original_queries.csv`) and reused across σ levels. At R=50, N=5 (250 raw per query), effective pool sizes after deduplication vary substantially by method:

| Method | Unique after dedup | Dedup rate | Mean Pareto front size |
|--------|-------------------|------------|----------------------|
| DiCE | ~50 | 98.0% | ~26 |
| NICE | ~10 | 99.6% | 10 (deterministic) |
| MOC | ~2,319 | 7.2% | ~306 |
| LORE | ~415 | 83.4% | ~67 |

### Perturbation Model

The perturbation model is feature-type aware:

- **Continuous features**: additive Gaussian noise clamped to [0, 1]:
  $\tilde{x}_j^{(m)} = \mathrm{clamp}(x_j + \delta_j^{(m)},\, 0,\, 1)$, $\delta_j^{(m)} \sim \mathcal{N}(0, \sigma^2)$

- **Categorical features**: uniform random resampling with probability σ:
  $\tilde{x}_j^{(m)} = \mathrm{Uniform}(\mathcal{V}_j)$ with probability σ, else $x_j$

A single σ parameter jointly controls both continuous noise scale and categorical flip probability. Three levels are used in the paper: σ ∈ {0.03, 0.05, 0.07}.

### Robustness Metrics

For each candidate $x'_k$ in the original pool, matched to its nearest neighbour $\tilde{x}'^{(m)}_k$ in each perturbed pool:

- **Geometric instability** $f_2$: mean L2 displacement $\frac{1}{M}\sum_m \|x'_k - \tilde{x}'^{(m)}_k\|_2$
- **Intervention instability** $f_3$: mean Jaccard dissimilarity of feature-change sets $\frac{1}{M}\sum_m \bigl(1 - J(\mathcal{I}(x'_k, x),\, \mathcal{I}(\tilde{x}'^{(m)}_k, \tilde{x}^{(m)}))\bigr)$

Jaccard is used (not Dice–Sørensen) because $1 - J$ is a proper metric satisfying the triangle inequality, which is relied upon in the theoretical bounds.

### Selection Strategies

Five post-hoc strategies map the Pareto front (or full pool) to a single candidate:

| Strategy | Code name | Description |
|----------|-----------|-------------|
| **Min-Proximity** | `min_proximity` | Closest to the original query; ignores robustness (standard baseline) |
| **Equal-Weight Scalarisation** | `weighted_sum_equal` | Min-max normalise, then minimise $\frac{1}{3}f_1 + \frac{1}{3}f_2 + \frac{1}{3}f_3$ |
| **Proximity-Heavy Scalarisation** | `weighted_sum_prox_heavy` | Normalise, minimise $0.6f_1 + 0.2f_2 + 0.2f_3$ |
| **Pareto-Knee** | `pareto_knee` | Restrict to Pareto front; pick member closest to ideal point (0,0,0) in normalised space |
| **Pareto-Lex (robustness-first)** | `pareto_lex` | Restrict to Pareto front; lexicographic order $(f_2, f_3, f_1)$ |
| **Random** | `random` | Uniformly random Pareto-optimal solution (sanity baseline) |

### Stability Profiles and AUC

By evaluating at multiple σ levels, instability profiles (instability vs. σ) are constructed and summarised by the trapezoidal AUC:
$$\mathrm{AUC_{geo}} = \int_{\sigma_1}^{\sigma_S} \bar{f}_2(\sigma)\, d\sigma, \qquad \mathrm{AUC_{int}} = \int_{\sigma_1}^{\sigma_S} \bar{f}_3(\sigma)\, d\sigma$$

Lower AUC indicates greater overall robustness across the tested perturbation range.

## Experimental Results Summary

Results from the paper (n=10 queries per dataset, R=50, N=5, σ ∈ {0.03, 0.05, 0.07}, M=5):

**Computation time (hours, full pipeline per dataset):**

| Dataset | DiCE | NICE | MOC | LORE |
|---------|------|------|-----|------|
| Adult Income | 40.4 | 0.20 | 11.2 | 5.1 |
| COMPAS | 44.3 | 0.09 | 6.1 | 3.8 |
| Diabetes | 31.0 | 0.22 | 25.8 | 5.2 |
| German Credit | 60.4 | 0.43 | 11.3 | 6.9 |
| HELOC | 54.4 | 0.08 | 9.2 | 4.4 |
| Lending Club | 16.3 | 0.23 | 12.1 | 6.9 |

**Mean objectives across all datasets (σ = 0.05):**

| Method | Proximity (f₁) ↓ | Geo. Inst. (f₂) ↓ | Int. Inst. (f₃) ↓ |
|--------|-----------------|-------------------|-------------------|
| DiCE | 1.746 | 0.944 | 0.366 |
| NICE | **0.960** | 0.431 | 0.256 |
| MOC | 2.545 | **0.672** | **0.177** |
| LORE | 1.723 | 0.742 | 0.280 |

**Selection strategy comparison (aggregated, σ = 0.05):**

| Strategy | Proximity ↓ | Geo. Inst. ↓ | Int. Inst. ↓ |
|----------|------------|-------------|-------------|
| Min-Proximity | **1.210** | 0.612 | 0.299 |
| Equal-Weight Scalarisation | 1.391 | 0.430 | **0.194** |
| Prox-Heavy Scalarisation | 1.250 | 0.501 | 0.233 |
| Pareto-Knee | 1.370 | 0.447 | 0.197 |
| Pareto-Lex (robustness-first) | 1.615 | **0.379** | 0.222 |
| Random (baseline) | 1.715 | 0.752 | 0.288 |

## Design Patterns

- **Registry Pattern** — CF methods and datasets are registered via `@register_method` / `@register_dataset` decorators. Registered methods: `dice`, `nice`, `gs`, `moc`, `lore`. Registered datasets: `adult-income`, `compas-recidivism`, `german-credit`, `lending-club`, `heloc`, `diabetes`.
- **Config-Driven** — All experiment, model, and method parameters are externalised in YAML with runtime override support (`overrides` dict passed to `run_pipeline`).
- **Factory Pattern** — `create_method()` and `load_dataset()` instantiate components by name from the registry.
- **Template Method** — `BasePerturbation` defines the perturbation contract (shared categorical resampling) while subclasses implement continuous noise strategies.
- **PyTorch Integration** — `PYTDataset` is a native `torch.utils.data.Dataset` with automatic MinMaxScaler + OneHotEncoder preprocessing; a `Transformer` object handles round-trip encode/decode for CF interpretability.
- **Workflow Orchestration** — Prefect `@flow` / `@task` for reproducible pipeline execution with per-stage timing and structured logging to JSONL.
- **Pool Reuse** — `run_pipeline` can reuse existing original pools from disk (`min_pool_size` parameter), skipping expensive regeneration when re-running with different σ values.

## Running Experiments

### Quick start (single combo)

```python
from src.orchestration.prefect_flow import run_pipeline

result = run_pipeline(
    dataset_name="adult",
    model_name="pytorch_classifier",
    cf_method_name="dice",
    experiment_name="robustness_experiment",
    seed=42,
    n_queries=10,
    sigmas=[0.03, 0.05, 0.07],
    overrides={"pool": {"runs": 50, "per_run": 5}, "perturbation": {"M": 5}},
)
```

### Parallel execution (multi-core)

```bash
# Dry-run — list what would execute without running
python -m scripts.run_parallel --dry-run

# Run all combos with the paper's budget scenario (n=10, M=3, R=50, N=5)
python -m scripts.run_parallel --scenario budget --workers 6

# Run a specific combo (useful for reruns)
python -m scripts.run_parallel --only adult,dice

# Run with a timeout per combo
python -m scripts.run_parallel --scenario lean --timeout 43200

# Resume after interruption (skips combos already marked OK)
python -m scripts.run_parallel --scenario lean --resume

# Exclude slow methods
python -m scripts.run_parallel --scenario lean --exclude heloc,gs german,gs
```

**Pre-defined scenarios:**

| Scenario | n_queries | σ levels | M | pool_runs × per_run |
|----------|-----------|----------|---|---------------------|
| `budget` | 10 | 3 | 3 | 50 × 5 = 250 (paper default) |
| `lean` | 10 | 3 | 5 | 50 × 5 = 250 |
| `moderate` | 15 | 3 | 5 | 50 × 5 = 250 |
| `practical` | 20 | 3 | 5 | 50 × 5 = 250 |
| `full` | 50 | 5 | 20 | 50 × 5 = 250 |

### Post-hoc selection analysis

```bash
# Analyse all methods and datasets found under results/
python -m scripts.selection_analysis

# Restrict to specific methods / datasets
python -m scripts.selection_analysis --methods dice nice moc lore --datasets adult compas
```

Outputs under `results/selection_analysis/`:
- `tables/`: aggregated, per-method, and per-dataset AUC and win-rate CSVs
- `figures/`: stability curves, proximity vs. instability scatter, win-rate bar charts

### Timing benchmark

```bash
python -m scripts.timing_benchmark
```

Runs mini experiments (n=2, R=3, N=2) for all combos and extrapolates to full-scale estimates. Results saved to `results/tables/timing_benchmark.csv`.

## Running Tests

```bash
pytest
```

## Contributing

Contributions are welcome. Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{bakir2026robust,
  author    = {Bakir, Volkan and Goktas, Polat and Akyüz, Süreyya},
  title     = {Robust Counterfactual Explanations via Stochastic Multi-Objective Optimisation},
  journal   = {arXiv preprint},
  year      = {2026},
  note      = {Preprint, under review}
}
```

## License

This project is part of a thesis research effort. See [LICENSE.md](LICENSE.md) for details.