# CF-Benchmark: Perturbation-Aware Counterfactual Robustness Framework

A perturbation-aware framework that formulates counterfactual explanation selection as a **stochastic multi-objective optimisation problem**, evaluating explanations under input uncertainty through geometric and intervention instability metrics.

## Overview

Counterfactual explanations identify minimal changes to an input that flip a model's prediction. However, they become unreliable when data distributions shift or models are retrained. Most generation techniques prioritise validity and proximity while neglecting robustness under uncertainty.

This framework addresses this gap by:

1. **Building candidate counterfactual pools** from multiple generation runs per input instance
2. **Perturbing input instances** in the neighbourhood of the original and evaluating each candidate explanation under these perturbations
3. **Defining a multi-objective vector** for each candidate comprising:
   - **(i) Proximity** to the original instance
   - **(ii) Geometric instability** under perturbations (L1, L2, L∞, Cosine, Mahalanobis distances)
   - **(iii) Intervention instability** measuring consistency of feature-level modifications (Jaccard Index, Dice–Sørensen Coefficient)
4. **Selecting Pareto-efficient solutions** that are not dominated across conflicting objectives, avoiding arbitrary scalarisation
5. **Generating instability profiles** that illustrate robustness as a function of perturbation magnitude, summarised via AUC scores

## Architecture

```
YAML Configs
    │
    ├── Dataset Configs ─→ Data Loaders ─→ PYTDataset (preprocessing + train/test split)
    │                                            │
    ├── Model Config ────→ PYTModel ◄────── Trainer (BCELoss, Adam)
    │                          │
    ├── CF Method Config ─→ CF Method Registry ───→ CFPoolBuilder ─→ Deduplicator ─→ CSV Pool
    │                          ├── DiCE (OPT)       │
    │                          ├── NICE (IB)        │
    │                          ├── GS (GEO)         │
    │                          ├── MOC (EVO)        │
    │                          └── LORE (RULE)      │
    │                                               │
    └── Experiment Config                      Perturbations (Gaussian / Uniform)
                                                    ├── Continuous: N(0, σ²) or U(-ε, ε)
                                                    └── Categorical: random one-hot re-sampling
                                                    │
                                               Robustness Evaluation
                                                    ├── Proximity
                                                    ├── GeometricInstability
                                                    └── InterventionInstability
                                                    │
                                               Selection Strategies
                                                    ├── Min-Proximity
                                                    ├── Min-Geo
                                                    ├── Weighted-Sum (equal / prox-heavy)
                                                    ├── Pareto-Knee
                                                    ├── Pareto-Lex
                                                    └── Random
                                                    │
                                               Stability Profiles & Analysis
```

## Project Structure

```
cf_benchmark/
├── main.py                          # Entry point
├── pyproject.toml                   # Project metadata and dependencies
├── configs/                         # YAML configuration files
│   ├── cf_method/
│   │   ├── dice.yaml                # DiCE generation parameters
│   │   ├── nice.yaml                # NICE generation parameters
│   │   ├── gs.yaml                  # GrowingSpheres parameters
│   │   ├── moc.yaml                 # MOC parameters
│   │   └── lore.yaml                # LORE parameters
│   ├── dataset/
│   │   ├── adult.yaml               # Adult Income dataset
│   │   ├── compas.yaml              # COMPAS Recidivism dataset
│   │   ├── german.yaml              # German Credit dataset
│   │   ├── lending.yaml             # Lending Club dataset
│   │   ├── heloc.yaml               # HELOC dataset
│   │   └── credit_default.yaml      # Credit Default dataset
│   ├── experiment/
│   │   └── robustness_experiment.yaml
│   └── model/
│       └── pytorch_classifier.yaml  # Neural network architecture
├── scripts/
│   ├── run_parallel.py              # Parallel multi-core experiment runner
│   ├── timing_benchmark.py          # Timing estimation for all combos
│   └── selection_analysis.py        # Post-hoc selection strategy analysis
├── notebooks/
│   └── analysis.ipynb               # Results analysis notebook
├── results/                         # Method-scoped outputs
│   ├── models/                      # Trained model checkpoints (shared)
│   ├── <method>/                    # Per-method results (dice, nice, gs, moc, lore)
│   │   ├── figures/                 # Generated plots
│   │   ├── logs/                    # Experiment logs
│   │   ├── pools/                   # Counterfactual pool CSVs
│   │   ├── raw/                     # Raw experiment outputs
│   │   └── tables/                  # Summary tables
│   └── selection_analysis/          # Cross-method selection comparison
│       ├── tables/                  # Aggregated, per-method, per-dataset CSVs
│       └── figures/                 # Stability curves, trade-off scatter, win-rate bar
└── src/
    ├── cf_methods/                   # Counterfactual generation
    │   ├── base_cf_method.py         # Abstract base class
    │   ├── dice_method.py            # DiCE-X wrapper (optimisation-based)
    │   ├── nice_method.py            # NICE wrapper (instance-based)
    │   ├── gs_method.py              # GrowingSpheres wrapper (geometric)
    │   ├── moc_method.py             # MOC wrapper (evolutionary)
    │   ├── lore_method.py            # LORE wrapper (rule-based)
    │   ├── method_factory.py         # Factory pattern
    │   └── registry.py               # @register_method decorator
    ├── data/
    │   ├── data_module.py            # Data loading entry point
    │   ├── registry.py               # @register_dataset decorator
    │   ├── datasets/                 # Dataset implementations (auto-download)
    │   │   ├── adult.py
    │   │   ├── compas.py
    │   │   ├── german.py
    │   │   ├── lending.py
    │   │   ├── heloc.py
    │   │   └── credit_default.py
    │   └── preprocessing/
    │       ├── py_dataset.py         # PyTorch Dataset with auto-preprocessing
    │       └── transform.py          # Transformer (MinMaxScaler + OHE) for CF methods
    ├── evaluation/
    │   ├── experiment.py             # Experiment orchestration
    │   ├── aggregator.py             # Result aggregation
    │   ├── selectors.py              # Selection strategies (Pareto-Knee, Lex, WS, etc.)
    │   ├── plotting.py               # Figures (stability curves, Pareto fronts, distributions)
    │   └── stability_curve.py        # Stability profile analysis
    ├── models/
    │   ├── base_model.py             # Abstract model base
    │   ├── pytorch_classifier.py     # Configurable neural network
    │   └── trainer.py                # Training loop with checkpointing
    ├── orchestration/
    │   ├── prefect_flow.py           # Workflow orchestration (Prefect @flow)
    │   └── tasks.py                  # Prefect @task definitions
    ├── perturbations/
    │   ├── base_perturbation.py      # Perturbation base class + categorical resampling
    │   ├── gaussian.py               # Gaussian noise (continuous + categorical)
    │   └── uniform.py                # Uniform noise (continuous + categorical)
    ├── pool/
    │   ├── pool_builder.py           # CF pool generation & CSV persistence
    │   └── deduplicator.py           # Duplicate CF removal
    ├── robustness/
    │   ├── geometric.py              # Geometric instability (L1, L2, L∞, Cosine, Mahalanobis)
    │   ├── intervention.py           # Intervention instability (Jaccard, Dice–Sørensen)
    │   ├── matcher.py                # CF matching logic
    │   └── score.py                  # Composite robustness scoring
    └── utils/
        ├── constants.py              # Paths, distance type enums
        ├── config_loader.py          # YAML configuration loader
        ├── logger.py                 # Logging setup
        └── seed.py                   # Reproducibility seed management
```

## Installation

### Prerequisites

- Python ≥ 3.10.13
- [DiCE-X](https://github.com/Dice-Extended/dice-x) (custom fork of DiCE)
- [NICEx](https://pypi.org/project/NICEx/) (Nearest Instance Counterfactual Explanations)

### Setup

```bash
# Clone the repository
git clone https://github.com/creaturerigger/cf_benchmark.git
cd cf_benchmark

# Install dependencies
pip install -e .

# Clone and install DiCE-X (custom DiCE fork)
cd ..
git clone https://github.com/Dice-Extended/dice-x.git
cd DiCE-X
pip install -e .
cd ../cf_benchmark

# Install NICE (no-deps to avoid pinned-version conflicts)
pip install NICEx --no-deps
```

## Datasets

| Dataset | Target | Classes | Source |
|---------|--------|---------|--------|
| **Adult Income** | `income` | ≤50K / >50K | UCI Repository |
| **COMPAS Recidivism** | `twoyearrecid` | 0 / 1 | OpenML |
| **German Credit** | `credit_risk` | 0 / 1 | UCI Repository |
| **Lending Club** | `loan_status` | 0 / 1 | Kaggle |
| **HELOC** | `RiskPerformance` | Good / Bad | FICO |
| **Credit Default** | `default` | 0 / 1 | UCI Repository |

Datasets are auto-downloaded and preprocessed on first use. Each dataset loader handles feature engineering, type conversion, and domain-specific cleaning.

## Counterfactual Methods

| Method | Strategy Family | Package | Key Property |
|--------|----------------|---------|-------------|
| **DiCE** | Optimisation-based (OPT) | [DiCE-X](https://github.com/Dice-Extended/dice-x) | Diversity-focused; gradient-based search over differentiable loss |
| **NICE** | Instance-based (IB) | [NICEx](https://pypi.org/project/NICEx/) | Nearest-neighbour with sparsity/proximity/plausibility optimisation; deterministic |
| **GS** | Geometric (GEO) | [GrowingSpheres](https://github.com/thibaultlaugel/growingspheres) | Iterative hypersphere expansion to find decision boundary crossing |
| **MOC** | Evolutionary (EVO) | [MOC](https://github.com/indyfree/moc) | Multi-objective counterfactuals via NSGA-II genetic algorithm |
| **LORE** | Rule-based (RULE) | [LORE-SA](https://github.com/simonepri/lore-sa) | Neighbourhood generation + decision tree + rule extraction |

Methods are registered via the `@register_method` decorator and instantiated at runtime through `create_method(cfg, model, dataframe, target_column, continuous_features)`. New methods only need to subclass `BaseCounterfactualGenerationMethod` and implement `generate()`.

## Configuration

All parameters are defined in YAML configuration files under `configs/`:

**Model** (`configs/model/pytorch_classifier.yaml`):
```yaml
layers:
  - units: 20
    activation: relu
output_activation: sigmoid
epochs: 50
learning_rate: 0.001
batch_size: 64
```

**CF Method — DiCE** (`configs/cf_method/dice.yaml`):
```yaml
total_CFs: 5
desired_class: opposite
backend: PYT
algorithm: gradient
iterations: 500
```

**CF Method — NICE** (`configs/cf_method/nice.yaml`):
```yaml
method:
  name: nice
nice:
  optimization: sparsity   # sparsity | proximity | plausibility | none
  justified_cf: true
  distance_metric: HEOM
  num_normalization: minmax
```

**Experiment** (`configs/experiment/robustness_experiment.yaml`):
```yaml
pool:
  runs: 15
  per_run: 5
perturbation:
  type: gaussian       # gaussian | uniform
  sigma: 0.05          # noise magnitude (controls both continuous and categorical perturbation)
  M: 20                # number of perturbed copies per query
```

## Key Concepts

### Counterfactual Pool

For each input instance, multiple counterfactual generation runs produce a **candidate pool**. The `CFPoolBuilder` orchestrates this process, running the CF method multiple times and collecting results into deduplicated CSV files with UUID-based query–CF linking.

### Robustness Evaluation

Each candidate counterfactual is evaluated under a distribution of **perturbed inputs** in the neighbourhood of the original instance:

- **Perturbation model** — Continuous features receive additive noise (Gaussian $\mathcal{N}(0,\sigma^2)$ or Uniform $U(-\varepsilon,\varepsilon)$) on [0, 1]-scaled values. Categorical features (one-hot encoded) are randomly re-sampled with probability $\sigma$, activating a uniformly random category within each group. A single $\sigma$ knob controls both perturbation strengths.

- **Geometric Instability** — Measures the distance between the original CF and the CF derived from a perturbed input using L1, L2, L∞, Cosine, or Mahalanobis distance. Lower values indicate greater geometric instability.

- **Intervention Instability** — Measures the consistency of feature-level modifications (which features are changed) using Jaccard Index or Dice–Sørensen Coefficient. Higher overlap indicates greater intervention instability.

### Selection Strategies

Given a multi-objective vector (proximity, geometric instability, intervention instability) for each candidate CF, the framework applies multiple selection strategies and compares their outcomes:

| Strategy | Description |
|----------|-------------|
| **Min-Proximity** | Pick the candidate closest to the original input |
| **Min-Geo** | Pick the candidate with lowest geometric instability |
| **Weighted-Sum (equal)** | Normalise objectives, equal weights |
| **Weighted-Sum (prox-heavy)** | Normalise objectives, 60% proximity / 20% geo / 20% intervention |
| **Pareto-Knee** | Extract Pareto front, pick member closest to the origin (L2 on normalised objectives) |
| **Pareto-Lex** | Extract Pareto front, lexicographic ordering (geo → intervention → proximity) |
| **Random** | Uniformly random baseline |

### Pareto Selection

Candidates are evaluated on a multi-objective vector of (proximity, geometric instability, intervention instability). **Pareto-efficient** solutions — those not dominated on all objectives — are selected, avoiding arbitrary weighting of conflicting objectives.

### Stability Profiles

Robustness is examined as a function of perturbation magnitude. **Stability profiles** plot how explanation quality degrades as input uncertainty increases, summarised via area-under-curve (AUC) scores over the perturbation spectrum.

## Design Patterns

- **Registry Pattern** — CF methods and datasets are dynamically registered via `@register_method` and `@register_dataset` decorators, enabling plug-and-play extensibility. Currently registered methods: `dice`, `nice`, `gs`, `moc`, `lore`. Currently registered datasets: `adult-income`, `compas-recidivism`, `german-credit`, `lending-club`, `heloc`, `credit-default`.
- **Config-Driven** — All experiment, model, and method parameters are externalised in YAML files
- **Factory Pattern** — `create_method()` and `load_dataset()` instantiate components by name from the registry
- **Template Method** — `BasePerturbation` defines the perturbation contract (including shared categorical resampling logic) while subclasses implement continuous noise strategies
- **PyTorch Integration** — Native `torch.utils.data.Dataset` subclass with automatic StandardScaler/OneHotEncoder preprocessing
- **Workflow Orchestration** — Prefect `@flow`/`@task` decorators for reproducible pipeline execution with logging and timing

## Running Experiments

### Single combo

```bash
python main.py --dataset adult --method dice --n-queries 10 --sigmas 0.03 0.05 0.07
```

### Parallel execution (multi-core)

```bash
# Dry run — see what will execute
python -m scripts.run_parallel --scenario lean --dry-run

# Run all combos in parallel (uses ncpus - 1 workers)
python -m scripts.run_parallel --scenario lean --timeout 7200

# Exclude expensive combos
python -m scripts.run_parallel --scenario lean --exclude heloc,gs --workers 12

# Retry specific combos
python -m scripts.run_parallel --only adult,dice compas,moc
```

Pre-defined scenarios: `lean` (q=10, σ=3, M=5), `moderate` (q=15, σ=3, M=5), `practical` (q=20, σ=3, M=5), `full` (q=50, σ=5, M=20).

### Post-hoc selection analysis

```bash
python -m scripts.selection_analysis
```

Generates Tables 1–3 (aggregated, per-method, per-dataset) and Figures 1–4 (stability curves, trade-off scatter, win-rate bar, proximity vs robustness) under `results/selection_analysis/`.

## Running Tests

```bash
pytest
```

## Contributing

Contributions are welcome. Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

## Citation

If you use this framework in your research, please cite:

```bibtex
@InProceedings{cf_benchmark_framework_bakir_2026,
  author    = {Bakir, Volkan and Goktas, Polat and Akyuz, Sureyya},
  title     = {Robust Counterfactual Explanations via Stochastic Multi-Objective Optimisation},
  year      = {2026}
}
```

## License

This project is part of a thesis research effort. See repository for [licensing](LICENSE.md) details.