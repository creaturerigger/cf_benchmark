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
    ├── CF Method Config ─→ DiCE Method (Any method
    │                                    can replace) ─→ CFPoolBuilder ─→ Deduplicator ─→ CSV Pool
    │                                                         │
    └── Experiment Config                                Perturbations (Gaussian / Uniform)
                                                              │
                                                         Robustness Evaluation
                                                              ├── Proximity
                                                              ├── GeometricInstability
                                                              └── InterventionInstability
                                                              │
                                                         Pareto Selection & Instability Profiles
```

## Project Structure

```
cf_benchmark/
├── main.py                          # Entry point
├── pyproject.toml                   # Project metadata and dependencies
├── configs/                         # YAML configuration files
│   ├── cf_method/
│   │   └── dice.yaml                # DiCE generation parameters
│   ├── dataset/
│   │   ├── adult.yaml               # Adult Income dataset
│   │   ├── compas.yaml              # COMPAS Recidivism dataset
│   │   ├── german.yaml              # German Credit dataset
│   │   └── lending.yaml             # Lending Club dataset
│   ├── experiment/
│   │   └── robustness_experiment.yaml
│   └── model/
│       └── pytorch_classifier.yaml  # Neural network architecture
├── scripts/
│   └── run_experiments.py           # Experiment execution script
├── notebooks/
│   └── analysis.ipynb               # Results analysis notebook
├── results/
│   ├── figures/                     # Generated plots
│   ├── models/                      # Trained model checkpoints
│   ├── pools/                       # Counterfactual pool CSVs
│   ├── raw/                         # Raw experiment outputs
│   └── tables/                      # Summary tables
└── src/
    ├── cf_methods/                   # Counterfactual generation
    │   ├── base_cf_method.py         # Abstract base class
    │   ├── dice_method.py            # DiCE-X wrapper
    │   ├── method_factory.py         # Factory pattern
    │   └── registry.py               # @register_method decorator
    ├── data/
    │   ├── data_module.py            # Data loading entry point
    │   ├── registry.py               # @register_dataset decorator
    │   ├── datasets/                 # Dataset implementations (auto-download)
    │   │   ├── adult.py
    │   │   ├── compas.py
    │   │   ├── german.py
    │   │   └── lending.py
    │   └── preprocessing/
    │       └── py_dataset.py         # PyTorch Dataset with auto-preprocessing
    ├── evaluation/
    │   ├── experiment.py             # Experiment orchestration
    │   ├── aggregator.py             # Result aggregation
    │   └── stability_curve.py        # Stability profile analysis
    ├── models/
    │   ├── base_model.py             # Abstract model base
    │   ├── pytorch_classifier.py     # Configurable neural network
    │   └── trainer.py                # Training loop with checkpointing
    ├── orchestration/
    │   ├── prefect_flow.py           # Workflow orchestration
    │   └── tasks.py                  # Prefect task definitions
    ├── perturbations/
    │   ├── base_perturbation.py      # Perturbation base class
    │   ├── gaussian.py               # Gaussian noise perturbation
    │   └── uniform.py                # Uniform noise perturbation
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
```

## Datasets

| Dataset | Target | Classes | Source |
|---------|--------|---------|--------|
| **Adult Income** | `income` | ≤50K / >50K | UCI Repository |
| **COMPAS Recidivism** | `twoyearrecid` | 0 / 1 | OpenML |
| **German Credit** | `credit_risk` | 0 / 1 | UCI Repository |
| **Lending Club** | `loan_status` | 0 / 1 | Kaggle |

Datasets are auto-downloaded and preprocessed on first use. Each dataset loader handles feature engineering, type conversion, and domain-specific cleaning.

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

**CF Method** (`configs/cf_method/dice.yaml`):
```yaml
total_CFs: 5
desired_class: opposite
backend: PYT
algorithm: gradient
iterations: 500
```

**Experiment** (`configs/experiment/robustness_experiment.yaml`):
```yaml
pool:
  runs: 15
  per_run: 5
perturbation:
  type: gaussian
  sigma: 0.05
  M: 20
```

## Key Concepts

### Counterfactual Pool

For each input instance, multiple counterfactual generation runs produce a **candidate pool**. The `CFPoolBuilder` orchestrates this process, running the CF method multiple times and collecting results into deduplicated CSV files with UUID-based query–CF linking.

### Robustness Evaluation

Each candidate counterfactual is evaluated under a distribution of **perturbed inputs** in the neighbourhood of the original instance:

- **Geometric Instability** — Measures the distance between the original CF and the CF derived from a perturbed input using L1, L2, L∞, Cosine, or Mahalanobis distance. Lower values indicate greater geometric instability.

- **Intervention Instability** — Measures the consistency of feature-level modifications (which features are changed) using Jaccard Index or Dice–Sørensen Coefficient. Higher overlap indicates greater intervention instability.

### Pareto Selection

Candidates are evaluated on a multi-objective vector of (proximity, geometric instability, intervention instability). **Pareto-efficient** solutions — those not dominated on all objectives — are selected, avoiding arbitrary weighting of conflicting objectives.

### Stability Profiles

Robustness is examined as a function of perturbation magnitude. **Stability profiles** plot how explanation quality degrades as input uncertainty increases, summarised via area-under-curve (AUC) scores over the perturbation spectrum.

## Design Patterns

- **Registry Pattern** — CF methods and datasets are dynamically registered via `@register_method` and `@register_dataset` decorators, enabling plug-and-play extensibility
- **Config-Driven** — All experiment, model, and method parameters are externalised in YAML files
- **Factory Pattern** — `create_method()` and `load_dataset()` instantiate components by name from the registry
- **PyTorch Integration** — Native `torch.utils.data.Dataset` subclass with automatic StandardScaler/OneHotEncoder preprocessing

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