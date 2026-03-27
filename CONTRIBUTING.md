# Contributing to CF-Benchmark

Thank you for your interest in contributing to this project. This document outlines the guidelines and workflow for contributing.

## Getting Started

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/cf_benchmark.git
   cd cf_benchmark
   ```

2. Install the project in development mode:
   ```bash
   pip install -e .
   ```

3. Install the [DiCE-X](https://github.com/Dice-Extended/dice-x) dependency:
   ```bash
   cd ..
   git clone https://github.com/Dice-Extended/dice-x
   cd DiCE-X
   pip install -e .
   cd ../cf_benchmark
   ```

4. Verify everything works:
   ```bash
   pytest
   ```

## Development Workflow

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, keeping commits focused and well-described.

3. Run the tests before submitting:
   ```bash
   pytest
   ```

4. Push your branch and open a pull request against `main`.

## Project Conventions

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use type hints where practical.
- Keep functions and methods focused on a single responsibility.

### Configuration

- All tuneable parameters belong in YAML configuration files under `configs/`.
- Avoid hardcoding values that may change between experiments.

### Registry Pattern

- New **datasets** should be registered using the `@register_dataset` decorator in `src/data/datasets/`.
- New **CF methods** should be registered using the `@register_method` decorator in `src/cf_methods/`.

### Adding a New Dataset

1. Create a new file in `src/data/datasets/` (e.g., `my_dataset.py`).
2. Implement a loader function that downloads, preprocesses, and returns a `pd.DataFrame`.
3. Decorate the function with `@register_dataset("my_dataset")`.
4. Add a corresponding YAML config in `configs/dataset/`.

### Adding a New CF Method

1. Create a new file in `src/cf_methods/` (e.g., `my_method.py`).
2. Subclass `BaseCounterfactualGenerationMethod` and implement the `generate()` method.
3. Register the method with `@register_method("my_method")`.
4. Add a corresponding YAML config in `configs/cf_method/`.

### Adding a New Perturbation Type

1. Create a new file in `src/perturbations/` (e.g., `my_perturbation.py`).
2. Subclass `BasePerturbation` and implement the required interface.

## Testing

- Place tests alongside Source modules in `tests/` subdirectories (e.g., `src/cf_methods/tests/`).
- Name test files with the `test_` prefix.
- Run the full suite with:
  ```bash
  pytest
  ```

## Commit Messages

Write clear, concise commit messages:

```
<type>: <short summary>

<optional body with more detail>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

Examples:
- `feat: add COMPAS dataset loader`
- `fix: handle missing values in German Credit preprocessing`
- `docs: update README with stability profile section`

## Pull Requests

- Keep pull requests focused on a single concern.
- Reference any related issues in the PR description.
- Ensure all tests pass before requesting review.
- Provide a brief description of what the PR changes and why.

## Reporting Issues

When opening an issue, include:

- A clear description of the problem or feature request.
- Steps to reproduce (for bugs).
- Expected vs. actual behaviour.
- Python version and OS.

## Questions

If you have questions about the codebase or the research behind it, feel free to open a discussion or issue on the repository.
