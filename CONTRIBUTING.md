# Contributing to DieselWolf

Thanks for taking the time to contribute! Please follow these guidelines to keep the project tidy and consistent.

## Development environment

1. Install Python 3.8 or later.
2. Install dependencies from `pyproject.toml`:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Style

We use **Black** and **Ruff** to maintain code quality. The hooks run automatically via `pre-commit`.
Run `pre-commit run --all-files` before pushing to ensure your code is formatted and linted.

## Pull requests

- Create feature branches from `main`.
- Ensure tests pass (`pytest` will be added in future).
- Reference roadmap items in your PR description.

Happy hacking!
