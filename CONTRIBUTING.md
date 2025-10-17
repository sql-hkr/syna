## Contributing to Syna

Thanks for your interest in contributing to Syna — a lightweight ML framework inspired by DeZero. We welcome bug reports, feature requests, documentation improvements, examples, and tests. This file explains how to get the project running locally, the preferred workflows, and the minimal requirements for a contribution to be accepted.

### Quick start (local development)

1. Clone the repository and create a branch for your change:

    ```bash
    git clone https://github.com/sql-hkr/syna.git
    git checkout -b feat/your-short-description
    ```

2. Use Python 3.11 or later and create a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Upgrade packaging tools and install the package editable with development dependencies. The project uses `uv_build` as its build backend; we recommend using the `uv` tool when available. A pip fallback is shown below.

    ```bash
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install uv
    # Prefer uv to install the project and dev dependencies
    uv sync
    ```

    If you prefer pip or `uv` is unavailable, run:

    ```bash
    python -m pip install -e .[dev]
    # or, if your pip does not recognize the `[dev]` extra
    python -m pip install -e .
    python -m pip install ruff shibuya sphinx torch torchvision
    ```

4. Run the test suite to ensure everything is green. This project uses Python's builtin `unittest` framework:

    ```bash
    uv run -m unittest discover -s tests
    ```

    To run a specific test module or test case, use for example:

    ```bash
    uv run -m unittest tests.test_functions
    ```

### Tests

- Tests live in the `tests/` folder. Add unit tests for any bug fixes or new features.
- Use Python's built-in `unittest` framework for new tests. Keep tests small, fast and deterministic when possible.
- Aim for coverage of edge cases and error paths for logic changes.

### Linting & formatting

- We use `ruff` for linting and auto-fixes. Run:

    ```bash
    ruff check .
    ruff format .
    ```

- When fixing/implementing code, make sure lint errors are resolved or justified in your PR.

### Documentation

- Documentation sources are in `docs/`. HTML builds can be generated with Sphinx:

    ```bash
    cd docs
    make html
    ```

- If you add public APIs or change behavior, update the appropriate docs and examples.

### Examples

- Example scripts are in `examples/`. Use `uv` to run examples so the environment and dependencies are managed consistently:

    ```
    uv run examples/dqn.py
    ```

### Pull Request process

1. Open an issue first for non-trivial changes (design or API changes) so maintainers can provide feedback before you invest time.
2. Work on a topic branch (not `main`). Use a descriptive branch name: `fix/short-desc`, `feat/short-desc`, or `doc/short-desc`.
3. Commit messages should be short and descriptive. If your changes are a bug fix or feature, reference the issue number if one exists.
4. Include or update tests that cover your changes.
5. Run the test suite, linter, and build the docs locally before opening the PR.
6. In the PR description include:
   - What the change does and why
   - Any notable design decisions or trade-offs
   - How to run tests/examples to verify the change

### PR checklist

- [ ] Branched from current `main`
- [ ] Tests added/updated
- [ ] Linting passes (`ruff`) and formatting applied
- [ ] Documentation updated (if applicable)
- [ ] Clear PR description and linked issue (if any)

### Code of conduct

Be respectful and professional. Follow common community standards when discussing issues and reviewing contributions.

### Reporting security issues

If you discover a security vulnerability, please contact the maintainers directly (see `pyproject.toml` author email) instead of creating a public issue.

### License

By contributing you agree that your contributions will be licensed under the project's MIT License.

### Need help?

Open an issue describing what you'd like to do, or reach out via the author email in `pyproject.toml`.

Thank you for helping improve Syna!
