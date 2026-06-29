# Contributing to LiteResearcher

Thanks for your interest in improving LiteResearcher. This guide covers how to
report issues, propose changes, and submit pull requests.

## Repository layout

| Directory | Purpose |
|-----------|---------|
| `inference/` | Inference & evaluation harness |
| `training/` | RL training stack (GRPO + difficulty-aware curriculum), built on [verl](https://github.com/volcengine/verl) |
| `datagen/` | Training-data and corpus synthesis pipeline |
| `environment/` | Local search/browse environment (Milvus + BGE-M3, PostgreSQL) |
| `docs/` | Project page and trajectory case viewer |

Each component has its own `README.md` and dependency file; start there for
component-specific setup.

## Reporting issues

- Search [existing issues](https://github.com/simplexai-labs/LiteResearcher/issues)
  before opening a new one.
- Include the component (`inference` / `training` / `datagen` / `environment`),
  your environment (OS, Python, GPU/driver, key package versions), the exact
  command you ran, and the full error output.
- For reproducibility problems, note the model checkpoint and dataset revision
  you used.

## Development setup

```bash
git clone https://github.com/simplexai-labs/LiteResearcher.git
cd LiteResearcher
# Install the component you are working on, e.g.:
cd inference && pip install -r requirements.txt
```

## Pull requests

1. Fork the repo and create a topic branch from `main`
   (e.g. `fix/eval-timeout`, `feat/corpus-dedup`).
2. Keep each PR focused on a single concern. Unrelated cleanups belong in
   separate PRs.
3. Match the existing code style of the component you touch. The `training/`
   stack inherits verl's `pre-commit` config — run `pre-commit run --all-files`
   there before pushing.
4. Update the relevant `README.md` when you change behavior, flags, or paths.
5. Write a clear PR description: what changed, why, and how you verified it.

## Commit messages

Use concise, conventional-style prefixes where they fit
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`) and explain the *why* in the
body when it is not obvious.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
