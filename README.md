# Hierarchical ultrafast molecular force fields

## Installation

Install this project with [poetry](https://python-poetry.org):

```
git clone git@gitlab.kit.edu:kit/ag_wenzel/humf.git
cd humf
poetry install
```

Activate the environment:

```
poetry shell
```

If you intend to contribute, install [pre-commit](https://pre-commit.com) hooks:

```
pre-commit install
```

## Testing

Run tests with [pytest](https://docs.pytest.org):

```
pytest
```

## Contributing

- pre-commit is configured to use [ruff](https://github.com/astral-sh/ruff) to lint and format Python files, and to use [isort](https://pycqa.github.io/isort/) to sort imports in Python files.
