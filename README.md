# Hierarchical ultrafast molecular force fields

## Installation

Install this project with [poetry](https://python-poetry.org).
Note that if you have a virtual environment activated (e.g. a `conda` environment), `poetry install` will install this project and its dependencies into that environment. Otherwise, poetry will create a virtual environment.

```
git clone git@gitlab.kit.edu:kit/ag_wenzel/humf.git
cd humf
poetry install
```

Activate the environment.
Note that this is not necessary if you had a virtual environment active when running `poetry install`.

```
poetry shell
```

If you intend to contribute, install [pre-commit](https://pre-commit.com) hooks:

```
pre-commit install
```

## Testing

Run tests with [pytest](https://docs.pytest.org).

```
pytest
```

## Contributing

The pre-commit tool uses [ruff](https://github.com/astral-sh/ruff) to lint and format Python files, and [isort](https://pycqa.github.io/isort/) to sort imports in Python files. Consider integrating these tools into your workflow, e.g. using the vscode plugins for [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort).
