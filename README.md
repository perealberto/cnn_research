# CNN Research Framework

A lightweight, modular codebase for experimenting with Convolutional Neural Networks (CNNs) in research settings.

## Installation

Project and environment.

```bash
git clone https://github.com/perealberto/cnn_research.git
cd cnn_research
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Pre-commit and linters.

```bash
pre-commit install
pre-commit migrate-config
pre-commit run --all-files
```

Additional tools.

```bash
jupyter labextension install jupyterlab-jupytext
```

## Repository Layout

```bash
src/           # Python package with reusable modules
notebooks/     # Exploratory notebooks (paired with .py)
```
