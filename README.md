# ne-syntax-consistency

Detecting named entity (NE) annotations that are inconsistent with morphology and dependency syntax in Estonian text.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run marimo edit notebooks/
```

Put your credentials in a `.env` file in the project root (it is gitignored):

```bash
cp .env.example .env
```
