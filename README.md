# Netflix Recommendation App


![Python Version](https://img.shields.io/badge/python-3.10-green)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Flake8](https://img.shields.io/badge/code%20quality-flake8-blue)](https://github.com/PyCQA/flake8)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


## Introduction
This app recommends movies based on other movies you liked.
The Model was trained with [kaggle dataset](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies) using cosine similarity based on its features.

## Project Repositories

### Github:
Clonning the Repo

```bash
git clonegit@github.com:joao-victor-campos/netflix-recommendation-app.git
cd netflix-recommendation-app
```

### Huggingface:

If you want to use the application, go to our :laughing: [Huggingface Repository](https://huggingface.co/spaces/joao-victor-campos/netflix-recommensation-model) 


## Development

### Install dependencies

```bash
make requirements
```

### Code Style
Apply code style with black and isort
```bash
make apply-style
```

Perform all checks (flake8, black and mypy)
```bash
make checks
```

