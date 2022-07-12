# globals
VERSION := $(shell grep __version__ spark_pipeline/__metadata__.py | head -1 | cut -d \" -f2 | cut -d \' -f2)

.PHONY: requirements-dev
## install development requirements
requirements-dev:
	@python -m pip install -U -r requirements.dev.txt

.PHONY: requirements-minimum
## install prod requirements
requirements-minimum:
	@python -m pip install -U -r requirements.txt

.PHONY: requirements
## install requirements
requirements: requirements-dev requirements-minimum