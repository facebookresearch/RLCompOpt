define HELP
CompilerGym experiments. Available targets:

	make init
		Install the build and runtime python dependencies. This should be run
		once before any other targets.

	make install
		Install the package itself.
endef
export HELP

# Configurable paths to binaries.
PYTHON ?= python3

.DEFAULT_GOAL := help

help:
	@echo "$$HELP"

init:
	$(PYTHON) -m pip install -r requirements.txt
	pre-commit install

install:
	$(PYTHON) setup.py install
	pre-commit install

.PHONY: init install
