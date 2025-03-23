# Project Makefile
.PHONY: clean-pyc uninstall-e install-e tests sync

all: clean-pyc uninstall-e install-e tests

clean-pyc:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.py[co]" -exec rm {} +

uninstall-e:
	pip uninstall -y cml-tools
	rm -rf src/*.egg-info

install-e: clean-pyc uninstall-e
	pip install --editable .

tests:
	@echo -n 'Checking cml tools is installed: version '
	@python -c 'import cml;print(cml.__version__)'
	@echo
	@echo 'Run an individual test file by invoking it as a module, e.g.'
	@echo '    python -m tests.test_online_norm -v'
	@echo
	@python -m unittest discover src/cml/test -v

tests-float32:
	@echo -n 'Checking cml tools is installed: version '
	@python -c 'import cml;print(cml.__version__)'
	@echo
	@echo 'Run an individual test file by invoking it as a module, e.g.'
	@echo '    python -m tests.test_online_norm -v'
	@echo
	@DTYPE=float python -m unittest discover src/cml/test -v

sync:
	rsync -avzP --exclude='__pycache__' --exclude='*pyc' $(shell pwd) /clio/projects --delete
