
.venv:
	python3 -m venv .venv
	.venv/bin/python -m pip install cython numpy

develop: .venv
	.venv/bin/python -m pip install -e .

