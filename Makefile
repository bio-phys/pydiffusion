
.venv:
	python3 -m venv .venv
	.venv/bin/python -m pip install cython 'numpy<1.25'

develop: .venv
	.venv/bin/python -m pip install -e .

