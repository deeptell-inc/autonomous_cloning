PY ?= python
export PYTHONPATH := $(CURDIR)
.PHONY: test slow quick all figures clean
test:            ## fast unit tests (< 1 min)
	$(PY) -m pytest tests/test_core.py -q
slow:            ## LM exactness + symbolic certification (minutes)
	$(PY) -m pytest tests/test_slow.py -q -m slow
quick:           ## theorems to M=6, quick robustness, Rydberg, symbolic, figures (~2 min)
	$(PY) scripts/verify_theorems.py 6 && $(PY) scripts/robustness.py --quick && $(PY) scripts/rydberg.py && $(PY) scripts/symbolic_proofs.py && $(PY) scripts/make_figures.py
all:             ## full reproduction (M to 8, 200 disorder samples)
	$(PY) scripts/verify_theorems.py 8 && $(PY) scripts/robustness.py && $(PY) scripts/rydberg.py && $(PY) scripts/symbolic_proofs.py && $(PY) scripts/make_figures.py
figures:
	$(PY) scripts/make_figures.py
clean:
	rm -f results/*.json figures/*.pdf
