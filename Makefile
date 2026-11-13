PY ?= python

.PHONY: install download extract eval test index serve clean

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) data/download_data.py

extract:
	$(PY) -m src.extract data/raw data/processed

# Evalue chunking x mode de recherche. Echoue avec un message clair
# tant que data/eval/questions.jsonl n'est pas etiquete.
eval: extract
	$(PY) -m src.evaluate --sizes 128 256 --strategies fixed section clause

test:
	$(PY) -m pytest -q

index: extract
	$(PY) -m src.index --strategy section --size 256 --out index

serve:
	$(PY) -m uvicorn src.app:app --reload

clean:
	rm -rf index .pytest_cache
