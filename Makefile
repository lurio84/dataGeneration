.PHONY: setup test gen train predict

setup:
	python3 -m venv venv && venv/bin/pip install -r requirements.txt

test:
	cd src && python3 -m pytest test_pipeline.py test_classifier.py -v \
	  --cov=. --cov-report=html:../htmlcov --cov-report=term-missing

gen:
	cd src && python3 generate_dataset.py --n 100 --seed 42

train:
	cd src && python3 -u classifier/train.py \
	  --cv-estimators 100 --cv-subsample 0.3 --n-jobs -1

predict:
	@echo "Usage: make predict INPUT=path/to/input.ply OUTPUT=path/to/output.ply"
	@echo "       make predict INPUT=input.ply OUTPUT=out.ply MODEL=lgbm"
	cd src && python3 classifier/predict.py $(INPUT) $(OUTPUT) $(if $(MODEL),--model $(MODEL),)
