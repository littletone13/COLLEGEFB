
# Makefile for CFB player-prop pipeline
# Requirements: python3, pytest (for tests), and PyYAML (for reading config.yaml)

PY ?= python3

# Pull values from config.yaml using PyYAML
YEAR := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['year'])")
WEEK := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['week'])")
MODEL_TAG := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['model_tag'])")
TITLE := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['title'])")
TEAMS_CSV := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['paths']['teams_csv'])")
BASELINES_CSV := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['paths']['baselines_csv'])")
ENHANCED_CSV := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['paths']['enhanced_csv'])")
EDGES_CSV := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['paths']['edges_csv'])")
REPORT_HTML := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['paths']['report_html'])")
TEMPLATE := $(shell $(PY) -c "import yaml; print(yaml.safe_load(open('config.yaml'))['paths']['template'])")
ASSUMED := $(shell $(PY) -c "import yaml; print(','.join(f\"{k}={v}\" for k, v in yaml.safe_load(open('config.yaml'))['assumed_prices'].items()))")

.PHONY: all baselines enhance report compile tests clean

all: baselines enhance report

compile:
	$(PY) -m compileall cfb/player_prop_sim.py scripts/enhance_player_prop_inputs.py scripts/generate_player_prop_baselines.py cfb/names.py || true

baselines:
	@echo '==> Generating baselines for year $(YEAR) week $(WEEK) -> $(BASELINES_CSV)'
	$(PY) scripts/generate_player_prop_baselines.py --year $(YEAR) --week $(WEEK) --output $(BASELINES_CSV)

enhance: baselines
	@echo '==> Enhancing inputs with CFBD + shares -> $(ENHANCED_CSV)'
	CFBD_API_KEY=$$CFBD_API_KEY $(PY) scripts/enhance_player_prop_inputs.py --year $(YEAR) --week $(WEEK) --in $(BASELINES_CSV) --out $(ENHANCED_CSV)

report: enhance
	@echo '==> Running sims + building HTML report -> $(REPORT_HTML)'
	$(PY) cfb_helper.py --teams $(TEAMS_CSV) --players $(ENHANCED_CSV) --out $(EDGES_CSV) --html $(REPORT_HTML) --title "$(TITLE)" --model-tag "$(MODEL_TAG)" --assumed "$(ASSUMED)" --template $(TEMPLATE)

tests:
	pytest -q

clean:
	rm -f $(EDGES_CSV) $(REPORT_HTML)
