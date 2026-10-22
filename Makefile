PYTHON ?= python
DATA_DIR ?= data
TRUTH_DIR ?= truth
MODEL_DIR ?= model
SEED ?= 42
DAYS ?= 21
NODES ?= 200
SELECTION_STRENGTH ?= 1.0
CAUSAL_EFFECT_SCALE ?= 1.0
TEST_DIR ?= build/test
QUERY_DIR ?= build/queries
BENCH_N ?= 300

.PHONY: data validate queries baseline train predict eval bench effect effect-validate diagnostics test clean

# Regenerates PUBLIC data and PRIVATE structural truth (not the trained model).
data:
	$(PYTHON) -m src.simulate --out "$(DATA_DIR)" --truth "$(TRUTH_DIR)" --seed $(SEED) --days $(DAYS) --nodes $(NODES) --selection-strength $(SELECTION_STRENGTH) --causal-effect-scale $(CAUSAL_EFFECT_SCALE)
	$(PYTHON) -m src.validate --data "$(DATA_DIR)" --truth "$(TRUTH_DIR)" --days $(DAYS)

validate:
	$(PYTHON) -m src.validate --data "$(DATA_DIR)" --truth "$(TRUTH_DIR)" --days $(DAYS)

# FROM and TO are UTC epoch seconds or ISO 8601; OUT is a directory.
queries:
	$(PYTHON) -m src.queries --data "$(DATA_DIR)" --from "$(FROM)" --to "$(TO)" --out "$(or $(OUT),$(QUERY_DIR))"

baseline:
	$(PYTHON) -m src.baseline --data "$(DATA_DIR)" --queries "$(or $(QUERIES),$(QUERY_DIR)/queries.jsonl)" --out "$(or $(OUT),$(TEST_DIR)/baseline.csv)"

# UNTIL is REQUIRED: exclusive cutoff of completed trips (UTC epoch or ISO 8601).
train:
	$(PYTHON) -m src.eta_model train --data "$(DATA_DIR)" --model "$(MODEL_DIR)" --until "$(UNTIL)" $(if $(TRAIN_QUERIES),--queries "$(TRAIN_QUERIES)",)

predict:
	$(PYTHON) -m src.eta_model predict --data "$(DATA_DIR)" --model "$(MODEL_DIR)" --queries "$(or $(QUERIES),$(TEST_DIR)/queries.jsonl)" --out "$(or $(OUT),$(TEST_DIR)/catboost.csv)"

# Exact assignment contract: QUERIES=... TRUTH=... PRED=... -> eval/report.md.
# Uses its own baseline by reading the graph. Does not depend on TEST_DIR.
eval:
	$(PYTHON) -m src.eval_contract --data "$(DATA_DIR)" --queries "$(or $(QUERIES),$(TEST_DIR)/queries.jsonl)" --truth "$(or $(TRUTH),$(TEST_DIR)/truth.csv)" --pred "$(or $(PRED),$(TEST_DIR)/catboost.csv)" --out "$(or $(OUT),eval/report.md)"

# Bench measures preloaded repeated one-query latency; use a realistic diverse-query
# workload as an additional check before claiming the full streaming SLA.
bench:
	$(PYTHON) -m src.bench --data "$(DATA_DIR)" --model "$(MODEL_DIR)" --queries "$(or $(QUERIES),$(TEST_DIR)/queries.jsonl)" --n $(BENCH_N) --out "$(or $(OUT),$(TEST_DIR)/bench.json)"

# Outcome/propensity features use only PUBLIC data; truth is NEVER read here.
effect:
	$(PYTHON) -m src.effect --data "$(DATA_DIR)" --out "$(or $(OUT),effect.json)"

# Five separate simulations, never touching the primary data/model directories.
effect-validate:
	$(PYTHON) -m src.effect_validation --work "$(or $(WORK),build/effect_validation)" --days $(DAYS) --nodes $(NODES) $(if $(REUSE),--reuse,)

diagnostics:
	$(PYTHON) -m src.diagnostics --data "$(DATA_DIR)" --test "$(TEST_DIR)"

test:
	$(PYTHON) -m unittest discover -s tests -v

# Safety: do not silently delete data or truth with 'make clean'.
clean:
	@echo "No automatic deletion. Delete build/, data/, truth/, model/ manually if intended."
