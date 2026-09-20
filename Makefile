study_name ?= LightGBM_tuning
dir ?= optuna
n_tune_trials ?= 100
model_type ?= 0
residual ?= 0
transform ?= 0
use_panel_id ?= 0
loss ?= 0

# Inference defaults
inf_dir ?= optuna
inf_study_name ?= LightGBM_tuning
inf_model_type ?= 0
inf_log_dir ?= logs
inf_residual ?= 0
inf_transform ?= 0
inf_use_panel_id ?= 0

ifeq ($(shell command -v uv 2>/dev/null),)
    RUN = python # Change to py or python3 depending on Python installation
else
    RUN = uv run python
endif



clean-data:
	$(RUN) scripts/clean_sgor_script.py
	$(RUN) scripts/get_panels.py

tune-lgbm:
	$(RUN) scripts/lightgbm_script.py \
		--study_name $(study_name) \
		--dir $(dir) \
		--n_tune_trials $(n_tune_trials) \
		--model_type $(model_type) \
		--residual $(residual) \
		--transform $(transform) \
		--use_panel_id $(use_panel_id)


run-lgbm-inference:
	$(RUN) scripts/lightgbm_inference.py \
		--dir $(inf_dir) \
		--study_name $(inf_study_name) \
		--model_type $(inf_model_type) \
		--log_dir $(inf_log_dir) \
		--residual $(inf_residual) \
		--transform $(inf_transform) \
		--use_panel_id $(inf_use_panel_id)

open-dashboard:
	ifeq ($(shell command -v uv 2>/dev/null),)
		uv run optuna-dashboard $(path_to_db_file)
	else
		optuna-dashboard $(path_to_db_file)
	endif