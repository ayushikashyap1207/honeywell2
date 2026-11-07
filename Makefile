SHELL := /bin/bash

.PHONY: demo up down logs ps migrate seed replay test clean calibrate evaluate drift shap track reports retrain index

# One-command live demo: infra -> schema -> batch seed -> app -> replay.
demo:
	docker compose up -d db redis kafka
	$(MAKE) migrate
	docker compose --profile seed run --rm --build seed
	docker compose --profile app --profile demo up -d --build
	@echo ""
	@echo "Dashboard:  http://localhost:8080"
	@echo "API docs:   http://localhost:8000/docs"
	@echo "MLflow:     http://localhost:5000 (docker compose up -d mlflow)"
	@echo "Follow logs with: make logs"

up:
	docker compose up -d
	$(MAKE) migrate

# Apply schema changes to an existing volume (safe to run repeatedly).
migrate:
	@for i in $$(seq 1 30); do \
		docker compose exec -T db pg_isready -U anomaly -d anomaly_detection >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	@sleep 2
	docker compose exec -T db psql -U anomaly -d anomaly_detection -f /docker-entrypoint-initdb.d/01_schema.sql

down:
	docker compose down

logs:
	docker compose --profile app logs -f --tail=100

ps:
	docker compose --profile app ps

seed:
	docker compose --profile seed run --rm --build seed

replay:
	docker compose --profile app build feature-worker
	docker compose --profile demo run --rm replay

test:
	python3 -m pytest -q

calibrate:
	python3 -m calibration.calibrate

evaluate:
	python3 evaluation.py

drift:
	python3 -m monitoring.drift

shap:
	python3 explainability_shap.py

index:
	python3 -m copilot.retrieval

track:
	python3 track_experiment.py

# Refresh the Week 2 model-report artifacts in dependency order.
reports: calibrate evaluate drift shap track
	python3 explainability.py
	python3 export_alerts_json.py
	python3 generate_dashboard.py

# Force a full retrain + report refresh (Prefect flow).
retrain:
	python3 flows/retrain.py --force

clean:
	docker compose --profile app --profile demo --profile seed down -v
