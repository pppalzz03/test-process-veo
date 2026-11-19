# Common commands. Nothing here is destructive: there is intentionally NO `down -v` / prune target.
COMPOSE ?= docker compose
PY      ?= python3

.PHONY: help install lint typecheck test test-unit test-integration test-rabbitmq parser-test \
        up down logs ps migrate env-check env-check-container build-engine report

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

install:               ## install package + dev deps
	$(PY) -m pip install -e '.[dev]'

lint:                  ## ruff
	ruff check . && ruff format --check .

typecheck:             ## mypy
	mypy shared services

test: test-unit test-integration  ## everything runnable without GPU/RabbitMQ

test-unit:             ## fast tests, no services
	$(PY) -m pytest tests/unit -q

test-integration:      ## needs PostgreSQL (TEST_DATABASE_URL or local postgres binaries)
	$(PY) -m pytest tests/integration -q

test-rabbitmq:         ## needs a DISPOSABLE broker: RABBITMQ_URL=amqp://...
	$(PY) -m pytest -m rabbitmq tests/integration/test_rabbitmq.py -q

parser-test:           ## C++ decode logic (no DeepStream needed)
	$(MAKE) -C deepstream/parser test

env-check:             ## Phase 1, read-only host inspection (run on the DGX Spark)
	scripts/env_check.sh

env-check-container:   ## also test the DeepStream image if already pulled
	scripts/env_check.sh --container-test

up:                    ## start the stack (needs .env)
	$(COMPOSE) up -d --build

down:                  ## stop containers, KEEP volumes
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

migrate:
	$(COMPOSE) run --rm migrate

build-engine:          ## usage: make build-engine VIDEO=/data/media/sample.mp4
	$(COMPOSE) run --rm --no-deps worker /app/scripts/build_engine.sh $(VIDEO)

report:                ## regenerate docs/benchmark-report.md from measured results
	$(PY) scripts/render_benchmark_report.py
