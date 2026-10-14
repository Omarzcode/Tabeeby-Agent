# All targets run inside the `tabeeby-agent` conda env (see environment.yml).
CONDA_ENV ?= tabeeby-agent
RUN = conda run --no-capture-output -n $(CONDA_ENV)

.PHONY: env app test eval bench mcp mcp-http telegram ingest ingest_default check-vectors lock deploy cloud-env docs

env:            ## create/update the conda env
	conda env update -n $(CONDA_ENV) -f environment.yml --prune

app:            ## run the API locally with auto-reload
	$(RUN) uvicorn app:app --reload --host 0.0.0.0 --port 8000

test:           ## offline test suite (no API keys needed)
	$(RUN) python -m pytest -q

eval:           ## live eval: real LLM + Qdrant, scored by code
	$(RUN) python -m evals.run_eval

bench:          ## live skill trigger bench
	$(RUN) python -m evals.skill_trigger_bench

mcp:            ## Tabeeby as a stdio MCP server (for Claude Desktop / Cursor)
	$(RUN) python -m mcp_server

mcp-http:       ## Tabeeby as a standalone streamable-HTTP MCP server on :8001
	$(RUN) python -m mcp_server --http

telegram:       ## Telegram bot locally via long polling (no public URL needed)
	$(RUN) python -m scripts.telegram_polling

ingest:         ## (re)build the Qdrant collection — DESTRUCTIVE with --reset
	$(RUN) python -m scripts.ingest_vezeeta --batch-size 64 --reset

ingest_default:
	$(RUN) python -m scripts.ingest_vezeeta

check-vectors:  ## are stored vectors from the configured embedding model?
	$(RUN) python -m scripts.check_vector_consistency

lock:           ## refresh uv.lock after editing pyproject.toml (FastAPI Cloud installs from it)
	$(RUN) uv lock

cloud-env:      ## push .env values to FastAPI Cloud as secrets
	$(RUN) bash scripts/push_env_to_fastapi_cloud.sh

deploy:         ## deploy to FastAPI Cloud
	$(RUN) fastapi deploy

docs:           ## build the LaTeX report (uses the `latex` conda env)
	cd ../docs && conda run -n latex tectonic tabeeby_report.tex
