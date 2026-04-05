#!/bin/sh
set -eu

export PORT="${PORT:-8000}"
export WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"
export GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-180}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

exec gunicorn api.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers "${WEB_CONCURRENCY}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  --bind "0.0.0.0:${PORT}" \
  --log-level "${LOG_LEVEL}" \
  --access-logfile - \
  --error-logfile -
