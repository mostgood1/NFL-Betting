#!/bin/bash
set -euo pipefail

echo "🚀 Starting NFL Betting on Render..."

# Ensure directories
mkdir -p nfl_compare/data nfl_compare/models logs

# Python version
python --version || true

# Install runtime deps if not already installed (Render does pip install -r requirements.txt)

# Seed minimal files if missing
if [ ! -f "nfl_compare/data/predictions.csv" ]; then
  echo "⚠️ predictions.csv not found; the UI/API will be empty until you generate it."
fi

# Environment
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

echo "Using PORT=${PORT:-5000} WEB_CONCURRENCY=${WEB_CONCURRENCY:-1} WEB_THREADS=${WEB_THREADS:-4}"

# Start app (avoid --preload to reduce boot memory/CPU; longer timeout for cold starts)
exec gunicorn app:app \
  --bind 0.0.0.0:${PORT:-5000} \
  --workers ${WEB_CONCURRENCY:-1} \
  --worker-class gthread \
  --threads ${WEB_THREADS:-4} \
  --timeout 180 \
  --keep-alive 5
