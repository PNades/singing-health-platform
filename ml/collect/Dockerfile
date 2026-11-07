# Collection site image (Cloud Run service strainless-collect, spec 2.8).
# Built by scripts/deploy_collect.ps1 from a staged context that holds only
# what the service needs (no data/, no .env): this file sits at its root.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app \
    PATH=/app/.venv/bin:$PATH \
    PORT=8080

# Only the `collect` dependency group, at the versions in uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group collect --no-cache

COPY . .
# The consent text and the quality checks must load at startup.
RUN python -c "from ml.collect.app import create_app; from ml.collect.storage import LocalStore; create_app(store=LocalStore('/tmp/x'))"

USER nobody
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 60 --error-logfile - "ml.collect.app:create_app()"
