FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./
COPY packages/ packages/
COPY conf/ conf/

# Serve only needs a subset of packages
RUN uv sync --frozen --no-dev --package adas-infra-serve --package adas-infra-core --package adas-infra-obs --package adas-infra-cli

ENV ADAS_SERVE_PORT=8080
EXPOSE 8080
EXPOSE 9000

ENTRYPOINT ["uv", "run", "adas-serve"]
CMD ["+profile=local_mock"]
