FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./
COPY packages/ packages/
COPY conf/ conf/

RUN uv sync --frozen --no-dev

COPY scripts/ scripts/
COPY tests/ tests/

ENV ADAS_DATA_ROOT=/data
ENV ADAS_STATE_DIR=/state
ENV ADAS_CHECKPOINT_DIR=/checkpoints

ENTRYPOINT ["uv", "run", "adas-train"]
CMD ["+profile=local_mock", "trainer.max_steps=1000"]
