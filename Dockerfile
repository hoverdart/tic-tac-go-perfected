FROM python:3.12-slim AS solver-builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir Cython==3.3.1 setuptools
COPY scripts/build_solver.py ./scripts/build_solver.py
COPY solver/push_solver/v4.py solver/push_solver/path_optimizer.py ./solver/push_solver/
RUN python scripts/build_solver.py

FROM python:3.12-slim

WORKDIR /app

COPY apps/api/requirements.txt ./apps/api/requirements.txt
RUN pip install --no-cache-dir -r apps/api/requirements.txt

COPY . .
COPY --from=solver-builder /build/solver/push_solver/*.so ./solver/push_solver/
COPY --from=solver-builder /build/solver/push_solver/native-build.json ./solver/push_solver/

CMD ["sh", "-c", "python -m apps.api.migrate && exec uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
