FROM python:3.12-slim

WORKDIR /app

COPY apps/api/requirements.txt ./apps/api/requirements.txt
RUN pip install --no-cache-dir -r apps/api/requirements.txt

COPY . .

CMD ["sh", "-c", "python -m apps.api.migrate && exec uvicorn apps.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
