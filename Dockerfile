FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY docker-requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

COPY models/retention/logistic_retention_v2 /app/models/retention/logistic_retention_v2
COPY services /app/services

ENV NEXUS_RETENTION_MODEL_URI=/app/models/retention/logistic_retention_v2

EXPOSE 8000

CMD ["uvicorn", "services.retention_api.main:app", "--host", "0.0.0.0", "--port", "8000"]