
FROM python:3.13-slim

WORKDIR /opt/nexus

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DBT_PROFILES_DIR=/opt/nexus/.dbt
ENV PYTHONPATH=/opt/nexus

# Install Git with bounded retries for transient repository failures.
RUN apt-get -o Acquire::Retries=3 update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-orchestration.txt /tmp/requirements.txt

# Preserve the project's pinned dependency versions.
RUN pip install --no-cache-dir --retries 5 --timeout 60 \
        -r /tmp/requirements.txt

COPY nexus_dbt /opt/nexus/nexus_dbt
COPY scripts /opt/nexus/scripts
COPY docs /opt/nexus/docs
COPY docker/nexus-data-platform/profiles.yml /opt/nexus/.dbt/profiles.yml

WORKDIR /opt/nexus/nexus_dbt

CMD ["dbt", "--version"]
