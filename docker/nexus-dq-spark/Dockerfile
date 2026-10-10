FROM apache/spark:4.1.1-python3

USER root

WORKDIR /opt/nexus

ENV PYTHONPATH=/opt/nexus
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip3 install --no-cache-dir pyspark==4.1.1 pyyaml==6.0.3

COPY scripts /opt/nexus/scripts
COPY docs /opt/nexus/docs

RUN chown -R spark:spark /opt/nexus

USER spark

WORKDIR /opt/nexus

CMD ["python3", "--version"]