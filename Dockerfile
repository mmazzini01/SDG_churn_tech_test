FROM apache/airflow:3.4.0-python3.12

# Same uv version as on the host, so it can read uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /bin/uv
COPY pyproject.toml uv.lock /tmp/project/

# uv turns the lock into exact pins; pip installs them the way the Airflow docs recommend.
# Pinning apache-airflow keeps pip from changing Airflow itself; pip check then fails the build if a
# pin breaks any other installed package (e.g. a provider), which pip install alone only warns about.
RUN /bin/uv export --project /tmp/project --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.txt \
 && pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt \
 && pip check
