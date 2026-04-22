FROM ghcr-proxy.private.krystof.io/cloudnative-pg/postgresql:17.6-standard-trixie

USER root

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        postgresql-17-pgvector; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
    rm -rf /var/lib/apt/lists/* /var/cache/* /var/log/*

USER 26