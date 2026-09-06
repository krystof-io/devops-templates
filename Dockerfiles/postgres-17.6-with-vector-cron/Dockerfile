# CNPG PostgreSQL 17.6 + pgvector + pg_cron.
#
# WHY A SIBLING IMAGE RATHER THAN ADDING pg_cron TO postgres-17.6-with-vector:
# that image's workflow force-pushes the MUTABLE :17.6 tag, and every CNPG
# Cluster in the public cluster (dev, prod, hindsight, openwebui — nine of them)
# pins exactly that tag. Adding a package there would reach all of them on their
# next pod restart. A separate image keeps the blast radius at zero for the cost
# of one Dockerfile and one workflow.
#
# WHO USES THIS: firecrawl-db. Firecrawl's job queue ("NuQ") is a Postgres
# schema, not Redis, and its DDL opens with `CREATE EXTENSION pg_cron` and then
# schedules 36 cron jobs — lock reapers every 15s, cleanup every 5min, nightly
# per-index REINDEX CONCURRENTLY. Without pg_cron the queue accumulates orphaned
# locked jobs and never reclaims them.
#
# WHY NOT CNPG 1.27's DECLARATIVE spec.postgresql.extensions: it needs
# PostgreSQL 18+ (it is built on extension_control_path, which does not exist in
# PG 17) AND it mounts extensions through a Kubernetes ImageVolume, whose
# feature gate is off on this cluster
# (kubernetes_feature_enabled{name="ImageVolume",stage="BETA"} 0 on k8s v1.34.1).
#
# The -standard- base keeps the PGDG apt source and keyring (it only clears
# /var/lib/apt/lists), which is why installing PGDG packages here works at all.
# postgresql-17-cron is present in the trixie-pgdg amd64 index.
FROM ghcr-proxy.private.krystof.io/cloudnative-pg/postgresql:17.6-standard-trixie

USER root

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        postgresql-17-pgvector \
        postgresql-17-cron; \
    apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
    rm -rf /var/lib/apt/lists/* /var/cache/* /var/log/*

USER 26
