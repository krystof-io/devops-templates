FROM debian:stable AS downloader

RUN apt-get update && apt-get install -y wget

# Copy versions file
COPY versions.yaml /tmp/versions.yaml

# Download script
COPY download-otel-agents.sh /tmp/download-otel-agents.sh
RUN chmod +x /tmp/download-otel-agents.sh

# Create directory structure
RUN mkdir -p /utilities/otel-agents

# Download all versions
RUN /tmp/download-otel-agents.sh

# Final minimal image
FROM busybox:latest

# Copy downloaded artifacts
COPY --from=downloader /utilities /utilities

# Copy versions file for reference
COPY versions.yaml /utilities/versions.yaml

LABEL maintainer="krystof.io"
LABEL description="Init container utilities for microservices"