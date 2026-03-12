FROM docker-private.private.krystof.io/debian:stable AS downloader

RUN apt-get update && apt-get install -y wget xz-utils

# Copy versions file
COPY versions.yaml /tmp/versions.yaml

# Download OpenTelemetry agents
COPY download-otel-agents.sh /tmp/download-otel-agents.sh
RUN chmod +x /tmp/download-otel-agents.sh
RUN mkdir -p /utilities/otel-agents
RUN /tmp/download-otel-agents.sh

# Download FFmpeg
COPY download-ffmpeg.sh /tmp/download-ffmpeg.sh
RUN chmod +x /tmp/download-ffmpeg.sh
RUN /tmp/download-ffmpeg.sh

# Final minimal image
FROM docker-private.private.krystof.io/busybox:stable

# Copy downloaded artifacts
COPY --from=downloader /utilities /utilities

# Copy versions file for reference
COPY versions.yaml /utilities/versions.yaml

LABEL maintainer="krystof.io"
LABEL description="Init container utilities for microservices"