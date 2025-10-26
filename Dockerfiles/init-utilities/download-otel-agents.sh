#!/bin/sh
set -e

echo "Downloading artifacts based on versions.yaml..."

# Parse YAML and download OTel agents (using basic grep/awk since we're in busybox)
OTEL_VERSIONS=$(grep -A 10 "versions:" /tmp/versions.yaml | grep -E "^\s+- " | sed 's/.*"\(.*\)".*/\1/')
OTEL_DEFAULT=$(grep "default:" /tmp/versions.yaml | head -1 | sed 's/.*"\(.*\)".*/\1/')

for VERSION in $OTEL_VERSIONS; do
    echo "Downloading OpenTelemetry Java agent v${VERSION}..."
    wget -q --show-progress \
        -O /utilities/otel-agents/opentelemetry-javaagent-${VERSION}.jar \
        "https://github.com/open-telemetry/opentelemetry-java-instrumentation/releases/download/v${VERSION}/opentelemetry-javaagent.jar" \
        || echo "Warning: Failed to download version ${VERSION}"
done

# Create latest symlink
echo "Creating symlink: latest -> ${OTEL_DEFAULT}"
ln -sf /utilities/otel-agents/opentelemetry-javaagent-${OTEL_DEFAULT}.jar \
       /utilities/otel-agents/opentelemetry-javaagent-latest.jar

echo "Download complete!"
ls -lh /utilities/otel-agents/