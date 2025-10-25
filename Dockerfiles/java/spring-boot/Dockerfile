# Generic Spring Boot Dockerfile
# This Dockerfile is used by all Spring Boot applications unless overridden
#
# Location: docker-templates repo at java/spring-boot/Dockerfile
#
# Build Args:
#   BASE_IMAGE - Base Docker image (default: eclipse-temurin:17-jre-alpine)
#   JAR_FILE - Path to the JAR file to copy (required)

ARG BASE_IMAGE=eclipse-temurin:17-jre-alpine
FROM ${BASE_IMAGE}

# Metadata
LABEL maintainer="krystof.io"
LABEL description="Generic Spring Boot application container"

# Build arguments
ARG JAR_FILE

# Create app user for security (don't run as root)
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Create common mount points for application data
# /app/data - Persistent application data (uploads, files, etc.)
# /app/logs - Application log files  
# /app/work - Application working directory (processing, temp files, etc.)
RUN mkdir -p /app/data /app/logs /app/work && \
    chown -R appuser:appgroup /app

# Copy application JAR
COPY ${JAR_FILE} app.jar

# Switch to non-root user
USER appuser

# Set minimal JVM options - container support and security
ENV JAVA_OPTS="-XX:+UseContainerSupport \
    -Djava.security.egd=file:/dev/./urandom"

# Declare volumes for common mount points
# These can be overridden by Kubernetes volume mounts
VOLUME ["/app/data", "/app/logs", "/app/work"]

# Expose default Spring Boot port
EXPOSE 8080

# Expose management port
EXPOSE 8081

# Run the application
ENTRYPOINT exec java ${JAVA_OPTS} -jar /app/app.jar