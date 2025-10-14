# Generic Spring Boot Dockerfile
# This Dockerfile is used by all Spring Boot applications unless overridden
#
# Location: docker-templates repo at java/spring-boot/Dockerfile
#
# Build Args:
#   BASE_IMAGE - Base Docker image (default: eclipse-temurin:17-jre-alpine)
#   JAR_FILE - Path to the JAR file to copy (required)
#   HEAP_SIZE - JVM heap size (default: 512m)
#   APP_NAME - Application name for logging/monitoring (default: app)

ARG BASE_IMAGE=eclipse-temurin:17-jre-alpine
FROM ${BASE_IMAGE}

# Metadata
LABEL maintainer="krystof.io"
LABEL description="Generic Spring Boot application container"

# Build arguments
ARG JAR_FILE
ARG HEAP_SIZE=512m
ARG APP_NAME=app

# Create app user for security (don't run as root)
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy application JAR
COPY ${JAR_FILE} app.jar

# Change ownership to app user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set JVM options
ENV JAVA_OPTS="-Xms${HEAP_SIZE} -Xmx${HEAP_SIZE} \
    -XX:+UseContainerSupport \
    -XX:MaxRAMPercentage=75.0 \
    -Djava.security.egd=file:/dev/./urandom"

# Expose default Spring Boot port
EXPOSE 8080

# Health check (assumes Spring Boot Actuator health endpoint)
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

# Run the application
ENTRYPOINT exec java ${JAVA_OPTS} -jar /app/app.jar