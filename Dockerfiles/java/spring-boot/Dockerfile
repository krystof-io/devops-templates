# Generic Spring Boot Dockerfile
# This Dockerfile is used by all Spring Boot applications unless overridden
#
# Location: docker-templates repo at java/spring-boot/Dockerfile
#
# Build Args:
#   BASE_IMAGE - Base Docker image (default: eclipse-temurin:17-jre-alpine)
#   JAR_FILE - Path to the JAR file to copy (required)
#   MAX_RAM_PERCENTAGE - Percentage of container memory to use for heap (default: 75.0)
#   APP_NAME - Application name for logging/monitoring (default: app)

ARG BASE_IMAGE=eclipse-temurin:17-jre-alpine
FROM ${BASE_IMAGE}

# Metadata
LABEL maintainer="krystof.io"
LABEL description="Generic Spring Boot application container"

# Build arguments
ARG JAR_FILE
ARG MAX_RAM_PERCENTAGE=75.0
ARG ADDITIONAL_JAVA_OPTS=""
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

# Set JVM options - Let Java automatically size heap based on container limits
ENV JAVA_OPTS="-XX:+UseContainerSupport \
    -XX:MaxRAMPercentage=${MAX_RAM_PERCENTAGE} \
    -XX:+UseG1GC \
    -XX:+UseStringDeduplication \
    -Djava.security.egd=file:/dev/./urandom \
    ${ADDITIONAL_JAVA_OPTS}"

# Expose default Spring Boot port
EXPOSE 8080

# Health check (assumes Spring Boot Actuator health endpoint)
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

# Run the application
ENTRYPOINT exec java ${JAVA_OPTS} -jar /app/app.jar