# Claude Context for devops-templates

## Repository Overview

Reusable DevOps templates for home lab deployments. Contains Helm charts and Dockerfiles published via GitHub Pages as a Helm repository.

**Helm repo:** `https://krystof-io.github.io/devops-templates`

## Repository Structure

```
devops-templates/
├── charts/
│   └── spring-boot-app/     # Generic Spring Boot Helm chart
├── Dockerfiles/
│   ├── java/spring-boot/    # Base Spring Boot Dockerfile
│   ├── node/frontend/       # Node.js frontend with nginx
│   ├── init-utilities/      # Init container with OTEL agents
│   └── actions-runner-*/    # GitHub Actions self-hosted runner
├── .github/workflows/
│   ├── release-charts.yaml  # Auto-publish charts to GitHub Pages
│   └── init-utilities.yaml  # Build init-utilities image
└── cr.yaml                  # Chart releaser config
```

## Helm Charts

### spring-boot-app

Generic chart for deploying any Spring Boot application. Key features:

- **Security:** ServiceAccount with no API token, pod security context, read-only root filesystem
- **Observability:** OpenTelemetry integration via init container, actuator endpoints
- **Reliability:** Configurable probes (liveness/readiness/startup), PDB, HPA
- **Networking:** Service, Ingress (nginx), Traefik IngressRoute support
- **Graceful shutdown:** terminationGracePeriodSeconds (30s), preStop hook (5s sleep)
- **Zero-downtime deploys:** RollingUpdate strategy with maxSurge:1, maxUnavailable:0

**Testing:**
```bash
helm lint charts/spring-boot-app
helm template test charts/spring-boot-app -f charts/spring-boot-app/sample-values.yaml
```

## Dockerfiles

| Image | Purpose |
|-------|---------|
| `java/spring-boot` | Base image for Spring Boot apps (Temurin JRE, non-root user) |
| `node/frontend` | Multi-stage Node.js build with nginx serving |
| `init-utilities` | Init container with OpenTelemetry Java agents |
| `actions-runner-*` | Self-hosted GitHub runner with JDK21, Maven, kubectl, Flux |

## GitHub Workflows

- **release-charts.yaml** - Publishes Helm charts to GitHub Pages on push to `charts/`
- **init-utilities.yaml** - Builds init-utilities image on push to `Dockerfiles/init-utilities/`
- **actions-runner-*.yml** - Builds custom GitHub runner image

## Development Notes

### Chart versioning
Update `Chart.yaml` version before committing. Uses semantic versioning.

### Private registry
Images pushed to `docker-private.build.krystof.io/krystof-io/`

### OpenTelemetry versions
Managed in `Dockerfiles/init-utilities/versions.yaml`. Current default: 2.21.0
