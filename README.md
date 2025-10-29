# DevOps Templates

Reusable Helm charts and Dockerfiles for home lab deployments.

## Helm Charts

This repository serves as a Helm chart repository via GitHub Pages.

### Add the repository:

```bash
helm repo add devops-templates https://krystof-io.github.io/devops-templates
helm repo update
```

### Available Charts:

- **spring-boot-app** - Generic Spring Boot application deployment chart

### Using with Flux:

```yaml
apiVersion: source.toolkit.fluxcd.io/v1beta2
kind: HelmRepository
metadata:
  name: devops-templates
  namespace: flux-system
spec:
  interval: 5m
  url: https://YOUR_USERNAME.github.io/devops-templates/
```

## Dockerfiles

Reference Dockerfiles for building container images.

## Development

### Testing Charts Locally

```bash
# Lint the chart
helm lint charts/spring-boot-app

# Test template rendering
helm template test charts/spring-boot-app

# Install locally
helm install test ./charts/spring-boot-app \
  --set image.repository=myapp \
  --set image.tag=latest

helm template spring-boot-hello . -f ./sample-values.yaml --namespace kio-dev > wow.yaml

```

### Publishing Charts

Charts are automatically published to GitHub Pages when pushed to main branch.
The GitHub Actions workflow handles packaging and indexing.