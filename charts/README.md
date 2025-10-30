# Helm Charts

## spring-boot-app

Generic Helm chart for deploying Spring Boot applications.

See [spring-boot-app/README.md](spring-boot-app/README.md) for details.

## Development

### Creating a New Chart

```bash
cd charts
helm create my-new-chart
```

### Chart Versioning

Follow semantic versioning:
- MAJOR version when you make incompatible changes
- MINOR version when you add functionality in a backwards compatible manner
- PATCH version when you make backwards compatible bug fixes

Update version in `Chart.yaml` before committing.

### Testing

```bash
helm lint charts/spring-boot-app

helm template test charts/spring-boot-app --debug -f ./charts/spring-boot-app/sample-values.yaml
```