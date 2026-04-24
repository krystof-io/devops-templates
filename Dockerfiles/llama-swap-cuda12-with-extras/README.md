## Manual build and push

Log in to the registry:

```bash
docker login <repohost>
```

Build:

```bash
docker build --progress=plain --build-arg CUDA_VERSION="12.8.1" \
  -t <repohost>/krystof-io/llama-swap-cuda12-with-extras:latest \
  .
```

Push:

```bash
docker push <repohost>/krystof-io/llama-swap-cuda12-with-extras:latest
```
