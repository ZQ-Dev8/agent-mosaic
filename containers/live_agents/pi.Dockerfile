FROM node:22.19.0-bookworm-slim

LABEL org.opencontainers.image.title="mosaic-live-pi" \
      org.opencontainers.image.description="Pinned Pi runtime for Agent Mosaic" \
      org.opencontainers.image.version="1.0.4" \
      org.opencontainers.image.revision="7c10bd4337495ee613f2224843ecdf349b80d1df" \
      org.opencontainers.image.source="https://www.npmjs.com/package/@earendil-works/pi-coding-agent" \
      org.mosaic-project.live-agent.harness="pi" \
      org.mosaic-project.live-agent.package="@earendil-works/pi-coding-agent@1.0.4" \
      org.mosaic-project.live-agent.package-integrity="sha512-+956nfMFHr5lDUVY/2Q4k+YzojzBuCaBXFgj0eSlXVGr7QVliVddKdc1Pz6yVg1dOlJQmb67doOVrlMsIcIdaw=="

RUN apt-get update \
  && apt-get install -y --no-install-recommends bash ca-certificates git ripgrep \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.0.4

RUN mkdir -p /root/.pi/agent /workspace/docs \
  && cat > /root/.pi/agent/models.json <<'JSON'
{
  "providers": {
    "openrouter": {
      "modelOverrides": {
        "openai/gpt-5.6-sol": {
          "compat": {
            "openRouterRouting": {
              "order": ["openai"],
              "only": ["openai"],
              "allow_fallbacks": false,
              "require_parameters": true
            }
          }
        }
      }
    }
  }
}
JSON

WORKDIR /workspace
ENTRYPOINT ["pi"]
