FROM nousresearch/hermes-agent:v2026.9.24@sha256:93b4e2877a2f48f4474b6dd2b99386ae32c4c552d32639df1a869b3f13c50b5a

LABEL org.opencontainers.image.title="mosaic-live-hermes" \
      org.opencontainers.image.description="Pinned Hermes runtime for Agent Mosaic" \
      org.opencontainers.image.version="0.21.5" \
      org.opencontainers.image.revision="f97608f178d1ffeca59860195ab7da295f7c8e5f" \
      org.opencontainers.image.source="https://github.com/NousResearch/hermes-agent" \
      org.mosaic-project.live-agent.harness="hermes" \
      org.mosaic-project.live-agent.base-index-digest="sha256:fca358f12efd65bfaaca05884166f15c0e2788375ca30d77061ac1ebc96452b7" \
      org.mosaic-project.live-agent.tirith-version="0.4.2" \
      org.mosaic-project.live-agent.tirith-archive-sha256="c550b1bfb0c8c872ab3421cd6ef756f260f7cf4981a18cedd49f141fa2d77569"

RUN /opt/hermes/.venv/bin/python <<'PY'
import hashlib
import io
import pathlib
import shutil
import tarfile
import urllib.request

url = "https://github.com/sheeki03/tirith/releases/download/v0.4.2/tirith-aarch64-unknown-linux-gnu.tar.gz"
expected = "c550b1bfb0c8c872ab3421cd6ef756f260f7cf4981a18cedd49f141fa2d77569"
with urllib.request.urlopen(url, timeout=60) as response:
    body = response.read()
if hashlib.sha256(body).hexdigest() != expected:
    raise ValueError("Tirith release archive checksum mismatch")
with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
    members = [m for m in archive.getmembers() if pathlib.PurePosixPath(m.name).name == "tirith"]
    if len(members) != 1 or not members[0].isfile():
        raise ValueError("Tirith archive must contain one regular executable")
    source = archive.extractfile(members[0])
    if source is None:
        raise ValueError("Tirith executable missing")
    dest = pathlib.Path("/usr/local/bin/tirith")
    with source, dest.open("xb") as output:
        shutil.copyfileobj(source, output)
    dest.chmod(0o755)
PY

RUN mkdir -p /workspace/docs \
  && chown hermes:hermes /workspace \
  && cat > /opt/data/config.yaml <<'YAML'
_config_version: 46

model:
  provider: "openrouter"
  default: "openai/gpt-5.6-sol"

agent:
  reasoning_effort: "medium"

provider_routing:
  only: ["openai"]
  order: ["openai"]
  require_parameters: true

delegation:
  fallback_providers: []

terminal:
  backend: "local"
YAML

WORKDIR /workspace

ENTRYPOINT ["/opt/hermes/docker/entrypoint-dispatch.sh"]
