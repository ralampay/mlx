from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from mlx.core.exceptions import MLXUserError


class ContainerCommandRunner(Protocol):
    def login(self, *, username: str, password: str, endpoint: str) -> None: ...
    def build(self, *, dockerfile: Path, image_uri: str, context: Path) -> None: ...
    def push(self, image_uri: str) -> None: ...


class DockerCommandRunner:
    def login(self, *, username: str, password: str, endpoint: str) -> None:
        subprocess.run(
            ["docker", "login", "--username", username, "--password-stdin", endpoint],
            input=password,
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def build(self, *, dockerfile: Path, image_uri: str, context: Path) -> None:
        subprocess.run(
            ["docker", "build", "-f", str(dockerfile), "-t", image_uri, str(context)],
            check=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
        )

    def push(self, image_uri: str) -> None:
        subprocess.run(["docker", "push", image_uri], check=True, stdout=sys.stderr, stderr=sys.stderr)


def source_digest(package_root: Path, dockerfile: Path) -> str:
    paths = sorted(package_root.rglob("*.py"))
    paths.extend([package_root / ".dockerignore", dockerfile])
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(package_root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


class PublishSageMakerImage:
    def __init__(
        self,
        *,
        ecr,
        repository_name: str,
        repository_uri: str,
        package_root: Path,
        dockerfile: Path,
        rebuild: bool,
        client_error: type[Exception],
        boto_error: type[Exception],
        command_runner: ContainerCommandRunner | None = None,
    ) -> None:
        self.ecr = ecr
        self.repository_name = repository_name
        self.repository_uri = repository_uri
        self.package_root = package_root
        self.dockerfile = dockerfile
        self.rebuild = rebuild
        self.client_error = client_error
        self.boto_error = boto_error
        self.command_runner = command_runner or DockerCommandRunner()

    def execute(self) -> str:
        digest_prefix = source_digest(self.package_root, self.dockerfile)[:16]
        tag = f"source-{digest_prefix}"
        if self.rebuild:
            tag = f"source-{digest_prefix}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
        image_uri = f"{self.repository_uri}:{tag}"
        existing = self._existing_digest(tag)
        if existing is not None and not self.rebuild:
            return f"{self.repository_uri}@{existing}"
        if not self.dockerfile.is_file():
            raise MLXUserError(
                f"Packaged SageMaker Dockerfile not found: {self.dockerfile}. "
                "Reinstall MLX or provide aws.image_uri."
            )
        try:
            auth = self.ecr.get_authorization_token()["authorizationData"][0]
            username, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
            self.command_runner.login(username=username, password=password, endpoint=auth["proxyEndpoint"])
            self.command_runner.build(dockerfile=self.dockerfile, image_uri=image_uri, context=self.package_root)
            self.command_runner.push(image_uri)
        except FileNotFoundError as exc:
            raise MLXUserError(
                "Docker is required to publish the SageMaker image, but the docker executable was not found."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() if isinstance(exc.stderr, str) else str(exc)
            raise MLXUserError(f"Unable to build or push the SageMaker image: {detail}") from exc
        return f"{self.repository_uri}@{self._existing_digest(tag, required=True)}"

    def _existing_digest(self, tag: str, *, required: bool = False) -> str | None:
        try:
            response = self.ecr.describe_images(
                repositoryName=self.repository_name,
                imageIds=[{"imageTag": tag}],
            )
        except (self.client_error, self.boto_error) as exc:
            response = getattr(exc, "response", {})
            if not required and response.get("Error", {}).get("Code") == "ImageNotFoundException":
                return None
            action = "published image digest lookup" if required else "image lookup"
            raise MLXUserError(f"AWS {action} failed: {exc}") from exc
        return str(response["imageDetails"][0]["imageDigest"])


__all__ = ["ContainerCommandRunner", "DockerCommandRunner", "PublishSageMakerImage", "source_digest"]
