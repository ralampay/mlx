"""Shared AWS transport operations; training policies remain mode-owned."""
from __future__ import annotations

import json
from typing import Mapping
from urllib.parse import urlparse

from mlx.core.exceptions import MLXUserError


class SubmitTrainingJob:
    def __init__(self, client, request, *, client_error, boto_error, wait,
                 action="training job submission"):
        self.client, self.request = client, request
        self.client_error, self.boto_error = client_error, boto_error
        self.wait, self.action = wait, action

    def execute(self):
        delays = (0, 2, 4, 8, 16)
        for index, delay in enumerate(delays):
            if delay:
                self.wait(delay)
            try:
                return self.client.create_training_job(**self.request)
            except self.client_error as exc:
                error = exc.response.get("Error", {})
                message = str(error.get("Message", "")).lower()
                retry = error.get("Code") == "ValidationException" and (
                    "role" in message or "assume" in message
                )
                if not retry or index == len(delays) - 1:
                    raise MLXUserError(f"AWS {self.action} failed: {exc}") from exc
            except self.boto_error as exc:
                raise MLXUserError(f"AWS {self.action} failed: {exc}") from exc
        raise MLXUserError("AWS training job submission failed after IAM propagation retries.")


class S3JsonStore:
    def __init__(self, client, *, client_error, boto_error):
        self.client = client
        self.client_error, self.boto_error = client_error, boto_error

    @staticmethod
    def _location(uri):
        parsed = urlparse(uri)
        return {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}

    def put(self, uri, value, *, action="manifest upload"):
        try:
            self.client.put_object(**self._location(uri),
                                   Body=json.dumps(value, sort_keys=True).encode("utf-8"),
                                   ContentType="application/json")
        except (self.client_error, self.boto_error) as exc:
            raise MLXUserError(f"AWS {action} failed: {exc}") from exc

    def get(self, uri, *, optional=False, action="manifest retrieval", label="AWS run manifest"):
        try:
            body = self.client.get_object(**self._location(uri))["Body"].read()
        except self.client_error as exc:
            if optional and exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise MLXUserError(f"AWS {action} failed: {exc}") from exc
        except self.boto_error as exc:
            raise MLXUserError(f"AWS {action} failed: {exc}") from exc
        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            if optional:
                return None
            raise MLXUserError(f"AWS {action} failed: {exc}") from exc
        if isinstance(value, Mapping):
            return value
        if optional:
            return None
        raise MLXUserError(f"{label} is invalid: {uri}")


class FindActiveTrainingAttempt:
    def __init__(self, client, *, describe, prefix, identity_key, identity, excluding,
                 client_error, boto_error, action="active training attempt lookup"):
        self.client, self.describe = client, describe
        self.prefix, self.identity_key, self.identity = prefix, identity_key, identity
        self.excluding = excluding
        self.client_error, self.boto_error, self.action = client_error, boto_error, action

    def execute(self):
        try:
            paginator = self.client.get_paginator("list_training_jobs")
            for status in ("InProgress", "Stopping"):
                for page in paginator.paginate(StatusEquals=status, NameContains=self.prefix):
                    for summary in page.get("TrainingJobSummaries", []):
                        name = summary["TrainingJobName"]
                        if name == self.excluding:
                            continue
                        if self.describe(name).get("HyperParameters", {}).get(self.identity_key) == self.identity:
                            return name
        except (self.client_error, self.boto_error) as exc:
            raise MLXUserError(f"AWS {self.action} failed: {exc}") from exc
        return None
