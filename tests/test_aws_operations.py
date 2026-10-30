from io import BytesIO
from types import SimpleNamespace

import pytest

from mlx.core.aws.operations import FindActiveTrainingAttempt, S3JsonStore, SubmitTrainingJob
from mlx.core.exceptions import MLXUserError


class ClientError(Exception):
    def __init__(self, code, message=""):
        self.response = {"Error": {"Code": code, "Message": message}}
        super().__init__(message)


class BotoError(Exception):
    pass


ERRORS = dict(client_error=ClientError, boto_error=BotoError)


@pytest.mark.parametrize("retry", [True, False])
def test_job_submission_retries_only_role_propagation(retry):
    calls, waits = [], []
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ClientError("ValidationException" if retry else "AccessDenied", "role missing")
        return {"TrainingJobArn": "job"}
    command = SubmitTrainingJob(SimpleNamespace(create_training_job=create), {"TrainingJobName": "test"},
                                wait=waits.append, **ERRORS)
    if retry:
        assert command.execute() == {"TrainingJobArn": "job"}
        assert waits == [2]
        assert calls[0] == calls[1]
    else:
        with pytest.raises(MLXUserError) as failure:
            command.execute()
        assert isinstance(failure.value.__cause__, ClientError)
        assert not waits


def test_job_retry_exhaustion_is_bounded():
    waits = []
    def create(**kwargs):
        raise ClientError("ValidationException", "cannot assume role")
    with pytest.raises(MLXUserError):
        SubmitTrainingJob(SimpleNamespace(create_training_job=create), {}, wait=waits.append, **ERRORS).execute()
    assert waits == [2, 4, 8, 16]


@pytest.mark.parametrize("body", [b"broken", b"[]"])
def test_manifest_required_and_optional_corruption_policy(body):
    store = S3JsonStore(SimpleNamespace(get_object=lambda **kw: {"Body": BytesIO(body)}), **ERRORS)
    assert store.get("s3://bucket/key", optional=True) is None
    with pytest.raises(MLXUserError):
        store.get("s3://bucket/key")


def test_optional_manifest_does_not_hide_access_errors():
    def get(**kwargs):
        raise ClientError("AccessDenied")
    with pytest.raises(MLXUserError):
        S3JsonStore(SimpleNamespace(get_object=get), **ERRORS).get("s3://bucket/key", optional=True)


def test_active_attempt_search_uses_injected_identity_and_excludes_current():
    described = []
    pages = SimpleNamespace(paginate=lambda **kw: [{"TrainingJobSummaries": [
        {"TrainingJobName": "current"}, {"TrainingJobName": "other"}]}])
    def describe(name):
        described.append(name)
        return {"HyperParameters": {"custom_id": "123"}}
    result = FindActiveTrainingAttempt(SimpleNamespace(get_paginator=lambda name: pages),
        describe=describe, prefix="prefix", identity_key="custom_id", identity="123",
        excluding="current", **ERRORS).execute()
    assert result == "other"
    assert described == ["other"]
